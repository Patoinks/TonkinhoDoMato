"""Training the emergent King Safety model (v4) from checkmate alone.

For a position P and the king of colour c (the defender), the attacker is c's
opponent. The attack game is ordinary legal chess; the attacker wins only by
checkmating c's king. Material, checks, sacrifices: nothing is scored except
that consequence. With U = 1 - KS:

    defender checkmated                       U = 1
    game over otherwise (stalemate, draw,
      the attacker itself mated)              U = 0
    attacker to move    U = max over its moves of  1 if the move mates,
                                                   else gamma * U(after it)
    defender to move    U = min over its moves of  U(after it)

so U = gamma ** (n - 1) when the attacker needs n moves to force mate.

Targets for one (P, c):
  1. exact: a rules-only forced-mate search (C++) up to `mate_moves` attacker
     moves. A mate found gives U exactly; a complete search without mate
     proves n > mate_moves, i.e. U <= gamma ** mate_moves.
  2. beyond that horizon: every legal continuation two plies deep (attacker
     and defender), the leaves valued by a lagged copy of the model
     (bootstrapping, as in value iteration), capped by the exact bound.
Positions come from real TonkAI games and from adversarial attack/defence
episodes in which the attacker plays the moves the model thinks hurt the
king most and the defender the ones it thinks protect it most.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from multiprocessing import get_context
import pickle
import time

import chess
import chess.pgn
import numpy as np
import torch
from torch.nn import functional as F

from . import fast_rules
from .encoding import pack_many, unpack_many
from .king_safety_model import KingSafetyModel, KingSafetyNet


@dataclass
class KSConfig:
    gamma: float = 0.8                 # each attacker move needed keeps (1 - gamma) of the danger away
    mate_moves: int = 3                # exact forced-mate horizon, in attacker moves
    node_limit: int = 150_000          # per root; a stopped search keeps only what it proved
    hidden: int = 256
    lr: float = 1e-3
    batch: int = 256
    buffer: int = 200_000              # training samples kept (packed)
    roots_per_iteration: int = 256     # each gives two samples: both kings
    steps_per_iteration: int = 150
    target_sync: int = 4               # iterations between copies of the bootstrap model
    episodes_per_iteration: int = 32
    episode_plies: int = 16
    explore: float = 0.1               # random moves in attack/defence episodes
    threat_explore: float = 0.3        # attacker moves spent trying moves that CREATE a mate threat
    threat_moves: int = 1              # threat = mate within this many moves if the defender passed
    explored_pool: int = 100_000       # positions reached by those episodes
    chunk: int = 32                    # roots per worker task
    seed: int = 7


# ------------------------------------------------------------ positions

def fens_from_checkpoint(path, limit=None) -> list[str]:
    """Every position of the real games stored in a V3 checkpoint's replay."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    fens = []
    for row in state["replay"]["episodes"]:
        if row["opponent"].startswith("counterfactual:"):
            continue
        board = chess.Board(row["initial_fen"] or chess.STARTING_FEN)
        fens.append(board.fen())
        for action in row["actions"]:
            board.push_uci(action)
            fens.append(board.fen())
    fens = list(dict.fromkeys(fens))
    return fens[:limit] if limit else fens


def fens_from_pgn(paths, limit=None) -> list[str]:
    fens = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            while (game := chess.pgn.read_game(stream)) is not None:
                board = game.board()
                fens.append(board.fen())
                for move in game.mainline_moves():
                    board.push(move)
                    fens.append(board.fen())
    fens = list(dict.fromkeys(fens))
    return fens[:limit] if limit else fens


# -------------------------------------------------------------- targets

def _fast(fen):
    return fast_rules.board(chess.Board(fen))


def search_targets(items, model: KingSafetyModel, cfg: KSConfig):
    """[(fen, defender_white)] -> (rows [n,782], U targets, exact n found, proven)."""
    rows, targets, founds, provens = [], [], [], []
    leaf_rows, leaf_meta, offset = [], [], 0
    for fen, defender in items:
        fast = _fast(fen)
        rows.append(fast.encode_here(defender))
        status = fast.ks_status(defender)
        if status:
            targets.append(1.0 if status == 1 else 0.0)
            founds.append(0 if status == 2 else -1)       # -1: already mated
            provens.append(cfg.mate_moves if status == 2 else 0)
            continue
        found, proven, _ = fast.forced_mate(not defender, cfg.mate_moves, cfg.node_limit)
        founds.append(found)
        provens.append(proven)
        if found:
            targets.append(cfg.gamma ** (found - 1))
            continue
        first, parent, leaf_status, leaf_row, leaves, count = fast.expand2(defender)
        attacker_to_move = chess.Board(fen).turn != defender
        leaf_meta.append((len(targets), first, parent, leaf_status, leaf_row, offset, count, attacker_to_move, proven))
        leaf_rows.append(leaves[:count])
        offset += count
        targets.append(None)
    # One forward pass for every leaf of the chunk.
    all_leaves = np.concatenate([r for r in leaf_rows if len(r)]) if any(len(r) for r in leaf_rows) else \
        np.zeros((0, 782), dtype=np.float32)
    u_all = 1.0 - model.ks_rows(all_leaves) if len(all_leaves) else np.zeros(0, dtype=np.float32)
    for index, first, parent, leaf_status, leaf_row, offset, count, attacker_to_move, proven in leaf_meta:
        u_net = u_all[offset:offset + count]
        first = np.asarray(first, dtype=np.int64)
        parent = np.asarray(parent, dtype=np.int64)
        leaf_status = np.asarray(leaf_status, dtype=np.int64)
        leaf_row = np.asarray(leaf_row, dtype=np.int64)
        leaf_u = np.where(leaf_status == 1, 1.0, 0.0)
        open_ = leaf_row >= 0
        leaf_u[open_] = u_net[leaf_row[open_]]
        if attacker_to_move:
            worst = np.full(len(first), np.inf)
            np.minimum.at(worst, parent, leaf_u)             # the defender's best reply
            per_move = np.where(first == 1, 1.0, np.where(first == 2, 0.0, cfg.gamma * worst))
            u = float(per_move.max())
        else:
            leaf_value = np.where(leaf_status == 1, 1.0, np.where(leaf_status == 2, 0.0, cfg.gamma * leaf_u))
            best = np.full(len(first), -np.inf)
            np.maximum.at(best, parent, leaf_value)          # the attacker's best follow-up
            per_move = np.where(first == 2, 0.0, best)
            u = float(per_move.min())
        if proven:                                           # exact: no mate within `proven` moves
            u = min(u, cfg.gamma ** proven)
        targets[index] = u
    return np.stack(rows).astype(np.float32), np.asarray(targets, dtype=np.float32), \
        np.asarray(founds, dtype=np.int16), np.asarray(provens, dtype=np.int16)


# ------------------------------------------------------------ episodes

def mating_moves(fast, attacker, cfg):
    """Attacker moves on a shortest forced mate (within mate_moves), or []."""
    found, _, _ = fast.forced_mate(attacker, cfg.mate_moves, cfg.node_limit)
    if not found:
        return []
    moves = []
    for code in fast.legal_moves():
        fast.push(code)
        if fast.ks_status(not attacker) == 1:
            moves.append(code)
        elif found > 1:
            after, _, _ = fast.forced_mate(attacker, found - 1, cfg.node_limit)
            if after:
                moves.append(code)
        fast.pop()
    return moves


def threat_moves(fast, cfg):
    """Moves that CREATE a mate threat: were the defender to pass, the attacker
    would force mate within threat_moves (only asked when no forced mate is
    already there, else every move would 'threaten'). Rules and mate only."""
    codes, threat, check = fast.threats(cfg.threat_moves, cfg.node_limit)
    return [code for code, depth in zip(codes, threat) if depth]


def attack_episode(fen, defender, model: KingSafetyModel, cfg: KSConfig, rng, counts=None):
    """Attack/defence exploration on `defender`'s king.

    The attacker plays a forced mate when the rules show one; otherwise, part
    of the time, a move that creates a new mate threat; otherwise what the
    model thinks hurts the king most. The defender plays what the model thinks
    protects it most. Both sometimes play at random (`explore`).
    Returns (visited fens, True if the defender was checkmated)."""
    board = chess.Board(fen)
    visited = [board.fen()]
    counts = counts if counts is not None else {}
    for _ in range(cfg.episode_plies):
        fast = _fast(board.fen())
        status = fast.ks_status(defender)
        if status:
            return visited, status == 1
        moves = list(board.legal_moves)
        codes = [fast_rules.code(move) for move in moves]
        u = 1.0 - model.ks_rows(fast.encode_after(codes, defender)[:len(codes)])
        for i, code in enumerate(codes):
            fast.push(code)
            status = fast.ks_status(defender)
            fast.pop()
            if status:
                u[i] = 1.0 if status == 1 else 0.0
        attacking = board.turn != defender
        choice = None
        if attacking:
            mating = mating_moves(fast, not defender, cfg)
            if mating:                                        # the rules already show a mate
                choice = codes.index(int(rng.choice(mating)))
                counts["mating"] = counts.get("mating", 0) + 1
            elif rng.random() < cfg.threat_explore:
                threats = threat_moves(fast, cfg)
                if threats:                                   # try creating a new mate threat
                    choice = codes.index(int(rng.choice(threats)))
                    counts["threat"] = counts.get("threat", 0) + 1
        if choice is None and rng.random() < cfg.explore:
            choice = int(rng.integers(len(moves)))
        if choice is None:
            target = u.max() if attacking else u.min()
            choice = int(rng.choice(np.flatnonzero(u == target)))
        counts["attacker_moves" if attacking else "defender_moves"] = \
            counts.get("attacker_moves" if attacking else "defender_moves", 0) + 1
        board.push(moves[choice])
        visited.append(board.fen())
    return visited, _fast(board.fen()).ks_status(defender) == 1


# ------------------------------------------------------------- workers

_worker = {}


def _model_from(version, blob, hidden):
    if _worker.get("version") != version:
        net = KingSafetyNet(hidden)
        net.load_state_dict(pickle.loads(blob))
        _worker.update(version=version, model=KingSafetyModel(net))
    return _worker["model"]


def _task_targets(version, blob, cfg, items):
    torch.set_num_threads(1)
    return search_targets(items, _model_from(version, blob, cfg.hidden), cfg)


def _task_episodes(version, blob, cfg, starts, seed):
    torch.set_num_threads(1)
    model = _model_from(version, blob, cfg.hidden)
    rng = np.random.default_rng(seed)
    fens, mates, counts = [], 0, {}
    for fen, defender in starts:
        visited, mated = attack_episode(fen, defender, model, cfg, rng, counts)
        fens.extend(visited)
        mates += int(mated)
    return fens, mates, len(starts), counts


# ------------------------------------------------------------- trainer

class KingSafetyTrainer:
    def __init__(self, cfg: KSConfig, roots, model: KingSafetyModel | None = None, workers: int = 8):
        self.cfg = cfg
        self.model = model or KingSafetyModel(KingSafetyNet(cfg.hidden))
        self.model.net.train()
        self.target = deepcopy(self.model.net).eval()
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=cfg.lr)
        self.roots = list(roots)
        self.explored = deque(maxlen=cfg.explored_pool)
        self.packed = deque(maxlen=cfg.buffer)
        self.labels = deque(maxlen=cfg.buffer)
        self.rng = np.random.default_rng(cfg.seed)
        self.iteration = self.samples = self.version = 0
        self.workers = workers
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) if workers > 1 else None

    def close(self):
        if self.pool:
            self.pool.shutdown(cancel_futures=True)

    def _blob(self):
        return pickle.dumps({k: v.detach().cpu() for k, v in self.target.state_dict().items()})

    def _run(self, fn, jobs):
        if self.pool is None:
            return [fn(*job) for job in jobs]
        return list(self.pool.map(fn, *zip(*jobs)))

    def _sample_roots(self, n):
        real = int(n * (0.5 if self.explored else 1.0))
        picks = [self.roots[int(i)] for i in self.rng.integers(len(self.roots), size=real)]
        if self.explored:
            picks += [self.explored[int(i)] for i in self.rng.integers(len(self.explored), size=n - real)]
        return picks

    def iterate(self) -> dict:
        cfg, started = self.cfg, time.perf_counter()
        blob = self._blob()
        # 1. Adversarial attack/defence episodes grow the pool of attack positions.
        starts = [(fen, bool(self.rng.integers(2))) for fen in self._sample_roots(cfg.episodes_per_iteration)]
        per = max(1, len(starts) // max(1, self.workers))
        jobs = [(self.version, blob, cfg, starts[i:i + per], int(self.rng.integers(2**31)))
                for i in range(0, len(starts), per)]
        mates = episodes = 0
        moves = {}
        for fens, mated, count, counts in self._run(_task_episodes, jobs):
            self.explored.extend(fens)
            mates, episodes = mates + mated, episodes + count
            for key, value in counts.items():
                moves[key] = moves.get(key, 0) + value
        # 2. Targets for both kings of fresh roots.
        roots = self._sample_roots(cfg.roots_per_iteration)
        items = [(fen, colour) for fen in roots for colour in (True, False)]
        jobs = [(self.version, blob, cfg, items[i:i + cfg.chunk]) for i in range(0, len(items), cfg.chunk)]
        rows, targets, founds, provens = [], [], [], []
        for r, t, f, p in self._run(_task_targets, jobs):
            rows.append(r), targets.append(t), founds.append(f), provens.append(p)
        rows, targets = np.concatenate(rows), np.concatenate(targets)
        founds, provens = np.concatenate(founds), np.concatenate(provens)
        self.packed.extend(pack_many(rows))
        self.labels.extend((1.0 - targets).tolist())            # stored as KS
        self.samples += len(rows)
        # 3. Learn.
        losses = []
        self.model.net.train()
        for _ in range(cfg.steps_per_iteration):
            idx = self.rng.integers(len(self.labels), size=min(cfg.batch, len(self.labels)))
            x = torch.from_numpy(unpack_many(np.stack([self.packed[i] for i in idx])))
            y = torch.tensor([self.labels[i] for i in idx], dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(self.model.net(x), y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        self.model.net.eval()
        self.iteration += 1
        if self.iteration % cfg.target_sync == 0:
            self.target.load_state_dict(self.model.net.state_dict())
            self.version += 1
        predicted = self.model.ks_rows(rows)
        mate = founds > 0
        return {"iteration": self.iteration, "samples": self.samples, "loss": float(np.mean(losses)),
                "seconds": time.perf_counter() - started, "target_ks_mean": float(1 - targets.mean()),
                "exact_mates": float(mate.mean()), "mated_now": float((founds < 0).mean()),
                "proven_safe_horizon": float((provens >= cfg.mate_moves).mean()),
                "ks_on_forced_mates": float(predicted[mate].mean()) if mate.any() else None,
                "ks_on_the_rest": float(predicted[~mate & (founds >= 0)].mean()) if (~mate).any() else None,
                "episode_mate_rate": mates / max(1, episodes), "explored_pool": len(self.explored),
                "attacker_threat_moves": moves.get("threat", 0) / max(1, moves.get("attacker_moves", 0)),
                "attacker_mating_moves": moves.get("mating", 0) / max(1, moves.get("attacker_moves", 0)),
                "buffer": len(self.labels)}

    def save(self, path, history):
        self.model.save(path, config=asdict(self.cfg), iterations=self.iteration, samples=self.samples,
                        history=history[-200:])
