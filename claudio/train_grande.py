"""Training of the grown Claudio (claudio_net.grow): actor processes play Claudio
against an opponent ladder, the learner runs on the GPU.

Opponents are only opponents (nothing is learned from their evaluations):
TinyOpponent, Fairy-Stockfish / Stockfish at a nominal Elo, or a frozen copy of
the starting network. A ladder plays them in order, e.g.
    --ladder tiny:2500,fairy700:1000,fairy900:1000,...,fairy1800:1000

Objectives (all organic - the results of its own games, the rules, checkmates):
- shaped: survival + the owner's victory measure (TonkAI's classic objective);
- victory: win 1, loss 0, a draw --draw-value (the owner: no survival module,
  draws worth nothing -> --draw-value 0). Claudio also decides with that draw value.
Optional: intuition <- calculation (--choice-weight), off by default.

--learn review (the owner's rule): each game is reviewed afterwards with Claudio's own
calculation (claudio/review.py). A position's value target is its forced material
state (0.5 + 0.5 tanh(balance / 5)) up to the decisive moment, and the game's result
only after it; each of Claudio's moves that won material is pushed above the other
moves of its position ("good"), each that lost material below them ("bad").

--learn notes: the same review, but ONLY its move notes teach (a good move above the
other moves of its position, a bad one below), from Claudio's own reviewed games. The
value's level is not trained on material (the calculation already counts material; the
network keeps what it knows beyond it) and the old replay is not replayed.

--colour white|black: Claudio always plays that colour (two separate AIs, joined
for evaluations as "white.pt+black.pt").

--learn winners (the owner's rule): learn only from decided games, from the side
that WON, whoever it was: at the winner's turns the move it played is pushed
above that position's other moves ("this is even better"). Nothing is labelled
bad, losses teach no "this loses", draws teach nothing, and the value's absolute
level is not trained (a ranking within each position cannot make it pessimistic).
When the opponent wins, its moves are learned too (with an engine opponent that
means learning from the engine's winning moves).

Progress is written for the training tracker app (history/<run>/launch.json +
history/<run>.jsonl, the same records train_cognitive.py writes).

    python claudio/train_grande.py --start claudio/claudio-grande-g0.pt --out-dir claudio/train-ladder \\
        --ladder tiny:2500,fairy700:1000,fairy900:1000 --objective victory --draw-value 0
    (stop early: create <out-dir>/STOP)
"""
import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from datetime import datetime, timezone
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FAIRY, ROOT, STOCKFISH  # noqa: E402  (frozen base code first)

TIMDGEI = ROOT / "checkpoints/obj-max-continuo/latest.pt"
ENGINES = {"sf": STOCKFISH, "fairy": FAIRY}
_w = {}


# ------------------------------------------------------------------ opponents
def parse_stage(text):
    """'tiny' | 'self' | 'fairy700' | 'sf1800' -> (kind, elo or None, label, tracker kind)."""
    text = text.strip().lower()
    if text == "tiny":
        return ("tiny", None, "TinyOpponent", "tiny")
    if text == "self":
        return ("self", None, "frozen self", "self")
    for prefix, name in (("fairy", "Fairy-Stockfish"), ("sf", "Stockfish")):
        if text.startswith(prefix) and text[len(prefix):].isdigit():
            elo = int(text[len(prefix):])
            return (prefix, elo, f"{name} {elo}", f"{name} {elo}")
    raise ValueError(f"unknown opponent {text!r}")


def parse_ladder(text):
    stages = []
    for part in text.split(","):
        name, _, games = part.rpartition(":")
        stages.append((parse_stage(name), int(games)))
    return stages


# --------------------------------------------------------------------- learner
def make_trainer_class():
    import chess
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tonkai import fast_rules
    from tonkai.cognitive_trainer import CognitiveTrainer
    from tonkai.encoding import unpack_many

    class ClaudioTrainer(CognitiveTrainer):
        choice_weight = 0.0
        choice_tau = 0.05
        loss_rows = 8           # sequences per step for the move-by-move losses (python-heavy)

        def _subset(self, batch, x, memories):
            rows = np.sort(self.rng.choice(len(x), min(self.loss_rows, len(x)), replace=False))
            sub = dict(batch, episodes=[batch["episodes"][i] for i in rows],
                       state_indices=batch["state_indices"][rows],
                       importance_weights=np.asarray(batch["importance_weights"])[rows])
            return sub, x[rows], memories[rows]

        def _reply_loss(self, batch, x, memories, offset, length):
            sub, sx, sm = self._subset(batch, x, memories)
            reply = self._move_loss(sub, sx, sm, offset, length, own=False)
            if self.review_weight:
                reply = reply + self.review_weight * self._review_loss(batch, x, memories, offset, length)
            if not self.choice_weight:
                return reply
            sub, sx, sm = self._subset(batch, x, memories)
            return reply + self.choice_weight * self._move_loss(sub, sx, sm, offset, length, own=True)

        def _codes(self, episode):
            """The episode's moves as C++ codes, converted once (the learner re-reads episodes often)."""
            cache = self.__dict__.setdefault("_code_cache", {})
            hit = cache.get(id(episode))
            if hit is None or hit[0] is not episode:
                hit = cache[id(episode)] = (episode, [fast_rules.code(chess.Move.from_uci(a)) for a in episode.actions])
                if len(cache) > 50_000:
                    cache.pop(next(iter(cache)))
            return hit[1]

        def _move_loss(self, batch, x, memories, offset, length, own):
            """own=False: TonkAI's prediction of the opponent's move (the base reply loss).
            own=True: intuition <- calculation on Claudio's own moves in its own games."""
            candidates, prefixes, labels, sizes, weights = [], [], [], [], []
            turns = x[:, offset:offset + length, 768].detach().cpu().numpy()
            for row, episode in enumerate(batch["episodes"]):
                if not episode.actions or (own and "(claudio)" not in (episode.opponent or "")):
                    continue
                indices = batch["state_indices"][row, offset:offset + length]
                side = turns[row] >= .5 if own else turns[row] < .5
                possible = np.flatnonzero((indices >= 0) & (indices < len(episode.actions)) & side)
                if not len(possible):
                    continue
                local = int(self.rng.choice(possible))
                index = int(indices[local])
                codes = self._codes(episode)
                fast = fast_rules.board(chess.Board(episode.initial_fen or chess.STARTING_FEN))
                fast.push_many(codes[:index])
                legal = list(fast.legal_moves())
                if own and len(legal) < 2:
                    continue
                white_to_move = fast.fen().split(" ", 2)[1] == "w"
                owner = white_to_move if own else not white_to_move
                labels.append(legal.index(codes[index]))
                candidates.append(fast_rules.encode_after_moves(fast, legal, owner))
                prefixes.append(memories[row, offset + local].expand(len(legal), -1))
                sizes.append(len(legal))
                weights.append(batch["importance_weights"][row])
            if not candidates:
                return x.new_zeros(())
            encoded = torch.from_numpy(np.concatenate(candidates)).to(self.device)
            context, _ = self.net.candidates(encoded, torch.cat(prefixes))
            if own:
                scores = (self.net.survival(context)[0].mean(-1) / self.choice_tau).split(sizes)
            else:
                scores = self.net.predict_logits(context).split(sizes)
            losses = torch.stack([F.cross_entropy(v[None], torch.tensor([label], device=self.device))
                                  for v, label in zip(scores, labels)])
            if own:
                self.last_choice_loss = float(losses.mean().detach())
            return (losses * losses.new_tensor(weights)).mean()

        review_weight = 0.0

        def _review_loss(self, batch, x, memories, offset, length):
            """Per-move notes from the game review: a good move above its alternatives, a bad one below."""
            labels_of = self.__dict__.setdefault("review_labels", {})
            candidates, prefixes, played, signs, sizes, weights, sizes_of_note = [], [], [], [], [], [], []
            for row, episode in enumerate(batch["episodes"]):
                notes = labels_of.get(hash(episode.actions))
                if not notes:
                    continue
                indices = batch["state_indices"][row, offset:offset + length]
                window = {int(index): local for local, index in enumerate(indices) if index >= 0}
                usable = [note for note in notes if note[0] in window]
                if not usable:
                    continue
                # heavier notes (bigger blunders / gains) are picked more often
                chances = np.array([note[2] if len(note) > 2 else 1.0 for note in usable], dtype=np.float64)
                ply, sign, *rest = usable[int(self.rng.choice(len(usable), p=chances / chances.sum()))]
                local = window[ply]
                codes = self._codes(episode)
                fast = fast_rules.board(chess.Board(episode.initial_fen or chess.STARTING_FEN))
                fast.push_many(codes[:ply])
                legal = list(fast.legal_moves())
                if len(legal) < 2:
                    continue
                sizes_of_note.append(rest[0] if rest else 1.0)
                owner = fast.fen().split(" ", 2)[1] == "w"
                played.append(legal.index(codes[ply]))
                signs.append(sign)
                candidates.append(fast_rules.encode_after_moves(fast, legal, owner))
                prefixes.append(memories[row, offset + local].expand(len(legal), -1))
                sizes.append(len(legal))
                weights.append(batch["importance_weights"][row])
            if not candidates:
                return x.new_zeros(())
            encoded = torch.from_numpy(np.concatenate(candidates)).to(self.device)
            context, _ = self.net.candidates(encoded, torch.cat(prefixes))
            scores = (self.net.survival(context)[0].mean(-1) / self.choice_tau).split(sizes)
            losses = []
            for v, index, sign in zip(scores, played, signs):
                logp = torch.log_softmax(v, 0)[index]
                # good: the played move above the others; bad: below them (-log(1 - p)).
                losses.append(-logp if sign > 0 else -torch.log1p(-logp.exp().clamp(max=1 - 1e-6)))
            losses = torch.stack(losses) * losses[0].new_tensor(sizes_of_note)
            self.last_choice_loss = float(losses.mean().detach())
            return (losses * losses.new_tensor(weights)).mean()

        def notes_update(self, replay):
            """One step on Claudio's reviewed games: only the review's move notes."""
            cfg = self.cfg
            if len(replay) < cfg.min_replay:
                return None
            batch = replay.sample(cfg.sequence_batch_size, cfg.burn_in, cfg.sequence_length, max(cfg.future_horizons))
            b, total, _ = batch["packed"].shape
            x = torch.from_numpy(unpack_many(batch["packed"].reshape(-1, batch["packed"].shape[-1]))
                                 .reshape(b, total, -1)).to(self.device)
            valid = torch.as_tensor(batch["mask"], device=self.device)
            offset, length = cfg.burn_in, cfg.sequence_length
            self.net.train()
            _, _, after = self._unroll(self.net, x, valid, offset, length, True, learner_window_only=True)
            loss = self._review_loss(batch, x, after, offset, length)
            if not loss.requires_grad:
                return None
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.steps += 1
            return float(loss.detach())

        def winner_update(self, replay, rows):
            """One step on the winners' games: at the winner's turns, its move above the others."""
            cfg = self.cfg
            if len(replay) < cfg.min_replay:
                return None
            batch = replay.sample(cfg.sequence_batch_size, cfg.burn_in, cfg.sequence_length, max(cfg.future_horizons))
            b, total, _ = batch["packed"].shape
            x = torch.from_numpy(unpack_many(batch["packed"].reshape(-1, batch["packed"].shape[-1]))
                                 .reshape(b, total, -1)).to(self.device)
            valid = torch.as_tensor(batch["mask"], device=self.device)
            offset, length = cfg.burn_in, cfg.sequence_length
            self.net.train()
            _, _, after = self._unroll(self.net, x, valid, offset, length, True, learner_window_only=True)
            usual, self.loss_rows = self.loss_rows, rows
            sub, sx, sm = self._subset(batch, x, after)
            self.loss_rows = usual
            loss = self._move_loss(sub, sx, sm, offset, length, own=True)
            if not loss.requires_grad:
                return None
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.steps += 1
            return float(loss.detach())

    return ClaudioTrainer


# ---------------------------------------------------------------------- actors
def _init_worker(out_dir, architecture, values, settings, start, learner_draw, objective, move_time):
    import psutil_free_priority  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from match_checkpoints import _historical_config
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.cognitive_trainer import CognitiveKnowledge
    from tonkai.king_safety_model import load_cached
    from claudio_agent import ClaudioConfig
    from claudio_net import build_net
    state = torch.load(TIMDGEI, map_location="cpu", weights_only=False)
    cfg = _historical_config(state)
    cfg.syzygy_path, cfg.tablebase_adjudicate = "", False
    net = build_net(architecture).eval().requires_grad_(False)
    values = {int(k): float(v) for k, v in values.items()}
    learner_cfg = ClaudioConfig(**settings)
    if learner_draw is not None:
        learner_cfg.draw_value = learner_draw
    _w.update(out_dir=Path(out_dir), net=net, values=values, ccfg=learner_cfg, frozen_cfg=ClaudioConfig(**settings),
              king=load_cached(cfg.king_safety_model), version=-1, shaped=objective == "shaped",
              proxy=CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder), kd_values=values),
              start=start, architecture=architecture, move_time=move_time, engine=None, engine_key=None, frozen=None)


def _opponent(stage, rng, seed, game):
    """A move function for this game's opponent (and the Claudio object when it is a frozen copy)."""
    import numpy as np
    from claudio_agent import Claudio
    kind, elo = stage[0], stage[1]
    if kind == "tiny":
        from tonkai.opponents import TinyOpponentAdapter
        return TinyOpponentAdapter(seed=int(rng.integers(2 ** 62))).choose, None
    if kind == "self":
        if _w["frozen"] is None:
            import torch
            from claudio_net import build_net
            frozen = build_net(_w["architecture"]).eval().requires_grad_(False)
            frozen.load_state_dict(torch.load(_w["start"], map_location="cpu", weights_only=False)["net"])
            _w["frozen"] = frozen
        rival = Claudio(_w["frozen"], _w["values"], _w["frozen_cfg"], rng=np.random.default_rng([seed, game, 2]),
                        king_model=_w["king"])
        return rival.choose, rival
    from tonkai.uci_opponent import UCIOpponent
    if _w["engine_key"] != (kind, elo):
        if _w["engine"] is not None:
            _w["engine"].close()
        _w["engine"], _w["engine_key"] = UCIOpponent(ENGINES[kind], elo, _w["move_time"]), (kind, elo)
    engine = _w["engine"]
    engine.reset()
    return (lambda board: engine.choose(board, rng)), None


def play(game, version, seed, stage, random_plies, max_plies, explore_plies, explore_temperature,
         colour="both", reviewed=False):
    import chess
    import numpy as np
    import torch
    from dataclasses import replace
    from tonkai import fast_rules
    from tonkai.encoding import pack_many
    from claudio_agent import Claudio

    net = _w["net"]
    if version != _w["version"]:
        net.load_state_dict(torch.load(_w["out_dir"] / f"actor-{version}.pt", map_location="cpu"))
        _w["version"] = version
    started = time.perf_counter()
    rng = np.random.default_rng([seed, game])
    me = {"white": chess.WHITE, "black": chess.BLACK}.get(colour, chess.WHITE if game % 2 == 0 else chess.BLACK)
    claudio = Claudio(net, _w["values"], _w["ccfg"], rng=np.random.default_rng([seed, game, 1]), king_model=_w["king"])
    claudio.spread = _w.get("spread", claudio.spread)          # its value-spread estimate carries across games
    answer, rival = _opponent(stage, rng, seed, game)
    board = chess.Board()
    while True:
        board.reset()
        for _ in range(random_plies):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        if not board.is_game_over(claim_draw=True):
            break
    # Opening exploration (AlphaZero style): Claudio (and a frozen copy) pick among
    # their own best moves with a little temperature for the first plies, then the best.
    players = [p for p in (claudio, rival) if p is not None]
    greedy = {id(p): p.cfg for p in players}
    exploring = {id(p): replace(p.cfg, temperature=explore_temperature) for p in players}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        player = claudio if board.turn == me else rival
        if player is None:
            board.push(answer(board))
            continue
        early = explore_temperature > 0 and len(board.move_stack) < explore_plies
        player.cfg = exploring[id(player)] if early else greedy[id(player)]
        board.push(player.choose(board))
    _w["spread"] = claudio.spread
    # Labels: every real position from Claudio's side (C++ encoding, one network pass), the
    # memory after each, and how surprising each opponent move was - all in one batch.
    fast = fast_rules.board(board.root())
    rows = [np.asarray(fast.encode_here(me), dtype=np.float32)]
    candidates, groups = [], []
    for ply, move in enumerate(board.move_stack):
        code = fast_rules.code(move)
        if fast.fen().split(" ", 2)[1] != ("w" if me else "b"):       # the opponent moved here
            codes = list(fast.legal_moves())
            candidates.append(fast_rules.encode_after_moves(fast, codes, me))
            groups.append((ply, len(codes), codes.index(code)))
        fast.push(code)
        rows.append(np.asarray(fast.encode_here(me), dtype=np.float32))
    surprises = np.zeros(len(rows), dtype=np.float32)
    with torch.inference_mode():
        latent = net.encode(torch.from_numpy(np.stack(rows)))
        memory, after = net.initial_memory(1), []
        for row in range(len(rows)):
            memory = net.memory_cell(latent[row:row + 1], memory)
            after.append(memory)
        if groups:
            memories = torch.cat([after[ply].expand(n, -1) for ply, n, _ in groups])
            context, _ = net.candidates(torch.from_numpy(np.concatenate(candidates)), memories)
            logits = net.predict_logits(context).split([n for _, n, _ in groups])
            for (ply, _, played), part in zip(groups, logits):
                surprises[ply + 1] = -float(np.log(max(float(part.softmax(0)[played]), 1e-12)))
    proxies = None
    review_labels = None
    if reviewed:
        # The game review: material state before the decisive moment, the result after it,
        # and a note (good/bad) for each of Claudio's own moves.
        from review import review
        values = [0.0] + [float(_w["values"][k]) for k in range(1, 6)]
        rv = review([m.uci() for m in board.move_stack], values)
        side = "white" if me else "black"
        sign = 1.0 if me else -1.0
        final = board.outcome(claim_draw=True)
        result_value = None if final is None or final.winner is None else float(final.winner == me)
        pick = rv["decisive"].get(side) if result_value is not None else None
        cut = pick["ply"] + 1 if pick else None
        proxies, last = [], 0.0
        for t, balance in enumerate(rv["balances"]):
            last = last if balance is None else sign * balance
            proxies.append(result_value if cut is not None and t > cut else 0.5 + 0.5 * float(np.tanh(last / 5.0)))
        proxies = np.asarray(proxies, dtype=np.float32)
        # weight: how big the move's score was (immediate + consequences), 1 at the threshold, at most 4
        review_labels = [(n["ply"], 1 if n["label"] == "good" else -1, min(4.0, abs(n["score"]) / 1.5))
                         for n in rv["notes"] if n["side"] == side and n["label"] in ("good", "bad")]
    elif _w["shaped"]:
        replay, proxy = board.root(), _w["proxy"]
        proxies = [proxy.position_proxy(replay, me)]
        for move in board.move_stack:
            replay.push(move)
            proxies.append(proxy.position_proxy(replay, me))
        proxies = np.asarray(proxies, dtype=np.float32)
    outcome = board.outcome(claim_draw=True)
    terminated = outcome is not None
    won = terminated and outcome.winner == me
    lost = terminated and outcome.winner is not None and outcome.winner != me
    result = "win" if won else "loss" if lost else "draw" if terminated else "truncated"
    # The winner's view of the game (for --learn winners): every position from its side.
    winner_packed = None
    if won:
        winner_packed = pack_many(np.stack(rows))
    elif lost:
        view = fast_rules.board(board.root())
        seen = [np.asarray(view.encode_here(not me), dtype=np.float32)]
        for move in board.move_stack:
            view.push(fast_rules.code(move))
            seen.append(np.asarray(view.encode_here(not me), dtype=np.float32))
        winner_packed = pack_many(np.stack(seen))
    return {"game": game, "colour": me, "packed": pack_many(np.stack(rows)), "lost": lost, "terminated": terminated,
            "result_value": (1.0 if won else 0.0 if lost else 0.5) if terminated else None,
            "surprises": surprises, "proxies": proxies, "actions": tuple(m.uci() for m in board.move_stack),
            "result": result, "end": outcome.termination.name if outcome else "max_plies",
            "winner": None if not (won or lost) else ("claudio" if won else "opponent"),
            "review_labels": review_labels,
            "winner_packed": winner_packed,
            "plies": len(board.move_stack), "seconds": time.perf_counter() - started}


# ------------------------------------------------------------------ telemetry
class Telemetry:
    """The records the training tracker app reads (history/<run>/launch.json + <run>.jsonl)."""

    def __init__(self, args, total, stages, checkpoint):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"claudio-{stamp}"
        self.history = ROOT / "history" / f"{self.run_id}.jsonl"
        folder = ROOT / "history" / self.run_id
        folder.mkdir(parents=True, exist_ok=True)
        launch = {"pid": os.getpid(), "run_id": self.run_id, "resume_games": 0, "campaign_start_games": 0,
                  "additional_games": total, "campaign_cap": total, "history": str(self.history),
                  "checkpoint": str(Path(checkpoint).resolve()), "objective": args.objective, "version": "claudio-grande",
                  "model_name": "Claudio (rede grande)", "opponent_mixture": {},
                  "ladder": [{"opponent": s[2], "games": n} for s, n in stages],
                  "stop_file": str(args.out_dir / "STOP")}
        (folder / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
        self.stream = open(self.history, "a", encoding="utf-8")
        self.write({"kind": "run", "config": {"objective": args.objective, "draw_value": args.draw_value,
                                               "choice_weight": args.choice_weight, "updates_per_game": args.updates,
                                               "sequence_batch_size": args.batch, "learning_rate": args.lr,
                                               "settings": args.set, "ladder": args.ladder, "start": str(args.start)}})

    def write(self, row):
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()

    def metrics(self, games, counts, opponents, draw_reasons, plies, started, state, extra):
        elapsed = max(1e-9, time.perf_counter() - started)
        self.write({"kind": "metrics", "games": games, **counts, "opponents": opponents,
                    "draw_reasons": dict(draw_reasons), "state": state,
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "games_per_second": games / elapsed, "positions_per_second": plies / elapsed, **extra})


# ------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=Path, required=True, help="Claudio checkpoint (grown)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "claudio/train-grande")
    parser.add_argument("--ladder", default="", help="opponents in order, name:games (tiny, self, fairyNNN, sfNNN)")
    parser.add_argument("--opponent", default="sf1800", help="the single opponent when there is no --ladder")
    parser.add_argument("--adaptive", default="", help="fairyNNN/sfNNN: start there and keep Claudio near --target-wr")
    parser.add_argument("--target-wr", type=float, default=0.5, help="adaptive: the win rate kept (wins/games)")
    parser.add_argument("--adapt-every", type=int, default=100, help="adaptive: games between Elo changes")
    parser.add_argument("--adapt-window", type=int, default=200, help="adaptive: games the win rate is measured on")
    parser.add_argument("--adapt-step", type=int, default=50, help="adaptive: Elo change per step")
    parser.add_argument("--adapt-band", type=float, default=0.05, help="adaptive: no change within target +- band")
    parser.add_argument("--replay-capacity", type=int, default=1_000_000, help="positions kept for learning")
    parser.add_argument("--eval-every", type=int, default=1000, help="games between fixed evaluations (0 = none)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-opponent", default="fairy1300", help="the fixed evaluation opponent")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--max-games", type=int, default=0, help="stop after this many games (0 = --hours/ladder)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--move-time", type=float, default=0.002)
    parser.add_argument("--objective", choices=("shaped", "victory"), default="shaped")
    parser.add_argument("--draw-value", type=float, default=0.25, help="what a draw is worth (victory objective)")
    parser.add_argument("--updates", type=int, default=2, help="learner steps per game")
    parser.add_argument("--batch", type=int, default=64, help="sequences per learner step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--colour", choices=("both", "white", "black"), default="both",
                        help="white/black: Claudio always plays that colour (a separate specialist)")
    parser.add_argument("--review-weight", type=float, default=0.5, help="review: weight of the good/bad move notes")
    parser.add_argument("--learn", choices=("outcomes", "winners", "review", "notes"), default="outcomes",
                        help="outcomes: TonkAI's usual learning; winners: only the winning side's moves, as better")
    parser.add_argument("--winner-updates", type=int, default=2, help="winners: learner steps per decided game")
    parser.add_argument("--winner-rows", type=int, default=32, help="winners: positions per learner step")
    parser.add_argument("--winner-tau", type=float, default=0.05, help="winners: softmax temperature of the values")
    parser.add_argument("--choice-weight", type=float, default=0.0)
    parser.add_argument("--choice-tau", type=float, default=0.05)
    parser.add_argument("--refresh", type=int, default=50, help="games between new actor weights")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--random-plies", type=int, default=0)
    parser.add_argument("--explore-plies", type=int, default=16, help="plies with temperature from the start")
    parser.add_argument("--explore-temperature", type=float, default=0.03)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--set", action="append", default=[], help="Claudio setting name=value for both players")
    parser.add_argument("--train-set", action="append", default=[],
                        help="learner setting name=value (e.g. value_weight=0, future_weight=0) - for studies")
    args = parser.parse_args()

    import psutil_free_priority  # noqa: F401
    import numpy as np
    import torch
    from dataclasses import asdict
    from match_checkpoints import _historical_config
    from tonkai.sequence_replay import SequenceEpisode
    from claudio_agent import ClaudioConfig
    from claudio_load import load_claudio, save_claudio

    adaptive = parse_stage(args.adaptive) if args.adaptive else None
    elo = adaptive[1] if adaptive else None
    if adaptive:
        stages = [(adaptive, args.max_games or 10 ** 9)]
    else:
        stages = parse_ladder(args.ladder) if args.ladder else [(parse_stage(args.opponent), args.max_games or 10 ** 9)]
    evaluation = parse_stage(args.eval_opponent) if args.eval_every else None
    total = sum(n for _, n in stages)
    if args.max_games:
        total = min(total, args.max_games)
    bounds = np.cumsum([n for _, n in stages])

    def stage_of(game):
        if adaptive:
            name = {"fairy": "Fairy-Stockfish", "sf": "Stockfish"}[adaptive[0]]
            return (adaptive[0], elo, f"{name} {elo}", f"{name} {elo}")
        return stages[min(int(np.searchsorted(bounds, game, side="right")), len(stages) - 1)][0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop = args.out_dir / "STOP"
    stop.unlink(missing_ok=True)
    net, values, config, history, king = load_claudio(args.start, with_config=True, with_king=True)
    config = asdict(ClaudioConfig.with_settings(args.set, base=config))
    architecture = torch.load(args.start, map_location="cpu", weights_only=False)["architecture"]
    state = torch.load(TIMDGEI, map_location="cpu", weights_only=False)
    cfg = _historical_config(state)
    cfg.syzygy_path, cfg.tablebase_adjudicate = "", False
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"           # cloud machines have no GPU
    cfg.learning_rate, cfg.sequence_batch_size = args.lr, args.batch
    cfg.num_workers, cfg.updates_per_game, cfg.objective = args.workers, args.updates, args.objective
    if args.objective == "victory":
        # The replay keeps the observed 0.5 for a draw; the learner reads it as --draw-value.
        import tonkai.survival_trainer as survival_trainer
        observed = survival_trainer.episode_result
        survival_trainer.episode_result = lambda e: (args.draw_value if observed(e) == 0.5 else observed(e))
    for item in args.train_set:
        key, _, raw = item.partition("=")
        if not hasattr(cfg, key):
            raise ValueError(f"unknown learner setting {key}")
        setattr(cfg, key, type(getattr(cfg, key))(raw) if getattr(cfg, key) is not None else float(raw))
    Trainer = make_trainer_class()
    Trainer.choice_weight, Trainer.choice_tau = args.choice_weight, args.choice_tau
    if args.learn == "notes":
        Trainer.review_weight = args.review_weight
    if args.learn == "review":
        # Value target = the review's per-position target (material state, the result after the
        # decisive moment); the move notes teach good/bad moves.
        cfg.objective, cfg.victory_weight, cfg.measure_lookahead = "shaped", 1.0, False
        Trainer.review_weight = args.review_weight
    if args.learn == "winners":
        Trainer.choice_tau = args.winner_tau
    net.requires_grad_(True).train()
    net.rnd_target.requires_grad_(False)                             # the fixed random target stays fixed
    trainer = Trainer(cfg, net=net)
    trainer.replay.load_state_dict(state["replay"])                 # the experience Claudio grew up with
    trainer.replay.capacity_positions = max(trainer.replay.capacity_positions, args.replay_capacity)
    trainer.games, trainer.steps = state["games"], state["steps"]
    del state
    from tonkai.sequence_replay import SequenceReplay
    winners = SequenceReplay(args.replay_capacity, seed=args.seed + 3) if args.learn == "winners" else None
    noted = SequenceReplay(args.replay_capacity, seed=args.seed + 4) if args.learn == "notes" else None

    def publish(version):
        path = args.out_dir / f"actor-{version}.pt"
        torch.save({k: v.detach().cpu() for k, v in trainer.net.state_dict().items()}, path)
        for old in args.out_dir.glob("actor-*.pt"):
            if int(old.stem.split("-")[1]) < version - 3:
                old.unlink(missing_ok=True)

    def save(name, finished, results):
        snapshot = copy.deepcopy(trainer.net).cpu().eval()
        save_claudio(args.out_dir / name, snapshot, architecture, values, ClaudioConfig(**config),
                     history + [{"trained_games": finished, "ladder": args.ladder or args.opponent, "updates": updates,
                                 "objective": args.objective, "draw_value": args.draw_value, "results": results,
                                 "per_opponent": {k: dict(v) for k, v in per_opponent.items()},
                                 "choice_weight": args.choice_weight}], king_model=king)

    telemetry = Telemetry(args, total, stages, args.out_dir / "claudio-grande-final.pt")
    version = 0
    publish(version)
    recent = deque(maxlen=500)
    counts = {"wins": 0, "draws": 0, "losses": 0, "truncated": 0}
    per_opponent, draw_reasons = {}, Counter()
    started, deadline = time.perf_counter(), time.perf_counter() + 3600 * args.hours
    game = finished = updates = plies = 0
    learn_seconds, losses = 0.0, deque(maxlen=200)
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init_worker,
                               initargs=(str(args.out_dir), architecture, values, config, str(args.start),
                                         args.draw_value if args.objective == "victory" else None, args.objective,
                                         args.move_time))

    def submit():
        nonlocal game
        stage = stage_of(game)
        future = pool.submit(play, game, version, args.seed, stage, args.random_plies, args.max_plies,
                             args.explore_plies, args.explore_temperature, args.colour, args.learn in ("review", "notes"))
        future.stage, future.eval_round = stage, None
        game += 1
        return future

    def submit_evaluation(round_):
        """Fixed evaluation: the current weights, no exploration, nothing learned from these games."""
        futures = set()
        for index in range(args.eval_games):
            future = pool.submit(play, index, version, args.seed + 1000, evaluation, 0, args.max_plies, 0, 0.0,
                                 args.colour, False)      # a specialist is evaluated on its own colour
            future.stage, future.eval_round = evaluation, round_
            futures.add(future)
        return futures

    window = deque(maxlen=args.adapt_window)
    evaluations, best = {}, None
    pending = {submit() for _ in range(min(2 * args.workers, total))}
    state_name, training_open = "completed", True
    while pending:
        done, pending = wait(pending, timeout=300, return_when=FIRST_COMPLETED)
        if not done:                          # a dead engine left a game hanging: stop cleanly
            state_name = "failed"
            break
        for future in done:
            r, stage = future.result(), future.stage
            if future.eval_round is not None:
                games_here = evaluations.setdefault(future.eval_round, [])
                games_here.append(r["result"])
                if len(games_here) == args.eval_games:
                    wr = games_here.count("win") / len(games_here)
                    telemetry.write({"kind": "evaluation", "games": future.eval_round, "opponent": evaluation[2],
                                     "win_rate": wr, "wins": games_here.count("win"),
                                     "draws": games_here.count("draw"), "losses": games_here.count("loss")})
                    if best is None or wr > best[1]:
                        best = (future.eval_round, wr)
                        shutil.copyfile(args.out_dir / f"eval-{future.eval_round}.pt",
                                        args.out_dir / "claudio-grande-best.pt")
                continue
            t0 = time.perf_counter()
            if winners is not None:
                # Only the winning side teaches, and only "this move is even better".
                if r["winner_packed"] is not None:
                    winners.add(SequenceEpisode(packed=r["winner_packed"], lost=False, terminated=True,
                                                opponent=f"winner {r['winner']} vs {stage[2]} (claudio)",
                                                actions=r["actions"], result_value=1.0))
                    for _ in range(args.winner_updates):
                        loss = trainer.winner_update(winners, args.winner_rows)
                        if loss is not None:
                            updates += 1
                            losses.append((0.0, 0.0, loss))
            elif noted is not None:
                # Only the review's notes teach: Claudio's own games, nothing from the old replay.
                if r.get("review_labels"):
                    trainer.__dict__.setdefault("review_labels", {})[hash(tuple(r["actions"]))] = r["review_labels"]
                    noted.add(SequenceEpisode(packed=r["packed"], lost=r["lost"], terminated=r["terminated"],
                                              opponent=f"{stage[2]} (claudio)", actions=r["actions"]))
                for _ in range(args.updates):
                    loss = trainer.notes_update(noted)
                    if loss is not None:
                        updates += 1
                        losses.append((0.0, 0.0, loss))
            else:
                episode = SequenceEpisode(packed=r["packed"], lost=r["lost"], terminated=r["terminated"],
                                          opponent=f"{stage[2]} (claudio)", surprises=r["surprises"],
                                          actions=r["actions"], result_value=r["result_value"], proxies=r["proxies"])
                if r.get("review_labels"):
                    trainer.__dict__.setdefault("review_labels", {})[hash(tuple(r["actions"]))] = r["review_labels"]
                trainer.ingest(episode, [], r["colour"])
                for _ in range(args.updates):
                    out = trainer.update()
                    if out is not None:
                        updates += 1
                        losses.append((out.get("value_loss", 0.0), out.get("reply_loss", 0.0),
                                       getattr(trainer, "last_choice_loss", 0.0)))
            learn_seconds += time.perf_counter() - t0
            finished += 1
            plies += r["plies"]
            key = {"win": "wins", "draw": "draws", "loss": "losses", "truncated": "truncated"}[r["result"]]
            counts[key] += 1
            row = per_opponent.setdefault(stage[3], {"wins": 0, "draws": 0, "losses": 0, "truncated": 0})
            row[key] += 1
            if r["result"] == "draw":
                draw_reasons[r["end"]] += 1
            recent.append(r["result"])
            window.append(r["result"])
            if adaptive and finished % args.adapt_every == 0 and len(window) >= args.adapt_every:
                wr = window.count("win") / len(window)
                if wr > args.target_wr + args.adapt_band:
                    elo = min(2850, elo + args.adapt_step)
                elif wr < args.target_wr - args.adapt_band:
                    elo = max(500, elo - args.adapt_step)
            if finished % args.refresh == 0:
                version += 1
                publish(version)
            if finished % 50 == 0 or finished == total:
                n = len(recent)
                mean = np.mean(np.array(losses), axis=0) if losses else [0, 0, 0]
                telemetry.metrics(finished, counts, per_opponent, draw_reasons, plies, started, "running", {
                    "stage": stage[2], "updates": updates, "adaptive_elo": elo,
                    "learner_share": round(learn_seconds / (time.perf_counter() - started), 3),
                    "recent_win_rate": round(recent.count("win") / n, 4),
                    "recent_draw_rate": round(recent.count("draw") / n, 4),
                    "best_evaluation": None if best is None else {"games": best[0], "win_rate": best[1]},
                    "value_loss": round(float(mean[0]), 5), "reply_loss": round(float(mean[1]), 5),
                    "choice_loss": round(float(mean[2]), 5)})
            if finished % args.save_every == 0:
                save(f"claudio-grande-{finished}.pt", finished, dict(counts))
            if evaluation and finished % args.eval_every == 0:
                version += 1
                publish(version)
                save(f"eval-{finished}.pt", finished, dict(counts))
                pending |= submit_evaluation(finished)
            if time.perf_counter() >= deadline or stop.exists():
                state_name, training_open = "stopped", False
            elif training_open and game < total:
                pending.add(submit())
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()                        # includes any hung game; engines die with their worker
    pool.shutdown(wait=False, cancel_futures=True)
    save("claudio-grande-final.pt", finished, dict(counts))
    telemetry.metrics(finished, counts, per_opponent, draw_reasons, plies, started, state_name,
                      {"updates": updates, "stage": "fim", "adaptive_elo": elo,
                       "best_evaluation": None if best is None else {"games": best[0], "win_rate": best[1]}})
    print(f"done ({state_name}): {finished} games, {updates} updates, {counts}, {per_opponent}", flush=True)


if __name__ == "__main__":
    main()
