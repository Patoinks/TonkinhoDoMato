"""Sequence learning of discounted survival, without a win reward.

Real losses supervise every earlier position. Other completed games are safe
absorbing outcomes; artificial move limits instead use a detached n-step target.
Future representations use a fixed copy of the initial encoder, so they cannot
collapse by moving the target towards their own predictions.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import time

import chess
import numpy as np
import torch
from torch.nn import functional as F

from .captures import episode_capture_targets, episode_owner_color
from . import fast_rules
from .encoding import encode_after_moves, unpack_many
from .sequence_replay import SequenceReplay
from .temporal_model import SurvivalNet
from .objective import episode_result, victory_weight

CAPTURE_PREFIX = "capture_out."
CAPTURE_CACHE = 4096


def _load_module(module, state):
    """Strict loading, except that the optional capture head may be new or gone.

    Returns (missing, unexpected) capture keys. Anything else still fails
    exactly as a strict load would.
    """
    result = module.load_state_dict(state, strict=False)
    other = [k for k in (*result.missing_keys, *result.unexpected_keys)
             if not k.startswith(CAPTURE_PREFIX)]
    if other:
        raise RuntimeError(f"Error(s) in loading state_dict: missing {result.missing_keys}, "
                           f"unexpected {result.unexpected_keys}")
    return list(result.missing_keys), list(result.unexpected_keys)


def _adapt_optimizer(state, added: int, dropped: int):
    """Match a saved optimizer to a model that gained/lost the capture head.

    The capture parameters are the last ones registered, so they are the last
    ids of the single parameter group: append fresh ids (Adam starts their
    moments lazily) or drop the trailing ones with their moments.
    """
    state = deepcopy(state)
    if len(state["param_groups"]) != 1:
        raise RuntimeError("capture head migration expects a single parameter group")
    params = state["param_groups"][0]["params"]
    for pid in params[len(params) - dropped:] if dropped else ():
        state["state"].pop(pid, None)
    if dropped:
        del params[len(params) - dropped:]
    start = max(params, default=-1) + 1
    params.extend(range(start, start + added))
    return state


def survival_targets(remaining, lost, terminated, gamma, bootstrap, steps):
    """Wins and draws deliberately share the same `lost=False` target.

    `remaining` and `steps` count half-moves. The death state itself has value
    zero. A position r moves before death has value 1 - gamma**r. Truncation is
    censored data, never a manufactured successful terminal outcome.
    """
    discount = torch.pow(torch.as_tensor(gamma, device=remaining.device), remaining.clamp_min(0))
    completed = torch.where(lost, 1.0 - discount, torch.ones_like(discount))
    n_discount = torch.pow(torch.as_tensor(gamma, device=remaining.device), steps)
    censored = 1.0 - n_discount + n_discount * bootstrap.detach()
    return torch.where(terminated, completed, censored)


class SurvivalTrainer:
    def __init__(self, cfg, net=None):
        cfg.validate()
        self.cfg = cfg
        self.device = cfg.resolved_device()
        self.net = (net or SurvivalNet(cfg)).to(self.device)
        self.target = deepcopy(self.net).eval().requires_grad_(False)
        self.future_encoder = deepcopy(self.net.encoder).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.replay = SequenceReplay(cfg.replay_capacity, seed=cfg.seed + 1)
        self.rng = np.random.default_rng(cfg.seed + 2)
        self.games = 0
        self.steps = 0
        # Optional V3 auxiliary target; zero leaves every V2 code path untouched.
        self.capture_weight = float(getattr(cfg, "capture_target_weight", 0.0) or 0.0)
        self.last_capture_loss = None
        self._capture_tables = OrderedDict()

    def _unroll(self, net, x, valid, burn_in, learn_length, gradients=False, *, learner_window_only=False):
        """Preserve all burn-in and learned states; optionally omit unused outputs.

        The learner consumes only the learning window. Its trailing recurrent
        states and burn-in fusion outputs never feed a loss. The target network
        still unrolls the complete future horizon for bootstrap targets.
        """
        if learner_window_only and not gradients:
            raise ValueError("Window-only outputs are for the learner, not the target network")
        memory = net.initial_memory(len(x))
        before, after = [], []
        # Encode all positions together. GRU time dependency remains sequential.
        with torch.set_grad_enabled(gradients):
            if getattr(self.cfg, "compact_sequences", False):
                from .execution_fastpaths import compact_encode
                latent = compact_encode(net.encode, x, valid)
            else:
                latent = net.encode(x.flatten(0, 1)).reshape(len(x), x.shape[1], -1)
        total, stop = x.shape[1], min(x.shape[1], burn_in + learn_length) if learner_window_only else x.shape[1]
        for t in range(stop):
            learning = gradients and burn_in <= t < burn_in + learn_length
            if t == burn_in or t == burn_in + learn_length:
                memory = memory.detach()
            with torch.set_grad_enabled(learning):
                features = latent[:, t] if learning else latent[:, t].detach()
                before.append(memory)
                candidate = net.memory_cell(features, memory)
                memory = torch.where(valid[:, t, None], candidate, memory)
                after.append(memory)
        # The fusion of step t reads only step t, so it runs once per span
        # (burn-in, learning window, tail) instead of once per step, with the
        # same gradient boundaries as the recurrence above.
        b, d = len(x), latent.shape[-1]

        def fuse(start, end, grad):
            if end <= start:
                return latent.new_zeros((b, 0, d))
            with torch.set_grad_enabled(grad):
                features = latent[:, start:end] if grad else latent[:, start:end].detach()
                memories = torch.stack(after[start:end], 1)
                fused = net.context_from_latent(features.reshape(-1, d), memories.reshape(-1, memories.shape[-1]))
            return fused.reshape(b, end - start, -1)

        low, high = min(burn_in, stop), min(burn_in + learn_length, stop)
        # Window-only placeholders keep the original layouts; no loss reads them.
        parts = [latent.new_zeros((b, low, d)) if learner_window_only else fuse(0, low, False)]
        parts += [fuse(low, high, gradients), fuse(high, stop, False)]
        if stop < total:
            parts.append(latent.new_zeros((b, total - stop, d)))
            before += [memory.new_zeros(memory.shape)] * (total - stop)
            after += [memory.new_zeros(memory.shape)] * (total - stop)
        return torch.cat(parts, 1), torch.stack(before, 1), torch.stack(after, 1)

    def _reply_loss(self, batch, x, memories, offset, length):
        """One legal reply distribution per sampled sequence, batched together."""
        candidates, prefixes, labels, sizes, weights = [], [], [], [], []
        for row, episode in enumerate(batch["episodes"]):
            if not episode.actions:
                continue
            indices = batch["state_indices"][row, offset:offset + length]
            # Plane 768 states whose turn it is relative to this episode's owner.
            possible = np.flatnonzero((indices >= 0) & (indices < len(episode.actions))
                                      & (x[row, offset:offset + length, 768].detach().cpu().numpy() < .5))
            if not len(possible):
                continue
            local = int(self.rng.choice(possible))
            index = int(indices[local])
            board = chess.Board(episode.initial_fen or chess.STARTING_FEN)
            if fast_rules.ENABLED:
                fast = fast_rules.board(board)
                fast.push_many_checked([fast_rules.code(chess.Move.from_uci(action))
                                        for action in episode.actions[:index]])
                # Legal-move order is still python-chess's. The native board
                # retains the history; this FEN is only for move enumeration.
                board = chess.Board(fast.fen())
            else:
                for action in episode.actions[:index]:
                    board.push_uci(action)
            moves = list(board.legal_moves)
            actual = chess.Move.from_uci(episode.actions[index])
            labels.append(moves.index(actual))
            if fast_rules.ENABLED:
                candidates.append(fast_rules.encode_after_moves(fast,
                                  [fast_rules.code(m) for m in moves], not board.turn))
            else:
                candidates.append(encode_after_moves(board, moves, not board.turn))
            prefixes.append(memories[row, offset + local].expand(len(moves), -1))
            sizes.append(len(moves))
            weights.append(batch["importance_weights"][row])
        if not candidates:
            return x.new_zeros(())
        encoded = torch.from_numpy(np.concatenate(candidates)).to(self.device)
        context, _ = self.net.candidates(encoded, torch.cat(prefixes))
        logits = self.net.predict_logits(context).split(sizes)
        losses = torch.stack([F.cross_entropy(logit[None], torch.tensor([label], device=self.device))
                              for logit, label in zip(logits, labels)])
        return (losses * losses.new_tensor(weights)).mean()

    def _proxy_targets(self, batch, offset, length):
        """(b, length) victory measures for the learning window, plus a mask."""
        indices = batch["state_indices"][:, offset:offset + length]
        values = np.zeros(indices.shape, dtype=np.float32)
        present = np.zeros(indices.shape, dtype=bool)
        for row, episode in enumerate(batch["episodes"]):
            proxies = getattr(episode, "proxies", None)
            if proxies is None:
                continue
            ok = (indices[row] >= 0) & (indices[row] < len(proxies))
            values[row, ok] = proxies[indices[row, ok]]
            present[row] = ok
        return (torch.as_tensor(values, device=self.device),
                torch.as_tensor(present, device=self.device))

    def _capture_table(self, episode_id, episode):
        """Future-capture targets for one episode, cached by episode id."""
        table = self._capture_tables.get(episode_id)
        if table is not None or episode_id in self._capture_tables:
            self._capture_tables.move_to_end(episode_id)
            return table
        if fast_rules.ENABLED and episode.actions and len(episode.actions) == len(episode.packed) - 1:
            root = chess.Board(episode.initial_fen or chess.STARTING_FEN)
            codes = [fast_rules.code(chess.Move.from_uci(action)) for action in episode.actions]
            table = fast_rules.board(root).capture_targets(codes, episode_owner_color(episode))
        else:
            table = episode_capture_targets(episode)
        self._capture_tables[episode_id] = table
        if len(self._capture_tables) > CAPTURE_CACHE:
            self._capture_tables.popitem(last=False)
        return table

    def _capture_loss(self, predicted, rows, squares, batch, offset, length):
        """MSE between predicted and actual future captures of the probed pieces.

        The target is what that piece really went on to capture in its game. It
        is a prediction target only: nothing rewards a capture or picks a move.
        """
        keep, targets = [], []
        clip = float(getattr(self.cfg, "capture_target_clip", 8.0))
        for j, row in enumerate(rows.detach().cpu().numpy()):
            i, t = divmod(int(row), length)
            table = self._capture_table(int(batch["episode_id"][i]), batch["episodes"][i])
            if table is None:
                continue
            state = int(batch["state_indices"][i, offset + t])
            if 0 <= state < len(table):
                keep.append(j)
                targets.append(min(float(table[state, squares[j]]), clip))
        if not keep:
            return predicted.new_zeros(())
        index = torch.as_tensor(keep, device=predicted.device)
        return F.mse_loss(predicted[index], predicted.new_tensor(targets))

    def _piece_loss(self, x, context, prior_memory, valid, batch=None, offset=0):
        # Position-dependent counterfactual value in survival units, no material table.
        self.last_capture_loss = None
        rows = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
        if not len(rows) or self.cfg.piece_value_weight == 0:
            return x.new_zeros(())
        if len(rows) > 64:
            selection = self.rng.choice(len(rows), 64, replace=False)
            rows = rows[torch.as_tensor(selection, device=self.device)]
        selected = x.flatten(0, 1)[rows]
        presence = selected[:, :768].reshape(-1, 12, 64).detach().cpu().numpy()
        presence[:, 5] = 0  # never remove kings
        presence[:, 11] = 0
        chosen_rows, planes, squares = [], [], []
        for row, pieces in enumerate(presence):
            indices = np.argwhere(pieces > .5)
            if not len(indices):
                continue
            plane, square = indices[int(self.rng.integers(len(indices)))]
            chosen_rows.append(row)
            planes.append(int(plane))
            squares.append(int(square))
        if not chosen_rows:
            return x.new_zeros(())
        rows = rows[torch.as_tensor(chosen_rows, device=self.device)]
        selected = x.flatten(0, 1)[rows]
        plane = torch.tensor(planes, device=self.device)
        square = torch.tensor(squares, device=self.device)
        current = context.flatten(0, 1)[rows]
        with torch.no_grad():
            removed = selected.clone()
            removed[torch.arange(len(rows), device=self.device), plane * 64 + square] = 0
            counterfactual, _ = self.net.observe(removed, prior_memory.flatten(0, 1)[rows])
            target = self.net.survival(current)[0].mean(-1) - self.net.survival(counterfactual)[0].mean(-1)
        if self.capture_weight > 0 and batch is not None and self.net.has_capture_head:
            predicted, captures = self.net.piece_values_and_captures(current, plane % 6, square, plane // 6)
            self.last_capture_loss = self._capture_loss(captures, rows, squares, batch, offset, x.shape[1])
        else:
            predicted = self.net.piece_values(current, plane % 6, square, plane // 6)
        return F.mse_loss(predicted, target)

    def update(self):
        if len(self.replay) < self.cfg.min_replay:
            return None
        cfg = self.cfg
        horizon = max(cfg.future_horizons)
        batch = self.replay.sample(cfg.sequence_batch_size, cfg.burn_in,
                                   cfg.sequence_length, horizon)
        b, total, _ = batch["packed"].shape
        unpacked = unpack_many(batch["packed"].reshape(-1, batch["packed"].shape[-1]))
        x = torch.from_numpy(unpacked.reshape(b, total, -1)).to(self.device)
        valid = torch.as_tensor(batch["mask"], device=self.device)
        offset, length = cfg.burn_in, cfg.sequence_length
        selection = slice(offset, offset + length)
        self.net.train()
        context, before, after = self._unroll(self.net, x, valid, offset, length, True, learner_window_only=True)
        with torch.no_grad():
            target_context, _, _ = self._unroll(self.target, x, valid, offset, length)
            target_values = self.target.survival(target_context.flatten(0, 1))[0].mean(-1).reshape(b, total)
            remaining = torch.as_tensor(batch["remaining"][:, selection], device=self.device)
            steps = remaining.clamp(0, horizon)
            endpoints = torch.arange(offset, offset + length, device=self.device)[None] + steps
            bootstrap = target_values.gather(1, endpoints)
            if cfg.objective == "victory":
                outcomes = torch.tensor([episode_result(e) if e.terminated else 0.0
                                         for e in batch["episodes"]], device=self.device)[:, None]
                targets = torch.where(torch.as_tensor(batch["terminated"], device=self.device)[:, None],
                                      outcomes, bootstrap.detach())
            else:
                targets = survival_targets(remaining,
                    torch.as_tensor(batch["terminal_lost"], device=self.device)[:, None],
                    torch.as_tensor(batch["terminated"], device=self.device)[:, None],
                    cfg.survival_gamma, bootstrap, steps)
                if cfg.objective == "shaped":
                    # Owner's victory measure, recorded per state during play.
                    # States without one (old episodes, experiments) keep survival.
                    proxies, has_proxy = self._proxy_targets(batch, offset, length)
                    weight = victory_weight(cfg)
                    targets = torch.where(has_proxy, (1 - weight) * targets + weight * proxies, targets)
            if getattr(cfg, "compact_sequences", False):
                from .execution_fastpaths import compact_encode
                future_targets = F.normalize(compact_encode(self.future_encoder, x, valid), dim=-1)
            else:
                future_targets = F.normalize(self.future_encoder(x.flatten(0, 1)), dim=-1).reshape(b, total, -1)
        learning = context[:, selection]
        mask = valid[:, selection]
        weights = torch.as_tensor(batch["importance_weights"], device=self.device)[:, None]
        weighted_mask = mask * weights
        means, logvars = self.net.survival(learning.flatten(0, 1))
        means, logvars = means.reshape(b, length, -1), logvars.reshape(b, length, -1)
        # Stable masks per episode keep genuinely different bootstrap datasets.
        head_masks = []
        for episode_id in batch["episode_id"]:
            generator = np.random.default_rng(int(episode_id) + cfg.seed * 100003)
            heads = generator.random(cfg.ensemble_size) < cfg.bootstrap_probability
            heads[int(episode_id) % cfg.ensemble_size] = True
            head_masks.append(heads)
        head_mask = torch.as_tensor(np.stack(head_masks), device=self.device)[:, None]
        residual = means - targets[..., None]
        variance_target = (residual.detach().square() + 1e-4).log().clamp(-8, 2)
        per_head = residual.square() + .05 * F.smooth_l1_loss(logvars, variance_target, reduction="none")
        effective = weighted_mask[..., None] * head_mask
        value_loss = (per_head * effective).sum() / effective.sum().clamp_min(1)
        predictions = self.net.future(learning.flatten(0, 1)).reshape(b, length, len(cfg.future_horizons), -1)
        future_errors, future_masks = [], []
        for i, distance in enumerate(cfg.future_horizons):
            target_latent = future_targets[:, offset + distance:offset + length + distance]
            # Smooth L1 on unit target features; predictor magnitude remains learnable.
            future_errors.append((predictions[:, :, i] - target_latent).square().mean(-1))
            future_masks.append(mask & valid[:, offset + distance:offset + length + distance])
        future_error = torch.stack(future_errors, -1)
        future_mask = torch.stack(future_masks, -1) * weights[..., None]
        future_loss = (future_error * future_mask).sum() / future_mask.sum().clamp_min(1)
        novelty_error = self.net.novelty_error(learning.detach().flatten(0, 1)).reshape(b, length)
        novelty_loss = (novelty_error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)
        reply_loss = self._reply_loss(batch, x, after, offset, length) if cfg.predict_weight else x.new_zeros(())
        piece_loss = self._piece_loss(x[:, selection], learning, before[:, selection], mask,
                                      batch=batch, offset=offset)
        loss = cfg.value_weight * value_loss + cfg.future_weight * future_loss + cfg.novelty_weight * novelty_loss
        loss = loss + cfg.predict_weight * reply_loss + cfg.piece_value_weight * piece_loss
        capture_loss = self.last_capture_loss
        if capture_loss is not None:
            loss = loss + self.capture_weight * capture_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite survival loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        if not torch.isfinite(gradient):
            raise FloatingPointError("Non-finite survival gradient")
        self.optimizer.step()
        with torch.no_grad():
            for target, current in zip(self.target.parameters(), self.net.parameters()):
                target.lerp_(current, cfg.target_tau)
        # Surprise starts priorities; new learning errors refine each sampled start.
        errors = (residual.detach().abs().mean(-1) + future_error.detach().mean(-1)) * mask
        priorities = (errors.sum(-1) / mask.sum(-1).clamp_min(1)).cpu().numpy()
        self.replay.update_priorities(batch["sample_keys"], priorities)
        self.steps += 1
        out = {"loss": float(loss.detach()), "survival_loss": float(value_loss.detach()),
               "value_loss": float(value_loss.detach()), "objective": cfg.objective,
               "future_loss": float(future_loss.detach()), "novelty_loss": float(novelty_loss.detach()),
               "reply_loss": float(reply_loss.detach()), "piece_loss": float(piece_loss.detach()),
               "learned_positions": int(mask.sum()), "gradient_norm": float(gradient)}
        if capture_loss is not None:
            out["capture_loss"] = float(capture_loss.detach())
        return out

    def state_dict(self):
        return {"format": "tonkai-survival-v2", "config": self.cfg.to_dict(),
                "model": self.net.state_dict(), "target": self.target.state_dict(),
                "future_encoder": self.future_encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
                "games": self.games, "steps": self.steps,
                "numpy_rng": deepcopy(self.rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def load_state_dict(self, state):
        if state.get("format") != "tonkai-survival-v2":
            raise ValueError("Legacy checkpoints require --init-from, not --resume")
        added, dropped = _load_module(self.net, state["model"])
        _load_module(self.target, state["target"])
        self.future_encoder.load_state_dict(state["future_encoder"])
        optimizer = state["optimizer"]
        if added or dropped:
            # Turning the capture target on/off for an existing checkpoint.
            optimizer = _adapt_optimizer(optimizer, len(added), len(dropped))
        self.optimizer.load_state_dict(optimizer)
        self.replay.load_state_dict(state["replay"])
        self.games, self.steps = state["games"], state["steps"]
        self.rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])

    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({**self.state_dict(), **extra}, temporary)
        # Windows readers/antivirus can briefly hold the previous checkpoint
        # without delete sharing. Retrying the atomic rename keeps that valid
        # checkpoint available; never delete it to work around a sharing lock.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
