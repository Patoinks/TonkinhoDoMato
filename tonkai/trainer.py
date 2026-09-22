"""Three losses on one shared encoder.

    total = value_weight   * CE(value_logits, real game result)
          + predict_weight * CE(predict_logits over legal replies, real reply)
          + piece_weight   * MSE(piece_value, counterfactual contribution)

The piece target is *TonkAI's own* estimate of what removing that piece does to
its survival prediction. It is not a chess piece value and nothing forces
utility to be a sum of piece values.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .encoding import INPUT_SIZE, PIECE_NAMES, PLANE_SIZE
from .model import TonkAINet
from .replay import PredictBuffer, ReplayBuffer
from .theory_buffer import SacrificeLog, TheoryBuffer

KING_PLANES = (5, 11)


class Trainer:
    def __init__(self, net: TonkAINet, cfg: Config, device: str, metrics,
                 rng: np.random.Generator | None = None):
        self.net = net
        self.inference_net = net
        self.cfg = cfg
        self.device = device
        self.metrics = metrics
        self.rng = rng or np.random.default_rng(cfg.seed)

        self.opt = torch.optim.AdamW(
            net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.replay = ReplayBuffer(cfg.replay_capacity, self.rng)
        self.predict_replay = PredictBuffer(
            cfg.predict_capacity, cfg.max_predict_candidates, self.rng
        )
        self.theory = TheoryBuffer(cfg.theory_capacity)
        self.sacrifice_log = SacrificeLog()

        self.games = 0
        self.steps = 0
        self.predict_batch_size = max(32, cfg.batch_size // 8)

    # -------------------------------------------------------------- ingest

    def add_game(self, record) -> None:
        self.games += 1
        if record.packed.shape[0]:
            self.replay.add_many(record.packed, record.result)
        for packed, true_idx in record.predict_samples:
            self.predict_replay.add(packed, true_idx)
        if record.theory_events:
            self.theory.extend(record.theory_events, record.result)
        if record.sacrifices:
            self.sacrifice_log.extend(record.sacrifices, record.result)

    def ready(self) -> bool:
        return len(self.replay) >= self.cfg.min_replay

    # -------------------------------------------------------------- losses

    def _value_and_piece_loss(self, batch_size: int):
        x_np, y_np = self.replay.sample(batch_size)
        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        latent = self.net.encode(x)
        vlogits = self.net.value_logits(latent)
        value_loss = F.cross_entropy(vlogits, y)

        piece_loss = torch.zeros((), device=self.device)
        if self.cfg.piece_value_weight > 0.0 and self.cfg.pieces_per_position > 0:
            piece_loss = self._piece_loss(x, x_np, latent, vlogits)

        return value_loss, piece_loss

    def _piece_loss(self, x, x_np, latent, vlogits):
        """Counterfactual contribution of one (or a few) pieces per position.

        target = U(position) - U(position with that piece erased)
        """
        cfg = self.cfg
        b = x.shape[0]
        dw = cfg.draw_weight

        p = torch.softmax(vlogits, dim=-1)
        u_orig = p[:, 0] + dw * p[:, 1] - p[:, 2]

        # candidate squares: any piece that is not a king
        planes = x_np[:, :PLANE_SIZE].reshape(b, 12, 64).copy()
        for kp in KING_PLANES:
            planes[:, kp, :] = 0.0
        flat = planes.reshape(b, PLANE_SIZE)

        total = torch.zeros((), device=self.device)
        used = 0
        for _ in range(cfg.pieces_per_position):
            # pick a uniformly random *set* bit per row, without a Python loop
            scores = self.rng.random((b, PLANE_SIZE)).astype(np.float32) * flat
            idx = scores.argmax(axis=1)
            valid = flat[np.arange(b), idx] > 0
            if not valid.any():
                continue

            vi = np.flatnonzero(valid)
            sel = idx[vi]
            rows = torch.from_numpy(vi).to(self.device)
            cols = torch.from_numpy(sel).to(self.device)

            x_masked = x[rows].clone()
            x_masked[torch.arange(len(vi), device=self.device), cols] = 0.0
            with torch.no_grad():
                vl_masked = self.net.value_logits(self.net.encode(x_masked))
                pm = torch.softmax(vl_masked, dim=-1)
                u_masked = pm[:, 0] + dw * pm[:, 1] - pm[:, 2]
            target = (u_orig[rows] - u_masked).detach()

            plane = sel // 64
            square = sel % 64
            ptype = torch.from_numpy((plane % 6).astype(np.int64)).to(self.device)
            pside = torch.from_numpy((plane // 6).astype(np.int64)).to(self.device)
            psq = torch.from_numpy(square.astype(np.int64)).to(self.device)

            pred = self.net.piece_values(latent[rows], ptype, psq, pside)
            total = total + F.mse_loss(pred, target)
            used += 1

            with torch.no_grad():
                vals = pred.detach().float().cpu().numpy()
                types = (plane % 6)
                for t in range(6):
                    sel_t = vals[types == t]
                    if sel_t.size:
                        self.metrics.piece_value[PIECE_NAMES[t]].extend(sel_t)

        return total / max(used, 1)

    def _predict_loss(self):
        if len(self.predict_replay) < 64:
            return torch.zeros((), device=self.device)
        b = min(self.predict_batch_size, len(self.predict_replay))
        x_np, mask_np, true_np = self.predict_replay.sample(b)
        k = x_np.shape[1]

        flat_mask = mask_np.reshape(-1)
        xv = torch.from_numpy(
            x_np.reshape(-1, INPUT_SIZE)[flat_mask]
        ).to(self.device)
        logits_valid = self.net.predict_logits(self.net.encode(xv))

        full = torch.full((b * k,), -1e9, device=self.device, dtype=logits_valid.dtype)
        pos = torch.from_numpy(np.flatnonzero(flat_mask)).to(self.device)
        full = full.index_copy(0, pos, logits_valid)
        logits = full.view(b, k)

        true_idx = torch.from_numpy(true_np).to(self.device)
        return F.cross_entropy(logits, true_idx)

    # ---------------------------------------------------------------- step

    def train_step(self, batch_size: int | None = None) -> dict:
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)

        value_loss, piece_loss = self._value_and_piece_loss(
            min(cfg.batch_size if batch_size is None else batch_size, len(self.replay))
        )
        self.predict_batch_size = max(32, (cfg.batch_size if batch_size is None else batch_size) // 8)
        predict_loss = self._predict_loss()

        total = (cfg.value_weight * value_loss
                 + cfg.predict_weight * predict_loss
                 + cfg.piece_value_weight * piece_loss)
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.opt.step()
        self.steps += 1

        out = {
            "value": float(value_loss.detach()),
            "predict": float(predict_loss.detach()),
            "piece": float(piece_loss.detach()),
        }
        self.metrics.value_loss.add(out["value"])
        self.metrics.predict_loss.add(out["predict"])
        self.metrics.piece_loss.add(out["piece"])
        self.metrics.train_batches += 1
        self.sync_inference()
        return out

    def sync_inference(self):
        if self.inference_net is not self.net:
            start = time.perf_counter()
            self.inference_net.load_state_dict(self.net.state_dict())
            self.metrics.inference_refresh_seconds = (
                getattr(self.metrics, "inference_refresh_seconds", 0.0)
                + time.perf_counter() - start)

    def train(self, n_steps: int) -> None:
        if not self.ready():
            return
        with self.metrics.timer("training"):
            for i in range(n_steps):
                if self.cfg.samples_per_game > 0:
                    batch, remainder = divmod(self.cfg.samples_per_game, n_steps)
                    size = batch + (i < remainder)
                    if size:
                        self.train_step(size)
                else:
                    self.train_step()

    # --------------------------------------------------------- checkpoints

    def save(self, path: str, extra: dict | None = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": self.cfg.to_dict(),
            "games": self.games,
            "steps": self.steps,
            "metrics": self.metrics.to_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": self.rng.bit_generator.state,
            "saved_at": time.time(),
        }
        if extra:
            payload.update(extra)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["model"])
        self.sync_inference()
        if load_optimizer and "optimizer" in payload:
            self.opt.load_state_dict(payload["optimizer"])
        self.games = payload.get("games", 0)
        self.steps = payload.get("steps", 0)
        if "numpy_rng" in payload:
            self.rng.bit_generator.state = payload["numpy_rng"]
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"].cpu().to(torch.uint8))
        return payload
