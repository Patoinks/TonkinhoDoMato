"""Where does the time actually go? Measure before optimising anything."""

from __future__ import annotations

import time

import chess
import numpy as np
import torch

import tonkai
from tonkai.config import Config
from tonkai.encoding import INPUT_SIZE, encode_after_moves, encode_into, pack_into


def bench(label, fn, n, unit="call"):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    print(f"  {label:38s} {1e6 * dt / n:9.1f} us/{unit}   {n / dt:10.0f}/s")
    return dt / n


def main():
    cfg = Config(device="cuda" if torch.cuda.is_available() else "cpu")
    dev = cfg.resolved_device()
    net, agent, opponent, worker, trainer, metrics = tonkai.build(cfg, dev)
    print(f"device: {dev}\n")

    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3", "d6"):
        board.push_san(san)
    moves = list(board.legal_moves)
    buf = np.empty((256, INPUT_SIZE), dtype=np.float32)
    one = np.empty(INPUT_SIZE, dtype=np.float32)
    fen = board.fen()
    print(f"test position: {len(moves)} legal moves\n")

    print("SELF-PLAY per-ply pieces")
    bench("list(board.legal_moves)", lambda: list(board.legal_moves), 3000)
    bench("board.fen()", lambda: board.fen(), 5000)
    bench("encode_into (1 position)", lambda: encode_into(board, chess.WHITE, one), 5000)
    bench(f"encode_after_moves ({len(moves)} moves)",
          lambda: encode_after_moves(board, moves, chess.WHITE, buf), 1000)
    x = encode_after_moves(board, moves, chess.WHITE, buf)
    row = np.empty(99, dtype=np.uint8)
    bench("pack_into (1 row)", lambda: pack_into(x[0], row), 20000)

    def fwd():
        t = torch.from_numpy(x).to(dev)
        with torch.inference_mode():
            v = net.value_logits(net.encode(t))
            return torch.softmax(v, -1).float().cpu().numpy()
    bench(f"forward+softmax+cpu ({len(moves)} rows)", fwd, 1000)

    def fwd_nocopy():
        t = torch.from_numpy(x).to(dev)
        with torch.inference_mode():
            return net.value_logits(net.encode(t))
    bench("forward only (no .cpu())", fwd_nocopy, 1000)

    bench("opponent.choose_move(fen)", lambda: opponent.choose_move(fen), 2000)
    bench("terminal() check",
          lambda: __import__("tonkai.chess_env", fromlist=["x"]).terminal(
              board, chess.WHITE, 300), 3000)

    print("\nTRAINING per-step pieces")
    for _ in range(12):
        trainer.add_game(worker.play_game())
    print(f"  (replay: {len(trainer.replay)} value / "
          f"{len(trainer.predict_replay)} predict)")

    bench("replay.sample(512)  [unpack 512x99]",
          lambda: trainer.replay.sample(512), 200)
    if len(trainer.predict_replay) >= 64:
        bench("predict_replay.sample(64) [unpack 64x48x99]",
              lambda: trainer.predict_replay.sample(64), 100)

    x_np, y_np = trainer.replay.sample(512)
    xt = torch.from_numpy(x_np).to(dev)
    bench("rng.random((512,768))",
          lambda: np.random.default_rng().random((512, 768)).astype(np.float32), 200)

    def value_only():
        lat = net.encode(xt)
        vl = net.value_logits(lat)
        loss = torch.nn.functional.cross_entropy(
            vl, torch.from_numpy(y_np).to(dev))
        loss.backward()
        net.zero_grad(set_to_none=True)
    bench("value fwd+bwd (512 rows)", value_only, 200)

    bench("FULL train_step()", trainer.train_step, 100, unit="step")

    print("\nFULL GAME")
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        worker.play_game()
    dt = time.perf_counter() - t0
    print(f"  self-play only                        {n / dt:10.1f} games/s")

    t0 = time.perf_counter()
    for _ in range(n):
        trainer.add_game(worker.play_game())
        trainer.train(cfg.updates_per_game)
    dt = time.perf_counter() - t0
    print(f"  self-play + {cfg.updates_per_game} updates              "
          f"{n / dt:10.1f} games/s")


if __name__ == "__main__":
    main()
