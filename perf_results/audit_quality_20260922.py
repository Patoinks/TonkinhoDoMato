"""Reproduce the frozen-checkpoint quality audit without training or saving weights.

Run from any working directory:
    python perf_results/audit_quality_20260922.py

The two KQvK positions are colour mirrors, not independent tactical examples.
Immediate mates are verified with python-chess, not a search engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import numpy as np
import torch

from tonkai.agent import TonkAIAgent
from tonkai.config import Config
from tonkai.model import TonkAINet


POSITIONS = [
    ("queen_endgame_white", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
    ("queen_endgame_black", "8/8/8/8/8/6k1/5q2/7K b - - 0 1"),
    (
        "scholars_mate",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    ),
]


def audit_checkpoint(relative_path: str) -> dict:
    path = ROOT / relative_path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(payload["config"])
    net = TonkAINet(cfg)
    net.load_state_dict(payload["model"])
    net.eval()
    agent = TonkAIAgent(net, cfg, "cpu")
    metrics = payload.get("metrics", {})
    item = {
        "checkpoint": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "games": payload.get("games"),
        "steps": payload.get("steps"),
        "saved_at": payload.get("saved_at"),
        "stored_config": payload.get("config", {}),
        "stored_metrics": {
            key: metrics.get(key)
            for key in ("games", "wins", "draws", "losses", "termination")
        },
        "stored_metrics_note": "Checkpoint metrics may cover only the current resumed run.",
        "initial_wdl": agent.value_of(chess.Board(), chess.WHITE)[0].tolist(),
        "positions": [],
    }
    for name, fen in POSITIONS:
        board = chess.Board(fen)
        assert board.is_valid(), fen
        choice = agent.evaluate_moves(board, board.turn)
        assert choice is not None
        order = np.argsort(-choice.utilities)
        ranks = {int(index): rank + 1 for rank, index in enumerate(order)}
        best = int(order[0])
        mates = []
        for index, move in enumerate(choice.moves):
            san = board.san(move)
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                mates.append({
                    "move": move.uci(),
                    "san": san,
                    "utility_rank": ranks[index],
                    "utility": float(choice.utilities[index]),
                    "wdl": choice.probs[index].tolist(),
                })
        assert mates, name
        item["positions"].append({
            "name": name,
            "fen": fen,
            "legal_moves": len(choice.moves),
            "mate_count": len(mates),
            "mates": mates,
            "argmax": choice.moves[best].uci(),
            "argmax_san": board.san(choice.moves[best]),
            "argmax_is_mate": any(
                mate["move"] == choice.moves[best].uci() for mate in mates
            ),
            "argmax_utility": float(choice.utilities[best]),
            "argmax_wdl": choice.probs[best].tolist(),
        })
    return item


def audit_history() -> list:
    summaries = []
    for path in sorted((ROOT / "history").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics = [row for row in rows if row.get("kind") == "metrics"]
        last = metrics[-1] if metrics else {}
        summaries.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run": next((row for row in rows if row.get("kind") == "run"), {}),
            "last_metrics": {
                key: last.get(key)
                for key in ("games", "wins", "draws", "losses", "termination", "final", "timestamp")
            },
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "scope": "Read-only frozen checkpoint inference. No training, no search. Each child is checked for immediate mate with python-chess.",
        "limitations": "Three positions per checkpoint; two are colour mirrors. This is a diagnostic, not an estimate of playing strength.",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "python_chess": chess.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "checkpoints": [audit_checkpoint(path) for path in (
            "checkpoints/latest.pt", "checkpoints/deployed_5020games.pt"
        )],
        "history": audit_history(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for checkpoint in report["checkpoints"]:
        position = next(row for row in checkpoint["positions"] if row["name"] == "scholars_mate")
        mate = position["mates"][0]
        print(f"{checkpoint['checkpoint']}: {mate['san']} rank {mate['utility_rank']}/{position['legal_moves']}; "
              f"P(loss)={mate['wdl'][2]:.9f}; argmax={position['argmax_san']}")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
