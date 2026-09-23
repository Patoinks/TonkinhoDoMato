"""Add evaluation games (a PGN) to a campaign's piece-value statistics.

Only the who-killed-whom record (and threats after the first move) is added,
under the given opponent kind; kills/deaths totals, game counts and training
data are untouched. A copy of the checkpoint is kept next to it first.

    python scripts/add_games_to_values.py history/matches/selfplay-v4-500.pgn --kind self
"""
import argparse
from pathlib import Path
import shutil
import sys

import chess
import chess.pgn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tonkai.captures import CaptureStats, parse_weights  # noqa: E402
from tonkai.cognitive_config import CognitiveConfig  # noqa: E402


def games(path, kind):
    with open(path, encoding="utf-8") as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            yield list(game.mainline_moves()), chess.WHITE, "unfinished", None, kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--kind", default="self", help="opponent kind the games count as")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("status") not in ("completed", "stopped"):
        parser.error("the campaign is not stopped: never write a checkpoint a training run is using")
    cfg = CognitiveConfig.from_dict(state["config"])
    stats = CaptureStats(cfg.hot_piece_kills, opponent_weights=parse_weights(cfg.value_opponents))
    stats.load_state_dict(state["capture_stats"])
    before = stats.piece_scores()
    added = stats.seed_new_measures(games(args.pgn, args.kind))
    backup = args.checkpoint.with_name(args.checkpoint.stem + ".before-values.pt")
    shutil.copy2(args.checkpoint, backup)
    state["capture_stats"] = stats.state_dict()
    torch.save(state, args.checkpoint)
    print(f"{added} games added as '{args.kind}'; backup {backup.name}")
    print("scores before:", before)
    print("scores now:   ", stats.piece_scores())
    print("values (pawn = 1):", stats.piece_values())


if __name__ == "__main__":
    main()
