"""Recover draw reasons from retained replay without interrupting a trainer.

Results are cached by monotonic episode ID because replay eventually evicts
games. Missing episodes remain unknown; no reasons are inferred from scores.
Only episodes in the requested session and published metrics range are counted.
"""
import argparse
import json
from pathlib import Path
import sys

import chess
import torch


def summarize(checkpoint, cache, run_id, initial_games, completed_games, draws):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "tonkai-survival-v2" or state.get("run_id") != run_id:
        raise ValueError("Checkpoint does not belong to the requested survival run")
    replay = state["replay"]
    if replay["next_episode_id"] != state["games"]:
        raise ValueError("Episode IDs no longer correspond to completed game indices")
    cache = Path(cache)
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if saved.get("run_id") != run_id:
        saved = {"run_id": run_id, "reasons": {}, "through_games": 0}
    reasons = saved["reasons"]
    for episode in replay["episodes"]:
        episode_id = episode["id"]
        if (episode_id < initial_games or not episode["terminated"] or episode["lost"]
                or str(episode_id) in reasons):
            continue
        board = chess.Board(episode.get("initial_fen") or chess.STARTING_FEN)
        for action in episode["actions"]:
            board.push_uci(action)
        outcome = board.outcome(claim_draw=True)
        if outcome is not None and outcome.winner is None:
            reasons[str(episode_id)] = outcome.termination.name.lower()
    saved["through_games"] = state["games"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    temporary.write_text(json.dumps(saved), encoding="utf-8")
    temporary.replace(cache)
    counts = {}
    for episode_id, reason in reasons.items():
        if initial_games <= int(episode_id) < initial_games + completed_games:
            counts[reason] = counts.get(reason, 0) + 1
    known = sum(counts.values())
    if known > draws:
        raise ValueError("Recovered draw count exceeds the published session total")
    return {"reasons": counts, "unknown": draws - known,
            "recovered": known, "checkpointGames": state["games"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--initial-games", type=int, required=True)
    parser.add_argument("--completed-games", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = summarize(args.checkpoint, args.cache, args.run_id, args.initial_games,
                       args.completed_games, args.draws)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
