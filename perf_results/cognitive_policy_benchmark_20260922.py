"""Matched-position latency only; does not measure chess strength or production GPS."""
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import torch

from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer
from tonkai.survival_agent import SurvivalAgent


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    source = root / "perf_results/cognitive-validation/verified/latest.pt"
    state = torch.load(source, weights_only=False, map_location="cpu")
    cfg = CognitiveConfig.from_dict(state["config"])
    cfg.planning_nodes = 96
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(state)
    boards = []
    for episode in state["replay"]["episodes"][-40:]:
        board = chess.Board(episode["initial_fen"] or chess.STARTING_FEN)
        for action in episode["actions"][:12]:
            board.push_uci(action)
        if not board.is_game_over(claim_draw=True):
            boards.append(board)
        if len(boards) == 16:
            break
    timings = {"v2": [], "v3": []}
    nodes, same_move = [], 0
    for repeat in range(3):
        for board in boards:
            selected = {}
            order = ("v2", "v3") if repeat % 2 == 0 else ("v3", "v2")
            for version in order:
                agent = (SurvivalAgent(trainer.net, cfg) if version == "v2"
                         else CognitiveAgent(trainer.net, cfg, trainer.knowledge))
                agent.observe(board, board.turn)  # same history, outside candidate timing
                started = time.perf_counter()
                choice = agent.evaluate(board, board.turn, explore=False)
                elapsed = time.perf_counter() - started
                selected[version] = choice.moves[int(choice.scores.argmax())].uci()
                if repeat:
                    timings[version].append(elapsed * 1000)
                    if version == "v3":
                        nodes.append(agent.diagnostics["planning_nodes"])
            if repeat:
                same_move += selected["v2"] == selected["v3"]
    result = {"source": str(source), "positions": len(boards), "measured_repeats": 2,
              "torch_threads": 1, "planning_nodes_cap": cfg.planning_nodes,
              "archive_entries": len(trainer.knowledge.memory.entries),
              "production_training_concurrent": True, "identical_selected_moves": same_move,
              "max_planning_nodes_observed": max(nodes),
              "latency_ms": {key: {"mean": statistics.mean(values), "median": statistics.median(values),
                                    "max": max(values)} for key, values in timings.items()},
              "scope": "candidate evaluation only; identical weights/positions; V3 intentionally changes policy; no strength claim"}
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
