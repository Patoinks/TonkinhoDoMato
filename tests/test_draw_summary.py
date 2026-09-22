import importlib.util
import json
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location("draw_summary", Path(__file__).parents[1] / "scripts/summarize_draws.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recover_repetition_cache_eviction_and_metrics_boundary(tmp_path):
    checkpoint, cache = tmp_path / "checkpoint.pt", tmp_path / "cache.json"
    episode = {"id": 5, "terminated": True, "lost": False, "initial_fen": "",
               "actions": ["g1f3", "g8f6", "f3g1", "f6g8"] * 2}
    state = {"format": "tonkai-survival-v2", "run_id": "run", "games": 6,
             "replay": {"next_episode_id": 6, "episodes": [episode]}}
    torch.save(state, checkpoint)
    summary = module.summarize(checkpoint, cache, "run", 4, 1, 0)
    assert summary["recovered"] == 0  # checkpoint ahead of published metrics
    state["replay"]["episodes"] = []  # evicted, but recoverable from cache
    torch.save(state, checkpoint)
    summary = module.summarize(checkpoint, cache, "run", 4, 2, 2)
    assert summary["reasons"] == {"threefold_repetition": 1}
    assert summary["unknown"] == 1  # never invent missing reasons


def test_only_draws_of_this_session_count(tmp_path):
    checkpoint, cache = tmp_path / "checkpoint.pt", tmp_path / "cache.json"
    episodes = [
        {"id": 0, "terminated": True, "lost": False, "initial_fen": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "actions": []},
        {"id": 1, "terminated": True, "lost": False, "initial_fen": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "actions": []},
        {"id": 2, "terminated": True, "lost": False, "initial_fen": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "actions": []},
        {"id": 3, "terminated": False, "lost": False, "initial_fen": "", "actions": []},
    ]
    torch.save({"format": "tonkai-survival-v2", "run_id": "run", "games": 4,
                "replay": {"next_episode_id": 4, "episodes": episodes}}, checkpoint)
    summary = module.summarize(checkpoint, cache, "run", 1, 3, 1)
    assert summary["reasons"] == {"stalemate": 1}
    assert summary["unknown"] == 0
