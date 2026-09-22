"""Export V2/V3 inference state and reference Python decisions for the browser."""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from export_web import GOLDEN_FENS, export_weights
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import SurvivalAgent
from tonkai.temporal_model import SurvivalNet
from tonkai.king_safety import king_senses


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def export(checkpoint, out):
    torch.set_num_threads(1)
    checkpoint, out = Path(checkpoint), Path(out)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = state.get("format")
    if kind not in ("tonkai-survival-v2", "tonkai-cognitive-v3"):
        raise ValueError(f"Unsupported temporal checkpoint: {kind}")
    cfg = CognitiveConfig.from_dict(state["config"])
    cfg.device = "cpu"
    net = SurvivalNet(cfg).eval()
    net.load_state_dict(state["model"])
    knowledge = None
    if kind == "tonkai-cognitive-v3":
        knowledge = CognitiveKnowledge(cfg, net.encoder)
        knowledge.load_state_dict(state["knowledge"])
        agent = CognitiveAgent(net, cfg, knowledge)
    else:
        agent = SurvivalAgent(net, cfg)
    # Keep all learned heads, including those used only during training. Frozen
    # retrieval encoder has its own namespace and must not use the live encoder.
    bundle = torch.nn.ModuleDict({"net": net})
    if knowledge:
        bundle["knowledge_encoder"] = knowledge.encoder
    out.mkdir(parents=True, exist_ok=True)
    tensors, blob, size = export_weights(bundle, str(out))
    if not all(torch.isfinite(t).all() for t in bundle.state_dict().values()):
        raise ValueError("Non-finite model weights")
    digest = hashlib.sha256(blob).hexdigest()
    filename = f"tonkai-{digest}.bin"
    (out / filename).write_bytes(blob)
    data = dict(format="float32-le", model_kind=kind, binary=filename, sha256=digest,
                bytes=size, tensors=tensors, draw_weight=cfg.draw_weight,
                games_trained=state["games"], pretrained_games=state.get("pretrained_games", 0),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                config={key: getattr(cfg, key) for key in ("memory_size", "ensemble_size", "memory_neighbors",
                    "memory_similarity", "memory_weight", "planning_nodes", "planning_candidates", "planning_replies",
                    "planning_uncertainty", "planning_gap", "calibration_min_samples", "robust_weight", "survival_gamma",
                    "skill_min_support", "skill_bonus", "piece_safety_weight", "piece_exchange_tolerance",
                    "piece_exchange_depth", "king_safety_weight")}, inference="deterministic; exploration disabled")
    if knowledge:
        data["knowledge"] = plain(dict(entries=[{k:e[k] for k in ("key", "target", "game", "trust", "own_turn")}
                                               for e in knowledge.memory.entries],
            skills=knowledge.skills.skills, calibration=knowledge.calibration.state_dict()))
    data["revision"] = hashlib.sha256(json.dumps({"weights": digest, "config": data["config"],
        "knowledge": data.get("knowledge")}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    (out / "tonkai.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    (out / "version.json").write_text(json.dumps({"sha256": digest, "revision": data["revision"], "model_kind": kind}), encoding="utf-8")

    cases = []
    fixtures = [(label, fen, [], colour) for label, fen, colour in GOLDEN_FENS]
    fixtures += [
        ("history_white", chess.STARTING_FEN, ["e2e4", "e7e5", "g1f3", "b8c6"], True),
        ("history_black", chess.STARTING_FEN, ["d2d4", "d7d5", "c2c4"], False),
        ("repeat_claim", chess.STARTING_FEN,
         ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"], True),
        ("mate", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", [], True),
        ("pinned_defender", "k3r3/8/8/8/5n2/3N4/4B3/4K3 b - - 0 1", [], False),
        ("multiple_attackers", "4k3/8/3n1n2/8/4N3/3P1P2/8/4K3 b - - 0 1", [], False),
        ("exchange_xray", "k3r3/8/8/8/3rP3/8/8/4R2K b - - 0 1", [], False),
        ("capture_promotion", "7k/8/8/8/8/8/1p6/R6K b - - 0 1", [], False),
        ("king_shelter", "5rk1/8/8/2b4q/8/8/5PPP/6K1 w - - 0 1", [], True),
        ("king_exposed", "5rk1/8/8/2b4q/8/8/8/6K1 w - - 0 1", [], True),
        ("king_pinned_defender", "k3r3/8/8/8/7q/8/4B3/4K3 w - - 0 1", [], True),
        ("king_endgame", "8/7k/8/8/3K4/8/4P3/8 w - - 0 1", [], True),
    ]
    for label, fen, moves, colour in fixtures:
        board = chess.Board(fen)
        for action in moves:
            board.push_uci(action)
        agent.reset()
        agent.observe(board, colour)
        with torch.no_grad():
            value = float(net.survival(agent.context)[0].mean())
        entry = dict(label=label, root=fen, history=moves, self="w" if colour else "b", value=value,
                     memory=agent.memory[0].tolist())
        if knowledge:
            entry["piece_senses"] = agent.sense_pieces(board, colour)
            entry["king_senses"] = king_senses(board)
        if board.turn == colour:
            choice = agent.choose(board, colour, np.random.default_rng(1), explore=False, temperature=0)
            entry["rank"] = [{"uci": move.uci(), "utility": float(score)}
                             for move, score in zip(choice.moves, choice.scores)]
            entry["best"] = choice.move.uci()
        else:
            replies, probabilities = agent.predict_reply(board, colour)
            entry["predict"] = [{"uci": m.uci(), "p": float(p)} for m, p in zip(replies, probabilities)]
        cases.append(entry)
    golden = dict(model_kind=kind, sha256=digest, tolerance=2e-4, cases=cases)
    (out / "golden.json").write_text(json.dumps(golden, allow_nan=False), encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = export(args.checkpoint, args.out)
    print(json.dumps({k:result[k] for k in ("model_kind", "sha256", "games_trained", "bytes")}))
