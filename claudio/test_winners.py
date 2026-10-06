"""--learn winners: only the winning side's moves are learned, as 'even better' than the
other moves of the same position. Checked on games already played (nothing is saved).

    python -m pytest claudio/test_winners.py -q
"""
import json
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

from tonkai import fast_rules  # noqa: E402
from tonkai.encoding import pack_many  # noqa: E402
from tonkai.sequence_replay import SequenceEpisode, SequenceReplay  # noqa: E402

GAMES = ROOT / "claudio/results/engine-grande-claudio-grande-g0-sf1800.json"
START = ROOT / "claudio/claudio-grande-g0.pt"
pytestmark = pytest.mark.skipif(not (GAMES.exists() and START.exists()), reason="needs Claudio's saved games")


def winner_view(moves, winner):
    """The positions of a game from the winner's side, as train_grande.play builds them."""
    view = fast_rules.board(chess.Board())
    rows = [np.asarray(view.encode_here(winner), dtype=np.float32)]
    for uci in moves:
        view.push(fast_rules.code(chess.Move.from_uci(uci)))
        rows.append(np.asarray(view.encode_here(winner), dtype=np.float32))
    return pack_many(np.stack(rows))


def winner_moves_probability(trainer, episodes):
    """Mean softmax probability (of the values, /tau) of the move each winner played."""
    net, tau, total, count = trainer.net, trainer.choice_tau, 0.0, 0
    with torch.no_grad():
        for moves, winner in episodes:
            board, memory = chess.Board(), net.initial_memory(1)
            for ply, uci in enumerate(moves):
                fast = fast_rules.board(board)
                x = torch.from_numpy(np.asarray(fast.encode_here(winner), dtype=np.float32)[None])
                memory = net.memory_cell(net.encode(x), memory)
                if board.turn == winner and ply % 5 == 0:
                    codes = list(fast.legal_moves())
                    context, _ = net.candidates(torch.from_numpy(fast_rules.encode_after_moves(fast, codes, winner)),
                                                memory)
                    p = (net.survival(context)[0].mean(-1) / tau).softmax(0)
                    total += float(p[codes.index(fast_rules.code(chess.Move.from_uci(uci)))])
                    count += 1
                board.push_uci(uci)
    return total / count


def test_winners_moves_become_more_likely_without_teaching_bad():
    import train_grande as tg
    from match_checkpoints import _historical_config
    from claudio_load import load_claudio
    games = json.loads(GAMES.read_text(encoding="utf-8"))["list"]
    decided = [(g["moves"].split(), (chess.WHITE if g["white"] else chess.BLACK) if g["result"] == "win"
                else (chess.BLACK if g["white"] else chess.WHITE)) for g in games if g["result"] in ("win", "loss")][:24]
    draws = [g for g in games if g["result"] == "draw"]
    assert decided and draws
    net, *_ = load_claudio(START, with_config=True)
    net.requires_grad_(True).train()
    net.rnd_target.requires_grad_(False)
    cfg = _historical_config(torch.load(tg.TIMDGEI, map_location="cpu", weights_only=False))
    cfg.device, cfg.sequence_batch_size, cfg.min_replay = "cpu", 16, 64
    Trainer = tg.make_trainer_class()
    Trainer.choice_tau = 0.05
    trainer = Trainer(cfg, net=net)
    replay = SequenceReplay(1_000_000, seed=1)
    for moves, winner in decided:
        replay.add(SequenceEpisode(packed=winner_view(moves, winner), lost=False, terminated=True,
                                   opponent="winner test (claudio)", actions=tuple(moves), result_value=1.0))
    before = winner_moves_probability(trainer, decided[:6])
    steps = [trainer.winner_update(replay, 16) for _ in range(12)]
    after = winner_moves_probability(trainer, decided[:6])
    assert all(loss is not None and np.isfinite(loss) for loss in steps)
    assert after > before                                         # the winners' moves are now preferred
