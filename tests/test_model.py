from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from tonkai.config import Config
from tonkai.encoding import INPUT_SIZE, encode, encode_after_moves
from tonkai.model import TonkAINet, probabilities, utility


@pytest.fixture
def net():
    torch.manual_seed(0)
    return TonkAINet(Config())


def test_parameter_budget(net):
    n = net.n_parameters()
    assert 100_000 <= n < 1_000_000, n


def test_shapes(net):
    x = torch.randn(7, INPUT_SIZE)
    latent = net.encode(x)
    assert latent.shape == (7, 128)
    assert net.value_logits(latent).shape == (7, 3)
    assert net.predict_logits(latent).shape == (7,)
    pv = net.piece_values(
        latent,
        torch.randint(0, 6, (7,)),
        torch.randint(0, 64, (7,)),
        torch.randint(0, 2, (7,)),
    )
    assert pv.shape == (7,)


def test_batched_evaluation_equals_one_at_a_time(net):
    """The batch shortcut must not change a single value."""
    board = chess.Board()
    moves = list(board.legal_moves)
    batch = torch.from_numpy(encode_after_moves(board, moves, chess.WHITE))
    with torch.no_grad():
        batched = net.value_logits(net.encode(batch))
        singles = []
        for mv in moves:
            board.push(mv)
            x = torch.from_numpy(encode(board, chess.WHITE))[None, :]
            singles.append(net.value_logits(net.encode(x))[0])
            board.pop()
    torch.testing.assert_close(batched, torch.stack(singles), atol=1e-5, rtol=1e-4)


def test_utility_is_bounded_and_ordered():
    logits = torch.tensor([[10.0, 0.0, 0.0],    # sure win
                           [0.0, 10.0, 0.0],    # sure draw
                           [0.0, 0.0, 10.0]])   # sure loss
    u = utility(logits, draw_weight=0.1)
    assert u[0] > u[1] > u[2]
    assert u[0] == pytest.approx(1.0, abs=1e-3)
    assert u[1] == pytest.approx(0.1, abs=1e-3)
    assert u[2] == pytest.approx(-1.0, abs=1e-3)


def test_probabilities_sum_to_one(net):
    p = probabilities(net.value_logits(net.encode(torch.randn(5, INPUT_SIZE))))
    torch.testing.assert_close(p.sum(-1), torch.ones(5))


def test_piece_head_receives_gradient(net):
    """The auxiliary loss must actually reach PieceHead."""
    x = torch.randn(4, INPUT_SIZE)
    latent = net.encode(x)
    pv = net.piece_values(
        latent, torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
    )
    pv.sum().backward()
    grads = [p.grad for p in net.piece_head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert net.type_embed.weight.grad is not None


def test_piece_value_depends_on_the_position_not_just_the_piece(net):
    """Two different boards must be able to value the same piece differently."""
    a = torch.from_numpy(encode(chess.Board(), chess.WHITE))[None, :]
    b = torch.from_numpy(
        encode(chess.Board("8/8/8/3R4/8/8/8/K6k w - - 0 1"), chess.WHITE)
    )[None, :]
    t = torch.tensor([3])   # rook
    s = torch.tensor([35])
    side = torch.tensor([0])
    with torch.no_grad():
        va = net.piece_values(net.encode(a), t, s, side)
        vb = net.piece_values(net.encode(b), t, s, side)
    assert not torch.allclose(va, vb)


def test_no_classical_piece_value_table_in_the_package():
    """Guard against someone quietly teaching pawn=1, knight=3, bishop=3...

    Looks at real code only - comments and docstrings are allowed to mention
    the thing they are promising not to do.
    """
    import ast
    import pathlib

    classic = [{1, 3, 5, 9}, {100, 320, 330, 500, 900}, {100, 300, 500, 900}]
    src = pathlib.Path(__file__).resolve().parents[1] / "tonkai"
    for f in sorted(src.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
                continue
            items = node.values if isinstance(node, ast.Dict) else node.elts
            nums = {
                abs(e.value)
                for e in items
                if isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
            }
            for table in classic:
                assert not table <= nums, (
                    f"{f.name}:{node.lineno} looks like a piece-value table: {nums}"
                )
