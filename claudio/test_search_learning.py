"""Learning from its own calculation (train_grande --learn search): what the search records
and the buffer that keeps it.

    python -m pytest claudio/test_search_learning.py -q
"""
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401

from tonkai import fast_rules  # noqa: E402
from tonkai.encoding import INPUT_SIZE, pack_many, unpack_many  # noqa: E402
from tonkai.temporal_model import SurvivalNet  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig  # noqa: E402
from train_grande import SearchBuffer  # noqa: E402

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
VALUES = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0}


def test_the_search_records_a_target_for_each_of_its_best_candidates():
    torch.manual_seed(0)
    cfg = ClaudioConfig.with_settings(["reply_top=6", "reply_mode=expect", "mate_attack=0"])
    me = Claudio(SurvivalNet(None), VALUES, cfg, rng=np.random.default_rng(0))
    me.trace = []
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
    me.choose(board)
    assert len(me.trace) == 1
    rows, memory, targets = me.trace[0]
    assert rows.shape == (6, INPUT_SIZE) and memory.shape == (6, me.net.memory_size) and targets.shape == (6,)
    assert memory.dtype == np.float16 and ((targets >= 0) & (targets <= 1)).all()
    np.testing.assert_array_equal(unpack_many(pack_many(rows)), rows)      # the encoding packs losslessly


def test_a_candidate_that_allows_mate_in_one_is_taught_zero():
    torch.manual_seed(0)
    cfg = ClaudioConfig.with_settings(["reply_top=100", "reply_mode=expect", "mate_attack=0"])
    me = Claudio(SurvivalNet(None), VALUES, cfg, rng=np.random.default_rng(0))
    me.trace = []
    board = chess.Board("6k1/1b3ppp/8/8/8/6q1/5PP1/5RK1 w - - 0 1")    # ...Qxg2# after most moves
    me.choose(board)
    _, _, targets = me.trace[0]
    assert (targets == 0).any() and (targets > 0).any()


def test_buffer_keeps_the_latest_rows_up_to_its_capacity():
    buffer = SearchBuffer(5)
    packed = np.arange(7 * 99, dtype=np.uint8).reshape(7, 99)
    memory = np.ones((7, 4), dtype=np.float16)
    buffer.add(packed[:3], memory[:3], np.arange(3, dtype=np.float32))
    buffer.add(packed[3:], memory[3:], np.arange(3, 7, dtype=np.float32))
    assert buffer.size == 5
    assert sorted(buffer.target.tolist()) == [2, 3, 4, 5, 6]
