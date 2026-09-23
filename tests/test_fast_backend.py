"""Backend selection must survive fresh imports (Windows actor spawn)."""
import os
import subprocess
import sys

import pytest

from tonkai import fast_rules


@pytest.mark.parametrize("requested,missing,expected", [
    ("python", False, "python"), ("auto", True, "python"),
    ("cpp", True, "unavailable"), ("invalid", False, "must be"),
    ("cpp", False, "cpp"),
])
def test_backend_selection_in_fresh_process(requested, missing, expected):
    if requested == "cpp" and not missing and not fast_rules.AVAILABLE:
        pytest.skip("extension not built")
    code = "import sys; " + ("sys.modules['tonkai_fast'] = None; " if missing else "")
    code += "from tonkai.fast_rules import BACKEND; print(BACKEND)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "TONKAI_RULES_BACKEND": requested}, timeout=30)
    assert expected in result.stdout + result.stderr
    assert (result.returncode == 0) == (expected in ("cpp", "python"))


def test_chess960_is_not_silently_treated_as_standard_chess():
    import chess
    with pytest.raises(ValueError, match="standard chess"):
        fast_rules.board(chess.Board.from_chess960_pos(42))
