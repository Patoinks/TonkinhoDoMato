"""Claudio runs on a frozen copy of the committed TonkAI code (claudio/base,
from git HEAD at the start of the duel, with its own tonkai_fast build), never
on the shared working tree, which the other side may be changing."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = HERE / "base"

# Opponents and judge (never teachers). Windows: the binaries in engines/; elsewhere the ones
# setup_cloud.sh builds (same versions: Stockfish 11, Fairy-Stockfish 14). Env vars override.
if sys.platform == "win32":
    _SF = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
    _FAIRY = ROOT / "engines/fairy-stockfish/fairy-stockfish_x86-64-bmi2.exe"
else:
    _SF, _FAIRY = ROOT / "engines/linux/stockfish-11", ROOT / "engines/linux/fairy-stockfish"
STOCKFISH = Path(os.environ.get("CLAUDIO_STOCKFISH", _SF))
FAIRY = Path(os.environ.get("CLAUDIO_FAIRY", _FAIRY))
_shared = {ROOT, ROOT / "scripts"}
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in _shared]
for path in (str(BASE / "scripts"), str(BASE), str(HERE / "fast"), str(HERE)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
