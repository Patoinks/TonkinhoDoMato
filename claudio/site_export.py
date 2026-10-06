"""export_temporal_web.py run on Claudio's frozen code (claudio/base) instead of
the shared working tree; same arguments, same output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402  (frozen tonkai first)

sys.path.append(str(ROOT))          # only for export_web / export_temporal_web themselves
import runpy  # noqa: E402

runpy.run_path(str(ROOT / "export_temporal_web.py"), run_name="__main__")
