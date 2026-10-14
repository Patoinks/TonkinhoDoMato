"""Opt-in local completion hook. Failed/stopped/test runs never deploy a model."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "web_publish.local.json"


def publish_completed(checkpoint, *, completed, config_path=None):
    config_path = Path(config_path or CONFIG)
    if not completed or not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    checkpoint = Path(checkpoint).resolve()
    roots = [(ROOT / p).resolve() for p in config.get("checkpoint_roots", ["checkpoints"])]
    if not config.get("enabled") or not any(checkpoint.is_relative_to(root) for root in roots):
        return False
    print("[site] Final checkpoint saved; validating and publishing the playable model...", flush=True)
    # No shell interpolation. Publication is synchronous after actors/checkpoints
    # close, so returning from a successful run also confirms deployment outcome.
    try:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/publish_web_model.py"),
                                 "--checkpoint", str(checkpoint), "--config", str(config_path)], cwd=ROOT)
    except OSError as error:
        print(f"[site] Could not launch publication: {error}. Training checkpoint is safe.", flush=True)
        return False
    if result.returncode:
        print("[site] Publication failed; training checkpoint is safe. See history/web-publications/latest.json", flush=True)
    return result.returncode == 0


def publish_wins(wins_dir, *, config_path=None):
    """Start a background deploy of the win archive; returns the process or None.

    Only the real archive (history/wins) with an enabled local config is ever
    published, so test runs and other directories never reach the site.
    """
    config_path = Path(config_path or CONFIG)
    if not config_path.is_file() or Path(wins_dir).resolve() != (ROOT / "history/wins").resolve():
        return None
    if not json.loads(config_path.read_text(encoding="utf-8-sig")).get("enabled"):
        return None
    try:
        return subprocess.Popen([sys.executable, str(ROOT / "scripts/publish_wins.py"), "--config", str(config_path)],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as error:
        print(f"[site] Could not launch win publication: {error}", flush=True)
        return None
