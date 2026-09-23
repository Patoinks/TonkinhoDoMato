"""Publish the archive of every TonkAI win (and the longest one) to the site.

Only static files change: ``wins.pgn`` and ``longest-win.pgn`` are copied into
both ``public/tonkai`` (kept by future builds) and ``dist/tonkai`` (deployed
now). Nothing is rebuilt. Before deploying, the local ``dist/index.html`` must
match the live one, so a deploy from this directory can never roll back a site
that was published from somewhere else.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tonkai.win_history import _atomic_write, _file_lock  # noqa: E402

WINS = ROOT / "history" / "wins"
LONGEST = ROOT / "history" / "longest_external_win.pgn"
RECEIPT = ROOT / "history" / "web-publications" / "wins-latest.json"


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def publish(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if not config.get("enabled"):
        return "disabled"
    site = Path(config["site_directory"]).resolve(strict=True)
    url = config["public_url"].rstrip("/")
    source = WINS / "wins.pgn"
    featured = WINS / "featured_game.pgn"
    if not featured.is_file() and not source.is_file():
        return "nothing to publish"
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    # The model publisher also deploys this site and holds publish.lock while it
    # does; wait for it rather than deploy a dist it is rebuilding.
    deadline = time.monotonic() + 1800
    while (RECEIPT.parent / "publish.lock").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("the model publisher still holds publish.lock")
        time.sleep(10)
    with _file_lock(RECEIPT.parent / ".wins-site.lock"):
        with _file_lock(WINS / ".wins.lock"):
            text = source.read_text(encoding="utf-8") if source.is_file() else ""
        game = featured.read_text(encoding="utf-8") if featured.is_file() else ""
        games = text.count("[Event ")
        local_index = (site / "dist" / "index.html").read_bytes()
        if hashlib.sha256(fetch(url + "/")).digest() != hashlib.sha256(local_index).digest():
            raise RuntimeError("live index.html differs from the local dist; refusing to roll the site back")
        for folder in (site / "public" / "tonkai", site / "dist" / "tonkai"):
            if text:
                _atomic_write(folder / "wins.pgn", text)
            if game:            # the site's one history game
                _atomic_write(folder / "featured-game.pgn", game)
            if LONGEST.is_file():
                _atomic_write(folder / "longest-win.pgn", LONGEST.read_text(encoding="utf-8"))
        deploy = ([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c",
                   "firebase deploy --only hosting --project anw-games --non-interactive"]
                  if os.name == "nt" else
                  ["firebase", "deploy", "--only", "hosting", "--project", "anw-games", "--non-interactive"])
        for attempt in range(3):
            if subprocess.run(deploy, cwd=site, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=600).returncode == 0:
                break
            if attempt == 2:
                raise RuntimeError("firebase deploy failed")
            time.sleep(2 ** (attempt + 1))
        for name, expected in (("wins.pgn", text), ("featured-game.pgn", game)):
            if not expected:
                continue
            live = fetch(url + f"/tonkai/{name}?v=" + str(time.time_ns())).decode("utf-8")
            if live.replace("\r\n", "\n") != expected.replace("\r\n", "\n"):
                raise RuntimeError(f"the live {name} does not match the local copy")
    return f"published {games} wins" + (" + featured game" if game else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "web_publish.local.json"))
    arguments = parser.parse_args()
    record = {"started_at": datetime.now(timezone.utc).isoformat()}
    try:
        record["state"] = publish(arguments.config)
        return 0
    except BaseException as error:
        record["state"], record["error"] = "failed", repr(error)
        return 1
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(RECEIPT, json.dumps(record, indent=2))


if __name__ == "__main__":
    sys.exit(main())
