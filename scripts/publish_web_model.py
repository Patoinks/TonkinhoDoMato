"""Validate, build, deploy and verify an immutable completed model snapshot."""
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


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def publish(checkpoint, config_path, allow_stopped=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    site = Path(config["site_directory"]).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    source_time = checkpoint.stat().st_mtime_ns
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    implementation = hashlib.sha256()
    files = [*sorted((ROOT / "tonkai").glob('*.py')), ROOT / 'export_temporal_web.py', ROOT / 'export_web.py',
             *sorted((site / 'src').rglob('*.ts')), *sorted((site / 'src').rglob('*.tsx'))]
    for file in files:
        implementation.update(str(file).encode()); implementation.update(file.read_bytes())
    implementation_hash = implementation.hexdigest()
    history = ROOT / "history/web-publications"
    history.mkdir(parents=True, exist_ok=True)
    receipt = history / "latest.json"
    last_success = history / "last-success.json"
    previous = json.loads(last_success.read_text()) if last_success.exists() else {}
    if (previous.get("checkpoint_sha256") == source_hash
            and previous.get("implementation_sha256") == implementation_hash):
        print("[site] This checkpoint is already published.", flush=True)
        return
    if previous.get("checkpoint_mtime_ns", 0) > source_time:
        raise ValueError("Refusing to replace a newer published checkpoint with an older one")
    lock = history / "publish.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = history / f"{stamp}-{source_hash[:12]}"
    directory.mkdir(exist_ok=True)
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=source_hash, checkpoint_mtime_ns=source_time,
                  implementation_sha256=implementation_hash,
                  started_at=datetime.now(timezone.utc).isoformat(), directory=str(directory), state="validating")
    log = (directory / "publish.log").open("a", encoding="utf-8", buffering=1)
    assets = site / "public/tonkai"
    installed = False
    deployed = False

    def step(args, cwd=site, timeout=300):
        log.write(f"\nCOMMAND {args!r}\n"); log.flush()
        result = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {args[0]}; see {log.name}")

    try:
        atomic(receipt, record)
        # Copy once; all checks and the deployed bytes refer to this same snapshot.
        snapshot = directory / "checkpoint.pt"
        shutil.copy2(checkpoint, snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Checkpoint changed while being snapshotted")
        import torch
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        kind = state.get("format")
        run_id = state.get("run_id")
        allowed = ("completed", "stopped") if allow_stopped else ("completed",)
        if kind in ("tonkai-cognitive-v3", "tonkai-survival-v2") and state.get("status") not in allowed:
            raise ValueError("Only completed checkpoints can be published automatically "
                             "(a cleanly stopped one only with --allow-stopped, by hand)")
        candidate = directory / "candidate"
        exporter = "export_temporal_web.py" if kind in ("tonkai-cognitive-v3", "tonkai-survival-v2") else "export_web.py"
        del state
        step([sys.executable, str(ROOT / exporter), "--checkpoint", str(snapshot), "--out", str(candidate)], ROOT)
        manifest = json.loads((candidate / "tonkai.json").read_text())
        # Legacy exports also receive immutable URLs; keep tonkai.bin for tools.
        versioned_binary = f"tonkai-{manifest['sha256']}.bin"
        if manifest.get("binary") != versioned_binary:
            shutil.copy2(candidate / "tonkai.bin", candidate / versioned_binary)
            manifest["binary"] = versioned_binary
            atomic(candidate / "tonkai.json", manifest)
        if not (candidate / "version.json").exists():
            atomic(candidate / "version.json", {"sha256": manifest["sha256"], "model_kind": manifest.get("model_kind", "legacy")})
        record.update(model_sha256=manifest["sha256"], model_kind=manifest.get("model_kind", "legacy"))
        node = shutil.which("node")
        if not node:
            raise RuntimeError("Node.js is required for browser parity checks")
        step([node, "--import", "tsx", "scripts/verify-tonkai-model.ts", str(candidate)])
        shutil.copytree(assets, directory / "previous-assets")
        installed = True
        shutil.copytree(candidate, assets, dirs_exist_ok=True)
        if run_id and (ROOT / "history" / (run_id + ".jsonl")).exists():
            step([node, "scripts/write-tonkai-final-status.mjs", str(ROOT / "history" / (run_id + ".jsonl")),
                  str(ROOT / "history" / run_id / "launch.json")])
        record["state"] = "building"; atomic(receipt, record)
        step([node, "--import", "tsx", "--test", *[str(p) for p in (site / "tests").glob("*.test.ts")]])
        step([node, "node_modules/typescript/bin/tsc", "-b"])
        step([node, "scripts/sync-tonkai-win.mjs"])
        step([node, "node_modules/vite/bin/vite.js", "build"])
        step([node, "node_modules/@playwright/test/cli.js", "test", "tests/browser/tonkai.spec.ts"], timeout=600)
        # Serve/validate the built bundle, not only the development module graph.
        step([node, "scripts/smoke-tonkai-production.mjs"])
        record["state"] = "publishing"; atomic(receipt, record)
        firebase = shutil.which("firebase.cmd") if os.name == "nt" else shutil.which("firebase")
        if not firebase:
            raise RuntimeError("Firebase CLI is not installed")
        # Windows .cmd launch requires cmd.exe; all arguments are fixed literals.
        deploy = ([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "firebase deploy --only hosting --project anw-games --non-interactive"]
                  if os.name == "nt" else [firebase, "deploy", "--only", "hosting", "--project", "anw-games", "--non-interactive"])
        for attempt in range(3):
            try:
                step(deploy)
                deployed = True
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        url = config["public_url"].rstrip("/")
        with urllib.request.urlopen(url + "/tonkai/tonkai.json?v=" + manifest["sha256"], timeout=30) as response:
            live = json.load(response)
        if live["sha256"] != manifest["sha256"]:
            raise ValueError("Published manifest does not match the validated model")
        with urllib.request.urlopen(url + "/tonkai/" + live.get("binary", "tonkai.bin"), timeout=30) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError("Published binary failed its checksum")
        record.update(state="published", finished_at=datetime.now(timezone.utc).isoformat())
        atomic(last_success, record)
        print(f"[site] Published {record['model_kind']} {manifest['sha256'][:12]} to {url}/tonkai", flush=True)
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        # A failed candidate cannot silently become the next status-only deployment.
        if installed and not deployed and (directory / "previous-assets").exists():
            shutil.copytree(directory / "previous-assets", assets, dirs_exist_ok=True)
            shutil.copytree(directory / "previous-assets", site / "dist/tonkai", dirs_exist_ok=True)
        raise
    finally:
        atomic(receipt, record); atomic(directory / "result.json", record)
        log.close(); lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(ROOT / "web_publish.local.json"))
    parser.add_argument("--allow-stopped", action="store_true",
                        help="owner's explicit request: publish a cleanly stopped campaign checkpoint")
    arguments = parser.parse_args()
    publish(arguments.checkpoint, arguments.config, arguments.allow_stopped)
