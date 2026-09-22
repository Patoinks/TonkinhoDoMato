"""Read-only, incremental training telemetry. Never opens a checkpoint or engine."""
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import time


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def process_alive(pid):
    if not pid:
        return False
    if os.name != 'nt':
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None if ctypes.get_last_error() == 5 else False
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b''
        self.metrics = {}

    def poll(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            self.offset, self.pending, self.metrics = 0, b'', {}
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()
        # Final metrics always end with a newline. Hold an incomplete record
        # until the writer finishes it; do not mistake it for a failure.
        for line in lines:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if row.get('kind') == 'metrics':
                self.metrics[int(row['games'])] = row


class TrainingReader:
    def __init__(self, project, alive=process_alive):
        self.project = Path(project)
        self.alive = alive
        self.runs = []
        self.tails = {}
        self.scanned_at = -100.

    def scan(self):
        runs = []
        for path in (self.project / 'history').glob('*/launch.json'):
            try:
                launch = json.loads(path.read_text(encoding='utf-8-sig'))
                if 'history' in launch and 'checkpoint' in launch:
                    runs.append((path.stat().st_mtime_ns, launch))
            except (OSError, ValueError):
                continue
        self.runs = sorted(runs, key=lambda item: item[0])
        self.scanned_at = time.monotonic()

    def snapshot(self):
        if time.monotonic() - self.scanned_at >= 10:
            self.scan()
        if not self.runs:
            return None
        launch = self.runs[-1][1]
        campaign = launch.get('campaign_start_games', launch.get('resume_games', 0))
        identity = os.path.normcase(os.path.abspath(launch['checkpoint']))
        related = [item for _, item in self.runs if os.path.normcase(os.path.abspath(item['checkpoint'])) == identity
                   and item.get('campaign_start_games', item.get('resume_games', 0)) == campaign]
        points = {0: dict(games=0, plies=0, wins=0, draws=0, losses=0, truncated=0)}
        latest = None
        for run in related:
            offset = max(0, run.get('resume_games', 0) - campaign)
            base = points.get(offset, {})
            tail = self.tails.setdefault(run['history'], JsonlTail(run['history']))
            tail.poll()
            for n, row in sorted(tail.metrics.items()):
                count = offset + n
                counts = {key: int(row.get(key, 0)) for key in ('wins', 'draws', 'losses', 'truncated')}
                if offset and sum(counts.values()) == n:  # Older session-only counters.
                    counts = {key: value + base.get(key, 0) for key, value in counts.items()}
                points[count] = dict(games=count, **counts, timestamp=timestamp(row['timestamp']),
                    plies=base.get('plies', 0) + round(n * row.get('survival_plies_mean', 0)),
                    session_mean_turns=row.get('survival_plies_mean', 0) / 2,
                    metrics=row)
                if run['run_id'] == launch['run_id']:
                    latest = points[count]
        total = launch.get('campaign_cap', max(0, launch.get('resume_games', 0) - campaign) + launch['additional_games'])
        if latest is None:
            return dict(state='starting', launch=launch, total=total, points=[], games=max(points), age=0)
        row = latest['metrics']
        alive = self.alive(launch.get('pid'))
        age = max(0, time.time() - latest['timestamp'])
        if row.get('state') in ('completed', 'failed', 'stopped'):
            state = row['state']
        elif alive is False:
            state = 'interrupted'
        elif age > 120:
            state = 'stale'
        else:
            state = 'running'
        series = [p for _, p in sorted(points.items()) if 'timestamp' in p]
        recent = [p for p in series if p['timestamp'] >= latest['timestamp'] - 300]
        speed = row.get('games_per_second', 0)
        if len(recent) >= 2:
            elapsed = latest['timestamp'] - recent[0]['timestamp']
            if elapsed > 0:
                speed = (latest['games'] - recent[0]['games']) / elapsed
        # Non-overlapping intervals, smoothed over up to four metric snapshots.
        rates, duration = [], []
        for i in range(1, len(series)):
            a, b = series[max(0, i-4)], series[i]
            games, elapsed = b['games'] - a['games'], b['timestamp'] - a['timestamp']
            if games > 0 and elapsed > 0:
                rates.append((b['games'], games / elapsed))
                duration.append((b['games'], (b['plies']-a['plies']) / games / 2))
        return dict(state=state, launch=launch, total=total, points=series, games=latest['games'],
                    counts={key: latest[key] for key in ('wins','draws','losses','truncated')},
                    age=age, timestamp=latest['timestamp'], speed=speed, mean_speed=row.get('games_per_second', 0),
                    eta=(total-latest['games'])/speed if state == 'running' and speed > 0 else None,
                    turns=latest['session_mean_turns'], rates=rates, duration=duration,
                    draw_reasons=row.get('draw_reasons', {}), metrics=row)
