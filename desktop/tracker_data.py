"""Read-only, incremental training telemetry. Never opens a checkpoint or engine."""
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import time


RESULTS = ('wins', 'draws', 'losses', 'truncated')


def result_counts(row):
    """Mixture logs use singular outcomes; resigned is an exclusive loss there."""
    return dict(wins=int(row.get('wins', row.get('win', 0))),
                draws=int(row.get('draws', row.get('draw', 0))),
                losses=int(row['losses'] if 'losses' in row else row.get('loss', 0) + row.get('resigned', 0)),
                truncated=int(row.get('truncated', row.get('unfinished', 0))))


def opponent_identity(kind, launch, config):
    info = launch.get('opponent_catalog', {}).get(kind, {})
    if kind == 'external':
        info = info or launch.get('opponent', {})
        name = info.get('engine_name', 'Adversário externo')
    elif kind == 'fairy':
        name = info.get('engine_name', 'Fairy-Stockfish')
        info = {**info, 'elo': info.get('elo', config.get('fairy_elo')),
                'move_time': info.get('move_time', config.get('opponent_move_time'))}
    else:
        name = {'self': 'Tonkinho (self-play)', 'tiny': 'TinyOpponent · profundidade 2',
                'unknown': 'Sem adversário registado'}.get(kind, kind)
    identity = json.dumps([kind, name, info.get('elo'), info.get('move_time'), info.get('sha256')])
    return dict(id=identity, kind=kind, name=name, elo=info.get('elo'), move_time=info.get('move_time'))


def snapshot_versions(launch):
    """Game counts of the frozen self-play rivals saved next to the checkpoint, in order."""
    folder = Path(launch.get('checkpoint', '')).parent / 'snapshots'
    versions = []
    for path in folder.glob('tonkinho-jogo-*.pt'):
        try:
            versions.append(int(path.stem.rsplit('-', 1)[1]))
        except ValueError:
            continue
    return sorted(versions)


def rival_identity(version, launch):
    """'Tonkinho self-play · Checkpoint 3 (jogo 45500)': the frozen copy it played."""
    version = int(version)
    known = snapshot_versions(launch)
    ordinal = known.index(version) + 1 if version in known else None
    label = f'Checkpoint {ordinal} (jogo {version})' if ordinal else f'jogo {version}'
    name = f'Tonkinho self-play · {label}'
    return dict(id=json.dumps(['self', 'snapshot', version]), kind='self', name=name, elo=None,
                move_time=None, version=version)


def add_opponent(table, identity, counts, resigned=0):
    row = table.setdefault(identity['id'], {**identity, **dict.fromkeys(RESULTS, 0), 'resigned': 0})
    for key in RESULTS:
        row[key] += counts.get(key, 0)
    row['resigned'] += resigned


def opponent_totals(base, raw, count, counts, launch, config, exact_base):
    """Merge session-only mixture counters into the checkpoint's campaign prefix.

    Old single-engine sessions can be attributed from their counter deltas.
    Missing history or old mixed sessions remain explicitly unattributed.
    """
    table = {key: dict(value) for key, value in base.get('opponents', {}).items()}
    if isinstance(raw.get('opponents'), dict):
        versions = raw.get('self_versions') if isinstance(raw.get('self_versions'), dict) else {}
        for kind, results in raw['opponents'].items():
            counts_here = result_counts(results)
            if kind == 'self' and versions:
                # One row per frozen Tonkinho; anything not attributed stays generic.
                for version, per_version in versions.items():
                    add_opponent(table, rival_identity(version, launch), result_counts(per_version),
                                 int(per_version.get('resigned', 0)))
                    for key, value in result_counts(per_version).items():
                        counts_here[key] -= value
                if not any(value > 0 for value in counts_here.values()):
                    continue
                counts_here = {key: max(0, value) for key, value in counts_here.items()}
            add_opponent(table, opponent_identity(kind, launch, config), counts_here,
                         int(results.get('resigned', 0)) if kind != 'self' or not versions else 0)
    else:
        mixture = launch.get('opponent_mixture', {})
        mixed = any(value for key, value in mixture.items() if key != 'external') or any(
            config.get(key, 0) for key in ('mix_self', 'mix_tiny', 'mix_fairy'))
        if not mixed:
            session = result_counts(raw)
            if sum(session.values()) != count:
                session = ({key: counts[key] - base.get(key, 0) for key in RESULTS}
                           if exact_base else {})
            if session and all(value >= 0 for value in session.values()):
                add_opponent(table, opponent_identity('external', launch, config), session)
    # An incomplete historical prefix must not be assigned to the current engine.
    campaign = launch.get('campaign_start_games', launch.get('resume_games', 0))
    campaign_games = max(0, launch.get('resume_games', 0) - campaign) + count
    # New mixture runs exclude self-play from the global W/D/L. Its separate
    # table row is still useful, but must never be counted as an external result.
    self_excluded = isinstance(raw.get('opponents'), dict) and sum(counts.values()) < campaign_games
    included = [row for row in table.values() if not self_excluded or row['kind'] != 'self']
    missing = {key: counts[key] - sum(row[key] for row in included) for key in RESULTS}
    if any(value < 0 for value in missing.values()):
        # Incompatible/restarted telemetry: preserve honest totals, not guessed attribution.
        table = {key: row for key, row in table.items() if self_excluded and row['kind'] == 'self'}
        missing = counts
    if any(missing.values()):
        add_opponent(table, opponent_identity('unknown', launch, config), missing)
    return table


def score_rows(table):
    rows = []
    for value in table.values():
        row = dict(value)
        finished = sum(row[key] for key in ('wins', 'draws', 'losses'))
        points = row['wins'] + .5 * row['draws']
        row.update(games=finished + row['truncated'], finished=finished, points=points,
                   score=points / finished if finished else None,
                   win_rate=row['wins'] / finished if finished else None)
        rows.append(row)
    return sorted(rows, key=lambda row: (row['kind'] == 'unknown', -row['games'], row['name']))


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
        self.config = {}

    def poll(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            self.offset, self.pending, self.metrics, self.config = 0, b'', {}, {}
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
            elif row.get('kind') == 'run':
                self.config = row.get('config', {})


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
            exact_base = offset in points
            base = points[max(n for n in points if n <= offset)]
            # A resume starts from its saved prefix; later unsaved games are discarded.
            points = {n: point for n, point in points.items() if n <= offset}
            tail = self.tails.setdefault(run['history'], JsonlTail(run['history']))
            tail.poll()
            for n, row in sorted(tail.metrics.items()):
                count = offset + n
                counts = {key: int(row.get(key, 0)) for key in ('wins', 'draws', 'losses', 'truncated')}
                if offset and 'opponents' not in row and sum(counts.values()) == n:  # Older session-only counters.
                    counts = {key: value + base.get(key, 0) for key, value in counts.items()}
                mean_plies = row.get('survival_plies_mean', 0)
                if row.get('games_per_second', 0) > 0 and 'positions_per_second' in row:
                    mean_plies = row['positions_per_second'] / row['games_per_second']
                points[count] = dict(games=count, **counts, timestamp=timestamp(row['timestamp']),
                    plies=base.get('plies', 0) + round(n * mean_plies),
                    session_mean_turns=mean_plies / 2,
                    metrics=row, opponents=opponent_totals(base, row, n, counts, run, tail.config, exact_base))
                if run['run_id'] == launch['run_id']:
                    latest = points[count]
        total = launch.get('campaign_cap', max(0, launch.get('resume_games', 0) - campaign) + launch['additional_games'])
        if latest is None:
            return dict(state='starting', launch=launch, total=total, points=[], games=max(points), age=0,
                        opponents=score_rows(base.get('opponents', {})))
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
                    draw_reasons=row.get('draw_reasons', {}), metrics=row, config=tail.config,
                    opponents=score_rows(latest['opponents']))


def metric_rows(value, prefix=''):
    """Flatten every recorded field, including zeros, nulls and new metrics."""
    if isinstance(value, dict):
        result = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            result.extend(metric_rows(child, f'{prefix}.{key}' if prefix else str(key)))
        return result or [(prefix, '{}')]
    if isinstance(value, (list, tuple)):
        result = []
        for index, child in enumerate(value):
            result.extend(metric_rows(child, f'{prefix}[{index}]'))
        return result or [(prefix, '[]')]
    return [(prefix, json.dumps(value, ensure_ascii=False))]
