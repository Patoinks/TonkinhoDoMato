"""Read-only campaign analysis from cumulative logs; excludes synthetic experiments."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = ['cognitive-20260922T191855Z-45299b23', 'cognitive-20260922T192207Z-99df804b']
points = {0: dict(games=0, plies=0, wins=0, draws=0, losses=0, truncated=0)}
metrics = []
for run in runs:
    launch = json.loads((ROOT / 'history' / run / 'launch.json').read_text())
    offset = launch['resume_games'] - launch['campaign_start_games']
    base = points[offset]
    lines = (ROOT / 'history' / (run + '.jsonl')).read_bytes().decode().splitlines()
    for index, line in enumerate(lines):
        try: row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1: raise
            continue  # Writer may be in the middle of the final line.
        if row.get('kind') != 'metrics': continue
        count = row['games']
        point = dict(games=offset + count, plies=base['plies'] + round(count * row['survival_plies_mean']),
            **{key: row[key] for key in ('wins', 'draws', 'losses', 'truncated')},
            timestamp=row['timestamp'], loss=row['loss'], gps=row['games_per_second'])
        assert sum(point[key] for key in ('wins', 'draws', 'losses', 'truncated')) == point['games']
        points[point['games']] = point

def window(start, end):
    a, b = points[start], points[end]
    result = dict(start=start + 1, end=end, games=end-start,
        mean_plies=(b['plies']-a['plies'])/(end-start),
        **{key:b[key]-a[key] for key in ('wins','draws','losses','truncated')})
    result['nonloss_percent'] = 100 * (result['wins'] + result['draws']) / result['games']
    snapshots = [point['loss'] for n, point in points.items() if start < n <= end]
    result['mean_logged_training_losses'] = {key: float(np.mean([loss[key] for loss in snapshots]))
        for key in ('survival_loss', 'reply_loss', 'future_loss', 'piece_loss')}
    return result

end = max(points)
first, last = window(0, 1000), window(end-1000, end)
# Approximate uncertainty for the difference in duration. Resample contiguous
# 100-game block means, not cumulative averages or fictitious individual games.
# This does not remove nonstationarity or prove checkpoint playing strength.
early = np.array([window(n, n+100)['mean_plies'] for n in range(0,1000,100)])
late = np.array([window(n, n+100)['mean_plies'] for n in range(end-1000,end,100)])
rng = np.random.default_rng(122)
delta = rng.choice(late, (20000, len(late)), replace=True).mean(1) - rng.choice(early, (20000, len(early)), replace=True).mean(1)
report = dict(snapshot=points[end], first_1000=first, last_1000=last,
    duration_change_plies=last['mean_plies']-first['mean_plies'],
    duration_change_percent=100*(last['mean_plies']/first['mean_plies']-1),
    approximate_95_percent_block_bootstrap_interval_plies=np.quantile(delta, [.025,.975]).tolist(),
    block_size_games=100, bootstrap_repetitions=20000,
    windows_500=[window(n,min(n+500,end)) for n in range(0,end,500)],
    limitations=['Cumulative logs have no individual-game duration variance.',
      'Block interval assumes approximately independent 100-game blocks; training is nonstationary.',
      'Exploratory training policy and changing replay; no frozen-checkpoint evaluation.',
      'Synthetic counterfactual/generalization branches are excluded.',
      'Training loss snapshots are not held-out predictive accuracy.'])
out = Path(__file__).with_suffix('.json')
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
