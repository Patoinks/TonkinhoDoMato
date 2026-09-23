import importlib.util
import json
from pathlib import Path
import time

import pytest

spec = importlib.util.spec_from_file_location('tracker_data',Path(__file__).resolve().parents[1]/'desktop/tracker_data.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_tail_holds_partial_lines_and_final_state_overrides_same_game_count(tmp_path):
    path = tmp_path/'events.jsonl'
    path.write_bytes(b'{"kind":"metrics","games":25,')
    tail = module.JsonlTail(path); tail.poll()
    assert not tail.metrics
    with path.open('ab') as stream: stream.write(b'"state":"running"}\n')
    tail.poll(); assert tail.metrics[25]['state']=='running'
    with path.open('ab') as stream: stream.write(b'{"kind":"metrics","games":25,"state":"failed"}\n')
    tail.poll(); assert tail.metrics[25]['state']=='failed'
    tail.poll(); assert len(tail.metrics)==1


def test_campaign_resume_counts_and_process_failure_are_honest(tmp_path):
    stamp = time.time()
    def run(name, offset, count, losses, state='running'):
        directory = tmp_path/'history'/name; directory.mkdir(parents=True)
        history = directory/'events.jsonl'
        row = dict(kind='metrics',games=count,losses=losses,wins=0,draws=0,truncated=0,state=state,
                   timestamp=module.datetime.fromtimestamp(stamp,module.timezone.utc).isoformat(),
                   survival_plies_mean=50,games_per_second=4,loss={})
        history.write_text(json.dumps(row)+'\n')
        launch = dict(run_id=name,pid=123,resume_games=1000+offset,campaign_start_games=1000,
                      campaign_cap=100000,additional_games=100000-offset,history=str(history),
                      checkpoint=str(tmp_path/'checkpoints/latest.pt'))
        (directory/'launch.json').write_text(json.dumps(launch))
    run('one',0,200,200,'failed')
    time.sleep(.01)
    run('two',200,25,225)
    reader=module.TrainingReader(tmp_path,alive=lambda _:True)
    result=reader.snapshot()
    assert result['games']==225 and result['counts']['losses']==225
    assert result['total']==100000 and result['state']=='running'
    assert result['points'][-1]['plies']==225*50
    reader.alive=lambda _:False
    result=reader.snapshot()
    assert result['state']=='interrupted' and result['eta'] is None


def write_run(project, name, offset, rows, config=None, mixture=None, elo=800):
    directory = project/'history'/name
    directory.mkdir(parents=True)
    history = directory/'events.jsonl'
    launch = dict(run_id=name, pid=123, resume_games=1000+offset, campaign_start_games=1000,
                  campaign_cap=1000, additional_games=1000-offset, history=str(history),
                  checkpoint=str(project/'checkpoints/latest.pt'),
                  opponent=dict(engine_name='Rodent', elo=elo, move_time=.002),
                  opponent_mixture=mixture or {'external': 1})
    records = [dict(kind='run', config=config or {})]
    for row in rows:
        records.append(dict(kind='metrics', state='running',
                            **{'survival_plies_mean':50,
                               'timestamp':module.datetime.now(module.timezone.utc).isoformat(), **row}))
    history.write_text(''.join(json.dumps(row)+'\n' for row in records))
    (directory/'launch.json').write_text(json.dumps(launch))
    return history


def test_mixture_score_resume_resignation_and_repeated_poll(tmp_path):
    row = dict(games=6, wins=2, draws=1, losses=2, truncated=1,
               opponents={'self': {'win':1, 'loss':1}, 'tiny':{'draw':1,'unfinished':1},
                          'external':{'win':1,'resigned':1}})
    write_run(tmp_path, 'a', 0, [row], config={'mix_self':.3})
    reader = module.TrainingReader(tmp_path, alive=lambda _:True)
    first = reader.snapshot()
    assert sum(r['games'] for r in first['opponents']) == 6
    assert next(r for r in first['opponents'] if r['kind']=='tiny')['score'] == .5
    time.sleep(.01)
    row = dict(games=4, wins=3, draws=2, losses=4, truncated=1,
               opponents={'self':{'draw':1}, 'external':{'loss':1}, 'tiny':{'win':1,'loss':1}})
    history = write_run(tmp_path, 'b', 6, [row], config={'mix_self':.3})
    reader.scan()
    result = reader.snapshot()
    opponents = {r['kind']:r for r in result['opponents']}
    assert result['games'] == sum(r['games'] for r in opponents.values()) == 10
    assert opponents['external']['losses'] == 2 and opponents['external']['resigned'] == 1
    assert opponents['self']['score'] == .5
    assert opponents['tiny']['score'] == .5 and opponents['tiny']['truncated'] == 1
    assert reader.snapshot()['opponents'] == result['opponents']
    # A final metrics record at the same count replaces, rather than adds, results.
    with history.open('a') as stream:
        stream.write(json.dumps(dict(kind='metrics', state='completed', timestamp=module.datetime.now(module.timezone.utc).isoformat(), **row))+'\n')
    assert reader.snapshot()['opponents'] == result['opponents']


def test_resume_drops_unsaved_games_and_distinguishes_engine_elo(tmp_path):
    write_run(tmp_path, 'a', 0, [dict(games=2,wins=1,losses=1), dict(games=4,wins=3,losses=1)])
    time.sleep(.01)
    write_run(tmp_path, 'b', 2, [dict(games=1,wins=1,losses=2)], elo=1000)
    result = module.TrainingReader(tmp_path, alive=lambda _:True).snapshot()
    assert result['games'] == 3
    rows = {r['elo']:r for r in result['opponents']}
    assert rows[800]['games'] == 2 and rows[800]['score'] == .5
    assert rows[1000]['games'] == 1 and rows[1000]['score'] == 0
    assert max(p['games'] for p in result['points']) == 3


def test_missing_prefix_and_unattributed_old_mixture_are_explicit(tmp_path):
    write_run(tmp_path, 'a', 20, [dict(games=3,wins=1,losses=22,
               opponents={'tiny':{'win':1,'loss':2}})])
    rows = {r['kind']:r for r in module.TrainingReader(tmp_path).snapshot()['opponents']}
    assert rows['unknown']['games'] == 20
    assert rows['tiny']['score'] == pytest.approx(1/3)
    other = tmp_path/'older'
    write_run(other, 'a', 0, [dict(games=10,wins=2,losses=8)], config={'mix_self':.5})
    rows = module.TrainingReader(other).snapshot()['opponents']
    assert len(rows) == 1 and rows[0]['kind'] == 'unknown' and rows[0]['games'] == 10


def test_only_unfinished_games_have_no_score_and_fairy_elo_comes_from_run(tmp_path):
    write_run(tmp_path, 'a', 0, [dict(games=2,truncated=2,opponents={'fairy':{'unfinished':2}})],
              config={'fairy_elo':1200, 'opponent_move_time':.003})
    row = module.TrainingReader(tmp_path).snapshot()['opponents'][0]
    assert row['score'] is None and row['win_rate'] is None
    assert row['points'] == 0 and row['elo'] == 1200 and row['move_time'] == .003


def test_self_play_is_separate_from_external_totals_after_resume(tmp_path):
    write_run(tmp_path, 'a', 0, [dict(games=5,wins=1,losses=2,
              opponents={'self':{'win':2},'external':{'win':1,'loss':2}})])
    time.sleep(.01)
    write_run(tmp_path, 'b', 5, [dict(games=5,wins=2,losses=2,draws=1,
              games_per_second=2,positions_per_second=80,survival_plies_mean=60,
              opponents={'self':{'loss':3},'tiny':{'win':1,'draw':1}})])
    result = module.TrainingReader(tmp_path).snapshot()
    rows = {row['kind']:row for row in result['opponents']}
    assert set(rows) == {'self','tiny','external'}
    assert rows['self']['games'] == 5 and rows['self']['score'] == .4
    assert rows['external']['games'] == 3 and rows['tiny']['score'] == .75
    assert result['counts'] == dict(wins=2, losses=2, draws=1, truncated=0)
    assert result['turns'] == 20  # The external-only duration must not label all games.


def test_all_metrics_keep_unknown_fields_nested_arrays_zero_and_null():
    rows = dict(module.metric_rows({'opponents': {'self': {'wins': 0}},
                                   'loss': {'new_head': .125}, 'future': [1, None],
                                   'empty': {}, 'enabled': False}))
    assert rows['opponents.self.wins'] == '0'
    assert rows['loss.new_head'] == '0.125'
    assert rows['future[1]'] == 'null'
    assert rows['empty'] == '{}'
    assert rows['enabled'] == 'false'


def test_self_play_rows_name_each_frozen_checkpoint(tmp_path):
    from desktop.tracker_data import opponent_totals
    snapshots = tmp_path / "ckpt/snapshots"
    snapshots.mkdir(parents=True)
    for version in (44254, 44754, 45254):
        (snapshots / f"tonkinho-jogo-{version}.pt").write_bytes(b"")
    launch = {"checkpoint": str(tmp_path / "ckpt/latest.pt"), "campaign_start_games": 0, "resume_games": 0}
    raw = {"opponents": {"self": {"win": 5, "draw": 3, "loss": 4}, "tiny": {"win": 1, "loss": 9}},
           "self_versions": {"44754": {"win": 2, "draw": 1, "loss": 1}, "45254": {"win": 3, "draw": 2, "loss": 3}}}
    counts = {"wins": 1, "draws": 0, "losses": 9, "truncated": 0}
    table = opponent_totals({}, raw, 22, counts, launch, {"mix_self": .1, "mix_tiny": .5}, True)
    names = sorted(row["name"] for row in table.values() if row["kind"] == "self")
    assert names == ["Tonkinho self-play · Checkpoint 2 (jogo 44754)",
                     "Tonkinho self-play · Checkpoint 3 (jogo 45254)"]
    rows = {row["name"]: row for row in table.values()}
    assert rows["Tonkinho self-play · Checkpoint 3 (jogo 45254)"]["wins"] == 3
