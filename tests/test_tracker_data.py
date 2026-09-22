import importlib.util
import json
from pathlib import Path
import time

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
