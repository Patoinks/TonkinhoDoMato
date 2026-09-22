from tonkai.config import Config
import tonkai

def test_sample_budget_and_legacy_schedule():
    cfg=Config(device='cpu', samples_per_game=11, updates_per_game=3, min_replay=0)
    tr=tonkai.build(cfg)[4]
    calls=[]
    tr.train_step=lambda size=None: calls.append(size)
    tr.train(3)
    assert calls == [4,4,3]
    cfg.samples_per_game=0
    tr.train(2)
    assert calls[-2:] == [None,None]
    assert Config.from_dict({'batch_size':512,'updates_per_game':4}).samples_per_game == 0
