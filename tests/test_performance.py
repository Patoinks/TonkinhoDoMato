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

def test_parallel_records_refresh_and_shutdown():
    import numpy as np
    from tonkai.parallel import ActorPool
    cfg=Config(device='cpu', num_workers=2, max_plies=12, opponent_depth=1,
               opponent_nodes=50, weight_refresh_games=1)
    net,_,_,_,tr,m=tonkai.build(cfg)
    with ActorPool(cfg,net,m) as pool:
        rec=pool.next_game()
        assert rec.plies == 12
        assert rec.packed.shape == (12,99)
        assert rec.tonkai_positions == m.tonkai_positions
        tr.add_game(rec)
        assert tr.games == m.games == 1
        version=pool.version.value
        pool.refresh(net,1)
        assert pool.version.value == version+1
        assert np.isfinite(rec.packed).all()
    assert all(not p.is_alive() for p in pool.processes)

def test_parallel_actor_failure_is_reported():
    import pytest
    from tonkai.parallel import ActorPool
    cfg=Config(device='cpu', num_workers=2,max_plies=10)
    net,_,_,_,_,m=tonkai.build(cfg)
    with ActorPool(cfg,net,m) as pool:
        for p in pool.processes:
            p.terminate(); p.join()
        with pytest.raises(RuntimeError,match='exited'):
            pool.next_game()

def test_parallel_cli_sigint_checkpoints(tmp_path):
    import subprocess,sys
    code = """
import signal, threading, sys
import train
sys.argv=['train.py','--games','100000','--workers','2','--max_plies','12',
          '--device','cpu','--no-history','--checkpoint_dir',sys.argv[1]]
threading.Timer(3, lambda: signal.raise_signal(signal.SIGINT)).start()
train.main()
"""
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,
                          text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'stopping after this game' in result.stdout
    assert (tmp_path/'latest.pt').exists()
