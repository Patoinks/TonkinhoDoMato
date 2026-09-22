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
          '--device','cpu','--no-history','--win_pgn_path=',
          '--checkpoint_dir',sys.argv[1]]
threading.Timer(3, lambda: signal.raise_signal(signal.SIGINT)).start()
train.main()
"""
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,
                          text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'stopping after this game' in result.stdout
    assert (tmp_path/'latest.pt').exists()

def test_single_worker_training_repeats_under_seed():
    import numpy as np
    cfg=Config(device='cpu', seed=19,max_plies=16,opponent_depth=1,
               opponent_nodes=50,min_replay=1,samples_per_game=16)
    records=[]
    for _ in range(2):
        _,_,_,worker,tr,_=tonkai.build(cfg)
        run=[]
        for game in range(3):
            rec=worker.play_game(); run.append(rec.packed.copy())
            tr.add_game(rec); tr.train(cfg.updates_per_game)
        records.append(run)
    for a,b in zip(*records): np.testing.assert_array_equal(a,b)

def test_cpu_inference_replica_tracks_training_and_load(tmp_path):
    import torch,pytest
    if not torch.cuda.is_available(): pytest.skip('CUDA required for replica test')
    cfg=Config(device='cuda',inference_device='cpu',max_plies=12,
               min_replay=1,samples_per_game=12)
    net,agent,_,worker,tr,_=tonkai.build(cfg)
    assert agent.device == 'cpu' and next(net.parameters()).is_cuda
    tr.add_game(worker.play_game()); tr.train(1)
    for key,value in net.state_dict().items():
        assert torch.equal(value.cpu(),agent.net.state_dict()[key])
    path=str(tmp_path/'replica.pt'); tr.save(path)
    with torch.no_grad(): next(agent.net.parameters()).zero_()
    tr.load(path)
    for key,value in net.state_dict().items():
        assert torch.equal(value.cpu(),agent.net.state_dict()[key])

def test_valid_predict_sampling_preserves_bytes_and_rng():
    import numpy as np
    from tonkai.replay import PredictBuffer
    from tonkai.encoding import pack, encode
    import chess
    old=PredictBuffer(10,48,np.random.default_rng(23))
    new=PredictBuffer(10,48,np.random.default_rng(23))
    row=pack(encode(chess.Board(),True))
    for n in [1,2,17,48,64]:
        for buffer in [old,new]:buffer.add(np.tile(row,(n,1)),n-1)
    x,mask,truth=old.sample(32)
    valid,new_mask,new_truth=new.sample_valid(32)
    assert x[mask].tobytes()==valid.tobytes()
    assert mask.tobytes()==new_mask.tobytes()
    assert truth.tobytes()==new_truth.tobytes()
    assert old.rng.bit_generator.state == new.rng.bit_generator.state

def test_vectorized_pack_matches_every_clock_and_orientation():
    import numpy as np,chess
    from tonkai.encoding import encode,pack,pack_many,INPUT_SIZE
    from export_web import GOLDEN_FENS
    rows=[]
    for _,fen,_ in GOLDEN_FENS:
        b=chess.Board(fen)
        for clock in range(101):
            b.halfmove_clock=clock
            for color in [True,False]:rows.append(encode(b,color))
    x=np.stack(rows)
    assert pack_many(x).tobytes()==np.stack([pack(row) for row in x]).tobytes()
    assert pack_many(np.empty((0,INPUT_SIZE),dtype=np.float32)).shape==(0,99)

def test_piece_probe_preserves_float64_rng_and_rounding():
    import numpy as np,torch,chess
    from tonkai.encoding import encode,PLANE_SIZE
    cfg=Config(device='cpu',pieces_per_position=3)
    net,_,_,_,tr,_=tonkai.build(cfg)
    for count in [3,7,3]:
        x_np=np.tile(encode(chess.Board(),True),(count,1))
        flat=x_np[:,:PLANE_SIZE].copy().reshape(count,12,64)
        flat[:,[5,11]]=0;flat=flat.reshape(count,PLANE_SIZE)
        rng=np.random.default_rng();rng.bit_generator.state=tr.rng.bit_generator.state
        for _ in range(3):expected=rng.random((count,PLANE_SIZE)).astype(np.float32)*flat
        x=torch.from_numpy(x_np);latent=net.encode(x);vlogits=net.value_logits(latent)
        tr._piece_loss(x,x_np,latent,vlogits)
        assert tr._probe_scores.tobytes()==expected.tobytes()
        assert tr.rng.bit_generator.state==rng.bit_generator.state


def test_piece_probes_do_not_modify_replay_inputs_and_keep_gradients():
    import chess
    import numpy as np
    import torch
    from tonkai.encoding import encode

    cfg = Config(device='cpu', pieces_per_position=2)
    net, _, _, _, trainer, _ = tonkai.build(cfg)
    positions = np.stack([
        encode(chess.Board(), True),
        encode(chess.Board('7k/8/8/8/8/8/8/K7 w - - 0 1'), True),
    ])
    before = positions.copy()
    x = torch.from_numpy(positions)
    latent = net.encode(x)
    loss = trainer._piece_loss(x, positions, latent, net.value_logits(latent))
    loss.backward()
    assert positions.tobytes() == before.tobytes()
    # Target computation must be detached, while PieceHead still trains the
    # encoder. The king-only row must contribute neither a probe nor a metric.
    assert net.encoder[0].weight.grad is not None
    assert net.piece_head[-1].weight.grad is not None
    assert all(p.grad is None for p in net.value_head.parameters())
    assert sum(len(w.values) for w in trainer.metrics.piece_value.values()) == 2


def test_cuda_piece_metrics_preserve_pre_update_values_and_probe_order():
    import chess
    import numpy as np
    import pytest
    import torch
    from tonkai.encoding import encode, pack_many, PIECE_NAMES

    if not torch.cuda.is_available():
        pytest.skip('CUDA required for GPU metric verification')
    cfg = Config(device='cuda', pieces_per_position=2, replay_capacity=32,
                 predict_capacity=64, min_replay=1)
    net, _, _, _, trainer, metrics = tonkai.build(cfg)
    rows = np.stack([encode(chess.Board(), color) for color in (True, False)])
    trainer.replay.add_many(pack_many(rows), 2)
    expected = {name: [] for name in PIECE_NAMES}
    original = net.piece_values

    def capture(latent, types, squares, sides):
        values = original(latent, types, squares, sides)
        # Observe the original predictions to detect stale or mutated metrics.
        predictions = values.detach().cpu().numpy().copy()
        piece_types = types.cpu().numpy()
        for index, name in enumerate(PIECE_NAMES):
            expected[name].extend(map(float, predictions[piece_types == index]))
        return values

    net.piece_values = capture
    for size in (7, 3, 8):
        trainer.train_step(size)
        for name in PIECE_NAMES:
            assert list(metrics.piece_value[name].values) == expected[name]
    trainer.cfg.piece_value_weight = 0.0
    trainer.train_step(5)
    for name in PIECE_NAMES:
        assert list(metrics.piece_value[name].values) == expected[name]
