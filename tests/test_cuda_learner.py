import copy
import pytest
import torch
from tonkai.cognitive_config import CognitiveConfig
from tonkai.survival_trainer import SurvivalTrainer
from tonkai.cuda_learner import CudaUnroll


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_unroll_forward_backward_and_updated_weights():
    torch.manual_seed(7)
    cfg = CognitiveConfig(device='cuda', hidden_size=32, latent_size=24, memory_size=16,
                          burn_in=2, sequence_length=4)
    trainer = SurvivalTrainer(cfg)
    reference = copy.deepcopy(trainer.net)
    graph = CudaUnroll(trainer)
    x = torch.randn(2, 8, 782, device='cuda')
    valid = torch.tensor([[0,1,1,1,1,0,0,0], [1]*8], device='cuda', dtype=torch.bool)
    for _ in range(2):
        trainer.net.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
        a = graph(trainer.net, x, valid, 2, 4, True, learner_window_only=True)
        b = trainer._unroll(reference, x, valid, 2, 4, True, learner_window_only=True)
        for actual, expected in zip(a, b): torch.testing.assert_close(actual, expected)
        a[0].square().sum().backward(); b[0].square().sum().backward()
        for p,q in zip(trainer.net.parameters(), reference.parameters()):
            if q.grad is not None:
                torch.testing.assert_close(p.grad, q.grad)
                with torch.no_grad(): p.add_(p.grad, alpha=-.001); q.add_(q.grad, alpha=-.001)
        x = torch.randn_like(x)
    for _ in range(2):
        a = graph(trainer.target, x, valid, 2, 4)
        b = trainer._unroll(trainer.target, x, valid, 2, 4)
        for actual, expected in zip(a,b): torch.testing.assert_close(actual, expected)
        x = torch.randn_like(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cpu_experiments_refresh_current_weights_without_moving_learner():
    from tonkai.cognitive_trainer import CognitiveTrainer
    trainer = CognitiveTrainer(CognitiveConfig(device='cuda'))
    first, device = trainer._experiment_policy()
    assert device == 'cpu' and next(first.parameters()).device.type == 'cpu'
    assert next(trainer.net.parameters()).device.type == 'cuda'
    old = next(first.parameters()).clone()
    with torch.no_grad(): next(trainer.net.parameters()).add_(.01)
    fresh, _ = trainer._experiment_policy()
    assert fresh is first
    torch.testing.assert_close(next(fresh.parameters()), next(trainer.net.parameters()).cpu())
    assert not torch.equal(old, next(fresh.parameters()))
