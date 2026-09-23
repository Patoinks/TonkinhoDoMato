from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
import torch

from tonkai.inference_broker import InferenceBroker, EncoderClient


def test_broker_batches_without_replacing_inflight_weights():
    torch.set_num_threads(1)
    model = torch.nn.Linear(5, 3).eval()
    x = torch.randn(4, 5)
    expected = model(x).detach().clone()
    broker = InferenceBroker(get_context('spawn'), 3, wait_seconds=.02)
    try:
        old = broker.register(model)
        assert broker.register(model) == old
        with torch.no_grad(): model.weight.add_(1)
        new = broker.register(model)
        clients = [EncoderClient(broker.requests, broker.responses[i], i, version)
                   for i, version in enumerate((old, old, new))]
        with ThreadPoolExecutor(3) as pool:
            results = list(pool.map(lambda c: c.encode(x), clients))
        torch.testing.assert_close(results[0], expected)
        torch.testing.assert_close(results[1], expected)
        torch.testing.assert_close(results[2], model(x))
        broker.release(old); assert old in broker.models
        broker.release(old); assert old not in broker.models
        broker.release(new); assert not broker.models
        assert broker.stats['requests'] == 3
    finally:
        broker.close()
