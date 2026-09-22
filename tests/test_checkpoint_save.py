from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tonkai.survival_trainer import SurvivalTrainer


def test_checkpoint_retries_transient_windows_lock_without_losing_old_file(tmp_path, monkeypatch):
    path = tmp_path / 'latest.pt'
    torch.save({'games': 100}, path)
    replace = Path.replace
    calls = []
    def locked_then_available(source, target):
        calls.append(source)
        if len(calls) <= 2:
            assert torch.load(target, weights_only=False)['games'] == 100
            raise PermissionError('Windows sharing lock')
        return replace(source, target)
    monkeypatch.setattr(Path, 'replace', locked_then_available)
    monkeypatch.setattr('tonkai.survival_trainer.time.sleep', lambda _: None)
    trainer = SimpleNamespace(state_dict=lambda: {'games': 200})
    SurvivalTrainer.save(trainer, path, status='running')
    assert len(calls) == 3
    assert torch.load(path, weights_only=False) == {'games': 200, 'status': 'running'}


def test_persistent_lock_keeps_original_checkpoint_and_reports_failure(tmp_path, monkeypatch):
    path = tmp_path / 'latest.pt'
    torch.save({'games': 100}, path)
    original = path.read_bytes()
    monkeypatch.setattr(Path, 'replace', lambda *args: (_ for _ in ()).throw(PermissionError('locked')))
    ticks = iter([0., 6.])
    monkeypatch.setattr('tonkai.survival_trainer.time.monotonic', lambda: next(ticks))
    with pytest.raises(PermissionError):
        SurvivalTrainer.save(SimpleNamespace(state_dict=lambda: {'games': 200}), path)
    assert path.read_bytes() == original
    assert torch.load(path.with_suffix('.pt.tmp'), weights_only=False)['games'] == 200
