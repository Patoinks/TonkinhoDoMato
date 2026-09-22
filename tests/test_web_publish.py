import json
from types import SimpleNamespace

from tonkai.web_publish import publish_completed


def setup(tmp_path):
    checkpoint = tmp_path / 'production/latest.pt'
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b'checkpoint')
    config = tmp_path / 'publish.json'
    config.write_text(json.dumps(dict(enabled=True, checkpoint_roots=[str(checkpoint.parent)])))
    return checkpoint, config


def test_only_completed_authorized_checkpoints_publish(tmp_path, monkeypatch):
    checkpoint, config = setup(tmp_path)
    calls = []
    monkeypatch.setattr('tonkai.web_publish.subprocess.run', lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0))
    assert not publish_completed(checkpoint, completed=False, config_path=config)
    assert not publish_completed(tmp_path / 'test/latest.pt', completed=True, config_path=config)
    assert calls == []
    assert publish_completed(checkpoint, completed=True, config_path=config)
    assert calls[0][3] == str(checkpoint.resolve())


def test_disabled_or_failed_publication_preserves_checkpoint(tmp_path, monkeypatch):
    checkpoint, config = setup(tmp_path)
    monkeypatch.setattr('tonkai.web_publish.subprocess.run', lambda *a, **kw: SimpleNamespace(returncode=1))
    assert not publish_completed(checkpoint, completed=True, config_path=config)
    assert checkpoint.read_bytes() == b'checkpoint'
    config.write_text(json.dumps(dict(enabled=False)))
    assert not publish_completed(checkpoint, completed=True, config_path=config)
