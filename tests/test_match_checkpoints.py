from tonkai.cognitive_config import CognitiveConfig
from scripts.match_checkpoints import _historical_config, _init, _state


def test_old_checkpoint_does_not_inherit_v5_activity():
    current = CognitiveConfig(piece_activity_weight=.08)
    old = _historical_config({"format": "tonkinho-snapshot-v1"}, current)
    assert old.piece_activity_weight == 0
    assert current.piece_activity_weight == .08
    assert _historical_config({"config": {"piece_activity_weight": .08}}).piece_activity_weight == .08
    assert _historical_config({"config": {"objective": "shaped"}}).piece_activity_weight == 0


def test_match_keeps_each_checkpoints_own_policy_and_piece_values(monkeypatch):
    import scripts.match_checkpoints as match

    current = CognitiveConfig(piece_activity_weight=.08, mate_search_moves=3)
    rival = CognitiveConfig(piece_activity_weight=0, mate_search_moves=0)

    def load(path, fallback=None):
        return ("current", object(), object(), current, {1: 1, 5: 9}) if path == "current" else (
            "rival", object(), object(), rival, {1: 1, 5: 8})

    monkeypatch.setattr(match, "_load", load)
    _init("current", "rival", None, {"mate_search_moves": 3})
    assert _state["ccfg"].piece_activity_weight == .08
    assert _state["rival_cfg"].piece_activity_weight == 0
    assert _state["kd"][5] == 9 and _state["rival_kd"][5] == 8
