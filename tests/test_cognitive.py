from copy import deepcopy

import chess
import numpy as np
import pytest
import torch

from tonkai.cognitive_agent import CognitiveAgent, terminal_value
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_memory import ErrorCalibration, ExperienceMemory, SkillLibrary, structural_keys
from tonkai.cognitive_trainer import CognitiveTrainer, at_ply
from tonkai.encoding import encode, pack_many
from tonkai.sequence_replay import SequenceEpisode
from tonkai.survival_agent import play_episode, SurvivalChoice


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    options = dict(device="cpu", hidden_size=24, latent_size=16, memory_size=12, head_hidden=8,
                   piece_hidden=8, ensemble_size=3, sequence_length=8, burn_in=2,
                   sequence_batch_size=2, replay_capacity=1000, min_replay=1, max_plies=16,
                   num_workers=1, archive_capacity=12, protected_episodes=2, skill_capacity=8,
                   planning_nodes=12, experiment_plies=4, protected_replay_every=1)
    options.update(kwargs)
    return CognitiveConfig(**options)


class RandomOpponent:
    description = "random-test"

    def reset(self):
        pass

    def choose(self, board, rng):
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]


def scripted(trainer, colour=chess.WHITE):
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    board, positions = chess.Board(), []
    rng = np.random.default_rng(8)
    positions.append(encode(board, colour))
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        if board.turn == colour:
            agent.choose(board, colour, rng, explore=False)
            agent.trace[-1]["move"] = uci
            # Forced scripted moves differ from the policy's proposal; discard
            # candidate evidence belonging to the unplayed action.
            agent.trace[-1].pop("next_key", None)
            agent.trace[-1].pop("next_prediction", None)
        board.push_uci(uci)
        agent.observe(board, colour)
        positions.append(encode(board, colour))
    episode = SequenceEpisode(pack_many(np.stack(positions)), colour == chess.WHITE, True,
                              actions=tuple(move.uci() for move in board.move_stack))
    return episode, agent.trace, board


def test_calibration_detects_shared_overconfidence_and_roundtrips():
    calibration = ErrorCalibration(8)
    calibration.observe(np.full(20, .9), np.zeros(20))
    assert calibration.uncertainty(np.array([.9]), np.array([0]))[0] > .8
    restored = ErrorCalibration(8)
    restored.load_state_dict(calibration.state_dict())
    np.testing.assert_array_equal(restored.count, calibration.count)
    np.testing.assert_allclose(restored.uncertainty(np.array([.9]), [0]), calibration.uncertainty(np.array([.9]), [0]))


def test_structural_keys_are_side_relative_and_sensitive_to_positions():
    board = chess.Board()
    x = np.stack([encode(board, True), encode(board.mirror(), False)])
    keys = structural_keys(x, np.ones((2, 16)))
    np.testing.assert_allclose(keys[0], keys[1])
    board.push_uci("e2e4")
    changed = structural_keys(encode(board, True)[None], np.ones((1, 16)))
    assert not np.array_equal(keys[0], changed[0])


def test_associative_retrieval_uses_similar_keys_and_independent_games():
    memory = ExperienceMemory(config())
    a, b = np.array([1., 0.]), np.array([0., 1.])
    assert memory.query(a[None])["information"][0] == 1
    memory.add(a, .8, .2, 1, chess.STARTING_FEN)
    memory.add(a, .8, .2, 1, chess.STARTING_FEN)
    first = memory.query(a[None])
    assert first["support"][0] == 1
    assert first["value"][0] == pytest.approx(.8)
    assert memory.query(b[None])["support"][0] == 0
    memory.add(a, .8, .2, 2, chess.STARTING_FEN)
    assert memory.query(a[None])["information"][0] < first["information"][0]


def test_information_gain_downweights_conflicting_outcomes():
    memory = ExperienceMemory(config())
    key = np.array([1., 0.])
    memory.add(key, 0, .5, 1, "")
    memory.add(key, 1, .5, 2, "")
    noisy = memory.query(key[None])["information"][0]
    clean = ExperienceMemory(config())
    clean.add(key, .5, .5, 1, "")
    clean.add(key, .5, .5, 2, "")
    assert noisy < clean.query(key[None])["information"][0]


def test_retrieval_respects_whose_turn_it_is():
    memory = ExperienceMemory(config())
    key = np.array([1., 0.])
    memory.add(key, .9, .1, 1, "", own_turn=True)
    assert memory.query(key[None], own_turn=False)["support"][0] == 0
    memory.add(key, .2, .1, 2, "", own_turn=False)
    assert memory.query(key[None], own_turn=False)["value"][0] == pytest.approx(.2)


def test_protected_memories_survive_low_priority_repetition():
    memory = ExperienceMemory(config(archive_capacity=3))
    unique = memory.add(np.array([0., 1.]), 1, 0, 0, "rare")
    for game in range(30):
        memory.add(np.array([1., 0.]), .5, .5, game + 1, "common")
    assert len(memory.entries) == 3
    assert unique in [entry["id"] for entry in memory.entries]
    memory.audit([unique], 0, observed=True)
    assert next(entry["trust"] for entry in memory.entries if entry["id"] == unique) < 1


def test_skill_learning_needs_support_checks_legality_and_audits():
    library = SkillLibrary(config())
    key = np.array([1., 0.])
    actions = ("e2e4", "e7e5", "g1f3")
    library.learn(key, actions, True, .9, .4, 1)
    assert library.retrieve(key, chess.Board(), True) == ()
    library.learn(key, actions, True, .9, .4, 2)
    assert library.retrieve(key, chess.Board(), True) == actions
    blocked = chess.Board()
    blocked.remove_piece_at(chess.E2)
    assert library.retrieve(key, blocked, True) == ()
    for _ in range(8):
        library.audit(key, 0, observed=True)
    assert library.retrieve(key, chess.Board(), True) == ()


def test_planning_is_budgeted_and_never_changes_real_memory_or_board():
    trainer = CognitiveTrainer(config())
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    board = chess.Board()
    agent.observe(board, True)
    memory, history = agent.memory.clone(), list(board.move_stack)
    result = agent.evaluate(board, True, explore=False)
    assert agent.diagnostics["planning_nodes"] <= trainer.cfg.planning_nodes
    assert agent.diagnostics["planned_decisions"] > 0
    assert result.move in board.legal_moves
    assert history == board.move_stack
    torch.testing.assert_close(memory, agent.memory, rtol=0, atol=0)


def test_fast_path_avoids_search_when_value_is_safe_and_calibrated(monkeypatch):
    trainer = CognitiveTrainer(config(planning_gap=0, planning_uncertainty=.2))
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    trainer.knowledge.calibration.count[:] = 10000
    monkeypatch.setattr(trainer.net, "survival", lambda context: (
        torch.linspace(.6, .9, len(context))[:, None].expand(-1, trainer.cfg.ensemble_size),
        torch.zeros(len(context), trainer.cfg.ensemble_size)))
    agent.evaluate(chess.Board(), True, explore=False)
    assert agent.diagnostics["fast_decisions"] == 1
    assert agent.diagnostics["planning_nodes"] == 0


def test_robust_search_includes_an_unlikely_mating_reply(monkeypatch):
    trainer = CognitiveTrainer(config(planning_nodes=96, planning_replies=3))
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    board = chess.Board()
    board.push_uci("f2f3")
    board.push_uci("e7e5")
    monkeypatch.setattr(trainer.net, "candidates", lambda x, memory: (x, memory.expand(len(x), -1)))
    monkeypatch.setattr(trainer.net, "survival", lambda context: (
        torch.full((len(context), trainer.cfg.ensemble_size), .9),
        torch.zeros(len(context), trainer.cfg.ensemble_size)))
    # Make Qh4 extremely unlikely in the learned reply predictor.
    monkeypatch.setattr(trainer.net, "predict_logits", lambda x: -30 * x[:, 10 * 64 + chess.H4])
    values = agent._robust(board, True, [chess.Move.from_uci("g2g4")],
                           trainer.net.initial_memory(1), np.array([.9]), None, [0])
    assert values[0] < .5  # pessimistic branch protects against predictor agreement


def test_skill_plan_is_cancelled_when_an_opponent_changes_reply():
    trainer = CognitiveTrainer(config())
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    agent.plan = ("e2e4", "e7e5", "g1f3")
    agent.plan_prefix = ()
    board = chess.Board()
    agent.observe(board, True)
    board.push_uci("e2e4")
    agent.observe(board, True)
    board.push_uci("c7c5")
    agent.observe(board, True)
    agent.choose(board, True, np.random.default_rng(2), explore=False)
    assert agent.plan == ()


def test_supported_skill_actually_guides_two_decisions(monkeypatch):
    trainer = CognitiveTrainer(config())
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    board = chess.Board()
    key = agent.keys(torch.from_numpy(encode(board, True)[None]))[0]
    actions = ("e2e4", "e7e5", "g1f3")
    for game in (1, 2):
        trainer.knowledge.skills.learn(key, actions, True, .9, .4, game)

    def tied(board, colour, **kwargs):
        agent.observe(board, colour)
        moves = list(board.legal_moves)
        size = len(moves)
        agent._evaluation = {"keys": np.repeat(key[None], size, axis=0),
                             "uncertainty": np.zeros(size), "gap": 0,
                             "retrieved": {"support": np.zeros(size)}}
        return SurvivalChoice(moves, np.full(size, .5), np.full(size, .5),
                              np.zeros(size), np.zeros(size), np.zeros(size))

    monkeypatch.setattr(agent, "evaluate", tied)
    rng = np.random.default_rng(9)
    assert agent.choose(board, True, rng, explore=False).move.uci() == "e2e4"
    board.push_uci("e2e4")
    agent.observe(board, True)
    board.push_uci("e7e5")
    agent.observe(board, True)
    assert agent.choose(board, True, rng, explore=False).move.uci() == "g1f3"
    assert agent.diagnostics["skill_uses"] == 2


def test_terminal_rewards_equal_for_win_draw_and_zero_for_loss():
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    draw = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert terminal_value(mate, True) == terminal_value(draw, True) == 1
    assert terminal_value(mate, False) == 0


def test_fast_terminal_checks_match_authoritative_rules_and_claims():
    boards = []
    repeated = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 4:
        repeated.push_uci(uci)
        boards.append(repeated.copy(stack=True))
    boards.extend([chess.Board("7k/8/8/8/8/8/R7/K7 w - - 99 80"),
                   chess.Board("7k/8/8/8/8/8/R7/K7 w - - 150 100")])
    rng = np.random.default_rng(44)
    for game in range(8):
        board = chess.Board()
        for ply in range(60):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
            boards.append(board.copy(stack=True))
    for board in boards:
        outcome = board.outcome(claim_draw=True)
        for colour in (True, False):
            expected = None if outcome is None else (0 if outcome.winner is not None and outcome.winner != colour else 1)
            assert terminal_value(board, colour) == expected


def test_real_episode_updates_calibration_archive_skills_and_gradients():
    trainer = CognitiveTrainer(config())
    episode, trace, board = scripted(trainer, chess.BLACK)
    assert trainer.knowledge.calibration.count.sum() == 0
    result = trainer.ingest(episode, trace, chess.BLACK)
    assert result["calibration_samples"] == len(trace)
    assert len(trainer.knowledge.memory.entries) > 0
    assert trainer.knowledge.memory.episodes
    assert trainer.knowledge.skills.skills
    before = trainer.net.memory_cell.weight_hh.detach().clone()
    update = trainer.update()
    assert np.isfinite(update["loss"])
    assert not torch.equal(before, trainer.net.memory_cell.weight_hh)


def test_truncated_episodes_never_calibrate_as_observed_success():
    trainer = CognitiveTrainer(config(max_plies=4))
    agent = CognitiveAgent(trainer.net, trainer.cfg, trainer.knowledge)
    episode, board, colour, stats = play_episode(agent, RandomOpponent(), trainer.cfg, np.random.default_rng(17))
    assert not episode.terminated
    trainer.ingest(episode, agent.trace, colour)
    assert trainer.knowledge.calibration.count.sum() == 0
    assert not trainer.knowledge.memory.entries


def test_counterfactual_and_legal_perturbation_feed_replay_not_real_counters():
    trainer = CognitiveTrainer(config(experiment_plies=3))
    episode, trace, board = scripted(trainer, chess.WHITE)
    trainer.ingest(episode, trace, chess.WHITE)
    # Force the later learner decision so a preceding opponent move can vary.
    trace[0]["uncertainty"], trace[1]["uncertainty"] = 0, 10
    old_games = trainer.games
    old_calibration = trainer.knowledge.calibration.count.copy()
    before = len(trainer.replay)
    events = trainer.investigate(episode, trace, True, RandomOpponent(), force=True)
    assert {event["kind"] for event in events} == {"counterfactual", "generalization"}
    assert len(trainer.replay) > before
    assert trainer.games == old_games
    np.testing.assert_array_equal(old_calibration, trainer.knowledge.calibration.count)
    cf = next(event for event in events if event["kind"] == "counterfactual")
    assert cf["baseline"]["first_move"] != cf["alternative"]["first_move"]
    for event in events:
        for branch in (event["baseline"], event.get("alternative", event.get("variant"))):
            replay = chess.Board(branch["initial_fen"])
            for uci in branch["actions"]:
                move = chess.Move.from_uci(uci)
                assert move in replay.legal_moves
                replay.push(move)


def test_v3_checkpoint_restores_all_cognitive_state_and_next_update(tmp_path):
    trainer = CognitiveTrainer(config())
    episode, trace, _ = scripted(trainer)
    trainer.ingest(episode, trace, True)
    trainer.update()
    path = tmp_path / "v3.pt"
    trainer.save(path)
    restored = CognitiveTrainer(config())
    restored.load_state_dict(torch.load(path, weights_only=False))
    assert restored.games == trainer.games
    assert restored.knowledge.memory.next_id == trainer.knowledge.memory.next_id
    np.testing.assert_array_equal(restored.knowledge.calibration.count, trainer.knowledge.calibration.count)
    assert restored.update()["loss"] == pytest.approx(trainer.update()["loss"], abs=1e-7)
    for a, b in zip(restored.net.parameters(), trainer.net.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_v2_migration_keeps_weights_and_replay_without_fabricated_memories():
    from tonkai.survival_trainer import SurvivalTrainer
    source = SurvivalTrainer(config())
    new = CognitiveTrainer(config())
    new.load_state_dict(source.state_dict())
    for a, b in zip(source.net.parameters(), new.net.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert new.knowledge.memory.entries == []
    assert new.state_dict()["format"] == "tonkai-cognitive-v3"
