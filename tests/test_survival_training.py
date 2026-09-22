from __future__ import annotations

from copy import deepcopy

import chess
import numpy as np
import pytest
import torch

from tonkai.encoding import encode, pack_many
from tonkai.sequence_replay import SequenceEpisode
from tonkai.survival_agent import SurvivalAgent, play_episode
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_trainer import SurvivalTrainer, survival_targets
from tonkai.temporal_model import SurvivalNet


OPERA_GAME = (
    "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 "
    "Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 "
    "Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#"
).split()


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    values = dict(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24,
                  memory_size=16, head_hidden=12, piece_hidden=12, ensemble_size=3,
                  future_horizons=(1, 2, 4), burn_in=4, sequence_length=12,
                  sequence_batch_size=4, replay_capacity=512, min_replay=1,
                  max_plies=12, seed=27)
    values.update(overrides)
    return SurvivalConfig(**values)


def encoded_episode(sans, self_color):
    board = chess.Board()
    positions = [encode(board, self_color)]
    for san in sans:
        board.push_san(san)
        positions.append(encode(board, self_color))
    outcome = board.outcome(claim_draw=True)
    lost = bool(outcome and outcome.winner is not None and outcome.winner != self_color)
    return SequenceEpisode(pack_many(np.stack(positions)), lost, outcome is not None,
                           opponent="test-legal-game", surprises=np.zeros(len(positions), dtype=np.float32),
                           actions=tuple(move.uci() for move in board.move_stack))


def populated_trainer():
    torch.manual_seed(71)
    trainer = SurvivalTrainer(config())
    for episode in (
        encoded_episode(OPERA_GAME, chess.BLACK),
        encoded_episode(OPERA_GAME, chess.WHITE),
        encoded_episode("Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8".split(), chess.WHITE),
        encoded_episode(OPERA_GAME[:16], chess.BLACK),
    ):
        trainer.replay.add(episode)
        trainer.games += 1
    return trainer


def assert_same(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same(a, b)
    else:
        assert first == second


def test_completed_wins_and_draws_have_identical_survival_targets():
    remaining = torch.arange(49)[None].expand(2, -1)
    # Row 0 represents a win and row 1 a draw: both are completed non-losses.
    targets = survival_targets(remaining, torch.tensor([[False], [False]]),
                               torch.tensor([[True], [True]]), 0.985,
                               torch.zeros_like(remaining, dtype=torch.float32),
                               torch.full_like(remaining, 8))
    assert torch.equal(targets[0], targets[1])
    assert torch.equal(targets, torch.ones_like(targets))
    assert not targets.requires_grad


def test_every_position_receives_monotonic_credit_even_over_twenty_plies_before_loss():
    gamma = SurvivalConfig().survival_gamma
    remaining = torch.arange(65)
    targets = survival_targets(remaining, torch.tensor(True), torch.tensor(True), gamma,
                               torch.zeros(65), torch.ones(65))
    torch.testing.assert_close(targets, 1 - torch.tensor(gamma).pow(remaining))
    assert targets[0] == 0
    assert torch.all(targets[1:] > targets[:-1])
    assert 0 < targets[20] < targets[40] < targets[64] < 1


def test_truncated_episodes_bootstrap_instead_of_becoming_successes():
    bootstrap = torch.tensor([0.2, 0.35, 0.7], requires_grad=True)
    steps = torch.tensor([0, 3, 8])
    targets = survival_targets(torch.tensor([0, 3, 80]), torch.tensor(False),
                               torch.tensor(False), 0.985, bootstrap, steps)
    expected = 1 - torch.tensor(0.985).pow(steps) + torch.tensor(0.985).pow(steps) * bootstrap.detach()
    torch.testing.assert_close(targets, expected)
    assert targets[0] == bootstrap.detach()[0]
    assert torch.all(targets < 1)
    assert not targets.requires_grad


def test_default_learning_window_backpropagates_more_than_twenty_states_and_detaches_burnin():
    torch.manual_seed(5)
    cfg = config(sequence_length=SurvivalConfig().sequence_length)
    assert cfg.sequence_length >= 40
    trainer = SurvivalTrainer(cfg)
    x = torch.rand(1, cfg.burn_in + cfg.sequence_length, 782, requires_grad=True)
    valid = torch.ones(x.shape[:2], dtype=torch.bool)
    contexts, _, _ = trainer._unroll(trainer.net, x, valid, cfg.burn_in, cfg.sequence_length, True)
    means, _ = trainer.net.survival(contexts[:, cfg.burn_in + 30])
    means.square().sum().backward()  # a zero-survival target at the distant loss
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[:, :cfg.burn_in].abs().sum() == 0
    assert x.grad[:, cfg.burn_in].abs().sum() > 0
    assert trainer.net.memory_cell.weight_hh.grad.abs().sum() > 0


def test_update_has_finite_losses_and_gradients_and_trains_every_requested_component():
    trainer = populated_trainer()
    before = {key: value.detach().clone() for key, value in trainer.net.state_dict().items()}
    fixed_future = deepcopy(trainer.future_encoder.state_dict())
    metrics = trainer.update()
    assert metrics is not None and trainer.steps == 1
    assert all(np.isfinite(value) for value in metrics.values())
    for name in ("loss", "survival_loss", "future_loss", "novelty_loss", "reply_loss", "piece_loss", "gradient_norm"):
        assert metrics[name] > 0, name
    assert metrics["learned_positions"] > 0
    after = trainer.net.state_dict()
    for prefix in ("encoder.", "memory_cell.", "fusion.", "survival_heads.", "future_heads.",
                   "rnd_predictor.", "predict_head.", "piece_head."):
        assert any(not torch.equal(before[key], value) for key, value in after.items() if key.startswith(prefix)), prefix
        gradients = [parameter.grad for key, parameter in trainer.net.named_parameters()
                     if key.startswith(prefix) and parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), prefix
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0, prefix
    for key, value in after.items():
        if key.startswith("rnd_target."):
            assert torch.equal(value, before[key])
    assert_same(fixed_future, trainer.future_encoder.state_dict())
    assert all(parameter.grad is None for parameter in trainer.target.parameters())
    assert all(parameter.grad is None for parameter in trainer.future_encoder.parameters())


def test_checkpoint_roundtrip_restores_replay_optimizer_rng_and_next_update(tmp_path):
    trainer = populated_trainer()
    assert trainer.update() is not None
    path = tmp_path / "survival.pt"
    trainer.save(path, test_note="roundtrip")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["format"] == "tonkai-survival-v2" and saved["test_note"] == "roundtrip"
    assert not path.with_suffix(".pt.tmp").exists()
    expected_metrics = trainer.update()
    expected_state = deepcopy(trainer.state_dict())
    expected_random = torch.rand(5)
    restored = SurvivalTrainer(config())
    restored.load_state_dict(saved)
    assert restored.games == 4 and restored.steps == 1
    assert len(restored.replay) == len(trainer.replay)
    actual_metrics = restored.update()
    assert_same(expected_metrics, actual_metrics)
    assert_same(expected_state, restored.state_dict())
    assert torch.equal(expected_random, torch.rand(5))
    with pytest.raises(ValueError, match="Legacy"):
        restored.load_state_dict({"model": {}})


def observe_history(agent, board, color):
    replay = board.root()
    agent.observe(replay, color)
    for move in board.move_stack:
        replay.push(move)
        agent.observe(replay, color)


def board_after(sans):
    board = chess.Board()
    for san in sans.split():
        board.push_san(san)
    return board


def test_actor_candidate_scoring_and_prediction_do_not_commit_hypothetical_memory():
    torch.manual_seed(41)
    cfg = config()
    agent = SurvivalAgent(SurvivalNet(cfg), cfg)
    board = chess.Board()
    agent.observe(board, True)
    before = agent.memory.clone()
    context = agent.context.clone()
    first = agent.evaluate(board, True, explore=False)
    second = agent.evaluate(board, True, explore=False)
    moves, probabilities = agent.predict_reply(board, True)
    assert first.moves == second.moves == moves == list(board.legal_moves)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert probabilities.sum() == pytest.approx(1)
    assert torch.equal(agent.memory, before) and torch.equal(agent.context, context)
    assert not board.move_stack
    agent.reset()
    assert not agent.memory.count_nonzero() and agent.context is None
    agent.observe(board, True)
    assert torch.equal(agent.memory, before) and torch.equal(agent.context, context)


@pytest.mark.parametrize("change", ["undo", "skip", "fresh", "color", "different-branch"])
def test_actor_reconstructs_the_real_prefix_after_history_changes(change):
    torch.manual_seed(23)
    cfg = config()
    net = SurvivalNet(cfg)
    actor = SurvivalAgent(net, cfg)
    board = board_after("e4 e5 Nf3 Nc6 Bc4")
    color = True
    if change == "fresh":
        pass
    elif change == "skip":
        actor.observe(board.root(), color)
    else:
        observe_history(actor, board, color)
        if change == "undo":
            board.pop(); board.pop()
        elif change == "color":
            color = False
        else:
            board = board_after("d4 d5 c4 e6 Nc3 Nf6")
    actor.observe(board, color)
    expected = SurvivalAgent(net, cfg)
    observe_history(expected, board, color)
    assert torch.equal(actor.memory, expected.memory)
    assert torch.equal(actor.context, expected.context)


def test_actor_does_not_confuse_transposed_positions_with_different_histories():
    torch.manual_seed(31)
    cfg = config()
    net = SurvivalNet(cfg)
    first = board_after("Nf3 Nf6 Nc3 Nc6")
    second = board_after("Nc3 Nc6 Nf3 Nf6")
    assert first.fen() == second.fen() and len(first.move_stack) == len(second.move_stack)
    actor = SurvivalAgent(net, cfg)
    observe_history(actor, first, True)
    before = actor.memory.clone()
    actor.observe(second, True)
    expected = SurvivalAgent(net, cfg)
    observe_history(expected, second, True)
    assert not torch.equal(before, expected.memory)
    assert torch.equal(actor.memory, expected.memory)


class RandomLegalOpponent:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def choose(self, board, rng):
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
def test_collects_real_legal_episode_against_random_opponent_without_cross_game_memory(color):
    torch.manual_seed(61)
    cfg = config(max_plies=12)
    agent = SurvivalAgent(SurvivalNet(cfg), cfg)
    opponent = RandomLegalOpponent()
    rng = np.random.default_rng(81)
    for _ in range(2):
        episode, board, actual_color, metrics = play_episode(agent, opponent, cfg, rng,
                                                            self_color=color, opponent_name="random-test")
        assert actual_color == color
        assert 0 < len(episode.actions) <= cfg.max_plies
        assert episode.packed.shape == (len(episode.actions) + 1, 99)
        replay = chess.Board()
        encoded = [encode(replay, color)]
        for uci in episode.actions:
            move = chess.Move.from_uci(uci)
            assert move in replay.legal_moves
            replay.push(move)
            encoded.append(encode(replay, color))
        assert replay.fen() == board.fen()
        np.testing.assert_array_equal(episode.packed, pack_many(np.stack(encoded)))
        outcome = board.outcome(claim_draw=True)
        assert episode.terminated == (outcome is not None)
        assert episode.lost == bool(outcome and outcome.winner is not None and outcome.winner != color)
        assert metrics["plies"] == len(episode.actions) and metrics["result"] == (outcome.result() if outcome else "*")
        assert np.isfinite(episode.surprises).all() and episode.surprises[0] == 0
        fresh = SurvivalAgent(agent.net, cfg)
        observe_history(fresh, board, color)
        assert torch.equal(agent.memory, fresh.memory)
    assert opponent.resets == 2
