"""Outcome semantics shared by replay, learning and inference.

Victory uses expected game score (win=1, draw=.5, loss=0), not a
calibrated win probability. Unfinished trajectories remain bootstrapped.
"""
import chess

from .encoding import unpack_many


def outcome_value(outcome, colour, objective="survival"):
    if outcome is None:
        return None
    if outcome.winner is None:
        return .5 if objective == "victory" else 1.0
    return float(outcome.winner == colour)


def episode_result(episode):
    """Recover an old replay's W/D/L from legal history, never guess a draw."""
    result = getattr(episode, "result_value", None)
    if result is not None or not episode.terminated:
        return result
    if episode.lost:
        return 0.0
    if len(episode.actions) != len(episode.packed) - 1:
        raise ValueError("Victory replay requires complete actions or an explicit result")
    board = chess.Board(episode.initial_fen or chess.STARTING_FEN)
    own_turn = unpack_many(episode.packed[:1])[0, 768] > .5
    colour = board.turn if own_turn else not board.turn
    for action in episode.actions:
        board.push_uci(action)
    result = outcome_value(board.outcome(claim_draw=True), colour, "victory")
    if result is None:
        raise ValueError("Completed replay has no verifiable outcome")
    return result


def observed_target(episode, ply, cfg):
    if cfg.objective == "victory":
        return episode_result(episode)
    return 1 - cfg.survival_gamma ** (len(episode.actions) - ply) if episode.lost else 1.0


def continuation_value(value, steps, cfg):
    # No living reward or discount: an earlier loss is still a loss.
    if cfg.objective == "victory":
        return value
    discount = cfg.survival_gamma ** steps
    return 1 - discount + discount * value
