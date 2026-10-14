"""Outcome semantics shared by replay, learning and inference.

Victory uses expected game score (win=1, draw=.5, loss=0), not a
calibrated win probability. Unfinished trajectories remain bootstrapped.

"shaped" is the owner's definition of victory for this network: a blend of
survival and a per-position victory measure (enemy king pressure, own king
safety, threats to high-K/D enemy pieces, protection of our own) recorded in
``SequenceEpisode.proxies``. A real checkmate is worth 1.0, the maximum.
"""
import chess

from .encoding import unpack_many

OBJECTIVES = ("survival", "victory", "shaped")


def victory_weight(cfg) -> float:
    return float(getattr(cfg, "victory_weight", 0.5))


def outcome_value(outcome, colour, objective="survival", weight=0.5):
    if outcome is None:
        return None
    if outcome.winner is None:
        if objective == "victory":
            return .5
        if objective == "shaped":
            return 1.0 - weight / 2       # survival credits a draw 1, victory .5
        return 1.0
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


def win_speed(won, remaining, gamma):
    """Owner's fast-win value: gamma ** half-moves until the win, 0 without one."""
    return float(gamma) ** max(0, remaining) if won else 0.0


def observed_target(episode, ply, cfg):
    if cfg.objective == "victory":
        return episode_result(episode)
    survival = 1 - cfg.survival_gamma ** (len(episode.actions) - ply) if episode.lost else 1.0
    proxies = getattr(episode, "proxies", None)
    if cfg.objective == "shaped" and proxies is not None and 0 <= ply < len(proxies):
        weight = victory_weight(cfg)
        value = (1 - weight) * survival + weight * float(proxies[ply])
        speed = float(getattr(cfg, "win_speed_weight", 0.0) or 0.0)
        if speed > 0 and episode.terminated:
            won = getattr(episode, "result_value", None) == 1.0
            fast = win_speed(won, len(episode.actions) - ply, getattr(cfg, "win_speed_gamma", 0.99))
            value = (1 - speed) * value + speed * fast
        return value
    return survival


def continuation_value(value, steps, cfg):
    # No living reward or discount: an earlier loss is still a loss.
    if cfg.objective == "victory":
        return value
    discount = cfg.survival_gamma ** steps
    survival = 1 - discount + discount * value
    if cfg.objective == "shaped":
        # The blended value stands in for both parts; only survival earns
        # living credit for the extra plies.
        weight = victory_weight(cfg)
        return (1 - weight) * survival + weight * value
    return survival
