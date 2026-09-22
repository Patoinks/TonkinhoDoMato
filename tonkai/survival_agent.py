"""A recurrent actor. Candidate branches never enter the real game memory."""
from dataclasses import dataclass

import chess
import numpy as np
import torch

from .encoding import encode, encode_after_moves, pack_many
from .sequence_replay import SequenceEpisode
from .objective import outcome_value


@dataclass
class SurvivalChoice:
    moves: list
    scores: np.ndarray
    mean: np.ndarray
    epistemic: np.ndarray
    aleatoric: np.ndarray
    novelty: np.ndarray
    index: int = 0

    @property
    def move(self):
        return self.moves[self.index]


class SurvivalAgent:
    def __init__(self, net, cfg, device="cpu"):
        self.net, self.cfg, self.device = net, cfg, device
        self.reset()

    def reset(self):
        self.memory = self.net.initial_memory(1, self.device)
        self.context = None
        self._position = None
        self._ply = -1
        self._history = ()
        self._root_fen = None

    @torch.inference_mode()
    def observe(self, board, self_color):
        position = (board.fen(), self_color)
        ply = len(board.move_stack)
        history = tuple(board.move_stack)
        root_fen = board.root().fen()
        if self._position == position and self._history == history and self._root_fen == root_fen:
            return self.context
        # Undo, a replaced board, or skipped observations require the real prefix.
        incremental = (self._position is not None and self._position[1] == self_color
                       and self._root_fen == root_fen and ply == self._ply + 1
                       and history[:-1] == self._history)
        if (self._position is None and ply > 0) or (self._position is not None and not incremental):
            self.reset()
            replay = board.root()
            self.observe(replay, self_color)
            for move in board.move_stack:
                replay.push(move)
                self.observe(replay, self_color)
            return self.context
        x = torch.from_numpy(encode(board, self_color)[None]).to(self.device)
        self.context, self.memory = self.net.observe(x, self.memory)
        self._position, self._ply = position, ply
        self._history, self._root_fen = history, root_fen
        return self.context

    @torch.inference_mode()
    def evaluate(self, board, self_color, *, explore=True, head=None):
        self.observe(board, self_color)
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal candidates in a terminal position")
        x = torch.from_numpy(encode_after_moves(board, moves, self_color)).to(self.device)
        context, _ = self.net.candidates(x, self.memory)
        means, log_variances = self.net.survival(context)
        mean = means.mean(-1)
        epistemic = means.std(-1, unbiased=False)
        aleatoric = log_variances.exp().mean(-1).sqrt()
        novelty = self.net.novelty_error(context)
        # RND error is bounded in the action score; it never rewrites survival labels.
        novelty = novelty / (novelty + self.cfg.novelty_scale)
        scores = means[:, head] if explore and head is not None else mean
        if explore:
            scores = scores + self.cfg.curiosity_bonus * novelty + self.cfg.uncertainty_bonus * epistemic
        return SurvivalChoice(moves, scores.cpu().numpy(), mean.cpu().numpy(),
                              epistemic.cpu().numpy(), aleatoric.cpu().numpy(), novelty.cpu().numpy())

    def choose(self, board, self_color, rng, *, temperature=None, explore=True, head=None):
        choice = self.evaluate(board, self_color, explore=explore, head=head)
        temperature = self.cfg.temperature if temperature is None else temperature
        if temperature > 0 and explore:
            weights = np.exp((choice.scores - choice.scores.max()) / temperature)
            choice.index = int(rng.choice(len(weights), p=weights / weights.sum()))
        else:
            choice.index = int(choice.scores.argmax())
        return choice

    @torch.inference_mode()
    def predict_reply(self, board, self_color):
        self.observe(board, self_color)
        moves = list(board.legal_moves)
        x = torch.from_numpy(encode_after_moves(board, moves, self_color)).to(self.device)
        context, _ = self.net.candidates(x, self.memory)
        return moves, self.net.predict_logits(context).softmax(0).cpu().numpy()


def play_episode(agent, opponent, cfg, rng, *, self_color=None, explore=True,
                 opponent_name="external"):
    """Collect the complete ordered trajectory, including the initial state."""
    self_color = bool(rng.integers(2)) if self_color is None else self_color
    board = chess.Board()
    agent.reset()
    opponent.reset()
    head = int(rng.integers(cfg.ensemble_size)) if explore else None
    temperature = cfg.temperature * float(rng.uniform(0.7, 1.3)) if explore else 0.0
    positions = [encode(board, self_color)]
    surprises = [0.0]
    uncertainties = []
    agent.observe(board, self_color)
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < cfg.max_plies:
        if board.turn == self_color:
            choice = agent.choose(board, self_color, rng, temperature=temperature,
                                  explore=explore, head=head)
            move = choice.move
            surprise = 0.0
            uncertainties.append(float(choice.epistemic[choice.index]))
        else:
            moves, probabilities = agent.predict_reply(board, self_color)
            move = opponent.choose(board, rng)
            if move not in moves:
                raise ValueError(f"Opponent {opponent_name} returned an illegal move: {move}")
            surprise = -float(np.log(max(float(probabilities[moves.index(move)]), 1e-12)))
        board.push(move)
        agent.observe(board, self_color)
        positions.append(encode(board, self_color))
        surprises.append(surprise)
    outcome = board.outcome(claim_draw=True)
    lost = bool(outcome and outcome.winner is not None and outcome.winner != self_color)
    terminated = outcome is not None
    episode = SequenceEpisode(packed=pack_many(np.stack(positions)), lost=lost,
        terminated=terminated, opponent=opponent_name, surprises=np.asarray(surprises, dtype=np.float32),
        actions=tuple(move.uci() for move in board.move_stack),
        result_value=outcome_value(outcome, self_color, "victory"))
    result = board.result(claim_draw=True) if terminated else "*"
    return episode, board, self_color, {
        "plies": len(board.move_stack), "lost": lost, "terminated": terminated,
        "result": result, "opponent": opponent_name,
        "epistemic_mean": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "surprise_mean": float(np.mean(surprises)),
    }


class NeuralOpponent:
    """Frozen population member with its own recurrent state and random style."""
    def __init__(self, net, cfg, rng):
        self.agent = SurvivalAgent(net, cfg, "cpu")
        self.cfg, self.rng = cfg, rng
        self.reset()

    def reset(self):
        self.agent.reset()
        self.observed = -1
        self.board = None
        self.head = int(self.rng.integers(self.cfg.ensemble_size))
        self.temperature = self.cfg.temperature * float(self.rng.uniform(0.5, 1.8))

    def choose(self, board, rng=None):
        # Replay only unseen real moves, including the learner's moves.
        colour = board.turn
        if (self.board is None or self.observed > len(board.move_stack)
                or self.board.move_stack != board.move_stack[:self.observed]
                or self.board.root().fen() != board.root().fen()):
            self.agent.reset()
            self.board = board.root()
            self.observed = 0
            self.agent.observe(self.board, colour)
        for move in board.move_stack[self.observed:]:
            self.board.push(move)
            self.agent.observe(self.board, colour)
        self.observed = len(board.move_stack)
        return self.agent.choose(board, colour, self.rng, temperature=self.temperature,
                                 explore=True, head=self.head).move

    def close(self):
        pass
