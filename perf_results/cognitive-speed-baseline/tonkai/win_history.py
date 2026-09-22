"""Keep one complete PGN: TonkAI's longest win against an external opponent."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import chess
import chess.pgn


@contextmanager
def _file_lock(path):
    # A persistent lock file also coordinates independent train/evaluate/play
    # processes. Never unlink it: another process may already hold it open.
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class LongestWinRecorder:
    def __init__(self, path):
        self.path = Path(path) if path else None

    def _stored_plies(self):
        if not self.path.exists():
            return -1
        with self.path.open(encoding="utf-8") as stream:
            game = chess.pgn.read_game(stream)
            if game is None or game.errors or chess.pgn.read_game(stream) is not None:
                raise ValueError("existing win PGN must contain one valid game")
        white = game.headers.get("White") == "TonkAI"
        black = game.headers.get("Black") == "TonkAI"
        if white == black or game.headers.get("Result") != ("1-0" if white else "0-1"):
            raise ValueError("existing PGN is not a TonkAI win against an external opponent")
        return sum(1 for _ in game.mainline_moves())

    def consider(self, board, self_color, opponent_name, result, termination):
        """Save a strictly longer win; draws, losses and a disabled path do nothing.

        ``result`` is supplied explicitly so a human resignation is supported.
        Move legality and any terminal board result are checked before writing.
        """
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("result must be a completed PGN result")
        if self_color not in (chess.WHITE, chess.BLACK):
            raise ValueError("TonkAI's color is required")
        if self.path is None or result != ("1-0" if self_color else "0-1"):
            return False
        if not opponent_name or opponent_name.strip().casefold() == "tonkai":
            raise ValueError("an external opponent name is required")

        root = board.root()
        if not root.is_valid():
            raise ValueError("game starts from an invalid position")
        replay = root.copy()
        game = chess.pgn.Game()
        game.setup(root)
        node = game
        for move in board.move_stack:
            if move not in replay.legal_moves:
                raise ValueError(f"illegal move in win history: {move.uci()}")
            replay.push(move)
            node = node.add_variation(move)
        if replay.fen(en_passant="fen", promoted=True) != board.fen(en_passant="fen", promoted=True):
            raise ValueError("move history does not reproduce the final board")
        outcome = replay.outcome()
        if outcome is not None and outcome.result() != result:
            raise ValueError("supplied result contradicts the final board")
        if termination == "checkmate" and not replay.is_checkmate():
            raise ValueError("checkmate termination requires a checkmated board")

        plies = len(board.move_stack)
        game.headers.update({
            "Event": "TonkAI vs external opponent",
            "Site": "local",
            "Date": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
            "White": "TonkAI" if self_color else opponent_name,
            "Black": opponent_name if self_color else "TonkAI",
            "Result": result,
            "Termination": str(termination),
            "PlyCount": str(plies),
            "TonkAIColor": "white" if self_color else "black",
            "FinalFEN": board.fen(en_passant="fen"),
        })
        text = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name("." + self.path.name + ".lock")
        with _file_lock(lock_path):
            if plies <= self._stored_plies():
                return False
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                                 dir=self.path.parent, prefix="." + self.path.name + ".",
                                                 suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return True

    def consider_record(self, record, opponent_name):
        """Rebuild a winning worker record in the parent process only."""
        from .chess_env import RESULT_WIN, result_from_white_pov

        if self.path is None or record.result != RESULT_WIN:
            return False
        if record.self_color is None or len(record.moves_uci) != record.plies:
            raise ValueError("winning record lacks its complete move history or color")
        board = chess.Board()
        for uci in record.moves_uci:
            board.push_uci(uci)
        return self.consider(board, record.self_color, opponent_name,
                             result_from_white_pov(record.result, record.self_color), record.reason)
