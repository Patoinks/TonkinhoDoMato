"""Keep one complete PGN: TonkAI's longest win against an external opponent."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import io
import re
import os
from pathlib import Path
import tempfile
import time

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


def build_win_game(board, self_color, opponent_name, result, termination):
    """A validated PGN game for a TonkAI win; raises on anything inconsistent."""
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
    return game, plies


class WinArchive:
    """Every TonkAI win, kept as ``wins.pgn`` (all games) and ``wins.json``.

    The JSON is what the website reads. Draws, losses and a disabled archive
    (``directory`` falsy) do nothing. Writes are locked and atomic.

    ``flush_seconds`` > 0 (training): each win is appended to ``wins.pgn`` at
    once, but the whole ``wins.json`` is rewritten at most that often and on
    ``flush()`` - rewriting it on every win costs time that grows with it.
    """

    def __init__(self, directory, flush_seconds=0.0):
        self.directory = Path(directory) if directory else None
        self.flush_seconds = float(flush_seconds)
        self.pending = []
        self.flushed = time.monotonic()

    @property
    def json_path(self):
        return self.directory / "wins.json"

    def load(self) -> list:
        if self.directory is None or not self.json_path.exists():
            return []
        return json.loads(self.json_path.read_text(encoding="utf-8"))["wins"]

    def add(self, board, self_color, opponent_name, result, termination, **meta) -> bool:
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("result must be a completed PGN result")
        if self.directory is None or result != ("1-0" if self_color else "0-1"):
            return False
        game, plies = build_win_game(board, self_color, opponent_name, result, termination)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        pgn = game.accept(exporter) + "\n"
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"number": None,
                 "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "tonkai_color": "white" if self_color else "black",
                 "opponent": opponent_name, "plies": plies, "termination": str(termination),
                 "result": result, "final_fen": board.fen(en_passant="fen"),
                 "moves": [move.uci() for move in board.move_stack], "pgn": pgn, **meta}
        with _file_lock(self.directory / ".wins.lock"):
            with (self.directory / "wins.pgn").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(pgn + "\n")
            self.pending.append(entry)
            if self.flush_seconds <= 0 or time.monotonic() - self.flushed >= self.flush_seconds:
                self._write_pending()
        return True

    def flush(self):
        """Write the wins still only in memory into ``wins.json``."""
        if self.directory is None or not self.pending:
            return
        with _file_lock(self.directory / ".wins.lock"):
            self._write_pending()

    def _write_pending(self):
        # Under the lock: re-read, so wins other processes added are kept.
        wins = self.load()
        for entry in self.pending:
            entry["number"] = len(wins) + 1
            wins.append(entry)
        _atomic_write(self.json_path, json.dumps({"format": "tonkai-wins-v1", "wins": wins},
                                                 indent=1, allow_nan=False))
        self.pending = []
        self.flushed = time.monotonic()


def _atomic_write(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _win_game(game):
    if game is None or game.errors:
        return False
    names = {"tonkai", "tonkinho do mato"}
    white = game.headers.get("White", "").strip().casefold() in names
    black = game.headers.get("Black", "").strip().casefold() in names
    if white == black or game.headers.get("Result") != ("1-0" if white else "0-1"):
        return False
    board = game.end().board()
    outcome = board.outcome(claim_draw=True)
    return (outcome is not None and outcome.result() == game.headers["Result"]
            or outcome is None and game.headers.get("Termination", "").lower() in {"resigned", "resignation"})


def _recorded_time(value):
    if not value:
        return float("-inf")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def featured_win_pgn(wins):
    """Select one validated external win by rating, plies, then recording time.

    Older archives put the configured Elo in the opponent name. Tiny's 800 is
    the existing nominal estimate, not a measured rating. Unknown ratings rank
    below known ones; their numeric engine/version names are never ratings.
    """
    best_key, best = None, ""
    for index, entry in enumerate(wins):
        try:
            stream = io.StringIO(entry["pgn"])
            game = chess.pgn.read_game(stream)
            if not _win_game(game) or chess.pgn.read_game(stream) is not None:
                continue
            white = game.headers["White"].strip().casefold() in {"tonkai", "tonkinho do mato"}
            opponent = game.headers["Black" if white else "White"]
            rating = entry.get("opponent_elo", game.headers.get("OpponentElo"))
            if rating is None:
                match = re.search(r"nominal Elo\s+(\d+)|(?:^|[·; ])(\d+)\s+Elo\b", opponent, re.I)
                rating = next((v for v in match.groups() if v), None) if match else None
            if rating is None and opponent.startswith("TinyOpponent"):
                rating = 800
            elo = max(0, int(rating or 0))
            recorded = entry.get("recorded_utc", game.headers.get("RecordedUTC", ""))
            plies = sum(1 for _ in game.mainline_moves())
            key = (elo, plies, _recorded_time(recorded), index)
            if best_key is None or key > best_key:
                game.headers["OpponentElo"] = str(elo)
                game.headers["PlyCount"] = str(plies)
                if recorded:
                    game.headers["RecordedUTC"] = recorded
                game.headers["Event"] = "TonkAI - featured victory"
                best = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)) + "\n"
                best_key = key
        except (ValueError, TypeError, KeyError):
            continue
    return best


class FeaturedGameRecorder:
    """One external victory, ranked by opponent Elo, plies, then recency."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self._cached = (None, None)                 # (file stamp, stored key)

    def stored_key(self):
        if self.path is None or not self.path.exists():
            return None
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._cached[0] == stamp:
            return self._cached[1]
        with self.path.open(encoding="utf-8") as stream:
            game = chess.pgn.read_game(stream)
        # Legacy featured losses/draws must never block a win.
        key = ((int(game.headers.get("OpponentElo", 0)), sum(1 for _ in game.mainline_moves()),
                _recorded_time(game.headers.get("RecordedUTC", ""))) if _win_game(game) else None)
        self._cached = (stamp, key)
        return key

    def consider(self, board, self_color, opponent_name, opponent_elo, result, termination,
                 *, recorded_utc=None):
        if self.path is None:
            return False
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("result must be a completed PGN result")
        if result != ("1-0" if self_color else "0-1"):
            return False
        if opponent_name.strip().casefold() in {"tonkai", "tonkinho do mato"}:
            return False
        stored = self.stored_key()
        if stored is not None and (int(opponent_elo), len(board.move_stack)) < stored[:2]:
            return False                             # cannot beat the stored game: nothing to build
        game, plies = build_win_game(board, self_color, opponent_name, result, termination)
        if not _win_game(game):
            raise ValueError("featured game must be a completed external victory")
        recorded = recorded_utc or datetime.now(timezone.utc).isoformat(timespec="microseconds")
        key = (int(opponent_elo), plies, _recorded_time(recorded))
        game.headers["Event"] = "TonkAI - featured victory"
        game.headers["OpponentElo"] = str(int(opponent_elo))
        game.headers["RecordedUTC"] = recorded
        text = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.path.with_name("." + self.path.name + ".lock")):
            stored = self.stored_key()
            if stored is not None and key <= stored:
                return False
            _atomic_write(self.path, text)
        return True


class LongestWinRecorder:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self._cached = (None, None)                 # (file stamp, stored plies)

    def _stored_plies(self):
        if not self.path.exists():
            return -1
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._cached[0] == stamp:
            return self._cached[1]
        plies = self._read_plies()
        self._cached = (stamp, plies)
        return plies

    def _read_plies(self):
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
        if len(board.move_stack) <= self._stored_plies():
            return False                             # not longer: nothing to build
        game, plies = build_win_game(board, self_color, opponent_name, result, termination)
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
