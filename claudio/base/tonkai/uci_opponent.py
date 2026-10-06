"""Persistent, strength-limited UCI opponent; returns moves, never training labels."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import subprocess

import chess
import chess.engine


class UCIOpponent:
    """One engine process per actor, with the caller's complete move history.

    ``elo`` is the engine's nominal setting, not a measured rating at this time
    control. ``rng`` in :meth:`choose` is accepted for opponent compatibility;
    UCI does not expose Stockfish's internal random seed. Use a separate instance
    per concurrent actor and call ``reset()`` before each game.
    """

    def __init__(self, path: str | os.PathLike[str], elo: int = 1600,
                 move_time: float = 0.02, threads: int = 1, hash_mb: int = 16,
                 *, timeout: float = 5.0):
        for name, value in (("elo", elo), ("threads", threads), ("hash_mb", hash_mb)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("move_time", move_time), ("timeout", timeout)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if move_time < 0.001:
            raise ValueError("move_time must be at least 0.001 seconds (UCI milliseconds)")

        executable = Path(path).expanduser().resolve(strict=True)
        if not executable.is_file():
            raise ValueError(f"UCI executable is not a file: {executable}")
        self.path = str(executable)
        self.elo = elo
        self.move_time = float(move_time)
        self.threads = threads
        self.hash_mb = hash_mb
        self.timeout = float(timeout)
        self._game = object()
        self._engine: chess.engine.SimpleEngine | None = None
        launch = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        engine = chess.engine.SimpleEngine.popen_uci(
            self.path, timeout=self.timeout, **launch,
        )
        self._engine = engine
        try:
            options = engine.options
            required = {"Threads": threads, "Hash": hash_mb, "UCI_Elo": elo}
            for name, value in required.items():
                option = options.get(name)
                if option is None or option.type != "spin":
                    raise ValueError(f"Engine must expose the UCI spin option {name}")
                if ((option.min is not None and value < option.min)
                        or (option.max is not None and value > option.max)):
                    raise ValueError(f"{name}={value} outside engine range "
                                     f"[{option.min}, {option.max}]")
            strength = options.get("UCI_LimitStrength")
            if strength is None or strength.type != "check":
                raise ValueError("Engine must expose the UCI_LimitStrength check option")
            configured = {**required, "UCI_LimitStrength": True}
            engine.configure(configured)
            engine.ping()  # Complete initialization before starting game timing.
            with executable.open("rb") as binary:
                checksum = hashlib.file_digest(binary, "sha256").hexdigest()
            self._info = {
                "engine_name": engine.id.get("name", executable.name),
                "engine_author": engine.id.get("author", ""),
                "path": self.path,
                "sha256": checksum,
                "elo": elo,
                "elo_is_nominal": True,
                "elo_range": [options["UCI_Elo"].min, options["UCI_Elo"].max],
                "move_time": self.move_time,
                "threads": threads,
                "hash_mb": hash_mb,
                "timeout": self.timeout,
                "options": configured,
                "seedable": False,
            }
        except BaseException:
            self._abort()
            raise

    @property
    def description(self) -> str:
        return (f"{self._info['engine_name']}; nominal Elo {self.elo}; "
                f"{self.move_time:g}s/move; Threads={self.threads}; Hash={self.hash_mb}MiB")

    @property
    def info(self) -> dict:
        """Configuration/provenance metadata only; contains no position evaluation."""
        return {**self._info, "elo_range": list(self._info["elo_range"]),
                "options": dict(self._info["options"])}

    def _active(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            raise RuntimeError("UCI opponent is closed")
        return self._engine

    def choose(self, board: chess.Board, rng=None) -> chess.Move:
        """Return a legal move without changing ``board`` or discarding its stack."""
        engine = self._active()
        if not board.is_valid():
            raise ValueError("Cannot play an invalid chess position")
        if board.is_game_over():
            raise ValueError("Cannot choose a move in a finished game")
        try:
            result = engine.play(board, chess.engine.Limit(time=self.move_time),
                                 game=self._game, info=chess.engine.INFO_NONE,
                                 ponder=False)
        except TimeoutError as exc:
            self._abort()
            raise TimeoutError(f"UCI opponent exceeded {self.move_time + self.timeout:g}s") from exc
        except chess.engine.EngineError:
            self._abort()
            raise
        if result.move is None or result.move not in board.legal_moves:
            self._abort()
            raise chess.engine.EngineError("UCI opponent returned no legal move")
        return result.move

    def reset(self) -> None:
        """Send ``ucinewgame`` on the next choice, clearing the engine's game state."""
        self._active()
        self._game = object()

    def _abort(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def close(self) -> None:
        """Stop the process; safe after errors and when called more than once."""
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.quit()
            except (TimeoutError, chess.engine.EngineError):
                pass
            finally:
                engine.close()

    def __enter__(self) -> UCIOpponent:
        self._active()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
