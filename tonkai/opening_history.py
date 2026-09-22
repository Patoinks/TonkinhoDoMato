"""Bounded, colour-aware opening audit of the first five dispatched games."""
from collections import Counter
import json
from pathlib import Path

import chess
import chess.pgn


class OpeningHistory:
    def __init__(self, directory, rows=()):
        self.directory = Path(directory)
        self.rows = {row["game"]: row for row in rows}

    def add(self, number, board, colour, opponent):
        if not 1 <= number <= 5 or number in self.rows:
            return
        game = chess.pgn.Game.from_board(board)
        game.headers["Result"] = board.result(claim_draw=True)
        game.headers["Event"] = "TonkAI first five training games"
        game.headers["Round"] = str(number)
        game.headers["White"] = "TonkAI" if colour else opponent
        game.headers["Black"] = opponent if colour else "TonkAI"
        replay = board.root()
        moves, own = [], []
        for move in board.move_stack[:10]:
            san = replay.san(move)
            moves.append(san)
            if replay.turn == colour:
                own.append(san)
            replay.push(move)
        self.rows[number] = dict(game=number, colour="white" if colour else "black",
                                 opening_san=moves, own_first_move=own[0] if own else None,
                                 opening_uci=[m.uci() for m in board.move_stack[:10]],
                                 plies=len(board.move_stack), result=game.headers["Result"],
                                 pgn=str(game))
        self.write()

    def state(self):
        return [self.rows[key] for key in sorted(self.rows)]

    def report(self):
        rows = self.state()
        groups = {}
        for colour in ("white", "black"):
            selected = [row for row in rows if row["colour"] == colour]
            first = Counter(row["own_first_move"] for row in selected if row["own_first_move"])
            lines = Counter(" ".join(row["opening_uci"]) for row in selected)
            groups[colour] = dict(games=len(selected), first_moves=dict(first),
                distinct_opening_lines=len(lines),
                same_first_move=len(first) == 1 if len(selected) >= 2 else None,
                same_first_ten_plies=len(lines) == 1 if len(selected) >= 2 else None)
        return dict(complete=len(rows) == 5, definition="First five dispatched games; first 10 plies, grouped by TonkAI colour",
                    games=[{k:v for k,v in row.items() if k != "pgn"} for row in rows], by_colour=groups,
                    caveat="Five exploratory training games are a small sample, not proof of a fixed opening policy.")

    def write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, text in (("first-five.pgn", "\n\n".join(row["pgn"] for row in self.state()) + "\n"),
                           ("opening-report.json", json.dumps(self.report(), indent=2, ensure_ascii=False))):
            path = self.directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
