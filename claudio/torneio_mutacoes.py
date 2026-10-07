"""Mutation tournament by generations (the owner's format).

Generation 1: --mutants mutants of the strongest Claudio (--checkpoint). Each mutant changes a
random share (--rate) of the numeric settings by 20% to 50%, up or down (--spread). Its network
and its own measured piece values never change, so every mutant is as organic as the original;
switches and choices (memory, reply_mode, ...) and settings at 0 stay as the parent has them.
The mutants and their parent play a knockout bracket of best-of-5 matches; the champion is the
parent of the next generation. After --generations generations, every generation's champion
(and the original, as generation 0) plays a group stage, every pair best of --group-games.

A match: games with colours alternating, each pair of games from its own opening
(--random-plies random legal moves, seeded). More wins goes through (a draw counts 0, the
owner's rule); level on wins: extra games until one is won (at most --extra), then the one that
thought for less. Every finished game is appended to results/torneio-<label>.jsonl, so a stopped
tournament continues where it was (same command). Measurement only - nothing is trained.

    python claudio/torneio_mutacoes.py --checkpoint claudio/claudio-grande-g0-forte.pt --label geracoes
"""
import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from itertools import combinations
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import arena  # noqa: E402

_t = {}

# The numeric settings that change how Claudio plays, with the range they are kept in
# (calculation depths are capped so that a generation stays playable in time). value_spread_*
# only act with value_scaling on, a switch, so they are not here.
BOUNDS = {
    "safety_weight": (0.0, 2.0),
    "exchange_depth": (1, 8),
    "exchange_nodes": (1_000, 800_000),
    "tactic_depth": (1, 6),
    "tactic_top": (1, 8),
    "tactic_rounds": (1, 10),
    "tactic_nodes": (10_000, 8_000_000),
    "leaf_weight": (0.0, 1.0),
    "reply_top": (0, 12),
    "reply_weight": (0.0, 3.0),
    "reply_depth": (2, 3),
    "reply_branch": (1, 6),
    "mate_defence": (0, 3),
    "mate_attack": (0, 4),
    "mate_gate": (0.0, 1.0),
    "mate_nodes": (1_000, 80_000),
    "pawn_reach": (0, 3),
    "king_weight": (0.0, 1.0),
    "simplify_weight": (0.0, 1.0),
    "simplify_margin": (0.0, 10.0),
    "temperature": (0.0, 0.1),
    "draw_value": (0.0, 1.0),
}


def mutate(parent, rng, rate, spread):
    """{setting: value} of a mutant of `parent` (full settings): each numeric setting, with
    chance `rate`, times 1 +- U(spread); at least one setting changes."""
    low, high = spread
    while True:
        child = {}
        for name, (least, most) in BOUNDS.items():
            value = parent[name]
            if not value or rng.random() >= rate:
                continue
            sign = 1.0 if rng.random() < 0.5 else -1.0
            new = value * (1.0 + sign * rng.uniform(low, high))
            if isinstance(value, int) and not isinstance(value, bool):
                new = int(round(new))
                if new == value:                     # a small integer still moves one step
                    new = value + int(sign)
                new = int(min(most, max(least, new)))
            else:
                new = round(float(min(most, max(least, new))), 6)
            if new != value:
                child[name] = new
        if child:
            return {**parent, **child}


def changes(settings, base):
    return {k: v for k, v in settings.items() if v != base[k]}


# ------------------------------------------------------------------- the games
def _init(checkpoint):
    import psutil_free_priority  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from claudio_load import load_claudio
    net, values, config, _, king = load_claudio(checkpoint, with_config=True, with_king=True)
    _t.update(net=net, values=values, king=king)


def _claudio(settings, seed):
    import numpy as np
    from claudio_agent import Claudio, ClaudioConfig
    return Claudio(_t["net"], _t["values"], ClaudioConfig(**settings), rng=np.random.default_rng(seed),
                   king_model=_t["king"])


def play(key, a, b, a_settings, b_settings, a_white, seed, opening, random_plies, max_plies):
    import chess
    players = {a: _claudio(a_settings, [seed, opening, 1]), b: _claudio(b_settings, [seed, opening, 2])}
    board = arena.opening(seed, opening, random_plies)
    spent = {a: 0.0, b: 0.0}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        who = a if (board.turn == chess.WHITE) == a_white else b
        start = time.perf_counter()
        move = players[who].choose(board)
        spent[who] += time.perf_counter() - start
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None or outcome.winner is None else (a if (outcome.winner == chess.WHITE) == a_white else b)
    return {"key": key, "a": a, "b": b, "white": a if a_white else b, "winner": winner,
            "end": outcome.termination.name if outcome else "max_plies", "plies": len(board.move_stack),
            "time": spent, "moves": " ".join(m.uci() for m in board.move_stack)}


# ----------------------------------------------------------------- tournament
class Tournament:
    def __init__(self, args):
        self.args = args
        self.log = ROOT / "claudio/results" / f"torneio-{args.label}.jsonl"
        self.done = {}
        if self.log.exists():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.done[row["key"]] = row
        self.stream = open(self.log, "a", encoding="utf-8")
        self.pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                                        initargs=(args.checkpoint,))

    def run(self, jobs):
        """Plays the jobs not already in the log; returns {key: game}."""
        results, pending = {}, set()
        for job in jobs:
            if job[0] in self.done:
                results[job[0]] = self.done[job[0]]
            else:
                pending.add(self.pool.submit(play, *job))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                game = future.result()
                self.done[game["key"]] = results[game["key"]] = game
                self.stream.write(json.dumps(game) + "\n")
                self.stream.flush()
        return results

    def match(self, prefix, a, b, players, games, opening):
        """A best-of-`games` match; returns (winner, summary)."""
        args = self.args

        def job(g):
            return (f"{prefix}-{a}-{b}-{g}", a, b, players[a], players[b], g % 2 == 0, args.seed,
                    opening + g // 2, args.random_plies, args.max_plies)
        played = list(self.run([job(g) for g in range(games)]).values())
        extra = 0
        while sum(x["winner"] == a for x in played) == sum(x["winner"] == b for x in played) and extra < args.extra:
            played += list(self.run([job(games + extra)]).values())
            extra += 1
        wa, wb = sum(x["winner"] == a for x in played), sum(x["winner"] == b for x in played)
        ta, tb = sum(x["time"][a] for x in played), sum(x["time"][b] for x in played)
        winner = a if (wa, -ta) > (wb, -tb) else b
        return winner, {"a": a, "b": b, "wins": [wa, wb], "draws": len(played) - wa - wb,
                        "seconds": [round(ta, 1), round(tb, 1)], "winner": winner, "by_time": wa == wb}

    def bracket(self, generation, players, rng):
        """Knockout of best-of matches (byes in the first round when not a power of 2)."""
        names = list(players)
        rng.shuffle(names)
        size = 1 << (len(names) - 1).bit_length()
        byes, field = names[:size - len(names)], names[size - len(names):]
        rounds, level, won = [], 0, defaultdict(int)
        while len(byes) + len(field) > 1:
            level += 1
            pairs = [(field[i], field[i + 1]) for i in range(0, len(field), 2)]
            # all the round's games at once (the extra games after)
            prefix = f"g{generation}-r{level}"
            first = []
            for m, (a, b) in enumerate(pairs):
                for g in range(self.args.games):
                    first.append((f"{prefix}-{a}-{b}-{g}", a, b, players[a], players[b], g % 2 == 0, self.args.seed,
                                  generation * 10_000_000 + level * 100_000 + m * 100 + g // 2,
                                  self.args.random_plies, self.args.max_plies))
            self.run(first)
            matches, winners = [], []
            for m, (a, b) in enumerate(pairs):
                winner, summary = self.match(prefix, a, b, players, self.args.games,
                                             generation * 10_000_000 + level * 100_000 + m * 100)
                winners.append(winner)
                won[winner] += 1
                matches.append(summary)
            rounds.append(matches)
            field, byes = byes + winners, []
            print(f"  geracao {generation} ronda {level}: {len(pairs)} eliminatorias, ficam {len(field)}", flush=True)
        return field[0], rounds, dict(won)

    def groups(self, players):
        """Every pair, best of --group-games; wins, draws, losses, winrate = wins / games."""
        jobs, opening = [], 900_000_000
        for p, (a, b) in enumerate(combinations(players, 2)):
            for g in range(self.args.group_games):
                jobs.append((f"grupos-{a}-{b}-{g}", a, b, players[a], players[b], g % 2 == 0, self.args.seed,
                             opening + p * 100 + g // 2, self.args.random_plies, self.args.max_plies))
        games = self.run(jobs)
        table = {n: {"jogos": 0, "V": 0, "E": 0, "D": 0, "s": 0.0} for n in players}
        pairs = {}
        for game in games.values():
            a, b, w = game["a"], game["b"], game["winner"]
            for n in (a, b):
                row = table[n]
                row["jogos"] += 1
                row["V" if w == n else "E" if w is None else "D"] += 1
                row["s"] += game["time"][n]
            pair = pairs.setdefault(f"{a} vs {b}", {a: 0, b: 0, "empates": 0})
            pair[w if w else "empates"] += 1
        for row in table.values():
            row["winrate"] = row["V"] / max(1, row["jogos"])
            row["s_por_jogo"] = row["s"] / max(1, row["jogos"])
        return table, pairs

    def close(self):
        for process in list(getattr(self.pool, "_processes", {}).values()):
            process.kill()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "claudio/claudio-grande-g0-forte.pt"))
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--mutants", type=int, default=200, help="mutants per generation (their parent plays too)")
    parser.add_argument("--rate", type=float, default=0.3, help="chance of each numeric setting mutating")
    parser.add_argument("--spread", type=float, nargs=2, default=(0.2, 0.5), help="change, as a share up or down")
    parser.add_argument("--games", type=int, default=5, help="games per knockout match (best of)")
    parser.add_argument("--group-games", type=int, default=10, help="games per pair in the final group stage")
    parser.add_argument("--extra", type=int, default=4, help="extra games at most when level on wins")
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    import numpy as np
    from claudio_agent import ClaudioConfig
    from claudio_load import load_claudio
    base = asdict(ClaudioConfig.with_settings([], base=load_claudio(args.checkpoint, with_config=True)[2]))
    out = ROOT / "claudio/results" / f"torneio-{args.label}.json"
    report = {"label": args.label, "checkpoint": args.checkpoint, "base": base, "rate": args.rate,
              "spread": list(args.spread), "generations": []}
    tournament = Tournament(args)
    started = time.perf_counter()
    parent, champions = dict(base), {"original": dict(base)}
    print(f"{args.generations} geracoes x ({args.mutants} mutantes + o pai), melhor de {args.games}; "
          f"grupos melhor de {args.group_games}", flush=True)
    try:
        for generation in range(1, args.generations + 1):
            rng = np.random.default_rng([args.seed, generation])
            players = {f"G{generation}-pai": parent}
            for i in range(1, args.mutants + 1):
                players[f"G{generation}-m{i:03d}"] = mutate(parent, rng, args.rate, args.spread)
            champion, rounds, won = tournament.bracket(generation, players, rng)
            parent = players[champion]
            champions[f"G{generation}"] = parent
            by_setting = defaultdict(list)
            for name, settings in players.items():
                for setting in changes(settings, players[f"G{generation}-pai"]):
                    by_setting[setting].append(won.get(name, 0))
            report["generations"].append({
                "generation": generation, "champion": champion, "champion_changes": changes(parent, base),
                "from_parent": changes(parent, players[f"G{generation}-pai"]), "final": rounds[-1][0],
                "rounds": rounds, "rounds_won": won,
                "by_setting": {s: round(sum(v) / len(v), 3) for s, v in by_setting.items()},
                "entrants": {n: changes(s, base) for n, s in players.items()}})
            out.write_text(json.dumps(report, indent=1), encoding="utf-8")
            final = rounds[-1][0]
            print(f"GERACAO {generation}: campeao {champion} (final {final['a']} vs {final['b']} {final['wins']}, "
                  f"{final['draws']} empates) - muda do forte: {changes(parent, base)} "
                  f"({(time.perf_counter() - started) / 60:.0f} min)", flush=True)
        table, pairs = tournament.groups(champions)
        report.update(groups=table, group_pairs=pairs,
                      champions={n: changes(s, base) for n, s in champions.items()},
                      minutes=round((time.perf_counter() - started) / 60, 1))
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"FASE DE GRUPOS (cada par melhor de {args.group_games}):", flush=True)
        for name, row in sorted(table.items(), key=lambda kv: (-kv[1]["V"], kv[1]["D"])):
            print(f"  {name:9s} WR {100 * row['winrate']:5.1f}%  V/E/D {row['V']}/{row['E']}/{row['D']}  "
                  f"{row['s_por_jogo']:.1f} s/jogo  {changes(champions[name], base)}", flush=True)
    finally:
        tournament.close()


if __name__ == "__main__":
    main()
