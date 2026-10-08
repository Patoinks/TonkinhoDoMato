"""A/B/B/A exact baseline-versus-optimized G16 C++ verification.

The baseline extension claudio_fast_ref is built from pinned commit
2c0bf4df137604a0a62bd8f1de9f592d9e0fabf7 (original Claudio G16 before the three optimization PRs); the current
claudio_fast contains batch terminal, Pos::remove and capture generator. Both use identical G16 weights/config, memory and random seeds.

Run in Actions after compiling both extensions:
    python claudio/bench_g16_native_abba.py --positions 96 --repeats 2
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401
import chess  # noqa: E402
import chess.pgn  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import claudio_fast  # noqa: E402
import claudio_fast_ref  # noqa: E402

import claudio_agent  # noqa: E402
from claudio_load import load_claudio  # noqa: E402

REFERENCES = {"baseline": claudio_fast_ref, "candidate": claudio_fast}


def real_positions(count):
    pgn = Path(__file__).resolve().parent / "results/matches/g16-vs-g15-100-bo30.pgn"
    selected = []
    with pgn.open("r", encoding="utf-8") as stream:
        game_id = 0
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if game.errors:
                raise ValueError(f"Invalid game {game_id}: {game.errors}")
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= 4 and ply % 6 == 0 and not board.is_game_over():
                    selected.append((game_id, ply, board.copy(stack=True)))
                board.push(move)
            game_id += 1
    # Evenly sample the entire recorded match, retaining chronological
    # position order for per-game GRU state. Not just its first opening moves.
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} recorded positions; requested {count}")
    indexes = sorted({min(len(selected) - 1, round(i * (len(selected) - 1) / (count - 1)))
                      for i in range(count)}) if count > 1 else [len(selected) // 2]
    return [selected[i] for i in indexes]



def completed_game_rows(count):
    """Complete BO30 games, replaying every recorded position, both colours."""
    pgn = Path(__file__).resolve().parent / "results/matches/g16-vs-g15-100-bo30.pgn"
    games = []
    with pgn.open("r", encoding="utf-8") as stream:
        game_id = 0
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if game.errors:
                raise ValueError(f"Invalid game {game_id}: {game.errors}")
            if game.headers.get("Result", "*") != "*":
                board = game.board()
                rows = []
                for ply, move in enumerate(game.mainline_moves()):
                    if board.is_game_over():
                        raise AssertionError(f"Game {game_id} continues after terminal ply {ply}")
                    rows.append((game_id, ply, board.copy(stack=True)))
                    board.push(move)
                if len(rows) >= 16:
                    games.append((game_id, rows, game.headers["Result"]))
            game_id += 1
    if len(games) < count:
        raise ValueError(f"Only {len(games)} complete BO30 games; requested {count}")
    games.sort(key=lambda item: len(item[1]))
    indexes = ([len(games) // 2] if count == 1 else
               [round(i * (len(games) - 1) / (count - 1)) for i in range(count)])
    return [games[i] for i in indexes]


def benchmark_complete_games(games, loaded, repeats):
    """ABBA decision-call totals per recorded game; not self-play or PGN parsing."""
    totals = {"baseline": [], "candidate": []}
    decisions_per_pass = 0
    for game_id, rows, result in games:
        expected = None
        per_game = {"baseline": [], "candidate": []}
        decisions_per_pass += len(rows)
        for repeat in range(repeats):
            for label in ("baseline", "candidate", "candidate", "baseline"):
                decisions, times, _ = evaluate(rows, loaded, REFERENCES[label])
                if expected is None:
                    expected = decisions
                elif decisions != expected:
                    changed = [(i, x, y) for i, (x, y) in enumerate(zip(expected, decisions))
                               if x != y]
                    raise AssertionError(
                        f"Full-game divergence id={game_id} version={label}: {changed[:8]}")
                per_game[label].append(sum(times))
        for label in totals:
            totals[label].extend(per_game[label])
        print(f"FULL GAME id={game_id} PGN_result={result} plies={len(rows)} "
              f"baseline_mean={statistics.mean(per_game['baseline']):.3f}ms "
              f"candidate_mean={statistics.mean(per_game['candidate']):.3f}ms "
              f"changed_moves=0", flush=True)
    b, c = metrics(totals["baseline"]), metrics(totals["candidate"])
    print(f"FULL GAMES: {len(games)} completed BO30 games, "
          f"{decisions_per_pass * repeats * 2} decisions/version; "
          f"all plies, both colours, same recorded moves and GRU history; "
          f"zero changed decisions", flush=True)
    print(f"FULL GAME TIME baseline mean={b[0]:.3f} median={b[1]:.3f} "
          f"p95={b[2]:.3f}ms; candidate mean={c[0]:.3f} "
          f"median={c[1]:.3f} p95={c[2]:.3f}ms; "
          f"time_saved={(1 - c[0] / b[0]) * 100:+.2f}%", flush=True)
    print("Recorded-game replay, not live games or a winrate test.", flush=True)


def native_of(module, board):
    original = claudio_agent.claudio_fast
    try:
        claudio_agent.claudio_fast = module
        return claudio_agent.native_board(board)
    finally:
        claudio_agent.claudio_fast = original


def exact_terminal_statuses(fast, codes):
    # Older G16 C++ does not yet provide the terminal batching API.
    if hasattr(fast, "terminal_after_moves"):
        return fast.terminal_after_moves(codes)
    statuses = []
    for move in codes:
        fast.push(move)
        try:
            statuses.append(fast.terminal_strict())
        finally:
            fast.pop()
    return statuses


def equal(a, b):
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(
            equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def compare_native(rows):
    values = [0.0, 1.0, 2.13, 2.73, 3.98, 9.06]
    checked = 0
    # Explicit tactical edge positions beyond the normal BO30 sampling:
    # castling, en passant, quiet/capture promotions, in-check evasion,
    # fifty-move horizon and history-dependent repetition.
    edges = [
        chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
        chess.Board("8/8/8/3pP3/8/8/4K3/7k w - d6 0 1"),
        chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1"),
        chess.Board("1r5k/P7/8/8/8/8/8/K7 w - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"),
        chess.Board("6k1/8/8/8/8/8/5Q2/6K1 w - - 96 80"),
    ]
    repetition = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repetition.push_uci(uci)
    edges.append(repetition)
    cases = rows[:min(len(rows), 32)] + [(9000 + i, 0, pos) for i, pos in enumerate(edges)]
    for _, _, board in cases:
        a, b = (native_of(module, board) for module in REFERENCES.values())
        assert a.fen() == b.fen(), "FEN state changed"
        codes = list(a.legal_moves())
        assert codes == list(b.legal_moves()), "Legal move order changed"
        top = codes[:min(4, len(codes))]
        if not top:
            continue
        functions = (
            lambda fast: exact_terminal_statuses(fast, codes),
            lambda fast: fast.exchange_after_moves(codes, values, 5, 200000),
            lambda fast: fast.deep_after_moves(top, values, 3, 5, 2000000, True),
            lambda fast: fast.root_search(top, [0.5, 0.49, 0.48, 0.47][:len(top)],
                                          [0.0] * len(top), 1.8069, 0.3, values, 3, 5, 2000000),
            lambda fast: fast.forced_mate(board.turn, 3, 20000, 1),
        )
        for fn in functions:
            x, y = fn(a), fn(b)
            if not equal(x, y):
                raise AssertionError(f"Native divergence in {fn} at {board.fen()}: {x!r} != {y!r}")
        assert a.fen() == b.fen() == board.fen(en_passant="fen"), "Candidate mutated native board"
        checked += 1
    print(f"NATIVE exact equivalence: {checked} positions, all five search/terminal methods identical", flush=True)


def evaluate(rows, loaded, module):
    net, values, conf, king = loaded
    original = claudio_agent.claudio_fast
    claudio_agent.claudio_fast = module
    try:
        players, last_game = {}, None
        outcomes, timings, groups = [], [], []
        for game_id, ply, board in rows:
            if last_game != game_id:
                players = {
                    side: claudio_agent.Claudio(
                        net, values, claudio_agent.ClaudioConfig.with_settings([], base=conf),
                        rng=np.random.default_rng([987, game_id, int(side)]), king_model=king)
                    for side in (chess.WHITE, chess.BLACK)
                }
                last_game = game_id
            actor = players[board.turn]
            begin = time.perf_counter()
            action = actor.choose(board)
            elapsed = (time.perf_counter() - begin) * 1000
            assert actor.moves == list(board.move_stack), "Historical plies were not reconstructed"
            assert action in board.legal_moves, f"Illegal move {action} in {board.fen()}"
            outcomes.append(action.uci())
            timings.append(elapsed)
            groups.append("endgame" if chess.popcount(board.occupied) <= conf["endgame_pieces"]
                          else "critical" if getattr(actor, "_depth", 0) >= conf["critical_depth"] else "normal")
        return outcomes, timings, groups
    finally:
        claudio_agent.claudio_fast = original


def metrics(samples):
    samples = sorted(samples)
    n = len(samples)
    return (statistics.mean(samples), statistics.median(samples),
            samples[min(n - 1, (95 * n + 99) // 100 - 1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--full-games", type=int, default=1,
                        help="Complete recorded BO30 games to replay per ply (0 to skip)")
    args = parser.parse_args()
    if args.positions <= 0 or args.repeats <= 0 or args.full_games < 0:
        parser.error("positions and repeats must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ckpt = Path(__file__).resolve().parent / "claudio-g16.pt"
    net, values, cfg, _, king = load_claudio(ckpt, with_config=True, with_king=True)
    assert cfg["critical_depth"] >= 6, "This must be the full G16 checkpoint"
    loaded = net, values, cfg, king
    rows = real_positions(args.positions)
    compare_native(rows)
    # Warm both native paths and the shared network outside timed samples.
    for label in ("baseline", "candidate", "candidate", "baseline"):
        evaluate(rows[:8], loaded, REFERENCES[label])
    print("WARMUP: 8 real BO30 positions, 2 passes per implementation", flush=True)
    expected = None
    timings = defaultdict(list)
    group_counts = None
    for repetition in range(args.repeats):
        for label in ("baseline", "candidate", "candidate", "baseline"):
            decisions, times, groups = evaluate(rows, loaded, REFERENCES[label])
            if expected is None:
                expected, group_counts = decisions, {n: groups.count(n) for n in set(groups)}
            else:
                mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(expected, decisions)) if a != b]
                if mismatches:
                    raise AssertionError(f"Changed {len(mismatches)} G16 decisions on {label}: {mismatches[:8]}")
            timings[label].extend(times)
            print(f"ABBA repeat={repetition + 1} {label}: "
                  f"mean={statistics.mean(times):.3f}ms positions={len(times)}", flush=True)
    base = metrics(timings["baseline"])
    cand = metrics(timings["candidate"])
    print(f"DECISION equivalence: 0 changed moves in {len(rows)} different positions, "
          f"{args.repeats * 2} repeats per version", flush=True)
    print(f"POSITION types: {group_counts}", flush=True)
    print(f"BASELINE: mean={base[0]:.3f} median={base[1]:.3f} p95={base[2]:.3f} ms", flush=True)
    print(f"CANDIDATE: mean={cand[0]:.3f} median={cand[1]:.3f} p95={cand[2]:.3f} ms", flush=True)
    print(f"TIME saved = {(1 - cand[0] / base[0]) * 100:+.2f}% ; "
          f"SPEED increase = {(base[0] / cand[0] - 1) * 100:+.2f}%", flush=True)
    print("Equivalent moves in sample alone are not an Elo or winrate non-inferiority proof.", flush=True)
    if args.full_games:
        benchmark_complete_games(completed_game_rows(args.full_games), loaded, args.repeats)


if __name__ == "__main__":
    main()
