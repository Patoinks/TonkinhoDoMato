"""Claudio's thinking time per section, ONE process, Claudio vs Timdgei. Read-only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401
import arena  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
settings = sys.argv[2:]
arena._init("", settings, [])
import claudio_agent  # noqa: E402

clock, decisions = {}, [0]
original = claudio_agent.Claudio.choose


def choose(self, board):
    self.clock = clock
    decisions[0] += 1
    return original(self, board)


claudio_agent.Claudio.choose = choose
start = time.perf_counter()
games = [arena.play(game, 99, 300) for game in range(GAMES)]
s = arena.summary(games)
print(f"{GAMES} games, score {s['score']:.2f}, Claudio {s['time']['claudio']['ms_per_decision']:.2f} ms/decision, "
      f"Timdgei {s['time']['timdgei']['ms_per_decision']:.2f}, {s['plies_per_game']:.0f} plies/game, "
      f"{time.perf_counter() - start:.0f} s")
for label, seconds in sorted(clock.items(), key=lambda kv: -kv[1]):
    print(f"  {label:14s} {1000 * seconds / decisions[0]:6.2f} ms per decision")
