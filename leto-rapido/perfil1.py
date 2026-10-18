"""One game in-process (no pool), for a native profiler."""
import sys, time
sys.argv = ["torneio.py"]
import torneio
torneio._init(str(torneio.CHECKPOINT), [], [], False)
t = time.perf_counter()
row = torneio.play(0, 2026, 300, 6)
print(row["plies"], round(time.perf_counter() - t, 2))
