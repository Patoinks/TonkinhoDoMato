"""Time per Claudio method around the native calls, one worker, two games."""
import sys, time, collections
sys.argv_settings = sys.argv[1:]; sys.argv = ["torneio.py"]
import torneio
import claudio_agent as A
T = collections.Counter(); N = collections.Counter()
def wrap(name):
    f = getattr(A.Claudio, name)
    def g(self, *a, **k):
        t = time.perf_counter()
        try: return f(self, *a, **k)
        finally: T[name] += time.perf_counter() - t; N[name] += 1
    setattr(A.Claudio, name, g)
for name in ("_root", "_deep_lines", "_scan", "_replies", "_own_answers", "_endings", "_material", "_dynamic_exchange", "_choose"):
    if hasattr(A.Claudio, name): wrap(name)
torneio._init(str(torneio.CHECKPOINT), sys.argv_settings, sys.argv_settings, False)
for game in range(2): torneio.play(game, 2026, 300, 6)
import claudio_agent  # mutant decisions only are timed below
total = T["_choose"]
for k, v in T.most_common(): print(f"{k:18s} {v:7.2f}s {v/total:6.1%} calls {N[k]:5d} {1000*v/max(1,N[k]):8.2f} ms/call")
