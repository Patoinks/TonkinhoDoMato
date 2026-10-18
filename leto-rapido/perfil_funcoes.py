"""Time per function inside a decision (pruned depth-7 Leto), two games, one process."""
import os; os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, collections
settings = ["root_prunings=True", "tactic_depth=7", "scan_prunings=True", "fast_net=True", "pawn_v4=True", "pawn_weight=0.5"] + sys.argv[1:]
sys.argv = ["torneio.py"]
import torneio
torneio._init(str(torneio.CHECKPOINT), settings, settings, False)
import claudio_agent as A, claudio_fast, pawnomics
from tonkai import fast_rules
T, N = collections.Counter(), collections.Counter()
def timed(owner, name, label=None):
    f = getattr(owner, name); label = label or name
    def g(*a, **k):
        t = time.perf_counter()
        try: return f(*a, **k)
        finally: T[label] += time.perf_counter() - t; N[label] += 1
    (object.__setattr__ if isinstance(owner, __import__("torch").nn.Module) else setattr)(owner, name, g)
for m in ("root_search", "deep_after_moves", "exchange_after_moves", "forced_mate", "legal_moves", "encode_here", "terminal_strict"):
    timed(claudio_fast.Board, m, "C++ " + m)
timed(fast_rules, "encode_after_moves", "encode_after_moves")
for name in dir(pawnomics):
    if callable(getattr(pawnomics, name)) and name.startswith(("pawn", "balance", "evaluate", "score")):
        timed(pawnomics, name, "pawnomics." + name)
for m in ("_replies", "_endings", "_material", "_scan", "_book", "_root", "_choose", "_catch_up", "_value", "_own_answers", "_remember"):
    if hasattr(A.Claudio, m): timed(A.Claudio, m)
for player in (torneio._state["base"][0], torneio._state["mutant"][0]):
    net, king = player[0], player[3]
    import rede_rapida; rede_rapida.acelerar(net); rede_rapida.acelerar_king(king)
    for m in ("candidates", "survival", "predict_logits", "encode", "memory_cell"):
        timed(net, m, "net." + m)
    if king is not None: timed(king, "ks_rows", "king.ks_rows")
import abertura
for m in ("principle_scores", "repertoire_move", "weight_at"):
    if hasattr(abertura, m): timed(abertura, m, "abertura." + m)
for game in range(int(os.environ.get("GAMES", "2"))): torneio.play(game, 2026, 300, 6)
total = T["_choose"]
for k, v in T.most_common(30): print(f"{k:34s} {v:6.2f}s {v/total:6.1%} calls {N[k]:6d} {1000*v/max(1,N[k]):7.3f} ms")
