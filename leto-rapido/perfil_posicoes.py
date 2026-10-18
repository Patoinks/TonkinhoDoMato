"""Time per function of one configuration's decisions on the positions of recorded games (m039 + --rapido).

    python leto-rapido/perfil_posicoes.py ../mega-torneio/m039-vs-leto-rapido.json 8
"""
import os; os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import collections, json, sys, time
import chess, numpy as np, torch
sys.path.insert(0, "."); import paths  # noqa
torch.set_num_threads(1)
import torneio, claudio_fast, abertura, rede_rapida
import claudio_agent as A
from claudio_load import load_claudio
from tonkai import fast_rules
M039 = []   # the champion (t0467) is part of torneio.RAPIDO now
net, values, config, _, king = load_claudio(str(torneio.CHECKPOINT), with_config=True, with_king=True)
cfg = A.ClaudioConfig.with_settings(["syzygy="] + torneio.RAPIDO + M039 + sys.argv[3:], base=config)
T, N = collections.Counter(), collections.Counter()
def timed(owner, name, label=None):
    f = getattr(owner, name); label = label or name
    def g(*a, **k):
        t = time.perf_counter()
        try: return f(*a, **k)
        finally: T[label] += time.perf_counter() - t; N[label] += 1
    (object.__setattr__ if isinstance(owner, torch.nn.Module) else setattr)(owner, name, g)
for m in ("root_search", "deep_after_moves", "exchange_after_moves", "forced_mate", "legal_moves", "encode_here", "terminal_strict"):
    timed(claudio_fast.Board, m, "C++ " + m)
timed(fast_rules, "encode_after_moves", "encode_after_moves")
for m in ("_replies", "_endings", "_scan", "_root", "_choose", "_catch_up", "_value", "_own_answers", "_tablebase"):
    if hasattr(A.Claudio, m): timed(A.Claudio, m)
timed(A.Claudio, "choose", "choose (total)")
timed(A, "native_board", "native_board")
rede_rapida.acelerar(net); rede_rapida.acelerar_king(king)
for m in ("candidates", "survival", "predict_logits", "encode", "memory_cell"):
    timed(net, m, "net." + m)
if king is not None: timed(king, "ks_rows", "king.ks_rows")
for m in ("principle_scores", "repertoire_move"):
    timed(abertura, m, "abertura." + m)
abertura.repertoire_move(chess.Board())            # the book is built once per process: not a decision cost
T.clear(); N.clear()
games = json.load(open(sys.argv[1], encoding="utf-8"))["games"][:int(sys.argv[2])]
decisions = 0
for g in games:
    players = {c: A.Claudio(net, values, cfg, rng=np.random.default_rng(1), king_model=king) for c in (chess.WHITE, chess.BLACK)}
    for q in players.values(): q._owner = 1   # one player per colour: the game memory then only catches up
    b = chess.Board()
    for i, u in enumerate(g["moves"].split()):
        if i >= 6:
            players[b.turn].choose(b); decisions += 1
        b.push_uci(u)
total = T["choose (total)"]
print(f"{decisions} decisões, {1000 * total / decisions:.2f} ms por decisão\n")
print("| parte | ms/decisão | % | chamadas/decisão |")
print("| --- | ---: | ---: | ---: |")
for k, v in T.most_common():
    print(f"| {k} | {1000 * v / decisions:.3f} | {v / total:.1%} | {N[k] / decisions:.2f} |")
