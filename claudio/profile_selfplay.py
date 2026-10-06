"""Where a self-play training game spends its time (one worker's view), and
where a learner step spends its time. Read-only: nothing is trained or saved.

    python claudio/profile_selfplay.py games 4
    python claudio/profile_selfplay.py learner 20
"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
import psutil_free_priority  # noqa: E402,F401

START = ROOT / "claudio/claudio-grande-g0.pt"


def games(n):
    import torch
    torch.set_num_threads(1)
    import train_grande as tg
    from claudio_load import load_claudio
    net, values, config, _, king = load_claudio(START, with_config=True, with_king=True)
    arch = torch.load(START, map_location="cpu", weights_only=False)["architecture"]
    out = ROOT / "claudio/results/profile-selfplay"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in net.state_dict().items()}, out / "actor-0.pt")
    tg._init_worker(str(out), arch, values, config, 1800, 0.002, "self", str(START), 0.25, "victory")
    import claudio_agent
    clocks = {"learner": {}, "frozen": {}}
    original = claudio_agent.Claudio.choose
    spent = {"learner": 0.0, "frozen": 0.0}
    decisions = {"learner": 0, "frozen": 0}

    def timed(self, board):
        who = "frozen" if self.net is tg._w["frozen"] else "learner"
        self.clock = clocks[who]
        start = time.perf_counter()
        move = original(self, board)
        spent[who] += time.perf_counter() - start
        decisions[who] += 1
        return move

    claudio_agent.Claudio.choose = timed
    total, plies = 0.0, 0
    for game in range(n):
        start = time.perf_counter()
        result = tg.play(game, 0, 7, 0, 300, 16, 0.03)
        total += time.perf_counter() - start
        plies += len(result["actions"])
    labels = total - spent["learner"] - spent["frozen"]
    print(f"{n} jogos, {plies / n:.0f} meios-lances/jogo, {total / n:.2f} s/jogo por processo")
    for who in ("learner", "frozen"):
        print(f"  {who:8s} {spent[who] / n:5.2f} s/jogo  {1000 * spent[who] / max(1, decisions[who]):5.2f} ms/lance")
        for label, seconds in sorted(clocks[who].items(), key=lambda kv: -kv[1]):
            print(f"      {label:13s} {1000 * seconds / max(1, decisions[who]):5.2f} ms/lance")
    print(f"  labels   {labels / n:5.2f} s/jogo (encodings, surpresas, empacotar)")


def learner(steps):
    import numpy as np
    import torch
    import train_grande as tg
    from match_checkpoints import _historical_config
    from claudio_load import load_claudio
    net, values, config, history, king = load_claudio(START, with_config=True, with_king=True)
    state = torch.load(tg.TIMDGEI, map_location="cpu", weights_only=False)
    cfg = _historical_config(state)
    cfg.syzygy_path, cfg.tablebase_adjudicate = "", False
    cfg.device, cfg.sequence_batch_size, cfg.objective = "cuda", 64, "victory"
    net.requires_grad_(True).train()
    net.rnd_target.requires_grad_(False)
    Trainer = tg.make_trainer_class()
    Trainer.choice_weight = 0.0
    trainer = Trainer(cfg, net=net)
    trainer.replay.load_state_dict(state["replay"])
    del state
    for _ in range(3):
        trainer.update()
    torch.cuda.synchronize()
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    for _ in range(steps):
        trainer.update()
    torch.cuda.synchronize()
    profile.disable()
    print(f"{steps} passos: {1000 * (time.perf_counter() - start) / steps:.0f} ms/passo")
    stats = pstats.Stats(profile)
    stats.sort_stats("cumulative").print_stats(28)


if __name__ == "__main__":
    kind, n = sys.argv[1], int(sys.argv[2])
    (games if kind == "games" else learner)(n)
