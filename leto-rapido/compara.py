"""Compare two tournament result files: identical moves per game, and the time ratio."""
import json, sys
a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
ga, gb = {g["game"]: g for g in a["games"]}, {g["game"]: g for g in b["games"]}
same = all(ga[k]["moves"] == gb[k]["moves"] for k in ga if k in gb)
diff = [k for k in ga if k in gb and ga[k]["moves"] != gb[k]["moves"]]
sa, sb = a["summary"]["seconds_per_game"], b["summary"]["seconds_per_game"]
print(f"identical moves: {same} (different games: {diff}); s/game {sa:.2f} -> {sb:.2f}; speedup x{sa / sb:.3f}")
