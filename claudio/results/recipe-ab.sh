cd /c/Users/Martim/Desktop/Github/TonkAI
D="--set tactic_top=4 --set root_pvs=true --set mate_attack=3 --set mate_gate=0.8 --set king_weight=0.047 --set repeat_as_draw=true"
python claudio/train_grande.py --start claudio/claudio-grande-g0.pt --out-dir claudio/train-A --hours 1 --max-games 2000 --batch 64 --updates 2 --choice-weight 0 --save-every 100000 > claudio/train-A.out 2>&1
python claudio/vs_engine.py --player claudio --checkpoint claudio/train-A/claudio-grande-final.pt $D --games 100 --label recipe-A-sf1800 2>&1 | grep WR
python claudio/train_grande.py --start claudio/claudio-grande-g0.pt --out-dir claudio/train-B --hours 1 --max-games 2000 --batch 64 --updates 2 --choice-weight 0.1 --choice-tau 0.2 --save-every 100000 > claudio/train-B.out 2>&1
python claudio/vs_engine.py --player claudio --checkpoint claudio/train-B/claudio-grande-final.pt $D --games 100 --label recipe-B-sf1800 2>&1 | grep WR
