cd /c/Users/Martim/Desktop/Github/TonkAI
S="--set tactic_top=4 --set mate_attack=2 --set king_weight=0.047 --set repeat_as_draw=true"
python claudio/arena.py --games 300 --label v5-pvs $S
python claudio/arena.py --games 300 --label v6-draw03 $S --set draw_value=0.3
python claudio/arena.py --games 300 --label v7-top6 $S --set tactic_top=6
