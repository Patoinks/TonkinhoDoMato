cd /c/Users/Martim/Desktop/Github/TonkAI
python claudio/arena.py --games 300 --label v1-rep --set repeat_as_draw=true
python claudio/arena.py --games 300 --label v2-ks --set king_weight=0.047
python claudio/arena.py --games 300 --label v3-top4 --set tactic_top=4
python claudio/arena.py --games 300 --label v4-all --set repeat_as_draw=true --set king_weight=0.047 --set tactic_top=4
