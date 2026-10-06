cd /c/Users/Martim/Desktop/Github/TonkAI
S="--set tactic_top=4 --set mate_attack=3 --set mate_gate=0.8 --set king_weight=0.047 --set repeat_as_draw=true"
python claudio/arena.py --games 300 --workers 5 --label v8-gate $S
python claudio/arena.py --games 300 --workers 5 --label v9-gate-exact $S --set root_pvs=false
python claudio/arena.py --games 300 --workers 5 --label v10-v4fix --set tactic_top=4 --set root_pvs=false --set king_weight=0.047 --set repeat_as_draw=true
python claudio/ablate_timdgei.py --games 200 --label no-memory --set memory_weight=0
python claudio/ablate_timdgei.py --games 200 --label no-tablebase --set syzygy_path=
python claudio/ablate_timdgei.py --games 200 --label no-passed-pawn --set passed_pawn_bonus=0
