cd /c/Users/Martim/Desktop/Github/TonkAI
T=claudio/claudio-sf2850-1723-recovered-pawn1.pt
D="--set tactic_top=4 --set root_pvs=true --set mate_attack=3 --set mate_gate=0.8 --set king_weight=0.047 --set repeat_as_draw=true"
python claudio/vs_engine.py --player claudio --games 100 --label claudio-sf1800
python claudio/vs_engine.py --player claudio --checkpoint $T --games 100 --label trained-stored-sf1800
python claudio/vs_engine.py --player claudio --checkpoint $T $D --games 100 --label trained-duelcfg-sf1800
python claudio/vs_engine.py --player timdgei --games 100 --label timdgei-sf1800
