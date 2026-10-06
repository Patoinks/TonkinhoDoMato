cd /c/Users/Martim/Desktop/Github/TonkAI
python claudio/train_claudio.py --minutes 60 --out claudio/claudio-trained.pt --log claudio/results/train.jsonl \
  --set tactic_top=4 --set root_pvs=false --set mate_attack=3 --set mate_gate=0.8 --set king_weight=0.047 \
  --set repeat_as_draw=true > claudio/results/train.out 2>&1
