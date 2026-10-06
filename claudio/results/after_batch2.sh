cd /c/Users/Martim/Desktop/Github/TonkAI
until [ -f claudio/results/arena-v7-top6.json ]; do sleep 5; done
python claudio/train_claudio.py --minutes 60 --out claudio/claudio-trained.pt --log claudio/results/train.jsonl \
  --set tactic_top=4 --set root_pvs=false --set mate_attack=3 --set mate_gate=0.8 --set king_weight=0.047 \
  --set repeat_as_draw=true > claudio/results/train.out 2>&1 &
bash claudio/results/batch3.sh
wait
