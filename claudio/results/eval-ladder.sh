cd /c/Users/Martim/Desktop/Github/TonkAI
for c in claudio/claudio-grande-g0.pt claudio/train-ladder/claudio-grande-2000.pt claudio/train-ladder/claudio-grande-4000.pt claudio/train-ladder/claudio-grande-6000.pt; do
  n=$(basename $c .pt)
  python claudio/vs_engine.py --player claudio --checkpoint $c --engine fairy --elo 1300 --games 100 --workers 2 --label ladder-$n-fairy1300 2>&1 | grep WR
done
