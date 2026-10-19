cd leto-rapido
for d in nezha loki anubis hades; do
  python torneio.py --games 24 --workers 6 --deus $d --seed 77 --out ../deus-$d.json 2>/dev/null | python -c "
import sys,json; d=json.loads(sys.stdin.read()); print('$d', 'score', round(d['mutant_score'],3), 'changed', d['style']['changed_rate'], 'plies', round(d['plies_per_game']))"
done
