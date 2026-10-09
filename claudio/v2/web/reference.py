"""History-preserving Python references for validating the browser port; no learning."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import paths
import chess
import numpy as np
import torch
from v2.package import load_bundle
from v2.rules import adjudicate


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--experience',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    games=json.loads(Path(args.experience).read_text())['games'][::6][:12]
    eligible=[(g,k) for g,row in enumerate(games) for k in range(len(row['moves'].split()))]
    selected={eligible[round(i*(len(eligible)-1)/95)] for i in range(96)}
    rows=[]
    for game,row in enumerate(games):
        board=chess.Board(); moves=[]
        for ply,uci in enumerate(row['moves'].split()):
            if (game,ply) in selected and adjudicate(board) is None:
                player=load_bundle(args.bundle,rng=np.random.default_rng(991))
                chosen=player.choose(board).uci()
                rows.append({'root':board.root().fen(),'moves':moves.copy(),'fen':board.fen(en_passant='fen'),
                             'move':chosen,'memory':player.memory.squeeze().tolist(),'stats':player.stats})
            board.push_uci(uci);moves.append(uci)
    # Force the conversion module, captures, promotion, draw claims and both colours.
    for fen in ['7k/1R6/7B/P7/8/8/4K3/8 w - - 53 143','7k/8/5KQ1/8/8/8/8/8 w - - 0 1',
                '8/8/8/8/8/2qk4/8/K7 b - - 0 1','4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1',
                '1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2']:
        board=chess.Board(fen);player=load_bundle(args.bundle)
        rows.append({'root':fen,'moves':[],'fen':fen,'move':player.choose(board).uci(),'memory':player.memory.squeeze().tolist(),'stats':player.stats})
    Path(args.out).write_text(json.dumps(rows,indent=2),encoding='utf-8');print(f'{len(rows)} Python decisions saved')


if __name__=='__main__':main()
