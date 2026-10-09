"""UCI G16 v2 with actual game-ending and draw handling."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
import torch
import chess
from v2.package import load_bundle
from v2.rules import adjudicate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--variant',choices=('full','all','teacher','finish','exact'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    player=load_bundle(args.checkpoint,args.variant)
    board=chess.Board()
    for raw in sys.stdin:
        words=raw.strip().split()
        if not words:continue
        if words[0]=='uci':print('id name G16 v2 checkpoint600\nid author Martim / Codex\nuciok',flush=True)
        elif words[0]=='isready':print('readyok',flush=True)
        elif words[0]=='ucinewgame':player.reset();board.reset()
        elif words[0]=='position':
            if words[1]=='startpos':board=chess.Board(); index=2
            else:board=chess.Board(' '.join(words[2:8])); index=8
            if index<len(words) and words[index]=='moves':
                for uci in words[index+1:]:board.push_uci(uci)
        elif words[0]=='go':
            ending=adjudicate(board)
            if ending:
                print(f"info string result {ending['result']} reason {ending['end']}",flush=True)
                print('bestmove 0000',flush=True)
            else:print('bestmove '+player.choose(board,terminal_checked=True).uci(),flush=True)
        elif words[0]=='quit':break


if __name__=='__main__':main()
