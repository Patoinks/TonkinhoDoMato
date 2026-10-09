"""Independent Torch valuation and python-chess material oracles. No learning."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import paths
import chess
from v2.package import load_bundle
from tonkai.encoding import encode


def valuation(net,base,weight,board,side):
    replay=board.root();rows=[encode(replay,side)]
    for move in board.move_stack:replay.push(move);rows.append(encode(replay,side))
    latent=net.encode(torch.from_numpy(np.stack(rows)))
    memory=net.initial_memory(1)
    for row in latent:memory=net.memory_cell(row[None],memory)
    context=net.context_from_latent(latent[-1:],memory)
    def raw(piece,square):
        relative=square if side else square^56
        if piece.color!=side:relative^=56
        return float(net.piece_values(context,torch.tensor([piece.piece_type-1]),torch.tensor([relative]),torch.tensor([0]))[0])
    items=sorted(board.piece_map().items());raws=[raw(p,s) for s,p in items]
    pawns=[abs(v) for v,(_,p) in zip(raws,items) if p.piece_type==chess.PAWN]
    unit=max(.001,float(np.median(pawns))) if pawns else .001
    def amount(piece,square):
        learned=max(-12,min(12,raw(piece,square)/unit))
        return (1-weight)*learned+weight*base[piece.piece_type]
    frame=[0.]*128
    for square,piece in items:
        if piece.piece_type!=chess.KING:frame[square]=amount(piece,square)
    for color in (False,True):
        for kind in range(2,6):
            for file in range(8):frame[64+int(color)*32+(kind-2)*8+file]=amount(chess.Piece(kind,color),file+(56 if color else 0))
    return frame


def after(board,move,values,palette):
    values=values.copy();from_=move.from_square;to=move.to_square;piece=board.piece_at(from_)
    captured=to+(-8 if board.turn else 8) if board.is_en_passant(move) else to
    gain=values[captured] if board.is_capture(move) else 0.
    moving=values[from_];values[from_]=0.;values[captured]=0.
    if move.promotion:
        promoted=palette[64+int(board.turn)*32+(move.promotion-2)*8+chess.square_file(to)]
        gain+=promoted-moving;moving=promoted
    if board.is_castling(move):
        rank=chess.square_rank(from_)*8;kingside=chess.square_file(to)==6
        rook_from=rank+(7 if kingside else 0);rook_to=rank+(5 if kingside else 3)
        values[rook_to]=values[rook_from];values[rook_from]=0.
    values[to]=moving
    return gain,values


def quiet(board,depth,values,palette):
    moves=list(board.legal_moves);check=board.is_check()
    if check and not moves:return -1000.
    if depth==0:return 0.
    best=-math.inf if check else 0.
    for move in moves:
        if not check and not board.is_capture(move):continue
        gain,next_values=after(board,move,values,palette);board.push(move)
        score=gain-quiet(board,depth-1,next_values,palette);board.pop();best=max(best,score)
    return best


def deep(board,depth,qdepth,values,palette,ply=1):
    if depth<=0:return quiet(board,qdepth,values,palette)
    moves=list(board.legal_moves)
    if not moves:return -(1000.-ply) if board.is_check() else 0.
    best=-math.inf
    for move in moves:
        gain,next_values=after(board,move,values,palette);board.push(move)
        score=gain-deep(board,depth-1,qdepth,next_values,palette,ply+1);board.pop();best=max(best,score)
    return best


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--legacy',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    player=load_bundle(args.bundle);net=player.net
    legacy=json.loads(Path(args.legacy).read_text());weight=legacy['config']['kd_weight'];base={int(k):v for k,v in legacy['kd_values'].items()}
    fens=[chess.STARTING_FEN,'4k3/8/8/2n1n3/3Q4/2N1N3/8/4K3 w - - 20 21',
          '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 21','1r2k3/P7/8/8/8/8/8/4K3 w - - 0 21',
          '4k3/8/8/8/8/8/2p5/3RK3 b - - 0 21','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 20 21',
          '4r1k1/8/8/8/3q4/8/4N3/4K3 w - - 20 21','7k/1R6/7B/P7/8/8/4K3/8 w - - 53 143']
    frames=[];material=[]
    with torch.inference_mode():
        for fen in fens:
            for side in (False,True):
                board=chess.Board(fen);frame=valuation(net,base,weight,board,side)
                frames.append({'root':fen,'side':side,'values':frame})
        board=chess.Board()
        for uci in ('e2e4','e7e5','g1f3','b8c6','f1c4','g8f6'):board.push_uci(uci)
        for side in (False,True):frames.append({'root':chess.STARTING_FEN,'moves':[m.uci() for m in board.move_stack],'side':side,'values':valuation(net,base,weight,board,side)})
    for ordinal,fen in enumerate(fens[1:7]):
        board=chess.Board(fen)
        # Deliberately unequal same-type values ensure type-only tables cannot pass.
        frame=[.37+(i+ordinal*11)%19*.213 for i in range(128)]
        for square,piece in board.piece_map().items():
            if piece.piece_type==chess.KING:frame[square]=0.
        rows=[]
        for move in list(board.legal_moves):
            gain,next_values=after(board,move,frame[:64],frame);board.push(move)
            rows.append({'uci':move.uci(),'gain':gain,'exchange':quiet(board,2,next_values,frame),'deep':deep(board,2,1,next_values,frame)})
            board.pop()
        material.append({'fen':fen,'values':frame,'rows':rows})
    Path(args.out).write_text(json.dumps({'weight':weight,'frames':frames,'material':material},indent=2),encoding='utf-8')
    print(f'{len(frames)} contextual frames; {sum(len(r["rows"]) for r in material)} independent material search cases')


if __name__=='__main__':main()
