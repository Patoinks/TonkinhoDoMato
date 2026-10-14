"""Board -> flat binary vector, always from TonkAI's own point of view.

Layout (782 floats):

    [  0 .. 767]  12 piece planes x 64 squares
                  planes 0-5  : SELF   pawn knight bishop rook queen king
                  planes 6-11 : OPPONENT, same order
    [768]         1 if it is SELF to move
    [769 .. 772]  castling rights: self K, self Q, opp K, opp Q
    [773 .. 780]  en-passant file, one-hot (all zero when there is none)
    [781]         halfmove clock / 100

When TonkAI plays black the board is flipped vertically, so "my back rank" is
always squares 0-7. TonkAI therefore only ever learns one perspective; it does
not know it is playing white or black.

Indices 0..780 are binary, which is why a position packs into 99 bytes.
"""

from __future__ import annotations

import chess
import numpy as np

N_PLANES = 12
N_SQUARES = 64
PLANE_SIZE = N_PLANES * N_SQUARES          # 768
EXTRA_SIZE = 14
INPUT_SIZE = PLANE_SIZE + EXTRA_SIZE       # 782

OFF_STM = PLANE_SIZE
OFF_CASTLE = PLANE_SIZE + 1
OFF_EP = PLANE_SIZE + 5
OFF_HALFMOVE = PLANE_SIZE + 13

# everything except the halfmove clock is a single bit
N_BINARY = INPUT_SIZE - 1                  # 781
PACK_BITS_BYTES = (N_BINARY + 7) // 8      # 98
PACKED_SIZE = PACK_BITS_BYTES + 1          # 99

PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING)
PIECE_NAMES = ("Pawn", "Knight", "Bishop", "Rook", "Queen", "King")
KING_PLANE_SELF = 5
KING_PLANE_OPP = 11

# new[i] = old[i ^ 56]  -> vertical flip
_FLIP = np.array([sq ^ 56 for sq in range(64)], dtype=np.intp)

_mask_scratch = np.empty(N_PLANES, dtype=np.uint64)


def encode_into(board: chess.Board, self_color: bool, out: np.ndarray) -> None:
    """Write one position into `out` (float32, length INPUT_SIZE)."""
    out[:] = 0.0

    m = _mask_scratch
    i = 0
    for color in (self_color, not self_color):
        for pt in PIECE_ORDER:
            m[i] = board.pieces_mask(pt, color)
            i += 1

    bits = np.unpackbits(m.view(np.uint8), bitorder="little")
    if self_color == chess.BLACK:
        bits = bits.reshape(N_PLANES, N_SQUARES)[:, _FLIP].reshape(PLANE_SIZE)
    out[:PLANE_SIZE] = bits

    out[OFF_STM] = 1.0 if board.turn == self_color else 0.0
    opp = not self_color
    out[OFF_CASTLE + 0] = float(board.has_kingside_castling_rights(self_color))
    out[OFF_CASTLE + 1] = float(board.has_queenside_castling_rights(self_color))
    out[OFF_CASTLE + 2] = float(board.has_kingside_castling_rights(opp))
    out[OFF_CASTLE + 3] = float(board.has_queenside_castling_rights(opp))

    if board.ep_square is not None:
        out[OFF_EP + chess.square_file(board.ep_square)] = 1.0

    out[OFF_HALFMOVE] = min(board.halfmove_clock, 100) / 100.0


def encode(board: chess.Board, self_color: bool) -> np.ndarray:
    out = np.empty(INPUT_SIZE, dtype=np.float32)
    encode_into(board, self_color, out)
    return out


def encode_after_moves(board: chess.Board, moves, self_color: bool,
                       out: np.ndarray | None = None) -> np.ndarray:
    """Encode the position resulting from each move, into ONE (N, 782) array.

    Keep python-chess responsible for each resulting position, then expand
    all piece bitboards together for the network's single forward pass.
    """
    n = len(moves)
    if out is None or out.shape[0] < n:
        out = np.empty((max(n, 1), INPUT_SIZE), dtype=np.float32)
    result = out[:n]
    if n == 0:
        return result
    result[:] = 0.0
    masks = np.empty((n, N_PLANES), dtype=np.uint64)
    opp = not self_color
    for i, mv in enumerate(moves):
        board.push(mv)
        try:
            j = 0
            for color in (self_color, opp):
                for pt in PIECE_ORDER:
                    masks[i, j] = board.pieces_mask(pt, color)
                    j += 1
            result[i, OFF_STM] = 1.0 if board.turn == self_color else 0.0
            result[i, OFF_CASTLE] = float(board.has_kingside_castling_rights(self_color))
            result[i, OFF_CASTLE + 1] = float(board.has_queenside_castling_rights(self_color))
            result[i, OFF_CASTLE + 2] = float(board.has_kingside_castling_rights(opp))
            result[i, OFF_CASTLE + 3] = float(board.has_queenside_castling_rights(opp))
            if board.ep_square is not None:
                result[i, OFF_EP + chess.square_file(board.ep_square)] = 1.0
            # Preserve encode_into's Python float division before assignment.
            result[i, OFF_HALFMOVE] = min(board.halfmove_clock, 100) / 100.0
        finally:
            board.pop()
    bits = np.unpackbits(masks.view(np.uint8), axis=1, bitorder="little")
    if self_color == chess.BLACK:
        # Reverse ranks inside every plane, matching the square ^ 56 mapping.
        bits = bits.reshape(n, N_PLANES, 8, 8)[:, :, ::-1, :].reshape(n, PLANE_SIZE)
    result[:, :PLANE_SIZE] = bits
    return result


# ------------------------------------------------------------------ packing

def pack(vec: np.ndarray) -> np.ndarray:
    """782 floats -> 99 bytes."""
    bits = vec[:N_BINARY].astype(np.uint8)
    packed = np.empty(PACKED_SIZE, dtype=np.uint8)
    packed[:PACK_BITS_BYTES] = np.packbits(bits, bitorder="little")
    packed[PACK_BITS_BYTES] = int(round(float(vec[OFF_HALFMOVE]) * 100.0))
    return packed


def pack_into(vec: np.ndarray, out: np.ndarray) -> None:
    out[:PACK_BITS_BYTES] = np.packbits(vec[:N_BINARY].astype(np.uint8),
                                        bitorder="little")
    out[PACK_BITS_BYTES] = int(round(float(vec[OFF_HALFMOVE]) * 100.0))


def pack_many(vectors: np.ndarray) -> np.ndarray:
    """Vectorized pack_into, including its float64 rounding of the clock."""
    out = np.empty((len(vectors), PACKED_SIZE), dtype=np.uint8)
    out[:, :PACK_BITS_BYTES] = np.packbits(
        vectors[:, :N_BINARY].astype(np.uint8), axis=1, bitorder="little")
    out[:, PACK_BITS_BYTES] = np.rint(
        vectors[:, OFF_HALFMOVE].astype(np.float64) * 100.0).astype(np.uint8)
    return out


def unpack_many(packed: np.ndarray) -> np.ndarray:
    """(N, 99) bytes -> (N, 782) float32."""
    packed = np.atleast_2d(packed)
    bits = np.unpackbits(packed[:, :PACK_BITS_BYTES], axis=1, bitorder="little")
    out = np.empty((packed.shape[0], INPUT_SIZE), dtype=np.float32)
    out[:, :N_BINARY] = bits[:, :N_BINARY]
    out[:, OFF_HALFMOVE] = packed[:, PACK_BITS_BYTES].astype(np.float32) / 100.0
    return out


# --------------------------------------------------------- plane <-> square

def plane_square_to_index(plane: int, square: int) -> int:
    return plane * N_SQUARES + square


def describe_plane(plane: int) -> tuple[str, str]:
    """(side, piece name) for a plane index."""
    side = "self" if plane < 6 else "opponent"
    return side, PIECE_NAMES[plane % 6]


def oriented_square(square: int, self_color: bool) -> int:
    """Real board square -> the square index used inside the encoding."""
    return square if self_color == chess.WHITE else square ^ 56


def real_square(oriented: int, self_color: bool) -> int:
    return oriented if self_color == chess.WHITE else oriented ^ 56
