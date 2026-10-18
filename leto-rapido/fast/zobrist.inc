// ------------------------------------------------------------------ fast core: keys (leto-rapido)
// Zobrist keys kept move by move, for the fast core's transposition table (Board.fast_core). The original
// search hashes python-chess's transposition key from scratch at every node; this one updates a 64-bit key
// with the squares a move changed. En passant counts only when a pawn of the side to move can take there.
inline bool& fast_core() { static thread_local bool on = false; return on; }

struct Zobrist {
    U64 piece[2][7][64], castle[64], ep[8], side;
    Zobrist() {
        U64 s = 0x9E3779B97F4A7C15ULL;
        auto next = [&]() {                     // splitmix64
            U64 z = (s += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        };
        for (auto& c : piece)
            for (auto& t : c)
                for (U64& v : t) v = next();
        for (U64& v : castle) v = next();
        for (U64& v : ep) v = next();
        side = next();
    }
};
const Zobrist Z;

HOT U64 z_square(const Pos& p, int s) {
    int t = p.type_at(s);
    return t ? Z.piece[p.color_at(s)][t][s] : 0;
}
HOT U64 z_castle(U64 rights) {
    U64 h = 0;
    for (; rights; rights &= rights - 1) h ^= Z.castle[lsb(rights)];
    return h;
}
HOT U64 z_ep(const Pos& p) {
    if (p.ep < 0) return 0;
    return (T.pawn[p.turn ^ 1][p.ep] & p.bb[PAWN] & p.occ[p.turn]) ? Z.ep[p.ep & 7] : 0;
}
inline U64 z_full(const Pos& p) {
    U64 h = p.turn == BLACK ? Z.side : 0;
    for (U64 all = p.occupied(); all; all &= all - 1) h ^= z_square(p, lsb(all));
    return h ^ z_castle(p.castling) ^ z_ep(p);
}
// The key after a move, from the key before: only the squares whose piece changed (2 to 4).
HOT U64 z_after(const Pos& before, const Pos& after, U64 key) {
    U64 touched = (before.occ[0] ^ after.occ[0]) | (before.occ[1] ^ after.occ[1]);
    for (; touched; touched &= touched - 1) {
        int s = lsb(touched);
        key ^= z_square(before, s) ^ z_square(after, s);
    }
    if (before.castling != after.castling) key ^= z_castle(before.castling) ^ z_castle(after.castling);
    return key ^ Z.side ^ z_ep(before) ^ z_ep(after);
}

// The capture subsequence of pseudo() into w (pseudo_captures without a vector); returns the end.
inline int* captures_into(const Pos& p, int* w) {
    const int us = p.turn;
    const U64 own = p.occ[us], enemy = p.occ[us ^ 1];
    for (U64 pieces = own & ~p.bb[PAWN]; pieces; pieces &= pieces - 1) {
        const int from = lsb(pieces);
        for (U64 targets = p.attacks_mask(from) & enemy; targets; targets &= targets - 1)
            *w++ = mv(from, lsb(targets));
    }
    for (U64 pawns = own & p.bb[PAWN]; pawns; pawns &= pawns - 1) {
        const int from = lsb(pawns);
        for (U64 caps = T.pawn[us][from] & enemy; caps; caps &= caps - 1) {
            const int to = lsb(caps);
            if (rank_of(to) == 0 || rank_of(to) == 7) {
                for (int t : {QUEEN, ROOK, BISHOP, KNIGHT}) *w++ = mv(from, to, t);
            } else {
                *w++ = mv(from, to);
            }
        }
    }
    for (int *m = w, *end = p.ep_into(w); m < end; ++m)
        if (enemy & bit(p.captured_square(*m))) *w++ = *m;
    return w;
}
