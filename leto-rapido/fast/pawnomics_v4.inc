// ------------------------------------------------------------------ pawnomics v4 (leto-rapido)
// Every pawn, in the side's own piece values (pawns): passed pawns by how likely and how soon they
// queen, and the structure of all pawns. One native call per decision (Board.pawnomics_v4_after_moves).
namespace pawnomics4 {
constexpr U64 FILE_A = 0x0101010101010101ULL;
inline U64 file_mask(int f) { return FILE_A << f; }
inline U64 adjacent_files(int f) { return (f > 0 ? file_mask(f - 1) : 0) | (f < 7 ? file_mask(f + 1) : 0); }
// Squares strictly in front of `s` for colour c (towards promotion), on the same file.
inline U64 front(int c, int s) {
    U64 f = file_mask(file_of(s));
    int r = rank_of(s);
    if (c == WHITE) return r < 7 ? f & (ALL << (8 * (r + 1))) : 0;
    return r > 0 ? f & (ALL >> (8 * (8 - r))) : 0;
}
inline U64 front_and_adjacent(int c, int s) {
    int f = file_of(s);
    U64 x = 0;
    if (f > 0) x |= front(c, s - 1);
    if (f < 7) x |= front(c, s + 1);
    return x;
}
inline U64 behind_or_level(int c, int s) {          // its rank and the ranks behind it, all files
    int r = rank_of(s);
    return c == WHITE ? (ALL >> (8 * (7 - r))) : (ALL << (8 * r));
}
inline int distance(int a, int b) {
    return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b)));
}
inline double value_of(const std::vector<double>& v, int type) { return type == KING ? 1000.0 : v[type]; }

// What `them` wins by capturing first on `s`, where a piece of the other side worth `standing` sits
// (static exchange, x-rays revealed as pieces leave). `occupied` already holds that piece on `s`.
inline double see(const Pos& p, int s, int them, double standing, U64 occupied, const std::vector<double>& v) {
    double gains[32];
    int n = 0, side = them;
    double piece = standing;
    for (;;) {
        U64 a = p.attackers(side, s, occupied) & occupied & ~bit(s);
        if (!a || n >= 31) break;
        int best = -1;
        double best_value = 1e18;
        for (U64 m = a; m; m &= m - 1) {
            int sq = lsb(m);
            double val = value_of(v, p.type_at(sq));
            if (val < best_value) { best_value = val; best = sq; }
        }
        if (best_value >= 1000.0 && (p.attackers(side ^ 1, s, occupied & ~bit(best)) & occupied & ~bit(best) & ~bit(s)))
            break;                                   // the king cannot take a defended piece
        gains[n] = piece - (n ? gains[n - 1] : 0.0);
        ++n;
        piece = best_value;
        occupied &= ~bit(best);
        side ^= 1;
    }
    for (int i = n - 1; i > 0; --i) gains[i - 1] = -std::max(-gains[i - 1], gains[i]);
    return n ? std::max(0.0, gains[0]) : 0.0;
}

inline double side_value(const Pos& p, int c, bool first, const std::vector<double>& v) {
    const int them = c ^ 1;
    const U64 own = p.bb[PAWN] & p.occ[c], enemy = p.bb[PAWN] & p.occ[them];
    const double pawn = v[PAWN], queen_gain = v[QUEEN] - v[PAWN];
    double enemy_material = 0.0;
    for (int t = KNIGHT; t <= QUEEN; ++t) enemy_material += v[t] * popcnt(p.bb[t] & p.occ[them]);
    const double catch_scale = std::max(0.25, 1.0 - enemy_material / 20.0);
    const int own_king = p.king(c), enemy_king = p.king(them);
    const U64 all = p.occupied();
    double total = 0.0;
    for (U64 m = own; m; m &= m - 1) {
        const int s = lsb(m), f = file_of(s), r = rank_of(s);
        const int rel = c == WHITE ? r : 7 - r;           // 1..6 counted from its own side
        const U64 ahead = front(c, s), ahead_adjacent = front_and_adjacent(c, s);
        const bool passed = !(enemy & (ahead | ahead_adjacent));
        const bool doubled = (own & ahead) != 0;          // an own pawn in front: this is the rear pawn
        const bool isolated = !(own & adjacent_files(f));
        const bool protected_ = (T.pawn[them][s] & own) != 0;   // own pawns attacking s
        const bool phalanx = (f > 0 && (own & bit(s - 1))) || (f < 7 && (own & bit(s + 1)));
        // structure, for every pawn
        if (doubled) total -= 0.20 * pawn;
        if (isolated) total -= ((enemy & file_mask(f)) ? 0.15 : 0.20) * pawn;
        if (protected_ || phalanx) total += 0.05 * pawn;
        if (!passed && !isolated && !doubled) {
            const int stop = c == WHITE ? s + 8 : s - 8;
            const bool support = (own & adjacent_files(f) & behind_or_level(c, s)) != 0;
            if (!support && stop >= 0 && stop < 64 && (T.pawn[c][stop] & enemy)) total -= 0.10 * pawn;   // backward
            if (!(enemy & ahead) && popcnt(enemy & ahead_adjacent) <= popcnt(own & adjacent_files(f)))
                total += 0.15 * pawn * rel / 6.0;          // a candidate passer
        }
        if (!passed || doubled) continue;
        // a passed pawn: how likely it queens, square by square on its way
        const int steps = 7 - rel - (rel == 1 ? 1 : 0);
        double chance = 1.0;
        for (U64 path = ahead; path;) {
            const int sq = c == WHITE ? lsb(path) : msb(path);
            path &= ~bit(sq);
            if (all & bit(sq)) {
                chance *= (p.occ[c] & bit(sq)) ? 0.85 : 0.35;   // an own piece steps aside; an enemy blockader
                continue;
            }
            const U64 there = (all & ~bit(s)) | bit(sq);
            if (see(p, sq, them, pawn, there, v) > 0.0) chance *= 0.45;          // lost there
            else if (p.attackers(them, sq, there) & there) chance *= 0.85;       // contested
            else chance *= 0.97;
        }
        const int queen_sq = c == WHITE ? 56 + f : f;
        if (enemy_king >= 0) {
            const int race = distance(enemy_king, queen_sq) - (first ? 0 : 1);
            if (enemy_material == 0.0 && race > steps) chance = 1.0;   // the rule of the square: unstoppable
            else if (race <= steps) chance *= 0.6;
        }
        if (own_king >= 0) {
            const int next = c == WHITE ? s + 8 : s - 8;
            if (next >= 0 && next < 64 && distance(own_king, next) <= 1) chance *= 1.1;
        }
        const U64 behind = file_mask(f) & ~ahead & ~bit(s);
        for (U64 rk = p.bb[ROOK] & p.occ[c] & behind; rk; rk &= rk - 1)
            if (!(T.between[lsb(rk)][s] & all)) { chance *= 1.1; break; }   // a rook behind it
        if (protected_) chance *= 1.15;
        for (U64 nb = own & adjacent_files(f); nb; nb &= nb - 1) {
            const int ns = lsb(nb);
            if (!(enemy & (front(c, ns) | front_and_adjacent(c, ns)))) { chance *= 1.1; break; }   // connected passers
        }
        chance = std::min(1.0, chance);
        total += queen_gain * chance * catch_scale * std::pow(0.92, std::max(0, steps));
    }
    return total;
}

inline double balance(const Pos& p, int me, const std::vector<double>& v) {
    return side_value(p, me, p.turn == me, v) - side_value(p, me ^ 1, p.turn != me, v);
}
}  // namespace pawnomics4
