// tonkai_fast - the V3 agent's hand-written chess rules, in C++.
//
// Every function here is a line-by-line port of a Python reference and must
// return EXACTLY what the reference returns (tests/test_fast_rules.py checks
// them against each other on many positions):
//
//   king_risks / king_detail  <- tonkai/king_safety.py  king_senses
//   piece_risks               <- tonkai/piece_safety.py piece_senses + position_risk
//   terminal                  <- tonkai/cognitive_agent.py terminal_value
//                                (python-chess outcome / fifty-move / threefold)
//
// Squares and bitboards follow python-chess (a1 = 0 .. h8 = 63). Legal moves
// are pseudo-legal moves that do not leave the mover's king attacked, which is
// the same set python-chess produces; move ORDER never matters to any result
// here. Floating point follows the Python expressions operation by operation.
// Python never fuses a*b+c; neither may we, or results differ in the last bit.
#ifdef _MSC_VER
#pragma fp_contract(off)
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using U64 = std::uint64_t;

namespace {

constexpr int BLACK = 0, WHITE = 1;
enum { NO_PIECE = 0, PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr U64 RANK1 = 0xFFULL, RANK8 = 0xFF00000000000000ULL;
constexpr U64 DARK = 0xAA55AA55AA55AA55ULL, LIGHT = 0x55AA55AA55AA55AAULL;
constexpr U64 ALL = ~0ULL;

inline U64 bit(int sq) { return U64(1) << sq; }
inline int lsb(U64 b) { return std::countr_zero(b); }
inline int msb(U64 b) { return 63 - std::countl_zero(b); }
inline int popcnt(U64 b) { return std::popcount(b); }
inline int file_of(int s) { return s & 7; }
inline int rank_of(int s) { return s >> 3; }

// Directions 0..3 increase the square index, 4..7 decrease it.
constexpr int DF[8] = {0, 1, 1, -1, 0, -1, -1, 1};
constexpr int DR[8] = {1, 0, 1, 1, -1, 0, -1, -1};
inline int opposite(int d) { return (d + 4) & 7; }

struct Tables {
    U64 knight[64], king[64], pawn[2][64], ray[8][64];
    U64 line[64][64], between[64][64];
    U64 file_line[64], rank_line[64], diag_line[64];
    std::vector<U64> open_rays[64];
    std::vector<std::pair<U64, U64>> shelter[2][64];

    Tables() {
        static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        auto on = [](int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; };
        for (int s = 0; s < 64; ++s) {
            int f = file_of(s), r = rank_of(s);
            knight[s] = king[s] = pawn[0][s] = pawn[1][s] = 0;
            for (auto& d : KN)
                if (on(f + d[0], r + d[1])) knight[s] |= bit((r + d[1]) * 8 + f + d[0]);
            for (int a = -1; a <= 1; ++a)
                for (int b = -1; b <= 1; ++b)
                    if ((a || b) && on(f + a, r + b)) king[s] |= bit((r + b) * 8 + f + a);
            for (int a : {-1, 1}) {
                if (on(f + a, r + 1)) pawn[WHITE][s] |= bit((r + 1) * 8 + f + a);
                if (on(f + a, r - 1)) pawn[BLACK][s] |= bit((r - 1) * 8 + f + a);
            }
            for (int d = 0; d < 8; ++d) {
                ray[d][s] = 0;
                for (int ff = f + DF[d], rr = r + DR[d]; on(ff, rr); ff += DF[d], rr += DR[d])
                    ray[d][s] |= bit(rr * 8 + ff);
            }
            file_line[s] = ray[0][s] | ray[4][s];
            rank_line[s] = ray[1][s] | ray[5][s];
            diag_line[s] = ray[2][s] | ray[3][s] | ray[6][s] | ray[7][s];
            // king_safety.OPEN_RAYS: two squares in each direction that has both.
            static const int KD[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
            for (auto& d : KD)
                if (on(f + 2 * d[0], r + 2 * d[1]))
                    open_rays[s].push_back(bit((r + d[1]) * 8 + f + d[0]) | bit((r + 2 * d[1]) * 8 + f + 2 * d[0]));
            // king_safety.SHELTER: per file around the king, rank +1/+2 (white) or -1/-2.
            for (int colour : {WHITE, BLACK})
                for (int file = std::max(0, f - 1); file < std::min(8, f + 2); ++file) {
                    U64 cells[2] = {0, 0};
                    for (int distance = 1; distance <= 2; ++distance) {
                        int rank = r + (colour == WHITE ? distance : -distance);
                        if (rank >= 0 && rank < 8) cells[distance - 1] = bit(rank * 8 + file);
                    }
                    shelter[colour][s].push_back({cells[0], cells[1]});
                }
        }
        for (int a = 0; a < 64; ++a)
            for (int b = 0; b < 64; ++b) {
                line[a][b] = between[a][b] = 0;
                for (int d = 0; d < 8; ++d)
                    if (ray[d][a] & bit(b)) {
                        line[a][b] = ray[d][a] | ray[opposite(d)][a] | bit(a);
                        between[a][b] = ray[d][a] & ray[opposite(d)][b];
                    }
            }
    }
};

const Tables T;

inline U64 ray_attacks(int d, int s, U64 occupied) {
    U64 r = T.ray[d][s], blockers = r & occupied;
    if (blockers) r ^= T.ray[d][d < 4 ? lsb(blockers) : msb(blockers)];
    return r;
}
inline U64 rook_attacks(int s, U64 occ) {
    return ray_attacks(0, s, occ) | ray_attacks(1, s, occ) | ray_attacks(4, s, occ) | ray_attacks(5, s, occ);
}
inline U64 bishop_attacks(int s, U64 occ) {
    return ray_attacks(2, s, occ) | ray_attacks(3, s, occ) | ray_attacks(6, s, occ) | ray_attacks(7, s, occ);
}

// Move: from | to << 6 | promotion piece type << 12.
inline int mv(int from, int to, int promo = 0) { return from | (to << 6) | (promo << 12); }
inline int mfrom(int m) { return m & 63; }
inline int mto(int m) { return (m >> 6) & 63; }
inline int mpromo(int m) { return (m >> 12) & 7; }

std::string square_name(int s) { return std::string{char('a' + file_of(s)), char('1' + rank_of(s))}; }
std::string uci(int m) {
    static const char P[] = " pnbrqk";
    std::string out = square_name(mfrom(m)) + square_name(mto(m));
    if (mpromo(m)) out += P[mpromo(m)];
    return out;
}

struct Pos {
    U64 bb[7] = {0, 0, 0, 0, 0, 0, 0};
    U64 occ[2] = {0, 0};
    int turn = WHITE;
    U64 castling = 0;  // python-chess castling_rights: bitmask of rook squares
    int ep = -1;
    int halfmove = 0;
    int fullmove = 1;

    U64 occupied() const { return occ[0] | occ[1]; }
    int type_at(int s) const {
        U64 m = bit(s);
        if (!(occupied() & m)) return NO_PIECE;
        for (int t = PAWN; t <= KING; ++t)
            if (bb[t] & m) return t;
        return NO_PIECE;
    }
    int color_at(int s) const { return int((occ[WHITE] >> s) & 1); }
    int king(int c) const {  // python-chess Board.king: msb of the king mask
        U64 k = bb[KING] & occ[c];
        return k ? msb(k) : -1;
    }
    void remove(int s) {
        U64 m = ~bit(s);
        for (int t = PAWN; t <= KING; ++t) bb[t] &= m;
        occ[0] &= m;
        occ[1] &= m;
    }
    void put(int type, int colour, int s) {
        remove(s);
        bb[type] |= bit(s);
        occ[colour] |= bit(s);
    }
    // python-chess attackers_mask(color, square, occupied)
    U64 attackers(int colour, int s, U64 occupied) const {
        U64 qr = bb[QUEEN] | bb[ROOK], qb = bb[QUEEN] | bb[BISHOP];
        U64 a = (T.king[s] & bb[KING]) | (T.knight[s] & bb[KNIGHT]) | (rook_attacks(s, occupied) & qr) |
                (bishop_attacks(s, occupied) & qb) | (T.pawn[colour ^ 1][s] & bb[PAWN]);
        return a & occ[colour];
    }
    // python-chess attacks_mask(square)
    U64 attacks_mask(int s) const {
        U64 m = bit(s);
        if (m & bb[PAWN]) return T.pawn[color_at(s)][s];
        if (m & bb[KNIGHT]) return T.knight[s];
        if (m & bb[KING]) return T.king[s];
        U64 a = 0;
        if (m & (bb[BISHOP] | bb[QUEEN])) a = bishop_attacks(s, occupied());
        if (m & (bb[ROOK] | bb[QUEEN])) a |= rook_attacks(s, occupied());
        return a;
    }
    // python-chess pin_mask(color, square)
    U64 pin_mask(int colour, int s) const {
        int k = king(colour);
        if (k < 0) return ALL;
        U64 sm = bit(s);
        const U64* lines[3] = {T.file_line, T.rank_line, T.diag_line};
        U64 sliders[3] = {bb[ROOK] | bb[QUEEN], bb[ROOK] | bb[QUEEN], bb[BISHOP] | bb[QUEEN]};
        for (int i = 0; i < 3; ++i) {
            U64 rays = lines[i][k];
            if (rays & sm) {
                for (U64 snipers = rays & sliders[i] & occ[colour ^ 1]; snipers; snipers &= snipers - 1) {
                    int sniper = lsb(snipers);
                    if ((T.between[sniper][k] & (occupied() | sm)) == sm) return T.line[k][sniper];
                }
                break;
            }
        }
        return ALL;
    }
    bool is_check() const {
        int k = king(turn);
        return k >= 0 && attackers(turn ^ 1, k, occupied()) != 0;
    }
    // python-chess clean_castling_rights (standard chess)
    U64 clean_castling() const {
        U64 c = castling & bb[ROOK];
        U64 w = c & RANK1 & occ[WHITE] & (bit(0) | bit(7));
        U64 b = c & RANK8 & occ[BLACK] & (bit(56) | bit(63));
        if (!(occ[WHITE] & bb[KING] & bit(4))) w = 0;
        if (!(occ[BLACK] & bb[KING] & bit(60))) b = 0;
        return w | b;
    }
    bool is_en_passant(int m) const {
        int from = mfrom(m), to = mto(m), diff = std::abs(to - from);
        return ep == to && (bb[PAWN] & bit(from)) && (diff == 7 || diff == 9) && !(occupied() & bit(to));
    }
    int captured_square(int m) const {
        return is_en_passant(m) ? mto(m) + (turn == WHITE ? -8 : 8) : mto(m);
    }
    bool is_zeroing(int m) const {
        U64 touched = bit(mfrom(m)) ^ bit(mto(m));
        return (touched & bb[PAWN]) || (touched & occ[turn ^ 1]);
    }
    bool reduces_castling(int m) const {
        U64 cr = clean_castling(), touched = bit(mfrom(m)) ^ bit(mto(m));
        return (touched & cr) || ((cr & RANK1) && (touched & bb[KING] & occ[WHITE])) ||
               ((cr & RANK8) && (touched & bb[KING] & occ[BLACK]));
    }

    // python-chess Board.push for standard moves (castling as e1g1).
    void make(int m) {
        int from = mfrom(m), to = mto(m), promo = mpromo(m), us = turn, them = us ^ 1;
        int type = type_at(from), previous_ep = ep;
        bool zeroing = is_zeroing(m);
        ++halfmove;
        if (us == BLACK) ++fullmove;
        if (zeroing) halfmove = 0;
        bool castle = type == KING && std::abs(file_of(to) - file_of(from)) == 2;
        U64 rook_from = castle ? bit(rank_of(from) * 8 + (file_of(to) == 6 ? 7 : 0)) : bit(to);
        castling &= ~rook_from & ~bit(from);
        if (type == KING) castling &= ~(us == WHITE ? RANK1 : RANK8);
        ep = -1;
        remove(from);
        int captured = type_at(to);
        if (type == PAWN) {
            int diff = to - from;
            if (diff == 16 && rank_of(from) == 1)
                ep = from + 8;
            else if (diff == -16 && rank_of(from) == 6)
                ep = from - 8;
            else if (to == previous_ep && (std::abs(diff) == 7 || std::abs(diff) == 9) && !captured)
                remove(us == WHITE ? to - 8 : to + 8);
        }
        if (promo) type = promo;
        if (castle) {
            int base = rank_of(from) * 8;
            bool kingside = file_of(to) == 6;
            remove(base + (kingside ? 7 : 0));
            put(ROOK, us, base + (kingside ? 5 : 3));
            put(KING, us, to);
        } else {
            put(type, us, to);
        }
        turn = them;
    }
    bool legal(int m) const {
        int k = king(turn);
        if (k < 0) return true;  // python-chess: no king -> every pseudo-legal move
        Pos next = *this;
        next.make(m);
        int after = next.king(turn);
        return after < 0 || !next.attackers(turn ^ 1, after, next.occupied());
    }
    bool attacked_for_king(U64 path, U64 occupied) const {
        for (; path; path &= path - 1)
            if (attackers(turn ^ 1, lsb(path), occupied)) return true;
        return false;
    }

    // Pseudo-legal moves, python-chess semantics (captures of the king included).
    void pseudo(std::vector<int>& out) const {
        int us = turn, them = us ^ 1;
        U64 own = occ[us], enemy = occ[them], all = occupied();
        for (U64 pieces = own & ~bb[PAWN]; pieces; pieces &= pieces - 1) {
            int from = lsb(pieces);
            for (U64 targets = attacks_mask(from) & ~own; targets; targets &= targets - 1)
                out.push_back(mv(from, lsb(targets)));
        }
        castles(out);
        for (U64 pawns = own & bb[PAWN]; pawns; pawns &= pawns - 1) {
            int from = lsb(pawns);
            auto add = [&](int to) {
                if (rank_of(to) == 0 || rank_of(to) == 7)
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT}) out.push_back(mv(from, to, p));
                else
                    out.push_back(mv(from, to));
            };
            for (U64 caps = T.pawn[us][from] & enemy; caps; caps &= caps - 1) add(lsb(caps));
            int step = us == WHITE ? 8 : -8, one = from + step;
            if (one >= 0 && one < 64 && !(all & bit(one))) {
                add(one);
                int start = us == WHITE ? 1 : 6, two = one + step;
                if (rank_of(from) == start && !(all & bit(two))) out.push_back(mv(from, two));
            }
        }
        ep_moves(out);
    }
    void ep_moves(std::vector<int>& out) const {
        if (ep <= 0 || (occupied() & bit(ep))) return;  // python: `not self.ep_square`
        U64 rank = turn == WHITE ? (0xFFULL << 32) : (0xFFULL << 24);
        for (U64 c = bb[PAWN] & occ[turn] & T.pawn[turn ^ 1][ep] & rank; c; c &= c - 1)
            out.push_back(mv(lsb(c), ep));
    }
    void castles(std::vector<int>& out) const {
        U64 backrank = turn == WHITE ? RANK1 : RANK8;
        U64 kings = occ[turn] & bb[KING] & backrank;
        if (!kings) return;
        int k = lsb(kings);
        int base = turn == WHITE ? 0 : 56;
        for (U64 rights = clean_castling() & backrank; rights; rights &= rights - 1) {
            int candidate = lsb(rights);
            bool a_side = candidate < k;
            int king_to = base + (a_side ? 2 : 6), rook_to = base + (a_side ? 3 : 5);
            U64 king_path = T.between[k][king_to], rook_path = T.between[candidate][rook_to];
            U64 all = occupied();
            if (!((all ^ bit(k) ^ bit(candidate)) & (king_path | rook_path | bit(king_to) | bit(rook_to))) &&
                !attacked_for_king(king_path | bit(k), all ^ bit(k)) &&
                !attacked_for_king(bit(king_to), all ^ bit(k) ^ bit(candidate) ^ bit(rook_to)))
                out.push_back(mv(k, king_to));
        }
    }
    void legal_moves(std::vector<int>& out) const {
        std::vector<int> all;
        all.reserve(64);
        pseudo(all);
        for (int m : all)
            if (legal(m)) out.push_back(m);
    }
    bool any_legal() const {
        std::vector<int> all;
        all.reserve(64);
        pseudo(all);
        for (int m : all)
            if (legal(m)) return true;
        return false;
    }
    // Legal captures ONTO square s (python: generate_legal_captures(to_mask=s)).
    void captures_to(int s, std::vector<int>& out) const {
        if (occ[turn ^ 1] & bit(s)) {
            for (U64 a = attackers(turn, s, occupied()); a; a &= a - 1) {
                int from = lsb(a);
                if ((bb[PAWN] & bit(from)) && (rank_of(s) == 0 || rank_of(s) == 7)) {
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT})
                        if (legal(mv(from, s, p))) out.push_back(mv(from, s, p));
                } else if (legal(mv(from, s))) {
                    out.push_back(mv(from, s));
                }
            }
        }
        if (s == ep) {
            std::vector<int> eps;
            ep_moves(eps);
            for (int m : eps)
                if (legal(m)) out.push_back(m);
        }
    }
    bool has_legal_ep() const {
        if (ep < 0) return false;
        std::vector<int> eps;
        ep_moves(eps);
        for (int m : eps)
            if (legal(m)) return true;
        return false;
    }
    // python-chess has_kingside/queenside_castling_rights(color)
    bool castling_side(int c, bool kingside) const {
        U64 backrank = c == WHITE ? RANK1 : RANK8;
        U64 king_mask = bb[KING] & occ[c] & backrank;
        if (!king_mask) return false;
        for (U64 rights = clean_castling() & backrank; rights; rights &= rights - 1) {
            U64 rook = rights & (~rights + 1);
            if (kingside ? rook > king_mask : rook < king_mask) return true;
        }
        return false;
    }
    bool insufficient(int c) const {
        U64 own = occ[c];
        if (own & (bb[PAWN] | bb[ROOK] | bb[QUEEN])) return false;
        if (own & bb[KNIGHT]) return popcnt(own) <= 2 && !(occ[c ^ 1] & ~bb[KING] & ~bb[QUEEN]);
        if (own & bb[BISHOP]) {
            bool same = !(bb[BISHOP] & DARK) || !(bb[BISHOP] & LIGHT);
            return same && !bb[PAWN] && !bb[KNIGHT];
        }
        return true;
    }
};

struct Key {  // python-chess _transposition_key
    U64 bb[7], white, black, castling;
    int turn, ep;
    bool operator==(const Key& o) const {
        for (int t = PAWN; t <= KING; ++t)
            if (bb[t] != o.bb[t]) return false;
        return white == o.white && black == o.black && castling == o.castling && turn == o.turn && ep == o.ep;
    }
    U64 hash() const {
        U64 h = 1469598103934665603ULL;
        auto mix = [&](U64 v) { h = (h ^ v) * 1099511628211ULL; h ^= h >> 29; };
        for (int t = PAWN; t <= KING; ++t) mix(bb[t]);
        mix(white); mix(black); mix(castling); mix(U64(turn)); mix(U64(ep + 1));
        return h;
    }
};

Key key_of(const Pos& p) {
    Key k{};
    for (int t = 0; t <= KING; ++t) k.bb[t] = p.bb[t];
    k.white = p.occ[WHITE];
    k.black = p.occ[BLACK];
    k.castling = p.clean_castling();
    k.turn = p.turn;
    k.ep = p.has_legal_ep() ? p.ep : -1;
    return k;
}

// ------------------------------------------------------------------ king

struct KingRow {
    int square;
    double risk, shelter, force, uncovered;
    int contacts, open_rays;
    bool in_check;
    U64 attackers, defenders, escapes;
};

std::array<KingRow, 2> king_senses(const Pos& p) {
    int kings[2] = {p.king(BLACK), p.king(WHITE)};
    if (kings[0] < 0 || kings[1] < 0) throw std::invalid_argument("King safety requires both kings");
    U64 zones[2] = {T.king[kings[0]] | bit(kings[0]), T.king[kings[1]] | bit(kings[1])};
    U64 relevant = zones[0] | zones[1];
    U64 eff_mask[64];
    int eff_side[64];
    U64 effective = 0;
    for (U64 pieces = p.occupied() & ~p.bb[KING]; pieces; pieces &= pieces - 1) {
        int sq = lsb(pieces);
        U64 mask = p.attacks_mask(sq);
        if (mask & relevant) {
            int colour = p.color_at(sq);
            eff_mask[sq] = mask & p.pin_mask(colour, sq);
            eff_side[sq] = colour;
            effective |= bit(sq);
        }
    }
    std::array<KingRow, 2> out{};
    for (int colour : {WHITE, BLACK}) {
        int king = kings[colour];
        U64 ring = T.king[king], zone = ring | bit(king);
        U64 attackers = 0, defenders = 0, hostile = 0, friendly = 0;
        int contacts = 0, n_attackers = 0;
        for (U64 e = effective; e; e &= e - 1) {
            int sq = lsb(e);
            if (!(eff_mask[sq] & zone)) continue;
            if (eff_side[sq] != colour) {
                attackers |= bit(sq);
                ++n_attackers;
                hostile |= eff_mask[sq] & zone;
                contacts += popcnt(eff_mask[sq] & zone);
            } else {
                defenders |= bit(sq);
                friendly |= eff_mask[sq] & zone;
            }
        }
        U64 threatened = hostile & ring;
        double uncovered = double(popcnt(threatened & ~friendly)) / double(std::max(1, popcnt(threatened)));
        U64 pawn_mask = p.bb[PAWN] & p.occ[colour];
        const auto& columns = T.shelter[colour][king];
        double shelter_sum = 0.0;
        for (auto& [first, second] : columns)
            shelter_sum += (pawn_mask & first) ? 1.0 : (pawn_mask & second) ? 0.5 : 0.0;
        double shelter = shelter_sum / double(columns.size());
        int open_rays = 0;
        for (U64 ray : T.open_rays[king]) open_rays += !(p.occupied() & ray);
        U64 occupancy = p.occupied() & ~bit(king);
        U64 escapes = 0;
        int n_escapes = 0;
        for (U64 cand = ring & ~p.occ[colour]; cand; cand &= cand - 1) {
            int sq = lsb(cand);
            if (sq != kings[colour ^ 1] && !p.attackers(colour ^ 1, sq, occupancy)) {
                escapes |= bit(sq);
                ++n_escapes;
            }
        }
        U64 enemy = p.occ[colour ^ 1];
        double force = std::min(1.0, double(4 * popcnt(p.bb[QUEEN] & enemy) + 2 * popcnt(p.bb[ROOK] & enemy) +
                                            popcnt((p.bb[BISHOP] | p.bb[KNIGHT]) & enemy)) / 8.0);
        double pressure = std::min(1.0, double(contacts) / 12.0);
        double coordination = std::min(1.0, double(std::max(0, n_attackers - 1)) / 3.0);
        double gate = std::min(1.0, double(n_attackers) / 2.0) * (0.35 + 0.65 * force);
        bool checked = p.attackers(colour ^ 1, king, p.occupied()) != 0;
        double inner = 0.25 * pressure;
        inner = inner + 0.15 * coordination;
        inner = inner + 0.18 * (1.0 - shelter) * force;
        inner = inner + 0.12 * double(open_rays) / 8.0;
        inner = inner + 0.18 * uncovered;
        inner = inner + 0.12 * (1.0 - std::min(1.0, double(n_escapes) / 3.0));
        double risk = std::min(1.0, gate * inner + (checked ? 0.15 : 0.0));
        out[colour] = KingRow{king, risk, shelter, force, uncovered, contacts, open_rays, checked,
                              attackers, defenders, escapes};
    }
    return out;
}

// ------------------------------------------------------------------ pieces

// value(Piece(type, colour), square) for a promotion, asked of Python only when
// an exchange actually promotes -- exactly when the Python reference asks.
struct Promotions {
    py::function value;
    double v[2][7][64];
    bool known[2][7][64] = {};
    explicit Promotions(py::function f) : value(std::move(f)) {}
    double get(int colour, int type, int sq) {
        if (!known[colour][type][sq]) {
            v[colour][type][sq] = value(colour, type, sq).cast<double>();
            known[colour][type][sq] = true;
        }
        return v[colour][type][sq];
    }
};

// CPython (3.12+) sum() over floats: 0 + first item, then Neumaier compensation.
double python_sum(const std::vector<double>& items) {
#if PY_VERSION_HEX < 0x030C0000
    double total = 0.0;
    for (double x : items) total += x;
    return total;
#else
    if (items.empty()) return 0.0;
    double total = 0.0 + items[0], compensation = 0.0;
    for (size_t i = 1; i < items.size(); ++i) {
        double x = items[i], t = total + x;
        if (std::fabs(total) >= std::fabs(x))
            compensation += (total - t) + x;
        else
            compensation += (x - t) + total;
        total = t;
    }
    if (compensation != 0.0 && std::isfinite(compensation)) total += compensation;
    return total;
#endif
}

// piece_safety._exchange_probe's `follow`: work and values are this probe's own copies.
double follow(Pos& work, double* values, Promotions& promos, int m, int remaining, int depth,
              int* defenders) {
    int from = mfrom(m), to = mto(m), promo = mpromo(m);
    int target = work.captured_square(m);
    double gain = values[target];
    double moving = values[from];
    double promoted = promo ? promos.get(work.turn, promo, to) : moving;
    if (promo) gain += promoted - moving;
    values[target] = 0.0;
    values[from] = 0.0;
    values[to] = promoted;
    work.make(m);
    bool root_with_defenders = remaining == depth && defenders != nullptr;
    if (remaining <= 1 && !root_with_defenders) return gain;
    std::vector<int> replies;
    work.captures_to(to, replies);
    if (root_with_defenders) {
        U64 from_squares = 0;
        for (int r : replies) from_squares |= bit(mfrom(r));
        *defenders = popcnt(from_squares);
    }
    if (remaining <= 1 || replies.empty()) return gain;
    int best = replies[0];
    for (size_t i = 1; i < replies.size(); ++i) {
        int r = replies[i];
        double a = values[mfrom(r)], b = values[mfrom(best)];
        if (a < b || (a == b && uci(r) < uci(best))) best = r;
    }
    double reply = follow(work, values, promos, best, remaining - 1, depth, nullptr);
    return gain - (reply > 0.0 ? reply : 0.0);
}

std::pair<double, double> piece_risks(const Pos& p, const std::vector<double>& root_values,
                                      Promotions& promos, double tolerance, int depth) {
    if (root_values.size() != 64) throw std::invalid_argument("values must have 64 entries");
    std::vector<int> by_square[2][64];
    Pos works[2];
    for (int colour : {WHITE, BLACK}) {
        Pos work = p;
        work.turn = colour ^ 1;
        if (work.turn != p.turn) work.ep = -1;
        std::vector<int> captures;
        for (U64 victims = work.occ[colour] & ~work.bb[KING]; victims; victims &= victims - 1)
            work.captures_to(lsb(victims), captures);
        if (work.ep >= 0) {
            std::vector<int> eps;
            work.ep_moves(eps);
            for (int m : eps)
                if (work.legal(m)) captures.push_back(m);
        }
        for (int m : captures) {
            int sq = work.captured_square(m);
            int type = work.type_at(sq);
            if (type != NO_PIECE && type != KING && work.color_at(sq) == colour &&
                std::find(by_square[colour][sq].begin(), by_square[colour][sq].end(), m) ==
                    by_square[colour][sq].end())
                by_square[colour][sq].push_back(m);
        }
        works[colour] = work;
    }
    std::vector<double> risks[2];
    for (U64 pieces = p.occupied(); pieces; pieces &= pieces - 1) {
        int sq = lsb(pieces), colour = p.color_at(sq);
        double risk = 0.0;
        const auto& attacks = by_square[colour][sq];
        if (p.type_at(sq) != KING && !attacks.empty()) {
            U64 attacker_squares = 0;
            for (int m : attacks) attacker_squares |= bit(mfrom(m));
            int attacker_count = popcnt(attacker_squares);
            double worst_balance = 0.0;
            int worst_defenders = 0;
            bool first = true;
            for (int m : attacks) {
                Pos work = works[colour];
                double values[64];
                std::copy(root_values.begin(), root_values.end(), values);
                int defenders = 0;
                double balance = -follow(work, values, promos, m, depth, depth, &defenders);
                if (first || balance < worst_balance || (balance == worst_balance && defenders < worst_defenders)) {
                    worst_balance = balance;
                    worst_defenders = defenders;
                    first = false;
                }
            }
            if (worst_balance < -tolerance) {
                risk = 0.5 + 0.5 * std::min(1.0, (-worst_balance - tolerance) / 4.0);
            } else if (std::abs(worst_balance) <= tolerance) {
                int deficit = std::max(0, attacker_count - worst_defenders);
                risk = worst_defenders == 0 ? 0.4 : 0.3 * double(deficit) / double(attacker_count);
            }
        }
        risks[colour].push_back(risk);  // rows are in square order, like sorted(piece_map)
    }
    // position_risk: min(1, max(risks, default=0) + .05 * sum(risks))
    auto position = [&](int c) {
        double top = 0.0;
        bool first = true;
        for (double r : risks[c])
            if (first || r > top) top = r, first = false;
        return std::min(1.0, top + 0.05 * python_sum(risks[c]));
    };
    return {position(WHITE), position(BLACK)};
}

// ------------------------------------------------------------------ encoding

// encoding.encode_into for one position, from `self`'s side, into a zeroed row.
void encode_row(const Pos& b, int self, float* row) {
    int plane = 0;
    for (int colour : {self, self ^ 1})
        for (int type = PAWN; type <= KING; ++type, ++plane)
            for (U64 m = b.bb[type] & b.occ[colour]; m; m &= m - 1) {
                int sq = lsb(m);
                row[plane * 64 + (self == WHITE ? sq : sq ^ 56)] = 1.0f;
            }
    row[768] = b.turn == self ? 1.0f : 0.0f;
    row[769] = b.castling_side(self, true) ? 1.0f : 0.0f;
    row[770] = b.castling_side(self, false) ? 1.0f : 0.0f;
    row[771] = b.castling_side(self ^ 1, true) ? 1.0f : 0.0f;
    row[772] = b.castling_side(self ^ 1, false) ? 1.0f : 0.0f;
    if (b.ep >= 0) row[773 + file_of(b.ep)] = 1.0f;
    row[781] = float(double(std::min(b.halfmove, 100)) / 100.0);
}

// ------------------------------------------------------------ king safety (v4)
//
// Only the rules of chess: legal moves and checkmate. Nothing here knows what
// a "safe" king looks like; the learned model has to discover that.

// Status of a position in the attack game on `defender`'s king:
// 0 play on, 1 the defender is checkmated, 2 over without that (stalemate,
// the attacker is checkmated, insufficient material, fifty-move rule).
int ks_status(const Pos& p, int defender) {
    if (!p.any_legal()) return (p.is_check() && p.turn == defender) ? 1 : 2;
    if ((p.insufficient(WHITE) && p.insufficient(BLACK)) || p.halfmove >= 100) return 2;
    return 0;
}

// Exact forced-mate search. attack(): the attacker is to move and must mate
// within n of its own moves; defend(): every defender move must still lose.
// Checking moves are tried first: that only orders the search - the answer is
// the same - since a move that does not give check cannot be checkmate.
struct MateSearch {
    long long nodes = 0, limit;
    bool aborted = false;
    explicit MateSearch(long long limit_) : limit(limit_) {}

    bool attack(const Pos& p, int n) {
        if (++nodes > limit) { aborted = true; return false; }
        std::vector<int> moves;
        p.legal_moves(moves);
        std::vector<Pos> kids;
        kids.reserve(moves.size());
        std::vector<int> checking, quiet;
        for (size_t i = 0; i < moves.size(); ++i) {
            Pos q = p;
            q.make(moves[i]);
            (q.is_check() ? checking : quiet).push_back(int(kids.size()));
            kids.push_back(q);
        }
        for (int i : checking) {
            if (!kids[i].any_legal()) return true;          // checkmate
            if (n > 1 && defend(kids[i], n - 1)) return true;
            if (aborted) return false;
        }
        if (n > 1)
            for (int i : quiet) {
                if (defend(kids[i], n - 1)) return true;
                if (aborted) return false;
            }
        return false;
    }

    bool defend(const Pos& q, int n) {
        if (++nodes > limit) { aborted = true; return false; }
        std::vector<int> moves;
        q.legal_moves(moves);
        if (moves.empty()) return false;                   // stalemate: the king survives
        for (int d : moves) {
            Pos r = q;
            r.make(d);
            if (!attack(r, n) || aborted) return false;
        }
        return true;
    }
};

// ------------------------------------------------------------------ board

class Board {
public:
    Board(U64 pawns, U64 knights, U64 bishops, U64 rooks, U64 queens, U64 kings, U64 white, U64 black, bool turn,
          U64 castling, int ep, int halfmove, int fullmove) {
        pos.bb[PAWN] = pawns;
        pos.bb[KNIGHT] = knights;
        pos.bb[BISHOP] = bishops;
        pos.bb[ROOK] = rooks;
        pos.bb[QUEEN] = queens;
        pos.bb[KING] = kings;
        pos.occ[WHITE] = white;
        pos.occ[BLACK] = black;
        pos.turn = turn ? WHITE : BLACK;
        pos.castling = castling;
        pos.ep = ep;
        pos.halfmove = halfmove;
        pos.fullmove = fullmove;
    }

    void push(int m) {
        states.push_back(pos);
        moves.push_back(m);
        keys.emplace_back();
        key_ready.push_back(false);
        pos.make(m);
    }
    void push_many(const std::vector<int>& ms) {
        for (int m : ms) push(m);
    }
    void push_many_checked(const std::vector<int>& ms) {
        for (int m : ms) {
            std::vector<int> legal;
            pos.legal_moves(legal);
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in replay history");
            push(m);
        }
    }
    // Complete standard FEN with the raw EP square (python-chess fen's
    // en_passant="fen" option). Used for move enumeration, never for history.
    std::string fen() const {
        static const char symbols[2][7] = {{' ', 'p', 'n', 'b', 'r', 'q', 'k'},
                                           {' ', 'P', 'N', 'B', 'R', 'Q', 'K'}};
        std::string out;
        for (int rank = 7; rank >= 0; --rank) {
            int empty = 0;
            for (int file = 0; file < 8; ++file) {
                int sq = rank * 8 + file, type = pos.type_at(sq);
                if (type == NO_PIECE) { ++empty; continue; }
                if (empty) { out += char('0' + empty); empty = 0; }
                out += symbols[pos.color_at(sq)][type];
            }
            if (empty) out += char('0' + empty);
            if (rank) out += '/';
        }
        out += pos.turn == WHITE ? " w " : " b ";
        U64 rights = pos.clean_castling();
        if (!rights) out += '-';
        if (rights & bit(7)) out += 'K';
        if (rights & bit(0)) out += 'Q';
        if (rights & bit(63)) out += 'k';
        if (rights & bit(56)) out += 'q';
        out += " " + (pos.ep < 0 ? std::string("-") : square_name(pos.ep));
        out += " " + std::to_string(pos.halfmove) + " " + std::to_string(pos.fullmove);
        return out;
    }
    std::vector<bool> checks_after(const std::vector<int>& ms) const {
        std::vector<bool> result;
        result.reserve(ms.size());
        for (int m : ms) {
            Pos next = pos;
            next.make(m);
            result.push_back(next.is_check());
        }
        return result;
    }
    void pop() {
        if (states.empty()) throw std::runtime_error("pop from an empty board");
        pos = states.back();
        states.pop_back();
        moves.pop_back();
        keys.pop_back();
        key_ready.pop_back();
    }
    bool turn() const { return pos.turn == WHITE; }
    int ply() const { return int(moves.size()); }

    // -1 none, 0 draw, 1 white wins, 2 black wins  (cognitive_agent.terminal_value)
    int terminal() {
        bool legal = pos.any_legal();
        if (!legal && pos.is_check()) return pos.turn == WHITE ? 2 : 1;
        if (pos.insufficient(WHITE) && pos.insufficient(BLACK)) return 0;
        if (!legal) return 0;
        if (pos.halfmove >= 150) return 0;
        if (repetition(5)) return 0;
        if (pos.halfmove >= 100) return 0;
        if (pos.halfmove >= 99) {
            std::vector<int> all;
            pos.legal_moves(all);
            for (int m : all)
                if (!pos.is_zeroing(m)) {
                    Pos next = pos;
                    next.make(m);
                    if (next.halfmove >= 100 && next.any_legal()) return 0;
                }
        }
        if (std::min(pos.halfmove, int(moves.size())) >= 7 && duplicate() && claim_threefold()) return 0;
        return -1;
    }

    std::pair<double, double> king_risks() const {
        auto rows = king_senses(pos);
        return {rows[WHITE].risk, rows[BLACK].risk};
    }
    py::dict king_detail() const {
        auto rows = king_senses(pos);
        py::dict out;
        for (int colour : {WHITE, BLACK}) {
            const KingRow& r = rows[colour];
            py::dict d;
            d["square"] = r.square;
            d["risk"] = r.risk;
            d["shelter"] = r.shelter;
            d["force"] = r.force;
            d["contacts"] = r.contacts;
            d["open_rays"] = r.open_rays;
            d["uncovered"] = r.uncovered;
            d["in_check"] = r.in_check;
            d["attackers"] = r.attackers;
            d["defenders"] = r.defenders;
            d["escapes"] = r.escapes;
            out[colour == WHITE ? "w" : "b"] = d;
        }
        return out;
    }
    std::pair<double, double> piece_risks_py(const std::vector<double>& values, py::function promotion,
                                             double tolerance, int depth) const {
        Promotions table(std::move(promotion));
        return piece_risks(pos, values, table, tolerance, depth);
    }
    // captures.track + capture_targets, retaining identity through EP, castling
    // and promotion. Integer targets are exactly the Python uint8 result.
    py::array_t<std::uint8_t> capture_targets(const std::vector<int>& ms, bool self_colour) const {
        Pos work = pos;
        std::array<int, 64> ids;
        ids.fill(-1);
        int next_id = 0;
        for (int s = 0; s < 64; ++s) if (work.occupied() & bit(s)) ids[s] = next_id++;
        std::vector<std::array<int, 64>> snapshots{ids};
        std::vector<int> capturers;
        snapshots.reserve(ms.size() + 1);
        capturers.reserve(ms.size());
        for (int m : ms) {
            std::vector<int> legal;
            work.legal_moves(legal);
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in capture history");
            int from = mfrom(m), to = mto(m), mover = ids[from];
            int target = work.captured_square(m);
            bool capture = (work.occ[work.turn ^ 1] & bit(target)) != 0;
            capturers.push_back(capture ? mover : -1);
            bool castle = work.type_at(from) == KING && std::abs(file_of(to) - file_of(from)) == 2;
            ids[from] = -1;
            if (capture) ids[target] = -1;
            if (castle) {
                int base = rank_of(from) * 8;
                int rook_from = base + (file_of(to) == 6 ? 7 : 0);
                int rook_to = base + (file_of(to) == 6 ? 5 : 3);
                ids[rook_to] = ids[rook_from]; ids[rook_from] = -1;
            }
            ids[to] = mover;
            work.make(m);
            snapshots.push_back(ids);
        }
        int n = int(snapshots.size());
        py::array_t<std::uint8_t> out({n, 64});
        auto r = out.mutable_unchecked<2>();
        std::array<int, 64> future{};
        for (int k = n - 1; k >= 0; --k) {
            for (int sq = 0; sq < 64; ++sq) {
                int id = snapshots[k][sq];
                r(k, self_colour ? sq : sq ^ 56) = id < 0 ? 0 : std::min(255, future[id]);
            }
            if (k > 0 && capturers[k - 1] >= 0) ++future[capturers[k - 1]];
        }
        return out;
    }

    // captures.activity: integer measurements, including the original rule
    // that castling alone does not count as the rook's first individual move.
    py::dict activity(const std::vector<int>& ms) const {
        Pos work = pos;
        U64 moved = 0;
        std::array<std::array<std::array<int, 7>, 6>, 2> totals{};
        for (int m : ms) {
            std::vector<int> legal;
            work.legal_moves(legal);
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in activity history");
            std::array<int, 64> destinations{};
            for (int action : legal) ++destinations[mfrom(action)];
            int turn = work.turn;
            U64 enemy = work.occ[turn ^ 1] & ~work.bb[KING];
            for (U64 own = work.occ[turn] & ~work.bb[KING]; own; own &= own - 1) {
                int sq = lsb(own), kind = work.type_at(sq);
                bool active = (moved & bit(sq)) != 0;
                if (active) {
                    totals[turn][0][kind] += destinations[sq];
                    ++totals[turn][1][kind];
                }
                int threats = 0;
                for (U64 targets = work.attacks_mask(sq) & enemy; targets; targets &= targets - 1)
                    if (work.attackers(turn ^ 1, lsb(targets), work.occupied())) ++threats;
                totals[turn][2][kind] += threats;
                ++totals[turn][3][kind];
                if (active) {
                    totals[turn][4][kind] += threats;
                    ++totals[turn][5][kind];
                }
            }
            int from = mfrom(m), to = mto(m), captured = work.captured_square(m);
            if (work.type_at(from) == KING && std::abs(file_of(to) - file_of(from)) == 2) {
                int base = rank_of(from) * 8;
                int rook_from = base + (file_of(to) == 6 ? 7 : 0);
                int rook_to = base + (file_of(to) == 6 ? 5 : 3);
                bool rook_moved = (moved & bit(rook_from)) != 0;
                moved &= ~(bit(rook_from) | bit(rook_to));
                if (rook_moved) moved |= bit(rook_to);
            }
            moved &= ~(bit(from) | bit(captured));
            moved |= bit(to);
            work.make(m);
        }
        static const char* names[] = {"mobility", "mobility_n", "threats", "threats_n", "threats_active",
                                      "threats_active_n"};
        py::dict out;
        for (int colour : {WHITE, BLACK}) {
            py::dict row;
            for (int metric = 0; metric < 6; ++metric) {
                py::dict values;
                int samples = metric < 2 ? 1 : metric < 4 ? 3 : 5;
                for (int kind = PAWN; kind <= QUEEN; ++kind)
                    if (totals[colour][samples][kind])
                        values[py::int_(kind)] = totals[colour][metric][kind];
                row[names[metric]] = values;
            }
            out[py::bool_(colour == WHITE)] = row;
        }
        return out;
    }

    // King safety (v4): (found, proven, nodes). found = n if the attacker forces
    // mate within n of its moves (smallest n), else 0. proven = the largest n
    // searched completely without finding mate (an exact upper bound on
    // vulnerability). A node limit stops the search early (then proven < max).
    std::tuple<int, int, long long> forced_mate(bool attacker_white, int max_moves, long long node_limit) const {
        int attacker = attacker_white ? WHITE : BLACK;
        long long used = 0;
        int found = 0, proven = 0;
        for (int n = 1; n <= max_moves; ++n) {
            MateSearch search(node_limit - used);
            bool mate = pos.turn == attacker ? search.attack(pos, n) : search.defend(pos, n);
            used += search.nodes;
            if (search.aborted) break;
            if (mate) {
                found = n;
                break;
            }
            proven = n;
        }
        return {found, proven, used};
    }
    int ks_status_py(bool defender_white) const { return ks_status(pos, defender_white ? WHITE : BLACK); }

    // King safety (v4) exploration: for every legal move of the side to move
    // (the attacker), in legal_moves() order: (move, threat, check). threat = n
    // when, were the defender allowed to pass, the attacker would force mate
    // within n moves (0 = no such threat within max_moves); check = the move
    // attacks the king now (a pass is then illegal, so no threat is computed).
    // Only the rules and checkmate - no notion of where pieces "should" go.
    std::tuple<std::vector<int>, std::vector<int>, std::vector<bool>> threats(int max_moves, long long node_limit) const {
        std::vector<int> moves, threat;
        std::vector<bool> check;
        pos.legal_moves(moves);
        for (int m : moves) {
            Pos q = pos;
            q.make(m);
            bool checking = q.is_check();
            check.push_back(checking);
            int found = 0;
            if (!checking) {
                Pos passed = q;                    // the defender passes: the attacker moves again
                passed.turn ^= 1;
                passed.ep = -1;
                for (int n = 1; n <= max_moves && !found; ++n) {
                    MateSearch search(node_limit);
                    if (search.attack(passed, n)) found = n;
                    if (search.aborted) break;
                }
            }
            threat.push_back(found);
        }
        return {moves, threat, check};
    }

    // King safety (v4): every legal continuation two plies deep, for `defender`.
    // Returns (first_status[i], leaf_parent[j], leaf_status[j], leaf_row[j],
    // rows): statuses as in ks_status; leaf_row is the row of `rows` (the leaf
    // encoded from the defender's side) or -1 for a finished leaf.
    py::tuple expand2(bool defender_white) const {
        int defender = defender_white ? WHITE : BLACK;
        std::vector<int> first;
        first.reserve(64);
        pos.legal_moves(first);
        std::vector<int8_t> first_status;
        std::vector<int32_t> leaf_parent, leaf_row;
        std::vector<int8_t> leaf_status;
        std::vector<Pos> open;
        for (size_t i = 0; i < first.size(); ++i) {
            Pos q = pos;
            q.make(first[i]);
            int status = ks_status(q, defender);
            first_status.push_back(int8_t(status));
            if (status) continue;
            std::vector<int> replies;
            q.legal_moves(replies);
            for (int r : replies) {
                Pos leaf = q;
                leaf.make(r);
                int ls = ks_status(leaf, defender);
                leaf_parent.push_back(int32_t(i));
                leaf_status.push_back(int8_t(ls));
                leaf_row.push_back(ls ? -1 : int32_t(open.size()));
                if (!ls) open.push_back(leaf);
            }
        }
        py::array_t<float> rows({std::max<py::ssize_t>(1, py::ssize_t(open.size())), py::ssize_t(782)});
        auto r = rows.mutable_unchecked<2>();
        std::fill(r.mutable_data(0, 0), r.mutable_data(0, 0) + rows.size(), 0.0f);
        for (size_t k = 0; k < open.size(); ++k) encode_row(open[k], defender, r.mutable_data(py::ssize_t(k), 0));
        auto as_array = [](auto& v) { return py::array(py::cast(v)); };
        return py::make_tuple(as_array(first_status), as_array(leaf_parent), as_array(leaf_status),
                              as_array(leaf_row), rows, int(open.size()));
    }
    // The current position encoded from one side (encoding.encode).
    py::array_t<float> encode_here(bool self_colour) const {
        py::array_t<float> out({py::ssize_t(782)});
        auto r = out.mutable_unchecked<1>();
        std::fill(r.mutable_data(0), r.mutable_data(0) + 782, 0.0f);
        encode_row(pos, self_colour ? WHITE : BLACK, r.mutable_data(0));
        return out;
    }

    // encoding.encode_after_moves: the position after each move, from self_color's side.
    py::array_t<float> encode_after(const std::vector<int>& ms, bool self_colour) const {
        int self = self_colour ? WHITE : BLACK, n = int(ms.size());
        py::array_t<float> out({std::max(n, 1), 782});
        auto r = out.mutable_unchecked<2>();
        for (int i = 0; i < std::max(n, 1); ++i)
            for (int j = 0; j < 782; ++j) r(i, j) = 0.0f;
        for (int i = 0; i < n; ++i) {
            Pos b = pos;
            b.make(ms[i]);
            int plane = 0;
            for (int colour : {self, self ^ 1})
                for (int type = PAWN; type <= KING; ++type, ++plane)
                    for (U64 m = b.bb[type] & b.occ[colour]; m; m &= m - 1) {
                        int sq = lsb(m);
                        r(i, plane * 64 + (self == WHITE ? sq : sq ^ 56)) = 1.0f;
                    }
            r(i, 768) = b.turn == self ? 1.0f : 0.0f;
            r(i, 769) = b.castling_side(self, true) ? 1.0f : 0.0f;
            r(i, 770) = b.castling_side(self, false) ? 1.0f : 0.0f;
            r(i, 771) = b.castling_side(self ^ 1, true) ? 1.0f : 0.0f;
            r(i, 772) = b.castling_side(self ^ 1, false) ? 1.0f : 0.0f;
            if (b.ep >= 0) r(i, 773 + file_of(b.ep)) = 1.0f;
            r(i, 781) = float(double(std::min(b.halfmove, 100)) / 100.0);
        }
        if (n == 0) return out;
        return out;
    }
    // (square, piece type, colour) in square order, like sorted(board.piece_map().items()).
    std::vector<std::tuple<int, int, bool>> pieces() const {
        std::vector<std::tuple<int, int, bool>> out;
        for (U64 b = pos.occupied(); b; b &= b - 1) {
            int sq = lsb(b);
            out.emplace_back(sq, pos.type_at(sq), pos.color_at(sq) == WHITE);
        }
        return out;
    }
    // Every promotion value a piece_risks call on this position could need.
    std::vector<std::tuple<int, int, int>> promotion_keys() const {
        std::vector<std::tuple<int, int, int>> out;
        for (int colour : {WHITE, BLACK}) {
            U64 seventh = colour == WHITE ? (0xFFULL << 48) : (0xFFULL << 8);
            U64 targets = 0;
            for (U64 pawns = pos.bb[PAWN] & pos.occ[colour] & seventh; pawns; pawns &= pawns - 1)
                targets |= T.pawn[colour][lsb(pawns)];
            for (; targets; targets &= targets - 1)
                for (int type : {KNIGHT, BISHOP, ROOK, QUEEN}) out.emplace_back(colour, type, lsb(targets));
        }
        return out;
    }
    std::vector<int> legal_moves() const {
        std::vector<int> out;
        pos.legal_moves(out);
        return out;
    }

private:
    Pos pos;
    std::vector<Pos> states;  // the position BEFORE each move
    std::vector<int> moves;
    std::vector<Key> keys;    // cached transposition key of states[i]
    std::vector<bool> key_ready;

    const Key& state_key(size_t i) {
        if (!key_ready[i]) {
            keys[i] = key_of(states[i]);
            key_ready[i] = true;
        }
        return keys[i];
    }
    bool irreversible(size_t i) const {
        const Pos& before = states[i];
        int m = moves[i];
        return before.is_zeroing(m) || before.reduces_castling(m) || before.has_legal_ep();
    }
    // python-chess is_repetition(count)
    bool repetition(int count) {
        int maybe = 1;
        for (size_t i = states.size(); i-- > 0;)
            if (states[i].occupied() == pos.occupied() && ++maybe >= count) break;
        if (maybe < count) return false;
        Key current = key_of(pos);
        for (long long i = (long long)moves.size() - 1;; --i) {
            if (count <= 1) return true;
            if (i + 1 < count - 1) break;
            if (irreversible(size_t(i))) break;
            if (state_key(size_t(i)) == current) --count;
        }
        return false;
    }
    // Does any position in the whole history occur twice? (terminal_value's gate)
    bool duplicate() {
        std::vector<std::pair<U64, int>> hashes;
        hashes.reserve(states.size() + 1);
        Key current = key_of(pos);
        hashes.emplace_back(current.hash(), -1);
        for (size_t i = 0; i < states.size(); ++i) hashes.emplace_back(state_key(i).hash(), int(i));
        std::sort(hashes.begin(), hashes.end());
        for (size_t a = 0; a + 1 < hashes.size(); ++a)
            for (size_t b = a + 1; b < hashes.size() && hashes[b].first == hashes[a].first; ++b) {
                const Key& x = hashes[a].second < 0 ? current : keys[hashes[a].second];
                const Key& y = hashes[b].second < 0 ? current : keys[hashes[b].second];
                if (x == y) return true;
            }
        return false;
    }
    // python-chess can_claim_threefold_repetition
    bool claim_threefold() {
        Key current = key_of(pos);
        std::vector<Key> seen{current};
        for (size_t i = moves.size(); i-- > 0;) {
            if (irreversible(i)) break;
            seen.push_back(state_key(i));
        }
        if (std::count(seen.begin(), seen.end(), current) >= 3) return true;
        std::vector<int> all;
        pos.legal_moves(all);
        for (int m : all) {
            Pos next = pos;
            next.make(m);
            Key k = key_of(next);
            if (std::count(seen.begin(), seen.end(), k) >= 2) return true;
        }
        return false;
    }
};

}  // namespace

PYBIND11_MODULE(tonkai_fast, m) {
    m.doc() = "C++ ports of the V3 agent's chess rules; results identical to the Python references.";
    py::class_<Board>(m, "Board")
        .def(py::init<U64, U64, U64, U64, U64, U64, U64, U64, bool, U64, int, int, int>())
        .def("push", &Board::push)
        .def("push_many", &Board::push_many)
        .def("push_many_checked", &Board::push_many_checked)
        .def("fen", &Board::fen)
        .def("checks_after", &Board::checks_after)
        .def("pop", &Board::pop)
        .def("copy", [](const Board& board) { return board; })
        .def_property_readonly("turn", &Board::turn)
        .def_property_readonly("ply", &Board::ply)
        .def("terminal", &Board::terminal)
        .def("king_risks", &Board::king_risks)
        .def("king_detail", &Board::king_detail)
        .def("piece_risks", &Board::piece_risks_py, py::arg("values"), py::arg("promotion"), py::arg("tolerance"),
             py::arg("depth"))
        .def("encode_after", &Board::encode_after, py::arg("moves"), py::arg("self_color"))
        .def("encode_here", &Board::encode_here, py::arg("self_color"))
        .def("forced_mate", &Board::forced_mate, py::arg("attacker_white"), py::arg("max_moves"),
             py::arg("node_limit"))
        .def("ks_status", &Board::ks_status_py, py::arg("defender_white"))
        .def("threats", &Board::threats, py::arg("max_moves"), py::arg("node_limit"))
        .def("expand2", &Board::expand2, py::arg("defender_white"))
        .def("capture_targets", &Board::capture_targets, py::arg("moves"), py::arg("self_color"))
        .def("activity", &Board::activity, py::arg("moves"))
        .def("pieces", &Board::pieces)
        .def("promotion_keys", &Board::promotion_keys)
        .def("legal_moves", &Board::legal_moves);
}
