#include <emscripten/bind.h>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <memory>
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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <deque>
#include <vector>


using U64 = std::uint64_t;

// The small functions every search node calls, always inlined (the compiler's choice changes as
// the file grows; their results do not depend on it).
#if defined(_MSC_VER)
#define HOT __forceinline
#else
#define HOT inline __attribute__((always_inline))
#endif

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

HOT U64 ray_attacks(int d, int s, U64 occupied) {
    U64 r = T.ray[d][s], blockers = r & occupied;
    if (blockers) r ^= T.ray[d][d < 4 ? lsb(blockers) : msb(blockers)];
    return r;
}
HOT U64 rook_attacks(int s, U64 occ) {
    return ray_attacks(0, s, occ) | ray_attacks(1, s, occ) | ray_attacks(4, s, occ) | ray_attacks(5, s, occ);
}
HOT U64 bishop_attacks(int s, U64 occ) {
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

static bool web_context_on = false;
static std::array<double,128> web_context_values{};
struct Pos {
    std::array<std::uint8_t,64> web_ids{};
    U64 bb[7] = {0, 0, 0, 0, 0, 0, 0};
    U64 occ[2] = {0, 0};
    int turn = WHITE;
    U64 castling = 0;  // python-chess castling_rights: bitmask of rook squares
    int ep = -1;
    int halfmove = 0;
    int fullmove = 1;

    U64 occupied() const { return occ[0] | occ[1]; }
    // The piece boards are disjoint (a square holds one piece), so the type (P=1 N=2 B=3 R=4 Q=5
    // K=6) is read bit by bit, without testing each board in turn: the same answer, no branches.
    HOT int type_at(int s) const {
        U64 low = bb[PAWN] | bb[BISHOP] | bb[QUEEN], mid = bb[KNIGHT] | bb[BISHOP] | bb[KING],
            high = bb[ROOK] | bb[QUEEN] | bb[KING];
        return int((low >> s) & 1) | (int((mid >> s) & 1) << 1) | (int((high >> s) & 1) << 2);
    }
    HOT bool single_king(int c) const {     // no king or one: what the shortcuts below assume
        U64 k = bb[KING] & occ[c];
        return (k & (k - 1)) == 0;
    }
    int color_at(int s) const { return int((occ[WHITE] >> s) & 1); }
    HOT int king(int c) const {  // python-chess Board.king: msb of the king mask
        U64 k = bb[KING] & occ[c];
        return k ? msb(k) : -1;
    }
    void remove(int s) {             // (make() clears its own two squares at once; this is for the rest)
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
    HOT U64 attackers(int colour, int s, U64 occupied) const {
        U64 qr = bb[QUEEN] | bb[ROOK], qb = bb[QUEEN] | bb[BISHOP];
        U64 a = (T.king[s] & bb[KING]) | (T.knight[s] & bb[KNIGHT]) | (rook_attacks(s, occupied) & qr) |
                (bishop_attacks(s, occupied) & qb) | (T.pawn[colour ^ 1][s] & bb[PAWN]);
        return a & occ[colour];
    }
    // attackers(colour, s, occupied) != 0: the short-range pieces first, a slider's
    // rays only when one of its pieces stands on the square's lines at all.
    HOT bool attacked(int colour, int s, U64 occupied) const {
        U64 them = occ[colour];
        if ((T.knight[s] & bb[KNIGHT] & them) || (T.pawn[colour ^ 1][s] & bb[PAWN] & them) ||
            (T.king[s] & bb[KING] & them))
            return true;
        U64 qr = (bb[QUEEN] | bb[ROOK]) & them, qb = (bb[QUEEN] | bb[BISHOP]) & them;
        if ((qr & (T.file_line[s] | T.rank_line[s])) && (rook_attacks(s, occupied) & qr)) return true;
        return (qb & T.diag_line[s]) && (bishop_attacks(s, occupied) & qb);
    }
    // python-chess attacks_mask(square)
    HOT U64 attacks_mask(int s) const {
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
    HOT bool is_check() const {
        int k = king(turn);
        return k >= 0 && attacked(turn ^ 1, k, occupied());
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
    HOT bool is_en_passant(int m) const {
        int from = mfrom(m), to = mto(m), diff = std::abs(to - from);
        return ep == to && (bb[PAWN] & bit(from)) && (diff == 7 || diff == 9) && !(occupied() & bit(to));
    }
    HOT int captured_square(int m) const {
        return is_en_passant(m) ? mto(m) + (turn == WHITE ? -8 : 8) : mto(m);
    }
    HOT bool is_zeroing(int m) const {
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
        if (web_context_on) {
            web_ids[to] = promo ? std::uint8_t(64 + us*32 + (promo-2)*8 + file_of(to)) : web_ids[from];
            if (type == KING && std::abs(file_of(to)-file_of(from)) == 2) {
                int base = rank_of(from)*8;
                web_ids[base + (file_of(to)==6 ? 5 : 3)] = web_ids[base + (file_of(to)==6 ? 7 : 0)];
            }
        }
        int captured = type_at(to);         // (lifting the piece first changes nothing: from != to)
        bool zeroing = is_zeroing(m);
        ++halfmove;
        if (us == BLACK) ++fullmove;
        if (zeroing) halfmove = 0;
        bool castle = type == KING && std::abs(file_of(to) - file_of(from)) == 2;
        U64 rook_from = castle ? bit(rank_of(from) * 8 + (file_of(to) == 6 ? 7 : 0)) : bit(to);
        castling &= ~rook_from & ~bit(from);
        if (type == KING) castling &= ~(us == WHITE ? RANK1 : RANK8);
        ep = -1;
        // remove(from) and the remove(to) inside put(), knowing what stands there.
        U64 fb = bit(from), tb = bit(to);
        if (type) {
            bb[type] &= ~fb;
            occ[0] &= ~fb;
            occ[1] &= ~fb;
        }
        if (captured) {
            bb[captured] &= ~tb;
            occ[0] &= ~tb;
            occ[1] &= ~tb;
        }
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
            bb[type] |= tb;                 // put(type, us, to): the square is already empty
            occ[us] |= tb;
        }
        turn = them;
    }
    bool legal(int m) const {
        int k = king(turn);
        if (k < 0) return true;  // python-chess: no king -> every pseudo-legal move
        Pos next = *this;
        next.make(m);
        int after = next.king(turn);
        return after < 0 || !next.attacked(turn ^ 1, after, next.occupied());
    }
    // What decides legal(m) for every move of this position, found once: the king,
    // the pieces pinned to it and the pieces giving check.
    struct Guard {
        int king;           // -1: no king, every pseudo-legal move is legal
        U64 pinned, checkers;
        bool plain;         // false: two kings or more - every move is made (legal(m))
    };
    Guard guard() const {
        Guard g{king(turn), 0, 0, single_king(turn)};
        if (g.king < 0) return g;
        int k = g.king, us = turn, them = us ^ 1;
        U64 all = occupied();
        g.checkers = attackers(them, k, all);
        U64 snipers = (((T.file_line[k] | T.rank_line[k]) & (bb[ROOK] | bb[QUEEN])) |
                       (T.diag_line[k] & (bb[BISHOP] | bb[QUEEN]))) & occ[them];
        for (; snipers; snipers &= snipers - 1) {
            U64 between = T.between[lsb(snipers)][k] & all;
            if (between && !(between & (between - 1)) && (between & occ[us])) g.pinned |= between;
        }
        return g;
    }
    // The side to move's pieces that alone stand between one of its sliders and the square target
    // (the enemy king): only moving one of these can uncover a check from that slider.
    U64 discoverers(int target) const {
        U64 out = 0, all = occupied(), own = occ[turn];
        U64 sliders = ((((T.file_line[target] | T.rank_line[target]) & (bb[ROOK] | bb[QUEEN])) |
                        (T.diag_line[target] & (bb[BISHOP] | bb[QUEEN]))) & own);
        for (; sliders; sliders &= sliders - 1) {
            U64 between = T.between[lsb(sliders)][target] & all;
            if (between && !(between & (between - 1)) && (between & own)) out |= between;
        }
        return out;
    }
    // legal(m) without making the move (castling and en passant still make it). A king
    // step is legal when its square is not attacked with the king lifted off the board;
    // any other move when it keeps a pin on its line and, in check, captures or blocks
    // the only checker - the moves that leave the king attacked, and only those.
    HOT bool legal_by(const Guard& g, int m) const {
        if (g.king < 0) return true;
        if (!g.plain) return legal(m);
        int from = mfrom(m), to = mto(m);
        if (from == g.king) {
            if (std::abs(file_of(to) - file_of(from)) == 2) return legal(m);
            return !attacked(turn ^ 1, to, occupied() ^ bit(from));
        }
        if (is_en_passant(m)) return legal(m);
        if ((g.pinned & bit(from)) && !(T.line[g.king][from] & bit(to))) return false;
        if (!g.checkers) return true;
        if (g.checkers & (g.checkers - 1)) return false;
        return ((g.checkers | T.between[lsb(g.checkers)][g.king]) & bit(to)) != 0;
    }
    bool attacked_for_king(U64 path, U64 occupied) const {
        for (; path; path &= path - 1)
            if (attacked(turn ^ 1, lsb(path), occupied)) return true;
        return false;
    }

    // Every pseudo-legal move (python-chess semantics, captures of the king included) to f, in one
    // fixed order: pieces by square, each one's targets by square, castles, pawns, en passant. f returns
    // false to stop there; each_pseudo then returns false. Nothing is generated past that point.
    template <class F>
    bool each_pseudo(F&& f) const {
        int us = turn, them = us ^ 1;
        U64 own = occ[us], enemy = occ[them], all = occupied();
        for (U64 pieces = own & ~bb[PAWN]; pieces; pieces &= pieces - 1) {
            int from = lsb(pieces);
            for (U64 targets = attacks_mask(from) & ~own; targets; targets &= targets - 1)
                if (!f(mv(from, lsb(targets)))) return false;
        }
        int special[16];
        for (int *m = special, *end = castles_into(special); m < end; ++m)
            if (!f(*m)) return false;
        for (U64 pawns = own & bb[PAWN]; pawns; pawns &= pawns - 1) {
            int from = lsb(pawns);
            auto add = [&](int to) {
                if (rank_of(to) == 0 || rank_of(to) == 7) {
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT})
                        if (!f(mv(from, to, p))) return false;
                    return true;
                }
                return bool(f(mv(from, to)));
            };
            for (U64 caps = T.pawn[us][from] & enemy; caps; caps &= caps - 1)
                if (!add(lsb(caps))) return false;
            int step = us == WHITE ? 8 : -8, one = from + step;
            if (one >= 0 && one < 64 && !(all & bit(one))) {
                if (!add(one)) return false;
                int start = us == WHITE ? 1 : 6, two = one + step;
                if (rank_of(from) == start && !(all & bit(two)) && !f(mv(from, two))) return false;
            }
        }
        for (int *m = special, *end = ep_into(special); m < end; ++m)
            if (!f(*m)) return false;
        return true;
    }
    // In check, with one king (g.plain): pseudo()'s moves in pseudo()'s order without those legal_by
    // rejects for not answering the check - a move other than the king's or en passant that neither
    // takes the only checker nor blocks it (in double check: every such move). No castles in check.
    template <class F>
    bool each_evasion(const Guard& g, F&& f) const {
        int us = turn, them = us ^ 1, k = g.king;
        U64 own = occ[us], enemy = occ[them], all = occupied();
        U64 block = (g.checkers & (g.checkers - 1)) ? 0 : (g.checkers | T.between[lsb(g.checkers)][k]);
        for (U64 pieces = own & ~bb[PAWN]; pieces; pieces &= pieces - 1) {
            int from = lsb(pieces);
            for (U64 targets = attacks_mask(from) & ~own & (from == k ? ALL : block); targets; targets &= targets - 1)
                if (!f(mv(from, lsb(targets)))) return false;
        }
        for (U64 pawns = own & bb[PAWN]; pawns; pawns &= pawns - 1) {
            int from = lsb(pawns);
            auto add = [&](int to) {
                if (rank_of(to) == 0 || rank_of(to) == 7) {
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT})
                        if (!f(mv(from, to, p))) return false;
                    return true;
                }
                return bool(f(mv(from, to)));
            };
            for (U64 caps = T.pawn[us][from] & enemy & block; caps; caps &= caps - 1)
                if (!add(lsb(caps))) return false;
            int step = us == WHITE ? 8 : -8, one = from + step;
            if (one >= 0 && one < 64 && !(all & bit(one))) {
                if ((block & bit(one)) && !add(one)) return false;
                int start = us == WHITE ? 1 : 6, two = one + step;
                if (rank_of(from) == start && !(all & bit(two)) && (block & bit(two)) && !f(mv(from, two)))
                    return false;
            }
        }
        int special[16];
        for (int *m = special, *end = ep_into(special); m < end; ++m)
            if (!f(*m)) return false;
        return true;
    }
    // The moves to try: in check (one king), only those that can answer it; otherwise all of them.
    template <class F>
    bool each_candidate(const Guard& g, F&& f) const {
        return (g.king >= 0 && g.plain && g.checkers) ? each_evasion(g, f) : each_pseudo(f);
    }
    // The same, deciding check first (cheaper than the guard when there is none).
    template <class F>
    bool each_answer(F&& f) const {
        int k = king(turn);
        if (k >= 0 && single_king(turn) && attacked(turn ^ 1, k, occupied())) return each_evasion(guard(), f);
        return each_pseudo(f);
    }
    // A side with at most 16 pieces has at most 16 x 27 + 2 pseudo-legal moves: they fit this buffer
    // and are appended at once (any other board, one by one).
    static constexpr int BUFFER = 448;
    void pseudo(std::vector<int>& out) const {
        if (popcnt(occ[turn]) <= 16) {
            int buffer[BUFFER], *w = buffer;
            each_pseudo([&](int m) { *w++ = m; return true; });
            out.insert(out.end(), buffer, w);
        } else {
            each_pseudo([&](int m) { out.push_back(m); return true; });
        }
    }
    // Pseudo-captures in the exact same order as pseudo(), omitting quiet
    // moves and castles before the quiescence search's legality filter.
    // This does not change any capture, en passant, underpromotion or order.
    void pseudo_captures(std::vector<int>& out) const {
        if (popcnt(occ[turn]) > 16) {
            each_pseudo([&](int m) {
                if (occ[turn ^ 1] & bit(captured_square(m))) out.push_back(m);
                return true;
            });
            return;
        }
        int buffer[BUFFER], *w = buffer;
        const int us = turn;
        const U64 own = occ[us], enemy = occ[us ^ 1];
        for (U64 pieces = own & ~bb[PAWN]; pieces; pieces &= pieces - 1) {
            const int from = lsb(pieces);
            for (U64 targets = attacks_mask(from) & enemy; targets; targets &= targets - 1)
                *w++ = mv(from, lsb(targets));
        }
        for (U64 pawns = own & bb[PAWN]; pawns; pawns &= pawns - 1) {
            const int from = lsb(pawns);
            for (U64 caps = T.pawn[us][from] & enemy; caps; caps &= caps - 1) {
                const int to = lsb(caps);
                if (rank_of(to) == 0 || rank_of(to) == 7) {
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT})
                        *w++ = mv(from, to, p);
                } else {
                    *w++ = mv(from, to);
                }
            }
        }
        // En passant, when its square really holds an enemy piece to take (what the old filter asked).
        for (int *m = w, *end = ep_into(w); m < end; ++m)
            if (occ[us ^ 1] & bit(captured_square(*m))) *w++ = *m;
        out.insert(out.end(), buffer, w);
    }
    int* ep_into(int* w) const {
        if (ep <= 0 || (occupied() & bit(ep))) return w;  // python: `not self.ep_square`
        U64 rank = turn == WHITE ? (0xFFULL << 32) : (0xFFULL << 24);
        for (U64 c = bb[PAWN] & occ[turn] & T.pawn[turn ^ 1][ep] & rank; c; c &= c - 1)
            *w++ = mv(lsb(c), ep);
        return w;
    }
    void ep_moves(std::vector<int>& out) const {
        int buffer[16];
        out.insert(out.end(), buffer, ep_into(buffer));
    }
    void castles(std::vector<int>& out) const {
        int buffer[4];
        out.insert(out.end(), buffer, castles_into(buffer));
    }
    int* castles_into(int* w) const {
        U64 backrank = turn == WHITE ? RANK1 : RANK8;
        U64 kings = occ[turn] & bb[KING] & backrank;
        if (!kings) return w;
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
                *w++ = mv(k, king_to);
        }
        return w;
    }
    void legal_moves(std::vector<int>& out) const {
        Guard g = guard();
        each_candidate(g, [&](int m) {
            if (legal_by(g, m)) out.push_back(m);
            return true;
        });
    }
    bool any_legal() const {
        Guard g = guard();
        if (g.king >= 0 && g.plain) {
            // A king step first: there usually is one. (Castling is legal only when the
            // step towards it is, so it never answers differently.)
            U64 lifted = occupied() ^ bit(g.king);
            for (U64 to = T.king[g.king] & ~occ[turn]; to; to &= to - 1)
                if (!attacked(turn ^ 1, lsb(to), lifted)) return true;
            // Double check: only the king moves (or, on boards no game reaches, an en passant that takes
            // one checker and blocks the other).
            if (g.checkers & (g.checkers - 1)) return has_legal_ep();
        }
        return !each_candidate(g, [&](int m) { return (g.plain && mfrom(m) == g.king) || !legal_by(g, m); });
    }
    // Legal captures ONTO square s (python: generate_legal_captures(to_mask=s)).
    void captures_to(int s, std::vector<int>& out) const {
        if (occ[turn ^ 1] & bit(s)) {
            U64 a = attackers(turn, s, occupied());
            Guard g = a ? guard() : Guard{-1, 0, 0, true};
            for (; a; a &= a - 1) {
                int from = lsb(a);
                if ((bb[PAWN] & bit(from)) && (rank_of(s) == 0 || rank_of(s) == 7)) {
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT})
                        if (legal_by(g, mv(from, s, p))) out.push_back(mv(from, s, p));
                } else if (legal_by(g, mv(from, s))) {
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
    bool has_legal_ep() const {          // ep_moves(), each made: without a list
        if (ep <= 0 || (occupied() & bit(ep))) return false;
        U64 rank = turn == WHITE ? (0xFFULL << 32) : (0xFFULL << 24);
        for (U64 c = bb[PAWN] & occ[turn] & T.pawn[turn ^ 1][ep] & rank; c; c &= c - 1)
            if (legal(mv(lsb(c), ep))) return true;
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

// Material identity matters to this TT when equal-type pieces exchange squares.
HOT double web_piece_value(const Pos& p, int square, const std::vector<double>& fallback) {
    int type=p.type_at(square);
    if (!web_context_on) return fallback[std::min(type,int(QUEEN))];
    return type == KING ? 10000.0 : type < PAWN ? 0.0 : web_context_values[p.web_ids[square]];
}
HOT U64 web_material_key(const Pos& p) {
    U64 hash=key_of(p).hash();
    if (web_context_on) for (U64 rest=p.occupied();rest;rest&=rest-1) {
        int square=lsb(rest); hash ^= U64(p.web_ids[square]+1)*0x9e3779b97f4a7c15ULL;
        hash=(hash<<11)|(hash>>53);
    }
    return hash;
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
    int pawn_reach = 0;     // > 0: the attacker's pawns this far or farther from the enemy king are not moved
    explicit MateSearch(long long limit_, int pawn_reach_ = 0) : limit(limit_), pawn_reach(pawn_reach_) {}

    // The mover's own king is safe after the move (Pos::legal, on the position already made).
    static HOT bool safe_after(const Pos& before, const Pos& after) {
        if (before.king(before.turn) < 0) return true;
        int k = after.king(before.turn);
        return k < 0 || !after.attacked(before.turn ^ 1, k, after.occupied());
    }

    // Can move m give check to the enemy king on square king (not attacked yet)? discoverers are
    // p.discoverers(king). Every move that gives check passes (a few that do not pass too): a castle,
    // a promotion, en passant, a discoverer moving (a discovered check), a knight or pawn landing
    // where it attacks the king, a slider landing on one of its lines with nothing in between once
    // it has left its square.
    static HOT bool may_check(const Pos& p, int m, int king, U64 discoverers) {
        int from = mfrom(m), to = mto(m);
        U64 f = bit(from), t = bit(to);
        if ((discoverers & f) || mpromo(m)) return true;
        if (p.bb[KNIGHT] & f) return (T.knight[king] & t) != 0;
        if (p.bb[PAWN] & f) return (T.pawn[p.turn ^ 1][king] & t) != 0 || p.is_en_passant(m);
        if (p.bb[KING] & f) return std::abs(file_of(to) - file_of(from)) == 2;
        U64 straight = T.file_line[king] | T.rank_line[king], diagonal = T.diag_line[king];
        U64 lines = (p.bb[QUEEN] & f) ? (straight | diagonal) : (p.bb[ROOK] & f) ? straight : diagonal;
        return (lines & t) && !(T.between[to][king] & (p.occupied() ^ f));
    }

    // The quiet moves of each recursion level (a deque keeps references valid).
    struct Lists {
        std::vector<int> quiet;
    };
    std::deque<Lists> levels;
    int level = 0;
    struct Down {
        int& l;
        explicit Down(int& x) : l(x) { ++l; }
        ~Down() { --l; }
    };
    Lists& lists_here() {
        while (int(levels.size()) <= level) levels.emplace_back();
        return levels[level];
    }

    // The checking moves first, then the quiet ones, each group in generation order. A checking move
    // is searched as soon as it is found (generating the rest first counts no node), a quiet one is
    // made when its turn comes, and moves that cannot give check are not made to find that out.
    bool attack(const Pos& p, int n) {
        if (++nodes > limit) { aborted = true; return false; }
        int king = p.king(p.turn ^ 1);
        U64 skip = 0;                       // the attacker's pawns this far or farther from the king
        if (pawn_reach > 0 && king >= 0)
            for (U64 pawns = p.bb[PAWN] & p.occ[p.turn]; pawns; pawns &= pawns - 1) {
                int s = lsb(pawns);
                if (std::max(std::abs((s & 7) - (king & 7)), std::abs((s >> 3) - (king >> 3))) >= pawn_reach)
                    skip |= bit(s);
            }
        if (n == 1 && king < 0) return false;          // no king: nothing gives check
        // The shortcuts need the enemy king not attacked yet and one attacking king at most (a second
        // king could step next to the enemy king: safe_after only looks at the first one).
        bool direct = king >= 0 && p.single_king(p.turn) && !p.attacked(p.turn, king, p.occupied());
        U64 disc = direct ? p.discoverers(king) : 0;
        if (n == 1 && direct) {
            // The last move: only a checkmate counts, in any order (no node is counted below this one).
            return !p.each_pseudo([&](int m) {
                if ((skip & bit(mfrom(m))) || !may_check(p, m, king, disc)) return true;
                Pos q = p;
                q.make(m);
                return !(safe_after(p, q) && q.is_check() && !q.any_legal());
            });
        }
        Lists& here = lists_here();
        Down down(level);
        std::vector<int>& quiet = here.quiet;
        quiet.clear();
        bool mate = false;
        p.each_pseudo([&](int m) {
            if (skip & bit(mfrom(m))) return true;
            if (direct && !may_check(p, m, king, disc)) {
                quiet.push_back(m);         // surely no check (if legal: seen when made)
                return true;
            }
            Pos q = p;
            q.make(m);
            if (!safe_after(p, q)) return true;
            if (!q.is_check()) {
                quiet.push_back(m);
                return true;
            }
            if (!q.any_legal() || (n > 1 && defend(q, n - 1))) {
                mate = true;
                return false;
            }
            return !aborted;
        });
        if (mate) return true;
        if (aborted || n == 1) return false;
        for (int m : quiet) {
            Pos q = p;
            q.make(m);
            if (!safe_after(p, q)) continue;
            if (defend(q, n - 1)) return true;
            if (aborted) return false;
        }
        return false;
    }

    bool defend(const Pos& q, int n) {
        if (++nodes > limit) { aborted = true; return false; }
        bool any = false, escaped = false;  // legality tested only for the replies actually tried
        q.each_pseudo([&](int d) {
            Pos r = q;
            r.make(d);
            if (!safe_after(q, r)) return true;
            any = true;
            if (!attack(r, n) || aborted) {
                escaped = true;
                return false;
            }
            return true;
        });
        return any && !escaped;             // no legal reply here is stalemate: the king survives
    }
};

// Options of the deep (tactical) search, for this thread: Board.search_options. All off: the
// original search, exactly.
struct SearchOptions {
    bool promotions = false;    // the exchange search (quiescence) also tries pawns promoting without capture
    bool quiet_checks = false;  // ...and, at its first half-move, moves that give check
    int extensions = 0;         // a move that gives check is searched one half-move deeper, this many per line
};
SearchOptions& search_options() {
    static thread_local SearchOptions options;
    return options;
}

// Capture-only negamax. In check, every legal evasion is searched; at the
// horizon only checkmate has a non-zero value. A window is translated through
// the immediate material gain before searching the opponent's position.
struct ExchangeSearch {
    static constexpr double MATE = 1000.0;
    const std::vector<double>& values;
    long long nodes = 0, limit;
    bool aborted = false;

    ExchangeSearch(const std::vector<double>& values_, long long limit_) : values(values_), limit(limit_) {}

    // Options (a deep search sets them from SearchOptions; off for the plain exchange calculation).
    bool promotions = false, quiet_checks = false;
    int top = -1;                          // the depth of the first half-move (quiet checks there only)

    // Quiet moves the options add, after the captures: pawns promoting without capture, and at the first
    // half-move moves that give check - knights, pawns and sliders landing where they attack the king,
    // any move of a piece that uncovers a slider (each made to be sure). Generated directly; legal_by.
    void add_quiet(const Pos& p, const Pos::Guard& g, std::vector<int>& moves, bool checks) const {
        int us = p.turn, them = us ^ 1, step = us == WHITE ? 8 : -8;
        U64 all = p.occupied(), own = p.occ[us], empty = ~all;
        U64 pawns = p.bb[PAWN] & own;
        auto add = [&](int m) { if (p.legal_by(g, m)) moves.push_back(m); };
        if (promotions)
            for (U64 seventh = pawns & (us == WHITE ? 0x00FF000000000000ULL : 0xFF00ULL); seventh; seventh &= seventh - 1) {
                int from = lsb(seventh), to = from + step;
                if (empty & bit(to))
                    for (int t : {QUEEN, ROOK, BISHOP, KNIGHT}) add(mv(from, to, t));
            }
        int king = p.king(them);
        if (!checks || king < 0 || !p.single_king(us) || p.attacked(us, king, all)) return;
        U64 disc = p.discoverers(king);
        U64 straight = rook_attacks(king, all), diagonal = bishop_attacks(king, all);
        auto checking = [&](int m) {
            if (!p.legal_by(g, m)) return;
            Pos next = p;
            next.make(m);
            if (next.is_check()) moves.push_back(m);
        };
        for (U64 pieces = own & ~p.bb[PAWN] & ~p.bb[KING]; pieces; pieces &= pieces - 1) {
            int from = lsb(pieces);
            U64 f = bit(from), reach = p.attacks_mask(from) & empty, target;
            if (disc & f) target = reach;                                   // discovered (or direct) check
            else if (p.bb[KNIGHT] & f) target = reach & T.knight[king];
            else if (p.bb[QUEEN] & f) target = reach & (straight | diagonal);
            else if (p.bb[ROOK] & f) target = reach & straight;
            else target = reach & diagonal;
            for (; target; target &= target - 1) checking(mv(from, lsb(target)));
        }
        U64 king_moves = (disc & own & p.bb[KING]) ? p.attacks_mask(p.king(us)) & empty : 0;   // discovered by the king
        for (; king_moves; king_moves &= king_moves - 1) checking(mv(p.king(us), lsb(king_moves)));
        for (U64 rest = pawns; rest; rest &= rest - 1) {
            int from = lsb(rest), one = from + step;
            if (one < 0 || one > 63 || !(empty & bit(one)) || rank_of(one) == 0 || rank_of(one) == 7) continue;
            bool discovering = disc & bit(from);
            if (discovering || (T.pawn[them][king] & bit(one))) checking(mv(from, one));
            int start = us == WHITE ? 1 : 6, two = one + step;
            if (rank_of(from) == start && (empty & bit(two)) && (discovering || (T.pawn[them][king] & bit(two))))
                checking(mv(from, two));
        }
    }

    // Move lists reused per recursion level (a deque keeps references valid).
    std::deque<std::vector<int>> lists;
    std::deque<std::vector<double>> keys;
    int level = 0;
    struct Level {
        int& l;
        explicit Level(int& x) : l(x) { ++l; }
        ~Level() { --l; }
    };
    std::vector<int>& list_here() {
        while (int(lists.size()) <= level) { lists.emplace_back(); keys.emplace_back(); }
        lists[level].clear();
        return lists[level];
    }

    HOT double move_gain(const Pos& p, int m) const {
        if (web_context_on) {
            int captured=p.captured_square(m), promo=mpromo(m);
            double gain=(p.occ[p.turn^1]&bit(captured)) ? web_piece_value(p,captured,values) : 0.0;
            if (promo) gain += web_context_values[64+p.turn*32+(promo-2)*8+file_of(mto(m))]
                               - web_piece_value(p,mfrom(m),values);
            return gain;
        }
        int to = mto(m);
        double gain = 0.0;
        if ((p.occupied() >> to) & 1) {             // the victim of type_at(captured_square(m)): no en passant
            int victim = p.type_at(to);
            if (victim <= QUEEN) gain = values[victim];
        } else if (to == p.ep && p.is_en_passant(m)) {
            int victim = p.type_at(p.captured_square(m));
            if (victim >= PAWN && victim <= QUEEN) gain = values[victim];
        }
        if (int promotion = mpromo(m)) gain += values[promotion] - values[PAWN];
        return gain;
    }

    double search(const Pos& p, int depth, double alpha, double beta) {
        if (++nodes > limit) { aborted = true; return 0.0; }
        bool check = p.is_check();
        std::vector<int>& moves = list_here();
        std::vector<double>& gains = keys[level];
        Level guard(level);
        if (check) {
            p.legal_moves(moves);
            if (moves.empty()) return -MATE;
        }
        if (depth == 0) return 0.0;

        if (!check) {
            // Standing pat (0) already reaches beta: that is the answer whatever the
            // captures are, so they are not generated - same value, same nodes.
            if ((0.0 > alpha ? 0.0 : alpha) >= beta) return 0.0;
            // Only captures matter here. Generate the exact capture subsequence
            // of pseudo() directly, avoiding quiet moves and castling entirely.
            // The legal filter preserves their original stable order.
            p.pseudo_captures(moves);
            bool checks = quiet_checks && depth == top, extra = promotions || checks;
            if (!moves.empty() || extra) {
                Pos::Guard g = p.guard();
                moves.erase(std::remove_if(moves.begin(), moves.end(), [&](int m) {
                    return !p.legal_by(g, m);
                }), moves.end());
                if (extra) add_quiet(p, g, moves, checks);
            }
        }
        // MVV-LVA, with stable move order for equal scores: an insertion sort
        // gives exactly std::stable_sort's order without allocating.
        gains.resize(2 * moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            gains[2 * i] = move_gain(p, moves[i]);
            gains[2 * i + 1] = web_piece_value(p,mfrom(moves[i]),values);
        }
        for (size_t i = 1; i < moves.size(); ++i) {
            int m = moves[i];
            double g = gains[2 * i], a = gains[2 * i + 1];
            size_t j = i;
            while (j > 0 && (g > gains[2 * (j - 1)] || (g == gains[2 * (j - 1)] && a < gains[2 * (j - 1) + 1]))) {
                moves[j] = moves[j - 1];
                gains[2 * j] = gains[2 * (j - 1)];
                gains[2 * j + 1] = gains[2 * (j - 1) + 1];
                --j;
            }
            moves[j] = m;
            gains[2 * j] = g;
            gains[2 * j + 1] = a;
        }

        double best = check ? -std::numeric_limits<double>::infinity() : 0.0;
        if (best > alpha) alpha = best;
        if (alpha >= beta) return best;
        for (size_t i = 0; i < moves.size(); ++i) {
            int m = moves[i];
            double gain = gains[2 * i];
            Pos next = p;
            next.make(m);
            double score = gain - search(next, depth - 1, gain - beta, gain - alpha);
            if (aborted) return best;
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
        return best;
    }
};

// Full-width material search: every legal move for `plies` half-moves, then
// the exact exchange search settles the captures left. Material and mate only.
static double material_search(const Pos& p, int plies, double alpha, double beta, ExchangeSearch& q, int qdepth) {
    if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }
    if (plies == 0) return q.search(p, qdepth, alpha, beta);
    std::vector<int> moves;
    p.legal_moves(moves);
    if (moves.empty()) return p.is_check() ? -ExchangeSearch::MATE : 0.0;
    std::stable_sort(moves.begin(), moves.end(), [&](int a, int b) { return q.move_gain(p, a) > q.move_gain(p, b); });
    double best = -std::numeric_limits<double>::infinity();
    for (int m : moves) {
        double gain = q.move_gain(p, m);
        Pos next = p;
        next.make(m);
        double score = gain - material_search(next, plies - 1, gain - beta, gain - alpha, q, qdepth);
        if (q.aborted) return best;
        if (score > best) best = score;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }
    return best;
}

// Deep material search with the classic prunings: iterative deepening, a
// transposition table (also the source of the main line), principal variation
// search, null move, late move reductions, killer and history move ordering.
// Leaves are settled by the exact exchange search. Material and mate only.
#ifndef LAZY_PICKS
#define LAZY_PICKS 1
#endif
struct DeepSearch {
    struct Entry {                        // 24 bytes: the table fits the cache better (moves fit 16 bits)
        U64 key = 0;
        double value = 0.0;
        std::uint16_t move = 0;
        std::int16_t depth = -1;
        std::uint16_t generation = 0;     // wraps sooner than before: the table is then cleared, as before
        std::uint8_t flag = 0;
    };
    struct TableCache {
        std::vector<Entry> entries = std::vector<Entry>(size_t(1) << 16);
        std::uint16_t generation = 0;
        // A session (Board.table_session): every search until it ends shares one table - the same
        // position is not searched again from scratch for each candidate of one decision.
        bool session = false;
        std::uint16_t session_generation = 0;
        std::uint16_t open() { return session ? session_generation : next(); }
        std::uint16_t next() {
            if (++generation == 0) {
                std::fill(entries.begin(), entries.end(), Entry{});
                generation = 1;
            }
            return generation;
        }
    };
    static TableCache& cache() {
        static thread_local TableCache storage;
        return storage;
    }
    enum { EXACT = 0, LOWER = 1, UPPER = 2 };
    ExchangeSearch q;
    int qdepth;
    std::vector<Entry>& table;
    const std::uint16_t generation;
    int killers[128][2] = {};
    int history[2][64][64] = {};
    struct Item {                                 // a move with its order rank and material gain
        double rank, gain;
        int move;
    };
    std::deque<std::vector<int>> deep_lists;      // move lists reused per ply
    std::deque<std::vector<Item>> deep_items;

    bool exact = false;   // true: no null move and no reductions - the same value as the full search
    bool lazy = true;     // finite piece values: the lazy ordering gives the full sort's order
    int extensions_left = 0;   // check extensions still allowed on the current line

    DeepSearch(const std::vector<double>& values, long long limit, int qdepth_, bool exact_ = false)
        : q(values, limit), qdepth(qdepth_), table(cache().entries), generation(cache().open()), exact(exact_),
          lazy(std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })),
          extensions_left(search_options().extensions) {
        q.promotions = search_options().promotions;
        q.quiet_checks = search_options().quiet_checks;
        q.top = qdepth;
    }

    Entry& slot(U64 key) { return table[key & (table.size() - 1)]; }

    static bool has_pieces(const Pos& p) {   // null move is unsafe with king and pawns only
        return (p.occ[p.turn] & (p.bb[KNIGHT] | p.bb[BISHOP] | p.bb[ROOK] | p.bb[QUEEN])) != 0;
    }

    double search(const Pos& p, int depth, double alpha, double beta, int ply, bool allow_null) {
        if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }
        if (depth <= 0) return q.search(p, qdepth, alpha, beta);
        U64 key = web_material_key(p);
        Entry& e = slot(key);
        int tt_move = 0;
        if (e.generation == generation && e.key == key) {
            tt_move = e.move;
            if (e.depth >= depth) {
                if (e.flag == EXACT) return e.value;
                if (e.flag == LOWER && e.value >= beta) return e.value;
                if (e.flag == UPPER && e.value <= alpha) return e.value;
            }
        }
        while (int(deep_lists.size()) <= ply) {
            deep_lists.emplace_back(); deep_items.emplace_back();
        }
        std::vector<int>& moves = deep_lists[ply];
        std::vector<Item>& items = deep_items[ply];
        moves.clear();
        const Pos::Guard guard = p.guard();
        bool check = guard.checkers != 0;         // = p.is_check() (attackers of the same king)
        if (guard.king >= 0 && guard.plain && check)
            p.each_evasion(guard, [&](int m) { moves.push_back(m); return true; });
        else
            p.pseudo(moves);                      // legality is checked lazily, move by move
        if (moves.empty()) return check ? -(ExchangeSearch::MATE - ply) : 0.0;
        const double eps = 1e-6;
        if (!exact && allow_null && !check && depth >= 3 && has_pieces(p)) {
            Pos null = p;
            null.turn ^= 1;
            null.ep = -1;
            double score = -search(null, depth - 3, -beta, -beta + eps, ply + 1, false);
            if (q.aborted) return 0.0;
            if (score >= beta) return score;
        }
        int k = std::min(ply, 127), side = p.turn;
        auto order = [&](int m, double gain) {
            if (m == tt_move) return 1e9;
            if (gain > 0) return 1e6 + 1000.0 * gain - web_piece_value(p,mfrom(m),q.values);
            if (m == killers[k][0]) return 9e5;
            if (m == killers[k][1]) return 8e5;
            return double(history[side][mfrom(m)][mto(m)]);
        };
        items.resize(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            double gain = q.move_gain(p, moves[i]);
            items[i] = Item{order(moves[i], gain), gain, moves[i]};
        }
        // The order of a stable sort by rank, made only as far as the node gets: the first moves by
        // selection (the earliest best of those left, the others keeping their order), the rest at
        // once by insertion sort when more are needed - the same order, most nodes cut off early.
        size_t sorted = 0;                            // moves[0, sorted) are in their final place
        auto place = [&](size_t j) {
            size_t n = items.size();
            if (j < sorted) return;
            if (lazy && j < LAZY_PICKS) {
                size_t top = j;
                for (size_t i = j + 1; i < n; ++i)
                    if (items[i].rank > items[top].rank) top = i;
                if (top != j) {
                    Item chosen = items[top];
                    for (size_t i = top; i > j; --i) items[i] = items[i - 1];
                    items[j] = chosen;
                }
                sorted = j + 1;
                return;
            }
            for (size_t i = j + 1; i < n; ++i) {      // stable: same order as std::stable_sort
                Item it = items[i];
                size_t at = i;
                while (at > j && it.rank > items[at - 1].rank) {
                    items[at] = items[at - 1];
                    --at;
                }
                items[at] = it;
            }
            sorted = n;
        };
        double best = -std::numeric_limits<double>::infinity(), start = alpha;
        int best_move = 0;
        size_t searched = 0;                          // legal moves searched so far (i of the legal list)
        // At the frontier (one half-move left) a successor is made only when its search needs it. One
        // that cannot be in check and stands pat at once (exchange search) costs its two nodes and is
        // worth 0 - what searching it returns - so it is counted, not made. Legality by the pins then.
        const bool frontier = depth == 1;
        const int enemy_king = p.king(p.turn ^ 1);
        // (Not with the enemy king attacked already - a quiet move leaves it so - nor with two kings.)
        const bool direct = frontier && enemy_king >= 0 && p.single_king(p.turn) &&
                            !p.attacked(p.turn, enemy_king, p.occupied());
        const U64 disc = direct ? p.discoverers(enemy_king) : 0;
        Pos next;
        for (size_t j = 0; j < items.size(); ++j) {
            place(j);
            int m = items[j].move;
            bool made = false, calm = false;          // calm: the successor is surely not in check
            if (!p.legal_by(guard, m)) continue;
            if (frontier) {
                calm = enemy_king < 0 || (direct && !MateSearch::may_check(p, m, enemy_king, disc));
            } else {
                next = p;
                next.make(m);
                made = true;
            }
            int ext = 0;                              // a check: one half-move deeper (SearchOptions)
            if (extensions_left > 0 && !calm) {
                if (!made) {
                    next = p;
                    next.make(m);
                    made = true;
                }
                ext = next.is_check() ? 1 : 0;
            }
            auto child = [&](int d, double a, double b) {
                d += ext;
                if (calm && d <= 0 && (qdepth == 0 || (0.0 > a ? 0.0 : a) >= b)) {
                    if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }    // this search's node
                    if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }    // the exchange search's
                    return 0.0;
                }
                if (!made) {
                    next = p;
                    next.make(m);
                    made = true;
                }
                extensions_left -= ext;
                double value = search(next, d, a, b, ply + 1, true);
                extensions_left += ext;
                return value;
            };
            size_t i = searched++;
            if (i == 0) best_move = m;
            double gain = items[j].gain;
            bool quiet = gain == 0.0 && !mpromo(m);
            int reduction = (!exact && depth >= 3 && i >= 3 && quiet && !check && !next.is_check()) ? 1 : 0;
            double score;
            if (i == 0) {
                score = gain - child(depth - 1, gain - beta, gain - alpha);
            } else {
                score = gain - child(depth - 1 - reduction, gain - alpha - eps, gain - alpha);
                if (!q.aborted && score > alpha && (reduction || score < beta))
                    score = gain - child(depth - 1, gain - beta, gain - alpha);
            }
            if (q.aborted) return best > -1e300 ? best : 0.0;
            if (score > best) { best = score; best_move = m; }
            if (best > alpha) alpha = best;
            if (alpha >= beta) {
                if (quiet) {
                    if (killers[k][0] != m) { killers[k][1] = killers[k][0]; killers[k][0] = m; }
                    history[side][mfrom(m)][mto(m)] += depth * depth;
                }
                break;
            }
        }
        if (!searched) return check ? -(ExchangeSearch::MATE - ply) : 0.0;   // no legal move
        e.key = key;
        e.generation = generation;
        e.value = best;
        e.depth = std::int16_t(depth);
        e.move = std::uint16_t(best_move);
        e.flag = best <= start ? UPPER : best >= beta ? LOWER : EXACT;
        return best;
    }

    // The main line from `p`, read back from the table.
    std::vector<int> line(Pos p, int max_len) {
        std::vector<int> out;
        std::vector<int> legal;
        for (int i = 0; i < max_len; ++i) {
            Entry& e = slot(web_material_key(p));
            if (e.generation != generation || e.key != web_material_key(p) || !e.move) break;
            p.legal_moves(legal);
            if (std::find(legal.begin(), legal.end(), e.move) == legal.end()) break;
            out.push_back(e.move);
            p.make(e.move);
        }
        return out;
    }
};

// ------------------------------------------------------------------ board

Pos parse_position(const std::string& fen);
class Board {
public:
    void context_values(const std::vector<double>& values) {
        if (values.size()!=128 || !std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}))
            throw std::invalid_argument("context requires 128 finite learned values");
        std::copy(values.begin(),values.end(),web_context_values.begin());
        for(int i=0;i<64;++i)pos.web_ids[i]=std::uint8_t(i);
        web_context_on=true;
    }
    std::vector<double> material_values() const {
        std::vector<double> out(64,0.0);
        for(int square=0;square<64;++square)if(pos.occupied()&bit(square))out[square]=web_context_values[pos.web_ids[square]];
        return out;
    }
    explicit Board(const std::string& fen) : pos(parse_position(fen)) {}
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

    // Owner's rule (strict repetition): a draw by repetition only once the
    // position HAS occurred three times (the player chose to repeat); no
    // "could claim" endings. Mate, stalemate, insufficient material, 5-fold
    // and the move rules (100/150 half-moves) as in terminal().
    int terminal_strict() {
        bool legal = pos.any_legal();
        if (!legal && pos.is_check()) return pos.turn == WHITE ? 2 : 1;
        if (pos.insufficient(WHITE) && pos.insufficient(BLACK)) return 0;
        if (!legal) return 0;
        if (pos.halfmove >= 100) return 0;
        if (repetition(3)) return 0;
        return -1;
    }
    // Exactly terminal_strict() after each candidate, including real game history.
    // Keep push/pop inside C++ instead of crossing the Python boundary twice
    // per candidate (especially for the reply search's ~200 positions).
    std::vector<int> terminal_after_moves(const std::vector<int>& codes) {
        std::vector<int> results;
        results.reserve(codes.size());
        for (int code : codes) {
            push(code);
            try {
                results.push_back(terminal_strict());
            } catch (...) {
                pop();
                throw;
            }
            pop();
        }
        return results;
    }

    std::pair<double, double> king_risks() const {
        auto rows = king_senses(pos);
        return {rows[WHITE].risk, rows[BLACK].risk};
    }


    // captures.track + capture_targets, retaining identity through EP, castling
    // and promotion. Integer targets are exactly the Python uint8 result.


    // captures.activity: integer measurements, including the original rule
    // that castling alone does not count as the rook's first individual move.


    // King safety (v4): (found, proven, nodes). found = n if the attacker forces
    // mate within n of its moves (smallest n), else 0. proven = the largest n
    // searched completely without finding mate (an exact upper bound on
    // vulnerability). A node limit stops the search early (then proven < max).
    std::tuple<int, int, long long> forced_mate(bool attacker_white, int max_moves, long long node_limit,
                                                int pawn_reach = 0) const {
        int attacker = attacker_white ? WHITE : BLACK;
        long long used = 0;
        int found = 0, proven = 0;
        for (int n = 1; n <= max_moves; ++n) {
            MateSearch search(node_limit - used, pawn_reach);
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
    std::tuple<double, bool, long long> exchange_gain(const std::vector<double>& values, int max_depth,
                                                        long long node_limit) const {
        if (values.size() != 6 || max_depth < 0 || node_limit < 0)
            throw std::invalid_argument("exchange_gain requires six values and non-negative limits");
        ExchangeSearch search(values, node_limit);
        double gain = search.search(pos, max_depth, -std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity());
        return {search.aborted ? 0.0 : gain, !search.aborted, search.nodes};
    }
    std::vector<double> exchange_after_moves(const std::vector<int>& codes, const std::vector<double>& values,
                                              int max_depth, long long node_limit) const {
        if (values.size() != 6 || max_depth < 0 || node_limit < 0)
            throw std::invalid_argument("exchange_after_moves requires six values and non-negative limits");
        std::vector<int> legal;
        pos.legal_moves(legal);
        std::vector<double> out;
        out.reserve(codes.size());
        for (int m : codes) {
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in exchange_after_moves");
            Pos next = pos;
            next.make(m);
            ExchangeSearch search(values, node_limit);
            double gain = search.search(next, max_depth, -std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity());
            if (search.aborted) throw std::runtime_error("exchange_after_moves exceeded node_limit");
            out.push_back(gain);
        }
        return out;
    }
    // Per candidate: what the opponent wins by force with ONE move of its own
    // (a fork, a double attack, a piece left undefended...) when we then reply
    // as well as we can, captures settled by the exact exchange search. At
    // least the capture-only result of exchange_after_moves; that value alone
    // when the node budget runs out. Rules only, no evaluation.
    std::vector<double> tactic_after_moves(const std::vector<int>& codes, const std::vector<double>& values,
                                           int max_depth, long long node_limit) const {
        if (values.size() != 6 || max_depth < 0 || node_limit < 0)
            throw std::invalid_argument("tactic_after_moves requires six values and non-negative limits");
        std::vector<int> legal;
        pos.legal_moves(legal);
        std::vector<double> out;
        out.reserve(codes.size());
        for (int m : codes) {
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in tactic_after_moves");
            Pos after = pos;
            after.make(m);
            ExchangeSearch search(values, node_limit);
            const double inf = std::numeric_limits<double>::infinity();
            double worst = search.search(after, max_depth, -inf, inf);   // captures only
            if (search.aborted) throw std::runtime_error("tactic_after_moves exceeded node_limit");
            std::vector<int> replies;
            after.legal_moves(replies);
            for (int r : replies) {
                if (after.occ[after.turn ^ 1] & bit(after.captured_square(r))) continue;   // captures: done above
                Pos threat = after;
                threat.make(r);
                std::vector<int> answers;
                threat.legal_moves(answers);
                double ours;
                if (answers.empty()) {
                    ours = threat.is_check() ? -ExchangeSearch::MATE : 0.0;
                } else {
                    // Captures first: the answers most likely to hold.
                    std::stable_sort(answers.begin(), answers.end(), [&](int a, int b) {
                        return search.move_gain(threat, a) > search.move_gain(threat, b);
                    });
                    ours = -inf;
                    for (int s : answers) {
                        double gain = search.move_gain(threat, s);
                        Pos next = threat;
                        next.make(s);
                        double value = gain - search.search(next, max_depth, -inf, inf);
                        if (search.aborted) break;
                        if (value > ours) ours = value;
                        if (ours >= -worst) break;           // this threat cannot beat the worst found
                    }
                }
                if (search.aborted) break;
                if (-ours > worst) worst = -ours;
            }
            out.push_back(worst);
        }
        return out;
    }
    // Per candidate: what the opponent wins by force, searching every move for
    // `plies` half-moves after ours (captures then settled exactly). NaN when
    // the node budget runs out for that candidate.
    std::vector<double> search_after_moves(const std::vector<int>& codes, const std::vector<double>& values,
                                           int plies, int max_depth, long long node_limit) const {
        if (values.size() != 6 || plies < 0 || max_depth < 0 || node_limit < 0)
            throw std::invalid_argument("search_after_moves requires six values and non-negative limits");
        std::vector<int> legal;
        pos.legal_moves(legal);
        std::vector<double> out;
        out.reserve(codes.size());
        const double inf = std::numeric_limits<double>::infinity();
        for (int m : codes) {
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in search_after_moves");
            Pos after = pos;
            after.make(m);
            ExchangeSearch q(values, node_limit);
            double value = material_search(after, plies, -inf, inf, q, max_depth);
            out.push_back(q.aborted ? std::numeric_limits<double>::quiet_NaN() : value);
        }
        return out;
    }
    // Per candidate: (what the opponent wins by force within `plies`
    // half-moves after ours, with prunings; the main line after our move).
    // NaN and an empty line when the node budget runs out.
    std::vector<std::pair<double, std::vector<int>>> deep_after_moves(
            const std::vector<int>& codes, const std::vector<double>& values, int plies, int max_depth,
            long long node_limit, bool exact = false) const {
        if (values.size() != 6 || plies < 0 || max_depth < 0 || node_limit < 0)
            throw std::invalid_argument("deep_after_moves requires six values and non-negative limits");
        std::vector<int> legal;
        pos.legal_moves(legal);
        std::vector<std::pair<double, std::vector<int>>> out;
        out.reserve(codes.size());
        const double inf = std::numeric_limits<double>::infinity();
        DeepSearch search(values, node_limit, max_depth, exact);   // one table for every candidate
        for (int m : codes) {
            if (std::find(legal.begin(), legal.end(), m) == legal.end())
                throw std::invalid_argument("illegal move in deep_after_moves");
            Pos after = pos;
            after.make(m);
            search.q.nodes = 0;
            search.q.aborted = false;
            double value = 0.0;
            // Exact mode goes straight to the full depth: at small depths the
            // iterations cost more than the move ordering they buy.
            for (int depth = exact ? plies : 1; depth <= plies && !search.q.aborted; ++depth) {
                double found = search.search(after, depth, -inf, inf, 1, true);
                if (!search.q.aborted) value = found;
            }
            if (search.q.aborted && plies > 0) {
                out.push_back({std::numeric_limits<double>::quiet_NaN(), {}});
                continue;
            }
            out.push_back({value, search.line(after, plies)});
        }
        return out;
    }
    // Claudio: which candidate is best once calculated? Candidate i scores
    // base[i] + weight * tanh((gains[i] - L_i) / 3), L_i = what the opponent wins
    // by force within `plies` half-moves after it (exact search, no prunings).
    // Candidates are searched in the given order, each only as far as needed to
    // know whether it beats the best so far (root PVS: a null window around the
    // loss that would tie, the exact value only for a new best). `floor` is the
    // best score among the moves not calculated. Returns (best candidate or -1,
    // best score, L_i or NaN, status_i): 0 = cannot beat the best even winning
    // everything, 1 = exact, 2 = proven not better, 3 = out of nodes.
    std::tuple<int, double, std::vector<double>, std::vector<int>> root_search(
            const std::vector<int>& codes, const std::vector<double>& base, const std::vector<double>& gains,
            double weight, double floor, const std::vector<double>& values, int plies, int max_depth,
            long long node_limit) const {
        if (values.size() != 6 || plies < 1 || max_depth < 0 || node_limit < 0 || weight <= 0.0
                || base.size() != codes.size() || gains.size() != codes.size())
            throw std::invalid_argument("root_search: bad arguments");
        std::vector<int> legal;
        pos.legal_moves(legal);
        const double inf = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double eps = 1e-9;
        DeepSearch search(values, node_limit, max_depth, true);   // one table for every candidate
        int best = -1;
        double best_score = floor;
        std::vector<double> losses(codes.size(), nan);
        std::vector<int> status(codes.size(), 0);
        for (size_t i = 0; i < codes.size(); ++i) {
            if (std::find(legal.begin(), legal.end(), codes[i]) == legal.end())
                throw std::invalid_argument("illegal move in root_search");
            if (base[i] + weight <= best_score) continue;          // status 0
            Pos after = pos;
            after.make(codes[i]);
            double r = (best_score - base[i]) / weight;
            if (std::isfinite(best_score) && r > -1.0) {
                // It beats the best only if L < T.
                double T = gains[i] - 3.0 * std::atanh(std::min(r, 1.0 - 1e-15));
                search.q.nodes = 0;
                search.q.aborted = false;
                double v = search.search(after, plies, T - eps, T, 1, true);
                if (search.q.aborted) { status[i] = 3; continue; }
                if (v >= T) { status[i] = 2; continue; }
            }
            search.q.nodes = 0;
            search.q.aborted = false;
            double loss = search.search(after, plies, -inf, inf, 1, true);
            if (search.q.aborted) { status[i] = 3; continue; }
            losses[i] = loss;
            status[i] = 1;
            double score = base[i] + weight * std::tanh((gains[i] - loss) / 3.0);
            if (score > best_score) { best_score = score; best = int(i); }
        }
        return {best, best_score, losses, status};
    }
    // Game review: what the side to move wins by force within `plies` half-moves
    // (exact search, captures settled by the exchange search), in `values`
    // units; +-(MATE - ply) for a forced mate; NaN when the node budget runs out.
    double forced_gain(const std::vector<double>& values, int plies, int max_depth, long long node_limit) const {
        if (values.size() != 6 || plies < 0 || max_depth < 0 || node_limit < 0)
            throw std::invalid_argument("forced_gain requires six values and non-negative limits");
        const double inf = std::numeric_limits<double>::infinity();
        DeepSearch search(values, node_limit, max_depth, true);
        double value = plies > 0 ? search.search(pos, plies, -inf, inf, 0, true)
                                 : search.q.search(pos, max_depth, -inf, inf);
        return search.q.aborted ? std::numeric_limits<double>::quiet_NaN() : value;
    }
    int ks_status_py(bool defender_white) const { return ks_status(pos, defender_white ? WHITE : BLACK); }
    // on: the deep searches that follow (this thread) share one transposition table, until off. Use it
    // only while the piece values, the exchange depth and the search mode stay the same.
    static void set_search_options(bool promotions, bool quiet_checks, int extensions) {
        search_options() = SearchOptions{promotions, quiet_checks, std::max(0, extensions)};
    }
    static void table_session(bool on) {
        DeepSearch::TableCache& cache = DeepSearch::cache();
        if (on) cache.session_generation = cache.next();
        cache.session = on;
    }

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

    // The current position encoded from one side (encoding.encode).


    // encoding.encode_after_moves: the position after each move, from self_color's side.

    std::vector<float> encode_here(bool white) const {
        std::vector<float> out(782, 0); encode_row(pos, white ? WHITE : BLACK, out.data()); return out;
    }
    std::vector<std::vector<float>> encode_after(const std::vector<int>& codes, bool white) const {
        std::vector<std::vector<float>> out;
        for (int code : codes) { Pos next = pos; next.make(code); std::vector<float> row(782, 0);
            encode_row(next, white ? WHITE : BLACK, row.data()); out.push_back(std::move(row)); }
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

namespace {
Pos parse_position(const std::string& fen);
bool dead_pawn_lock(const Pos& p);
std::string position_key(const std::string& fen) {
    size_t end = fen.rfind(' ');
    if (end == std::string::npos) throw std::invalid_argument("invalid FEN");
    end = fen.rfind(' ', end - 1);
    return fen.substr(0, end);
}

std::pair<std::vector<int>, std::vector<bool>> endings(Board& board, const std::vector<int>& codes,
                                                     const std::vector<std::string>& previous, bool check_dead) {
    std::unordered_set<std::string> seen(previous.begin(), previous.end());
    std::vector<int> terminal;
    std::vector<bool> repeated;
    terminal.reserve(codes.size()); repeated.reserve(codes.size());
    for (int code : codes) {
        board.push(code);
        try {
            int end = board.terminal_strict();
            const std::string fen = board.fen();
            if (check_dead && end == 4 && dead_pawn_lock(parse_position(fen))) end = 0;
            terminal.push_back(end);
            repeated.push_back(!seen.empty() && seen.contains(position_key(fen)));
        } catch (...) { board.pop(); throw; }
        board.pop();
    }
    return {terminal, repeated};
}

Pos parse_position(const std::string& fen) {
    std::istringstream stream(fen);
    std::string placement, turn, castling, ep;
    Pos p;
    if (!(stream >> placement >> turn >> castling >> ep >> p.halfmove >> p.fullmove))
        throw std::invalid_argument("invalid FEN for mate planner");
    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') { --rank; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        int colour = c >= 'A' && c <= 'Z' ? WHITE : BLACK;
        char lower = char(colour == WHITE ? c - 'A' + 'a' : c);
        const std::string symbols = " pnbrqk";
        auto type = symbols.find(lower);
        if (type == std::string::npos || rank < 0 || file > 7) throw std::invalid_argument("invalid placement");
        p.put(int(type), colour, rank * 8 + file++);
    }
    if (rank != 0 || file != 8 || (turn != "w" && turn != "b")) throw std::invalid_argument("invalid FEN");
    p.turn = turn == "w" ? WHITE : BLACK;
    p.castling = 0;
    for (char c : castling) {
        if (c == 'K') p.castling |= bit(7); if (c == 'Q') p.castling |= bit(0);
        if (c == 'k') p.castling |= bit(63); if (c == 'q') p.castling |= bit(56);
    }
    p.ep = ep == "-" ? -1 : (ep.at(1) - '1') * 8 + ep.at(0) - 'a';
    return p;
}

bool dead_pawn_lock(const Pos& p) {
    if (!p.bb[PAWN] || p.bb[KNIGHT] || p.bb[BISHOP] || p.bb[ROOK] || p.bb[QUEEN]) return false;
    if (popcnt(p.bb[KING] & p.occ[WHITE]) != 1 || popcnt(p.bb[KING] & p.occ[BLACK]) != 1) return false;
    U64 attacks[2] = {0, 0};
    for (int colour : {WHITE, BLACK}) {
        U64 own = p.bb[PAWN] & p.occ[colour];
        for (U64 pieces = own; pieces; pieces &= pieces - 1) {
            int square = lsb(pieces), target = square + (colour == WHITE ? 8 : -8);
            if (target < 0 || target >= 64 || !(p.bb[PAWN] & bit(target))) return false;
            attacks[colour] |= T.pawn[colour][square];
        }
        if (attacks[colour] & p.bb[PAWN] & p.occ[colour ^ 1]) return false;
    }
    for (int colour : {WHITE, BLACK}) {
        int king = p.king(colour);
        U64 blocked = (p.bb[PAWN] & p.occ[colour]) | attacks[colour ^ 1];
        if (blocked & bit(king)) return false;
        U64 reached = 0, frontier = bit(king);
        while (frontier) {
            reached |= frontier;
            U64 next = 0;
            for (U64 squares = frontier; squares; squares &= squares - 1) next |= T.king[lsb(squares)];
            frontier = next & ~blocked & ~reached;
        }
        if (reached & p.bb[PAWN] & p.occ[colour ^ 1]) return false;
    }
    return true;
}

struct ProofKey {
    Key board;
    int remaining, halfmove;
    bool operator==(const ProofKey& other) const {
        return remaining == other.remaining && halfmove == other.halfmove && board == other.board;
    }
};
struct ProofHash {
    size_t operator()(const ProofKey& key) const {
        return size_t(key.board.hash() ^ (U64(key.remaining) << 48) ^ (U64(key.halfmove) << 56));
    }
};
struct ProofAnswer { bool winning; int move; };
struct PositionHash { size_t operator()(const Key& key) const { return size_t(key.hash()); } };
Key raw_key(const Pos& p) { Key key = key_of(p); key.ep = p.ep; return key; }

// Exact AND/OR mate search. Its only goal is checkmate under the game rules.
// The static forbidden set prevents returning to positions of the actual game.
struct MatePlanner {
    int attacker;
    long long nodes = 0, limit;
    bool aborted = false;
    std::unordered_set<Key, PositionHash> forbidden;
    std::unordered_map<ProofKey, ProofAnswer, ProofHash> solved;
    MatePlanner(int attacker_, long long limit_, const std::vector<std::string>& previous)
        : attacker(attacker_), limit(limit_) {
        for (const auto& key : previous) forbidden.insert(raw_key(parse_position(key + " 0 1")));
        solved.reserve(65536);
    }

    std::string prefix(const Pos& p) const {
        Board b(p.bb[PAWN], p.bb[KNIGHT], p.bb[BISHOP], p.bb[ROOK], p.bb[QUEEN], p.bb[KING],
                  p.occ[WHITE], p.occ[BLACK], p.turn == WHITE, p.castling, p.ep, p.halfmove, p.fullmove);
        return position_key(b.fen());
    }
    ProofAnswer solve(const Pos& p, int remaining, bool root = false) {
        if (++nodes > limit) { aborted = true; return {false, 0}; }
        if (p.halfmove >= 100 || remaining <= 0) {
            return {p.turn != attacker && p.is_check() && !p.any_legal(), 0};
        }
        if ((p.insufficient(WHITE) && p.insufficient(BLACK)) || dead_pawn_lock(p)) return {false, 0};
        Key current = raw_key(p);
        if (!root && forbidden.contains(current)) return {false, 0};
        // Far from 50 moves, every path within this bound has identical rule eligibility.
        // Normalize only when even the longest possible non-zeroing continuation stays below 100.
        int clock_key = p.halfmove + 2 * remaining + 1 < 100 ? 0 : p.halfmove;
        ProofKey key{current, remaining, clock_key};
        if (auto found = solved.find(key); found != solved.end()) return found->second;
        std::vector<int> moves;
        p.legal_moves(moves);
        if (moves.empty()) return {p.turn != attacker && p.is_check(), 0};
        if (p.turn == attacker) {
            // Checking moves first is rule-based enumeration, no positional reward.
            std::stable_partition(moves.begin(), moves.end(), [&](int code) { Pos q = p; q.make(code); return q.is_check(); });
            for (int code : moves) {
                Pos q = p; q.make(code);
                auto result = solve(q, remaining - 1);
                if (aborted) return {false, 0};
                if (result.winning) { solved[key] = {true, code}; return {true, code}; }
            }
            solved[key] = {false, 0}; return {false, 0};
        }
        for (int code : moves) {
            Pos q = p; q.make(code);
            auto result = solve(q, remaining);
            if (aborted) return {false, 0};
            if (!result.winning) { solved[key] = {false, 0}; return {false, 0}; }
        }
        solved[key] = {true, 0}; return {true, 0};
    }
};

}
// Browser bindings only; the generated Board/Search/MatePlanner bodies are pinned originals.
namespace {
using emscripten::val;
std::unique_ptr<Board> web_board;
inline val js(double v); inline val js(float v); inline val js(int v); inline val js(bool v);
inline val js(long long v); inline val js(const std::string& v);
template<class A,class B> val js(const std::pair<A,B>& p);
template<class... T> val js(const std::tuple<T...>& t);
template<class T> val js(const std::vector<T>& values) {
    val out=val::array(); for(size_t i=0;i<values.size();++i) out.set(i,js(values[i])); return out;
}
inline val js(double v) { return val(v); }
inline val js(float v) { return val(v); }
inline val js(int v) { return val(v); }
inline val js(bool v) { return val(v); }
inline val js(long long v) { return val(double(v)); }
inline val js(const std::string& v) { return val(v); }
template<class A,class B> val js(const std::pair<A,B>& p) { val out=val::array();out.set(0,js(p.first));out.set(1,js(p.second));return out; }
template<class... T> val js(const std::tuple<T...>& t) { val out=val::array();size_t i=0;std::apply([&](auto...v){(out.set(i++,js(v)),...);},t);return out; }
template<class T> std::vector<T> vec(const val& value) { return emscripten::vecFromJSArray<T>(value); }
val call(const std::string& operation,const val& a) {
    if(operation=="init") { web_context_on=false;web_board=std::make_unique<Board>(a[0].as<std::string>());web_board->push_many_checked(vec<int>(a[1]));return val::undefined(); }
    if(operation=="options") { Board::set_search_options(a[0].as<bool>(), a[1].as<bool>(), a[2].as<int>()); return val::undefined(); }
    if(operation=="dead") return val(dead_pawn_lock(parse_position(a[0].as<std::string>())));
    if(operation=="plan") {
        Pos p=parse_position(a[0].as<std::string>());MatePlanner search(p.turn,a[3].as<double>(),vec<std::string>(a[1]));
        int found=0,move=0,proven=0;
        for(int n=1;n<=a[2].as<int>();++n) { auto result=search.solve(p,n,true);if(search.aborted) break;
            proven=n;if(result.winning){found=n;move=result.move;break;} }
        val out=val::object();out.set("moves",found);out.set("move",move);out.set("nodes",double(search.nodes));
        out.set("proven",proven);out.set("aborted",search.aborted);return out;
    }
    if(!web_board) throw std::runtime_error("uninitialized G16 board");
    Board& b=*web_board;
    if(operation=="context") { b.context_values(vec<double>(a[0])); return val::undefined(); }
    if(operation=="material") return js(b.material_values());
    if(operation=="legal") return js(b.legal_moves());
    if(operation=="fen") return val(b.fen());
    if(operation=="push") { b.push(a[0].as<int>());return val::undefined(); }
    if(operation=="pop") { b.pop();return val::undefined(); }
    if(operation=="terminal") return val(b.terminal());
    if(operation=="strict") return val(b.terminal_strict());
    if(operation=="here") return js(b.encode_here(a[0].as<bool>()));
    if(operation=="after") return js(b.encode_after(vec<int>(a[0]),a[1].as<bool>()));
    if(operation=="endings") return js(endings(b,vec<int>(a[0]),vec<std::string>(a[1]),a[2].as<bool>()));
    if(operation=="mate") return js(b.forced_mate(a[0].as<bool>(),a[1].as<int>(),a[2].as<double>(),a[3].as<int>()));
    if(operation=="exchange") return js(b.exchange_after_moves(vec<int>(a[0]),vec<double>(a[1]),a[2].as<int>(),a[3].as<double>()));
    if(operation=="deep") return js(b.deep_after_moves(vec<int>(a[0]),vec<double>(a[1]),a[2].as<int>(),a[3].as<int>(),a[4].as<double>(),true));
    if(operation=="root") return js(b.root_search(vec<int>(a[0]),vec<double>(a[1]),vec<double>(a[2]),a[3].as<double>(),a[4].as<double>(),vec<double>(a[5]),a[6].as<int>(),a[7].as<int>(),a[8].as<double>()));
    throw std::invalid_argument("unknown G16 rule operation");
}
}
EMSCRIPTEN_BINDINGS(g16_v2_rules) { emscripten::function("call",&call); }
