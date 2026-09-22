#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace tiny {

namespace {

const int MATE = 30000;
const int INF = 1000000;

// piece values used *inside* the opponent only
const int PIECE_VALUE[7] = {0, 100, 320, 330, 500, 900, 0};

// piece-square tables, written visually: first row is rank 8, files a..h.
// A white piece on square `sq` reads table[sq ^ 56]; a black piece reads
// table[sq] (that is the vertical mirror).
const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
const int PST_KNIGHT[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
const int PST_BISHOP[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
const int PST_ROOK[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};
const int PST_QUEEN[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
const int PST_KING[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
};

const int* PST[7] = {nullptr, PST_PAWN, PST_KNIGHT, PST_BISHOP,
                     PST_ROOK, PST_QUEEN, PST_KING};

const int KNIGHT_DF[8] = {1, 2, 2, 1, -1, -2, -2, -1};
const int KNIGHT_DR[8] = {2, 1, -1, -2, -2, -1, 1, 2};
const int KING_DF[8] = {0, 1, 1, 1, 0, -1, -1, -1};
const int KING_DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const int ROOK_DF[4] = {1, -1, 0, 0};
const int ROOK_DR[4] = {0, 0, 1, -1};
const int BISHOP_DF[4] = {1, 1, -1, -1};
const int BISHOP_DR[4] = {1, -1, 1, -1};

inline bool onBoard(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

}  // namespace

// ---------------------------------------------------------------- position

void Position::clear() {
    for (int i = 0; i < 64; ++i) sq[i] = EMPTY;
    kingSq[0] = kingSq[1] = -1;
    whiteToMove = true;
    castling = 0;
    ep = -1;
    halfmove = 0;
    fullmove = 1;
}

void Position::setStartPos() {
    setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Position::setFen(const std::string& fen) {
    clear();
    std::istringstream ss(fen);
    std::string boardPart, stmPart, castlePart, epPart;
    int hm = 0, fm = 1;
    if (!(ss >> boardPart >> stmPart)) return false;
    if (!(ss >> castlePart)) castlePart = "-";
    if (!(ss >> epPart)) epPart = "-";
    ss >> hm;
    ss >> fm;

    int rank = 7, file = 0;
    for (char c : boardPart) {
        if (c == '/') {
            --rank;
            file = 0;
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += c - '0';
            continue;
        }
        int piece = EMPTY;
        switch (c) {
            case 'P': piece = WP; break;
            case 'N': piece = WN; break;
            case 'B': piece = WB; break;
            case 'R': piece = WR; break;
            case 'Q': piece = WQ; break;
            case 'K': piece = WK; break;
            case 'p': piece = BP; break;
            case 'n': piece = BN; break;
            case 'b': piece = BB; break;
            case 'r': piece = BR; break;
            case 'q': piece = BQ; break;
            case 'k': piece = BK; break;
            default: return false;
        }
        if (!onBoard(file, rank)) return false;
        const int s = makeSquare(file, rank);
        sq[s] = static_cast<int8_t>(piece);
        if (pieceType(piece) == WK) kingSq[pieceColor(piece)] = int8_t(s);
        ++file;
    }

    whiteToMove = (stmPart == "w");
    castling = 0;
    for (char c : castlePart) {
        if (c == 'K') castling |= CR_WK;
        else if (c == 'Q') castling |= CR_WQ;
        else if (c == 'k') castling |= CR_BK;
        else if (c == 'q') castling |= CR_BQ;
    }
    ep = -1;
    if (epPart.size() >= 2 && epPart[0] != '-') {
        int f = epPart[0] - 'a';
        int r = epPart[1] - '1';
        if (onBoard(f, r)) ep = static_cast<int8_t>(makeSquare(f, r));
    }
    halfmove = static_cast<int16_t>(hm);
    fullmove = static_cast<int16_t>(fm < 1 ? 1 : fm);
    return true;
}

std::string Position::fen() const {
    static const char* NAMES = " PNBRQK";
    std::string out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int p = sq[makeSquare(file, rank)];
            if (p == EMPTY) {
                ++empty;
                continue;
            }
            if (empty) {
                out += char('0' + empty);
                empty = 0;
            }
            char c = NAMES[pieceType(p)];
            if (pieceColor(p) == 1) c = char(c - 'A' + 'a');
            out += c;
        }
        if (empty) out += char('0' + empty);
        if (rank) out += '/';
    }
    out += whiteToMove ? " w " : " b ";
    std::string cr;
    if (castling & CR_WK) cr += 'K';
    if (castling & CR_WQ) cr += 'Q';
    if (castling & CR_BK) cr += 'k';
    if (castling & CR_BQ) cr += 'q';
    out += cr.empty() ? "-" : cr;
    out += ' ';
    if (ep < 0) {
        out += '-';
    } else {
        out += char('a' + fileOf(ep));
        out += char('1' + rankOf(ep));
    }
    out += ' ';
    out += std::to_string(halfmove);
    out += ' ';
    out += std::to_string(fullmove);
    return out;
}

bool Position::isAttacked(int square, int byColor) const {
    const int f = fileOf(square), r = rankOf(square);

    // pawns
    const int pawnRank = (byColor == 0) ? r - 1 : r + 1;
    const int pawn = (byColor == 0) ? WP : BP;
    for (int df = -1; df <= 1; df += 2) {
        const int pf = f + df;
        if (onBoard(pf, pawnRank) && sq[makeSquare(pf, pawnRank)] == pawn)
            return true;
    }

    // knights
    const int knight = (byColor == 0) ? WN : BN;
    for (int i = 0; i < 8; ++i) {
        const int nf = f + KNIGHT_DF[i], nr = r + KNIGHT_DR[i];
        if (onBoard(nf, nr) && sq[makeSquare(nf, nr)] == knight) return true;
    }

    // king
    const int king = (byColor == 0) ? WK : BK;
    for (int i = 0; i < 8; ++i) {
        const int nf = f + KING_DF[i], nr = r + KING_DR[i];
        if (onBoard(nf, nr) && sq[makeSquare(nf, nr)] == king) return true;
    }

    // rook / queen rays
    const int rook = (byColor == 0) ? WR : BR;
    const int queen = (byColor == 0) ? WQ : BQ;
    for (int d = 0; d < 4; ++d) {
        int nf = f + ROOK_DF[d], nr = r + ROOK_DR[d];
        while (onBoard(nf, nr)) {
            const int p = sq[makeSquare(nf, nr)];
            if (p != EMPTY) {
                if (p == rook || p == queen) return true;
                break;
            }
            nf += ROOK_DF[d];
            nr += ROOK_DR[d];
        }
    }

    // bishop / queen rays
    const int bishop = (byColor == 0) ? WB : BB;
    for (int d = 0; d < 4; ++d) {
        int nf = f + BISHOP_DF[d], nr = r + BISHOP_DR[d];
        while (onBoard(nf, nr)) {
            const int p = sq[makeSquare(nf, nr)];
            if (p != EMPTY) {
                if (p == bishop || p == queen) return true;
                break;
            }
            nf += BISHOP_DF[d];
            nr += BISHOP_DR[d];
        }
    }

    return false;
}

bool Position::inCheck(int color) const {
    const int k = kingSquare(color);
    if (k < 0) return false;
    return isAttacked(k, color ^ 1);
}

int Position::generatePseudoInto(Move* out, bool capturesOnly) const {
    int n = 0;
    const int us = whiteToMove ? 0 : 1;
    const int them = us ^ 1;

    for (int from = 0; from < 64; ++from) {
        const int p = sq[from];
        if (p == EMPTY || pieceColor(p) != us) continue;
        const int type = pieceType(p);
        const int f = fileOf(from), r = rankOf(from);

        if (type == WP) {
            const int dir = (us == 0) ? 1 : -1;
            const int startRank = (us == 0) ? 1 : 6;
            const int promoRank = (us == 0) ? 7 : 0;

            const int nr = r + dir;
            // quiescence skips quiet pushes, but never skips a promotion
            if (onBoard(f, nr) && sq[makeSquare(f, nr)] == EMPTY &&
                (!capturesOnly || nr == promoRank)) {
                const int to = makeSquare(f, nr);
                if (nr == promoRank) {
                    for (int pr = WN; pr <= WQ; ++pr) {
                        Move m;
                        m.from = uint8_t(from);
                        m.to = uint8_t(to);
                        m.promo = uint8_t(pr);
                        out[n++] = m;
                    }
                } else {
                    Move m;
                    m.from = uint8_t(from);
                    m.to = uint8_t(to);
                    out[n++] = m;
                    const int nr2 = r + 2 * dir;
                    if (r == startRank && sq[makeSquare(f, nr2)] == EMPTY) {
                        Move d;
                        d.from = uint8_t(from);
                        d.to = uint8_t(makeSquare(f, nr2));
                        d.flags = FLAG_DOUBLE_PUSH;
                        out[n++] = d;
                    }
                }
            }

            for (int df = -1; df <= 1; df += 2) {
                const int cf = f + df, cr = r + dir;
                if (!onBoard(cf, cr)) continue;
                const int to = makeSquare(cf, cr);
                const int target = sq[to];
                const bool isEp = (ep >= 0 && to == ep);
                if (target == EMPTY && !isEp) continue;
                if (target != EMPTY && pieceColor(target) == us) continue;
                if (cr == promoRank) {
                    for (int pr = WN; pr <= WQ; ++pr) {
                        Move m;
                        m.from = uint8_t(from);
                        m.to = uint8_t(to);
                        m.promo = uint8_t(pr);
                        out[n++] = m;
                    }
                } else {
                    Move m;
                    m.from = uint8_t(from);
                    m.to = uint8_t(to);
                    if (isEp) m.flags = FLAG_EP;
                    out[n++] = m;
                }
            }
            continue;
        }

        if (type == WN || type == WK) {
            const int* dfs = (type == WN) ? KNIGHT_DF : KING_DF;
            const int* drs = (type == WN) ? KNIGHT_DR : KING_DR;
            for (int i = 0; i < 8; ++i) {
                const int nf = f + dfs[i], nr = r + drs[i];
                if (!onBoard(nf, nr)) continue;
                const int to = makeSquare(nf, nr);
                const int target = sq[to];
                if (target != EMPTY && pieceColor(target) == us) continue;
                if (capturesOnly && target == EMPTY) continue;
                Move m;
                m.from = uint8_t(from);
                m.to = uint8_t(to);
                out[n++] = m;
            }
            if (type == WK && !capturesOnly) {
                // castling: squares empty, king not in check and not passing
                // through an attacked square. The destination square is
                // covered by the generic legality filter.
                const uint8_t kSide = (us == 0) ? CR_WK : CR_BK;
                const uint8_t qSide = (us == 0) ? CR_WQ : CR_BQ;
                const int home = (us == 0) ? 4 : 60;
                if (from == home && !isAttacked(home, them)) {
                    if ((castling & kSide) && sq[home + 1] == EMPTY &&
                        sq[home + 2] == EMPTY && !isAttacked(home + 1, them)) {
                        Move m;
                        m.from = uint8_t(home);
                        m.to = uint8_t(home + 2);
                        m.flags = FLAG_CASTLE;
                        out[n++] = m;
                    }
                    if ((castling & qSide) && sq[home - 1] == EMPTY &&
                        sq[home - 2] == EMPTY && sq[home - 3] == EMPTY &&
                        !isAttacked(home - 1, them)) {
                        Move m;
                        m.from = uint8_t(home);
                        m.to = uint8_t(home - 2);
                        m.flags = FLAG_CASTLE;
                        out[n++] = m;
                    }
                }
            }
            continue;
        }

        // sliders
        const int* dfs;
        const int* drs;
        int ndir;
        if (type == WB) {
            dfs = BISHOP_DF;
            drs = BISHOP_DR;
            ndir = 4;
        } else if (type == WR) {
            dfs = ROOK_DF;
            drs = ROOK_DR;
            ndir = 4;
        } else {  // queen: all eight directions
            dfs = KING_DF;
            drs = KING_DR;
            ndir = 8;
        }

        for (int d = 0; d < ndir; ++d) {
            int nf = f + dfs[d], nr = r + drs[d];
            while (onBoard(nf, nr)) {
                const int to = makeSquare(nf, nr);
                const int target = sq[to];
                if (target != EMPTY && pieceColor(target) == us) break;
                if (!capturesOnly || target != EMPTY) {
                    Move m;
                    m.from = uint8_t(from);
                    m.to = uint8_t(to);
                    out[n++] = m;
                }
                if (target != EMPTY) break;
                nf += dfs[d];
                nr += drs[d];
            }
        }
    }
    return n;
}

int Position::generateLegalInto(Move* out, bool capturesOnly) const {
    Move pseudo[MAX_MOVES];
    const int np = generatePseudoInto(pseudo, capturesOnly);
    Position& self = const_cast<Position&>(*this);
    const int us = whiteToMove ? 0 : 1;
    int n = 0;
    for (int i = 0; i < np; ++i) {
        Undo u;
        self.make(pseudo[i], u);
        const bool ok = !self.inCheck(us);
        self.unmake(pseudo[i], u);
        if (ok) out[n++] = pseudo[i];
    }
    return n;
}

void Position::generatePseudo(std::vector<Move>& out) const {
    Move buf[MAX_MOVES];
    const int n = generatePseudoInto(buf);
    out.assign(buf, buf + n);
}

void Position::generateLegal(std::vector<Move>& out) const {
    Move buf[MAX_MOVES];
    const int n = generateLegalInto(buf);
    out.assign(buf, buf + n);
}

void Position::make(const Move& m, Undo& u) {
    u.castling = castling;
    u.ep = ep;
    u.halfmove = halfmove;
    u.captured = sq[m.to];

    const int p = sq[m.from];
    const int us = pieceColor(p);

    if (m.flags & FLAG_EP) {
        const int capSq = (us == 0) ? m.to - 8 : m.to + 8;
        u.captured = sq[capSq];
        sq[capSq] = EMPTY;
    }

    sq[m.to] = int8_t(m.promo ? (m.promo | (us << 3)) : p);
    sq[m.from] = EMPTY;

    if (m.flags & FLAG_CASTLE) {
        if (m.to == 6) {
            sq[5] = sq[7];
            sq[7] = EMPTY;
        } else if (m.to == 2) {
            sq[3] = sq[0];
            sq[0] = EMPTY;
        } else if (m.to == 62) {
            sq[61] = sq[63];
            sq[63] = EMPTY;
        } else if (m.to == 58) {
            sq[59] = sq[56];
            sq[56] = EMPTY;
        }
    }

    if (pieceType(p) == WP || u.captured != EMPTY) halfmove = 0;
    else ++halfmove;

    ep = -1;
    if (m.flags & FLAG_DOUBLE_PUSH) ep = int8_t((m.from + m.to) / 2);

    if (pieceType(p) == WK) {
        kingSq[us] = int8_t(m.to);
        castling &= uint8_t(us == 0 ? ~(CR_WK | CR_WQ) : ~(CR_BK | CR_BQ));
    }
    if (m.from == 0 || m.to == 0) castling &= uint8_t(~CR_WQ);
    if (m.from == 7 || m.to == 7) castling &= uint8_t(~CR_WK);
    if (m.from == 56 || m.to == 56) castling &= uint8_t(~CR_BQ);
    if (m.from == 63 || m.to == 63) castling &= uint8_t(~CR_BK);

    whiteToMove = !whiteToMove;
    if (us == 1) ++fullmove;
}

void Position::unmake(const Move& m, const Undo& u) {
    whiteToMove = !whiteToMove;
    const int us = whiteToMove ? 0 : 1;  // the side that had moved

    int moved = sq[m.to];
    if (m.promo) moved = WP | (us << 3);
    sq[m.from] = int8_t(moved);
    if (pieceType(moved) == WK) kingSq[us] = int8_t(m.from);

    if (m.flags & FLAG_EP) {
        sq[m.to] = EMPTY;
        const int capSq = (us == 0) ? m.to - 8 : m.to + 8;
        sq[capSq] = u.captured;
    } else {
        sq[m.to] = u.captured;
    }

    if (m.flags & FLAG_CASTLE) {
        if (m.to == 6) {
            sq[7] = sq[5];
            sq[5] = EMPTY;
        } else if (m.to == 2) {
            sq[0] = sq[3];
            sq[3] = EMPTY;
        } else if (m.to == 62) {
            sq[63] = sq[61];
            sq[61] = EMPTY;
        } else if (m.to == 58) {
            sq[56] = sq[59];
            sq[59] = EMPTY;
        }
    }

    castling = u.castling;
    ep = u.ep;
    halfmove = u.halfmove;
    if (us == 1) --fullmove;
}

int Position::evaluate() const {
    int score = 0;
    for (int i = 0; i < 64; ++i) {
        const int p = sq[i];
        if (p == EMPTY) continue;
        const int type = pieceType(p);
        const bool white = (pieceColor(p) == 0);
        const int v = PIECE_VALUE[type] + PST[type][white ? (i ^ 56) : i];
        score += white ? v : -v;
    }
    return score;
}

// ------------------------------------------------------------------ helpers

std::string moveToUci(const Move& m) {
    static const char* NAMES = " pnbrqk";
    std::string s;
    s += char('a' + fileOf(m.from));
    s += char('1' + rankOf(m.from));
    s += char('a' + fileOf(m.to));
    s += char('1' + rankOf(m.to));
    if (m.promo) s += NAMES[m.promo];
    return s;
}

uint64_t perft(Position& p, int depth) {
    if (depth == 0) return 1;
    std::vector<Move> moves;
    p.generateLegal(moves);
    if (depth == 1) return uint64_t(moves.size());
    uint64_t total = 0;
    for (const Move& m : moves) {
        Undo u;
        p.make(m, u);
        total += perft(p, depth - 1);
        p.unmake(m, u);
    }
    return total;
}

// ------------------------------------------------------------------- engine

Engine::Engine(const Config& cfg) : cfg_(cfg), rng_(cfg.seed ? cfg.seed : 1) {}

void Engine::setSeed(uint64_t seed) {
    cfg_.seed = seed;
    rng_ = seed ? seed : 1;
}

double Engine::nextRandom() {
    // xorshift64*
    rng_ ^= rng_ >> 12;
    rng_ ^= rng_ << 25;
    rng_ ^= rng_ >> 27;
    const uint64_t x = rng_ * 2685821657736338717ULL;
    return double(x >> 11) / double(1ULL << 53);
}

namespace {

int mvvLva(const Position& p, const Move& m) {
    const int victim = (m.flags & FLAG_EP) ? WP : pieceType(p.sq[m.to]);
    if (victim == 0 && !m.promo) return 0;
    const int attacker = pieceType(p.sq[m.from]);
    return 1000 + PIECE_VALUE[victim] * 8 - PIECE_VALUE[attacker] +
           (m.promo ? PIECE_VALUE[m.promo] : 0);
}

// Move the captures and promotions to the front, best victim first, and leave
// the quiet moves alone. Returns how many noisy moves ended up at the front.
// Cheaper than sorting the whole list, and quiescence can then just walk the
// first `nc` entries.
int orderMoves(const Position& p, Move* moves, int n) {
    int scores[MAX_MOVES];
    int nc = 0;
    for (int i = 0; i < n; ++i) {
        const int s = mvvLva(p, moves[i]);
        if (s > 0) {
            const Move tmp = moves[nc];
            moves[nc] = moves[i];
            moves[i] = tmp;
            scores[nc] = s;
            ++nc;
        }
    }
    for (int i = 1; i < nc; ++i) {  // insertion sort, descending
        const Move m = moves[i];
        const int s = scores[i];
        int j = i - 1;
        while (j >= 0 && scores[j] < s) {
            moves[j + 1] = moves[j];
            scores[j + 1] = scores[j];
            --j;
        }
        moves[j + 1] = m;
        scores[j + 1] = s;
    }
    return nc;
}

}  // namespace

int Engine::quiesce(Position& p, int alpha, int beta, int ply) {
    if (stats_.nodes >= nodeBudget_) {
        aborted_ = true;
        const int e = p.evaluate();
        return p.whiteToMove ? e : -e;
    }
    ++stats_.nodes;

    const int e = p.evaluate();
    const int stand = p.whiteToMove ? e : -e;
    if (stand >= beta) return stand;
    if (stand > alpha) alpha = stand;
    if (ply > 8) return stand;

    // only captures and promotions are generated here
    Move moves[MAX_MOVES];
    const int n = p.generateLegalInto(moves, /*capturesOnly=*/true);
    orderMoves(p, moves, n);

    int best = stand;
    for (int i = 0; i < n; ++i) {
        const Move& m = moves[i];
        Undo u;
        p.make(m, u);
        const int score = -quiesce(p, -beta, -alpha, ply + 1);
        p.unmake(m, u);
        if (score > best) best = score;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
        if (aborted_) break;
    }
    return best;
}

int Engine::negamax(Position& p, int depth, int alpha, int beta, int ply) {
    if (stats_.nodes >= nodeBudget_) {
        aborted_ = true;
        const int e = p.evaluate();
        return p.whiteToMove ? e : -e;
    }
    ++stats_.nodes;

    Move moves[MAX_MOVES];
    const int n = p.generateLegalInto(moves);

    if (n == 0) {
        const int us = p.whiteToMove ? 0 : 1;
        if (p.inCheck(us)) return -MATE + ply;  // mated
        return 0;                               // stalemate
    }
    if (p.halfmove >= 100) return 0;

    if (depth <= 0) {
        if (cfg_.quiescence) return quiesce(p, alpha, beta, ply);
        const int e = p.evaluate();
        return p.whiteToMove ? e : -e;
    }

    orderMoves(p, moves, n);

    int best = -INF;
    for (int i = 0; i < n; ++i) {
        const Move& m = moves[i];
        Undo u;
        p.make(m, u);
        const int score = -negamax(p, depth - 1, -beta, -alpha, ply + 1);
        p.unmake(m, u);
        if (score > best) best = score;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
        if (aborted_) break;
    }
    return best;
}

std::string Engine::chooseMoveUci(const std::string& fen) {
    Position p;
    if (!p.setFen(fen)) return "0000";

    Move moves[MAX_MOVES];
    const int nmoves = p.generateLegalInto(moves);
    if (nmoves == 0) return "0000";
    if (nmoves == 1) {
        ++stats_.searches;
        stats_.last_root_moves = 1;
        return moveToUci(moves[0]);
    }

    aborted_ = false;
    nodeBudget_ =
        stats_.nodes + uint64_t(cfg_.max_nodes < 1 ? 1 : cfg_.max_nodes);

    orderMoves(p, moves, nmoves);

    // Iterative deepening at the root. A depth's scores are only accepted once
    // EVERY root move has been searched at that depth, so the numbers being
    // compared always come from the same depth. Running out of nodes then just
    // falls back to a shallower - but still coherent - ranking, instead of
    // mixing deep scores with shallow ones and picking nonsense.
    //
    // Each root move gets a full window, because the stochastic pick below
    // needs a real score for every move, not just for the best one.
    std::vector<int> scores(static_cast<size_t>(nmoves), 0);
    std::vector<int> working(static_cast<size_t>(nmoves), 0);

    // Depth 0 is the guaranteed fallback and always completes. It uses
    // quiescence rather than the raw static evaluation, so that even the
    // cheapest settings are aware of captures instead of hanging pieces.
    // (quiesce degrades to stand-pat by itself once the budget is gone.)
    for (size_t i = 0; i < scores.size(); ++i) {
        Undo u;
        p.make(moves[i], u);
        if (cfg_.quiescence) {
            scores[i] = -quiesce(p, -INF, INF, 1);
        } else {
            const int e = p.evaluate();
            scores[i] = -(p.whiteToMove ? e : -e);
        }
        p.unmake(moves[i], u);
    }
    int completedDepth = 0;

    for (int d = 1; d <= cfg_.depth; ++d) {
        bool complete = true;
        for (size_t i = 0; i < working.size(); ++i) {
            Undo u;
            p.make(moves[i], u);
            working[i] = -negamax(p, d - 1, -INF, INF, 1);
            p.unmake(moves[i], u);
            if (aborted_) {
                complete = false;
                break;
            }
        }
        if (!complete) break;
        scores = working;
        completedDepth = d;
    }

    int best = -INF;
    for (int s : scores) best = std::max(best, s);

    ++stats_.searches;
    stats_.last_score = best;
    stats_.last_root_moves = int(scores.size());
    stats_.last_depth = completedDepth;
    stats_.last_hit_node_cap = aborted_;

    if (cfg_.randomness <= 0.0) {
        for (size_t i = 0; i < scores.size(); ++i)
            if (scores[i] == best) return moveToUci(moves[i]);
        return moveToUci(moves[0]);
    }

    // controlled stochastic selection among the near-best moves
    const double margin = cfg_.randomness * 120.0;
    const double temp = std::max(1.0, cfg_.randomness * 70.0);

    std::vector<size_t> cand;
    std::vector<double> weights;
    double total = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (double(best - scores[i]) > margin) continue;
        const double w = std::exp(double(scores[i] - best) / temp);
        cand.push_back(i);
        weights.push_back(w);
        total += w;
    }
    if (cand.empty() || total <= 0.0) {
        for (size_t i = 0; i < scores.size(); ++i)
            if (scores[i] == best) return moveToUci(moves[i]);
        return moveToUci(moves[0]);
    }

    double r = nextRandom() * total;
    for (size_t k = 0; k < cand.size(); ++k) {
        r -= weights[k];
        if (r <= 0.0) return moveToUci(moves[cand[k]]);
    }
    return moveToUci(moves[cand.back()]);
}

std::vector<std::string> Engine::legalMovesUci(const std::string& fen) {
    Position p;
    std::vector<std::string> out;
    if (!p.setFen(fen)) return out;
    std::vector<Move> moves;
    p.generateLegal(moves);
    out.reserve(moves.size());
    for (const Move& m : moves) out.push_back(moveToUci(m));
    return out;
}

}  // namespace tiny
