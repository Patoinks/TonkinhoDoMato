// TinyOpponent - a deliberately small chess opponent for TonkAI.
//
// Design rules (see README):
//   * correctness first: file/rank arithmetic everywhere, no clever bit tricks
//   * tiny alpha-beta, configurable depth + node cap
//   * material + piece-square static eval
//   * controlled randomness at the root, deterministic under a seed
//
// None of the knowledge in here is ever exposed to TonkAI. It is only an
// opponent.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tiny {

// piece encoding: bit 3 is the colour, bits 0-2 the type
enum : int {
    EMPTY = 0,
    WP = 1, WN = 2, WB = 3, WR = 4, WQ = 5, WK = 6,
    BP = 9, BN = 10, BB = 11, BR = 12, BQ = 13, BK = 14
};

inline int pieceType(int p) { return p & 7; }
inline int pieceColor(int p) { return p >> 3; }  // 0 = white, 1 = black

// squares are 0 = a1 .. 63 = h8, matching python-chess
inline int fileOf(int sq) { return sq & 7; }
inline int rankOf(int sq) { return sq >> 3; }
inline int makeSquare(int file, int rank) { return rank * 8 + file; }

enum : uint8_t {
    FLAG_EP = 1,
    FLAG_CASTLE = 2,
    FLAG_DOUBLE_PUSH = 4
};

struct Move {
    uint8_t from = 0;
    uint8_t to = 0;
    uint8_t promo = 0;   // promoted piece TYPE (WN..WQ), 0 when not a promotion
    uint8_t flags = 0;
};

struct Undo {
    int8_t captured = EMPTY;
    uint8_t castling = 0;
    int8_t ep = -1;
    int16_t halfmove = 0;
};

// castling right bits
enum : uint8_t { CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8 };

// No legal chess position has more than 218 legal moves.
enum : int { MAX_MOVES = 256 };

struct Position {
    int8_t sq[64];
    int8_t kingSq[2] = {-1, -1};  // cached: scanning 64 squares for the king
                                  // on every legality test dominated the search
    bool whiteToMove = true;
    uint8_t castling = 0;
    int8_t ep = -1;          // en-passant target square, -1 when none
    int16_t halfmove = 0;
    int16_t fullmove = 1;

    void clear();
    void setStartPos();
    bool setFen(const std::string& fen);
    std::string fen() const;

    int kingSquare(int color) const { return kingSq[color]; }
    bool isAttacked(int square, int byColor) const;
    bool inCheck(int color) const;

    // Fast paths: fill a caller-owned array, return the count. No allocation,
    // which matters because these run at every single search node.
    int generatePseudoInto(Move* out, bool capturesOnly = false) const;
    int generateLegalInto(Move* out, bool capturesOnly = false) const;

    // Convenience wrappers for perft and the bindings.
    void generatePseudo(std::vector<Move>& out) const;
    void generateLegal(std::vector<Move>& out) const;

    void make(const Move& m, Undo& u);
    void unmake(const Move& m, const Undo& u);

    int evaluate() const;  // centipawns, from WHITE's point of view
};

std::string moveToUci(const Move& m);
uint64_t perft(Position& p, int depth);

struct Config {
    int depth = 2;
    int max_nodes = 1000;
    double randomness = 0.35;
    uint64_t seed = 42;
    bool quiescence = true;
};

struct SearchStats {
    uint64_t nodes = 0;
    uint64_t searches = 0;
    int last_score = 0;
    int last_root_moves = 0;
    int last_depth = 0;      // depth actually completed at the root
    bool last_hit_node_cap = false;
};

class Engine {
public:
    explicit Engine(const Config& cfg);

    // returns a UCI move string, or "0000" when there is no legal move
    std::string chooseMoveUci(const std::string& fen);
    std::vector<std::string> legalMovesUci(const std::string& fen);

    void setSeed(uint64_t seed);
    void setDepth(int d) { cfg_.depth = d; }
    void setMaxNodes(int n) { cfg_.max_nodes = n; }
    void setRandomness(double r) { cfg_.randomness = r; }
    void setQuiescence(bool q) { cfg_.quiescence = q; }

    int depth() const { return cfg_.depth; }
    int maxNodes() const { return cfg_.max_nodes; }
    double randomness() const { return cfg_.randomness; }
    bool quiescence() const { return cfg_.quiescence; }

    const SearchStats& stats() const { return stats_; }
    void resetStats() { stats_ = SearchStats(); }

private:
    int negamax(Position& p, int depth, int alpha, int beta, int ply);
    int quiesce(Position& p, int alpha, int beta, int ply);
    double nextRandom();

    Config cfg_;
    SearchStats stats_;
    uint64_t rng_ = 42;
    uint64_t nodeBudget_ = 0;
    bool aborted_ = false;
};

}  // namespace tiny
