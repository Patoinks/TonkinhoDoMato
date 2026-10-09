// Extend the pinned native rules without modifying the other agent's source.
#include <pybind11/pybind11.h>
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) void register_base(pybind11::module_& module)
#define Board BoardV2
#include "../../fast/claudio_fast.cpp"
#undef Board
#pragma pop_macro("PYBIND11_MODULE")
#include <unordered_set>
#include <unordered_map>
#include <sstream>

namespace {
Pos parse_position(const std::string& fen);
bool dead_pawn_lock(const Pos& p);
std::string position_key(const std::string& fen) {
    size_t end = fen.rfind(' ');
    if (end == std::string::npos) throw std::invalid_argument("invalid FEN");
    end = fen.rfind(' ', end - 1);
    return fen.substr(0, end);
}

std::pair<std::vector<int>, std::vector<bool>> endings(BoardV2& board, const std::vector<int>& codes,
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
        BoardV2 b(p.bb[PAWN], p.bb[KNIGHT], p.bb[BISHOP], p.bb[ROOK], p.bb[QUEEN], p.bb[KING],
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

py::dict mating_plan(const std::string& fen, const std::vector<std::string>& previous,
                     int max_moves, long long nodes) {
    if (max_moves < 1 || max_moves > 16 || nodes < 0) throw std::invalid_argument("invalid mate budget");
    Pos p = parse_position(fen);
    MatePlanner search(p.turn, nodes, previous);
    int found = 0, move = 0, proven = 0;
    for (int n = 1; n <= max_moves; ++n) {
        auto answer = search.solve(p, n, true);
        if (search.aborted) break;
        proven = n;
        if (answer.winning) { found = n; move = answer.move; break; }
    }
    py::dict result;
    result["moves"] = found; result["move"] = move; result["nodes"] = search.nodes;
    result["proven"] = proven; result["aborted"] = search.aborted;
    result["states"] = search.solved.size();
    return result;
}
}

PYBIND11_MODULE(claudio_fast_v2, m) {
    register_base(m);
    py::object cls = m.attr("Board");
    cls.attr("terminal_repetition_after_moves") = py::cpp_function(&endings, py::is_method(cls),
        py::arg("codes"), py::arg("previous"), py::arg("check_dead") = true);
    m.def("mating_plan", &mating_plan, py::arg("fen"), py::arg("previous"),
          py::arg("max_moves") = 8, py::arg("nodes") = 2000000);
    m.def("dead_pawn_lock", [](const std::string& fen) { return dead_pawn_lock(parse_position(fen)); });
}
