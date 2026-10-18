    // ---------------------------------------------------------------- fast core: quiescence (leto-rapido)
    // The same rules as search() - stand pat 0, every evasion in check, mate at the horizon, the promotion
    // and quiet-check options - with the moves in a stack array and the best one picked as it is needed
    // (MVV-LVA), instead of a vector and a stable sort of all of them.
    double qs(const Pos& p, int depth, double alpha, double beta) {
        if (++nodes > limit) { aborted = true; return 0.0; }
        const int king = p.king(p.turn);
        const bool check = king >= 0 && p.attacked(p.turn ^ 1, king, p.occupied());
        int moves[256];
        int n = 0;
        if (check) {
            if (depth == 0) return p.any_legal() ? 0.0 : -MATE;
            const Pos::Guard g = p.guard();
            p.each_candidate(g, [&](int m) {
                if (p.legal_by(g, m)) moves[n++] = m;
                return n < 256;
            });
            if (!n) return -MATE;
        } else {
            if (depth == 0) return 0.0;
            if ((0.0 > alpha ? 0.0 : alpha) >= beta) return 0.0;
            if (popcnt(p.occ[p.turn]) <= 16) {
                n = int(captures_into(p, moves) - moves);
            } else {
                p.each_pseudo([&](int m) {
                    if (p.occ[p.turn ^ 1] & bit(p.captured_square(m))) moves[n++] = m;
                    return n < 256;
                });
            }
            const bool checks = quiet_checks && depth == top, extra = promotions || checks;
            if (n || extra) {
                const Pos::Guard g = p.guard();
                int w = 0;
                for (int i = 0; i < n; ++i)
                    if (p.legal_by(g, moves[i])) moves[w++] = moves[i];
                n = w;
                if (checks) {                              // rare: the vector version
                    std::vector<int> more;
                    add_quiet(p, g, more, true);
                    for (int m : more) if (n < 256) moves[n++] = m;
                } else if (promotions) {
                    const int us = p.turn, step = us == WHITE ? 8 : -8;
                    const U64 empty = ~p.occupied();
                    for (U64 seventh = p.bb[PAWN] & p.occ[us] & (us == WHITE ? 0x00FF000000000000ULL : 0xFF00ULL);
                         seventh; seventh &= seventh - 1) {
                        const int from = lsb(seventh), to = from + step;
                        if (empty & bit(to))
                            for (int t : {QUEEN, ROOK, BISHOP, KNIGHT})
                                if (n < 256 && p.legal_by(g, mv(from, to, t))) moves[n++] = mv(from, to, t);
                    }
                }
            }
        }
        double gain[256], rank[256];
        for (int i = 0; i < n; ++i) {
            gain[i] = move_gain(p, moves[i]);
            rank[i] = 1000.0 * gain[i] - values[std::min(p.type_at(mfrom(moves[i])), int(QUEEN))];
        }
        double best = check ? -std::numeric_limits<double>::infinity() : 0.0;
        if (best > alpha) alpha = best;
        if (alpha >= beta) return best;
        for (int i = 0; i < n; ++i) {
            int top_i = i;
            for (int j = i + 1; j < n; ++j)
                if (rank[j] > rank[top_i]) top_i = j;
            if (top_i != i) {
                std::swap(moves[i], moves[top_i]);
                std::swap(gain[i], gain[top_i]);
                std::swap(rank[i], rank[top_i]);
            }
            Pos next = p;
            next.make(moves[i]);
            double score = gain[i] - qs(next, depth - 1, gain[i] - beta, gain[i] - alpha);
            if (aborted) return best;
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
        return best;
    }
