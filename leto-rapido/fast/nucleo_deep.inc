    // ---------------------------------------------------------------- fast core: deep search (leto-rapido)
    // search() with the same rules (table, PVS, null move, late move reductions, killers, history, check
    // extensions, the frontier shortcut, exchange depth by square) and three changes for speed: the key is a
    // Zobrist key kept move by move, the moves live in a stack array (no deque of vectors), and a countermove
    // (the quiet answer that last refuted the opponent's move) is tried after the killers. Order among equal
    // ranks is no longer the generation order, so its values are not bit for bit the original's.
    int counter[2][64][64] = {};

    double search_fast(const Pos& p, U64 key, int depth, double alpha, double beta, int ply, bool allow_null, int prev) {
        ++node_total();
        if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }
        if (depth <= 0) {
            int d = (dynamic && ply < 256 && dest[ply] >= 0) ? exchange_plies(p, dest[ply]) : qdepth;
            return q.qs(p, d, alpha, beta);
        }
        Entry16& e = slot16(key);
        int tt_move = 0;
        if (e.key == (key ^ salt)) {
            tt_move = e.move;
            if (e.depth >= depth) {
                if (e.flag == EXACT) return e.value;
                if (e.flag == LOWER && e.value >= beta) return e.value;
                if (e.flag == UPPER && e.value <= alpha) return e.value;
            }
        }
        const Pos::Guard guard = p.guard();
        const bool check = guard.checkers != 0;
        int moves[Pos::BUFFER];
        int n = 0;
        if (guard.king >= 0 && guard.plain && check)
            p.each_evasion(guard, [&](int m) { moves[n++] = m; return n < Pos::BUFFER; });
        else
            p.each_pseudo([&](int m) { moves[n++] = m; return n < Pos::BUFFER; });
        if (!n) return check ? -(ExchangeSearch::MATE - ply) : 0.0;
        const double eps = 1e-6;
        if (!exact && allow_null && !check && depth >= 3 && has_pieces(p)) {
            Pos null = p;
            null.turn ^= 1;
            null.ep = -1;
            if (ply + 1 < 256) dest[ply + 1] = -1;
            const int null_r = (aggressive_prunings() && depth >= 6) ? 4 : 3;
            double score = -search_fast(null, key ^ Z.side ^ z_ep(p), depth - null_r, -beta, -beta + eps, ply + 1, false, 0);
            if (q.aborted) return 0.0;
            if (score >= beta) return score;
        }
        const int k = std::min(ply, 127), side = p.turn;
        const int answer = prev ? counter[side][mfrom(prev)][mto(prev)] : 0;
        double rank[Pos::BUFFER], gains[Pos::BUFFER];
        for (int i = 0; i < n; ++i) {
            const int m = moves[i];
            const double gain = q.move_gain(p, m);
            gains[i] = gain;
            rank[i] = m == tt_move ? 1e9
                    : gain > 0 ? 1e6 + 1000.0 * gain - q.values[std::min(p.type_at(mfrom(m)), int(QUEEN))]
                    : m == killers[k][0] ? 9e5
                    : m == killers[k][1] ? 8e5
                    : m == answer ? 7e5
                    : double(history[side][mfrom(m)][mto(m)]);
        }
        // One packed array (rank, index): the best few by selection (most nodes are cut after one or two
        // moves), the rest sorted once; moves and gains stay where they were generated.
        std::uint64_t order[Pos::BUFFER];
        for (int i = 0; i < n; ++i) order[i] = (std::uint64_t(rank[i] * 1024.0) << 20) | std::uint64_t(Pos::BUFFER - 1 - i);
        int sorted = 0;
        auto place = [&](int j) {
            if (j < sorted) return;
            if (j < 3) {
                int top_i = j;
                for (int i = j + 1; i < n; ++i)
                    if (order[i] > order[top_i]) top_i = i;
                std::swap(order[j], order[top_i]);
                sorted = j + 1;
                return;
            }
            std::sort(order + j, order + n, std::greater<std::uint64_t>());
            sorted = n;
        };
        double best = -std::numeric_limits<double>::infinity(), start = alpha;
        int best_move = 0;
        int searched = 0;
        const bool frontier = depth == 1;
        const int enemy_king = p.king(p.turn ^ 1);
        const bool direct = frontier && enemy_king >= 0 && p.single_king(p.turn) &&
                            !p.attacked(p.turn, enemy_king, p.occupied());
        const U64 disc = direct ? p.discoverers(enemy_king) : 0;
        Pos next;
        U64 next_key = 0;
        for (int j = 0; j < n; ++j) {
            place(j);
            const int at = Pos::BUFFER - 1 - int(order[j] & 0xFFFFF);
            const int m = moves[at];
            bool made = false, calm = false;
            if (!p.legal_by(guard, m)) continue;
            auto make_next = [&]() {
                next = p;
                next.make(m);
                next_key = z_after(p, next, key);
                _mm_prefetch(reinterpret_cast<const char*>(&slot16(next_key)), _MM_HINT_T0);   // the child's entry
                made = true;
            };
            if (frontier)
                calm = enemy_king < 0 || (direct && !MateSearch::may_check(p, m, enemy_king, disc));
            else
                make_next();
            int ext = 0;
            if (extensions_left > 0 && !calm) {
                if (!made) make_next();
                ext = next.is_check() ? 1 : 0;
            }
            auto child = [&](int d, double a, double b) {
                d += ext;
                if (calm && d <= 0 && (qdepth == 0 || (0.0 > a ? 0.0 : a) >= b)) {
                    if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }
                    if (++q.nodes > q.limit) { q.aborted = true; return 0.0; }
                    return 0.0;
                }
                if (!made) make_next();
                extensions_left -= ext;
                if (ply + 1 < 256) dest[ply + 1] = mto(m);
                double value = search_fast(next, next_key, d, a, b, ply + 1, true, m);
                extensions_left += ext;
                return value;
            };
            const int i = searched++;
            if (i == 0) best_move = m;
            const double gain = gains[at];
            const bool quiet = gain == 0.0 && !mpromo(m);
            int reduction = 0;
            if (!exact && depth >= 3 && i >= 3 && quiet && !check) {
                if (!made) make_next();
                reduction = next.is_check() ? 0 : 1;
            }
            if (reduction && aggressive_prunings() && depth >= 5 && i >= 6) reduction = 2;
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
                    if (prev) counter[side][mfrom(prev)][mto(prev)] = m;
                }
                break;
            }
        }
        if (!searched) return check ? -(ExchangeSearch::MATE - ply) : 0.0;
        e.key = key ^ salt;
        e.value = float(best);
        e.depth = std::int8_t(std::min(depth, 127));
        e.move = std::uint16_t(best_move);
        e.flag = best <= start ? UPPER : best >= beta ? LOWER : EXACT;
        return best;
    }
