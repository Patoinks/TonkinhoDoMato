"""Optional contextual material in the generated browser engine only.

The neural model is evaluated once at the real root. Compact identity bytes
carry each root piece's own value through the existing bounded C++ searches.
Promotion values come from that same neural context at the actual destination.
Rule/repetition hashes remain pure chess; only the material TT includes identity.
"""
import hashlib


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Pinned native source changed: {before[:70]}')
    return source.replace(before, after, 1)


def apply(source):
    source=once(source,'struct Pos {','''static bool web_context_on = false;
static std::array<double,128> web_context_values{};
struct Pos {
    std::array<std::uint8_t,64> web_ids{};''')
    source=once(source,'        int type = type_at(from), previous_ep = ep;', '''        int type = type_at(from), previous_ep = ep;
        if (web_context_on) {
            web_ids[to] = promo ? std::uint8_t(64 + us*32 + (promo-2)*8 + file_of(to)) : web_ids[from];
            if (type == KING && std::abs(file_of(to)-file_of(from)) == 2) {
                int base = rank_of(from)*8;
                web_ids[base + (file_of(to)==6 ? 5 : 3)] = web_ids[base + (file_of(to)==6 ? 7 : 0)];
            }
        }''')
    source=once(source,'// ------------------------------------------------------------------ king','''// Material identity matters to this TT when equal-type pieces exchange squares.
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
// ------------------------------------------------------------------ king''')
    source=once(source,'    HOT double move_gain(const Pos& p, int m) const {','''    HOT double move_gain(const Pos& p, int m) const {
        if (web_context_on) {
            int captured=p.captured_square(m), promo=mpromo(m);
            double gain=(p.occ[p.turn^1]&bit(captured)) ? web_piece_value(p,captured,values) : 0.0;
            if (promo) gain += web_context_values[64+p.turn*32+(promo-2)*8+file_of(mto(m))]
                               - web_piece_value(p,mfrom(m),values);
            return gain;
        }''')
    source=source.replace('values[std::min(p.type_at(mfrom(moves[i])), int(QUEEN))]', 'web_piece_value(p,mfrom(moves[i]),values)')
    source=source.replace('q.values[std::min(p.type_at(mfrom(m)), int(QUEEN))]', 'web_piece_value(p,mfrom(m),q.values)')
    # The three material-search key reads; do not touch repetition or mate keys.
    if source.count('key_of(p).hash()') != 4: raise ValueError('Unexpected material key calls')
    start=source.index('struct DeepSearch {')
    source=source[:start]+source[start:].replace('key_of(p).hash()', 'web_material_key(p)')
    source=once(source,'public:\n    explicit Board(','''public:
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
    explicit Board(''')
    return source


def digest():
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
