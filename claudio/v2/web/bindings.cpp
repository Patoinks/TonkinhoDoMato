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
