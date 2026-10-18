"""Insert Pawnomics v4 into claudio_fast.cpp (idempotent)."""
from pathlib import Path

p = Path(__file__).with_name("claudio_fast.cpp")
t = p.read_text(encoding="utf-8")
if "pawnomics_v4.inc" not in t:
    anchor = "// ------------------------------------------------------------------ board\n"
    assert t.count(anchor) == 1
    t = t.replace(anchor, '#include "pawnomics_v4.inc"\n\n' + anchor)
    m = "    static void keep_table(bool on) { DeepSearch::cache().keep = on; }"
    assert t.count(m) == 1
    t = t.replace(m, m + """
    double pawnomics_v4(const std::vector<double>& values, bool me_white) const {
        if (values.size() != 6) throw std::invalid_argument("pawnomics_v4 requires six values");
        return pawnomics4::balance(pos, me_white ? WHITE : BLACK, values);
    }
    std::vector<double> pawnomics_v4_after_moves(const std::vector<int>& codes, const std::vector<double>& values,
                                                 bool me_white) const {
        if (values.size() != 6) throw std::invalid_argument("pawnomics_v4 requires six values");
        std::vector<double> out;
        out.reserve(codes.size());
        for (int m : codes) {
            Pos after = pos;
            after.make(m);
            out.push_back(pawnomics4::balance(after, me_white ? WHITE : BLACK, values));
        }
        return out;
    }""")
    i = t.index('.def_static("table_owner"')
    t = t[:i] + ('.def("pawnomics_v4", &Board::pawnomics_v4)\n'
                 '        .def("pawnomics_v4_after_moves", &Board::pawnomics_v4_after_moves)\n        ') + t[i:]
    p.write_text(t, encoding="utf-8")
print("ok")
