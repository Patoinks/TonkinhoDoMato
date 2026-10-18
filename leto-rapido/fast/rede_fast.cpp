// leto-rapido: the Leto network's decision-time pieces (encoder, GRU memory, fusion, survival and predict heads,
// king safety) in C++, compiled for this machine (AVX-512 on Zen 4). One call per batch instead of numpy's
// ~25 operations; the input planes are mostly zero and each layer skips zero inputs (y += x_k * W_k).
// float32 like rede_rapida.py; the summation order differs, so results match to float32 rounding.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <immintrin.h>
#include <malloc.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

struct Dense {
    int in = 0, out = 0;
    std::vector<float> wt, b;           // wt[k * out + j] = W[j][k]
    Dense() = default;
    Dense(const Array& w, const Array& bias) {
        if (w.ndim() != 2 || bias.ndim() != 1 || bias.shape(0) != w.shape(0)) throw std::invalid_argument("bad layer");
        out = int(w.shape(0)); in = int(w.shape(1));
        wt.resize(size_t(in) * out); b.assign(bias.data(), bias.data() + out);
        const float* p = w.data();
        for (int j = 0; j < out; ++j)
            for (int k = 0; k < in; ++k) wt[size_t(k) * out + j] = p[size_t(j) * in + k];
    }
    // y (rows x out) = x (rows x in, row stride sx) W^T + b. AVX-512: 4 rows x 64 columns of accumulators
    // (16 registers) over all inputs, an input skipped when it is zero in all 4 rows.
    void apply(const float* x, int rows, int sx, float* y) const {
        const float* __restrict w = wt.data();
        if (out % 64) {                              // small outputs (heads): plain loops
            for (int r = 0; r < rows; ++r) {
                float* yr = y + size_t(r) * out;
                const float* xr = x + size_t(r) * sx;
                for (int j = 0; j < out; ++j) yr[j] = b[j];
                for (int k = 0; k < in; ++k) {
                    const float a = xr[k];
                    if (a != 0.0f)
                        for (int j = 0; j < out; ++j) yr[j] += a * w[size_t(k) * out + j];
                }
            }
            return;
        }
        // the inputs that are non-zero in some row of the batch (piece planes: ~40 of 782)
        int* ks = static_cast<int*>(alloca(sizeof(int) * size_t(in)));
        int nk = 0;
        for (int k = 0; k < in; ++k)
            for (int r = 0; r < rows; ++r)
                if (x[size_t(r) * sx + k] != 0.0f) { ks[nk++] = k; break; }
        for (int r0 = 0; r0 < rows; r0 += 4) {
            const int nr = std::min(4, rows - r0);
            const float* xr[4];
            for (int r = 0; r < 4; ++r) xr[r] = x + size_t(r0 + std::min(r, nr - 1)) * sx;
            for (int j0 = 0; j0 < out; j0 += 64) {
                __m512 acc[4][4];
                for (int c = 0; c < 4; ++c) {
                    const __m512 bias = _mm512_loadu_ps(b.data() + j0 + 16 * c);
                    for (int r = 0; r < 4; ++r) acc[r][c] = bias;
                }
                for (int q = 0; q < nk; ++q) {
                    const int k = ks[q];
                    const float a0 = xr[0][k], a1 = xr[1][k], a2 = xr[2][k], a3 = xr[3][k];
                    const float* wk = w + size_t(k) * out + j0;
                    const __m512 w0 = _mm512_loadu_ps(wk), w1 = _mm512_loadu_ps(wk + 16),
                                 w2 = _mm512_loadu_ps(wk + 32), w3 = _mm512_loadu_ps(wk + 48);
                    const __m512 v[4] = {_mm512_set1_ps(a0), _mm512_set1_ps(a1), _mm512_set1_ps(a2), _mm512_set1_ps(a3)};
                    for (int r = 0; r < 4; ++r) {
                        acc[r][0] = _mm512_fmadd_ps(v[r], w0, acc[r][0]);
                        acc[r][1] = _mm512_fmadd_ps(v[r], w1, acc[r][1]);
                        acc[r][2] = _mm512_fmadd_ps(v[r], w2, acc[r][2]);
                        acc[r][3] = _mm512_fmadd_ps(v[r], w3, acc[r][3]);
                    }
                }
                for (int r = 0; r < nr; ++r)
                    for (int c = 0; c < 4; ++c) _mm512_storeu_ps(y + size_t(r0 + r) * out + j0 + 16 * c, acc[r][c]);
            }
        }
    }
};

inline void silu(float* x, size_t n) {
    size_t i = 0;
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(x + i, _mm512_div_ps(v, _mm512_add_ps(one, _mm512_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), v)))));
    }
    for (; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

Array make(int rows, int cols) { return Array({rows, cols}); }

struct Net {
    Dense e1, e2, r1, r2;               // encoder: 782 -> 512 -> 128, residual 128 -> 512 -> 128
    Dense ih, hh;                        // GRU (input 128, hidden H)
    Dense fusion;                        // (latent, memory) -> context
    Dense s1;                            // survival heads, first layers stacked: context -> E*Hs
    std::vector<Dense> s2;               // each head's second layer: Hs -> 2
    Dense p1, p2;                        // predict head
    int hidden = 0;

    Net(const std::vector<Array>& w) {
        // order: e1 w,b  e2 w,b  r1 w,b  r2 w,b  ih w,b  hh w,b  fusion w,b  s1 w,b  p1 w,b  p2 w,b  then s2 w,b ...
        if (w.size() < 20 || (w.size() - 20) % 2) throw std::invalid_argument("bad weights");
        e1 = Dense(w[0], w[1]); e2 = Dense(w[2], w[3]); r1 = Dense(w[4], w[5]); r2 = Dense(w[6], w[7]);
        ih = Dense(w[8], w[9]); hh = Dense(w[10], w[11]); fusion = Dense(w[12], w[13]); s1 = Dense(w[14], w[15]);
        p1 = Dense(w[16], w[17]); p2 = Dense(w[18], w[19]);
        for (size_t i = 20; i < w.size(); i += 2) s2.emplace_back(w[i], w[i + 1]);
        hidden = hh.in;
    }

    std::vector<float> encode_raw(const float* x, int rows, int cols) const {
        if (cols != e1.in) throw std::invalid_argument("encode: bad input width");
        std::vector<float> a(size_t(rows) * e1.out), lat(size_t(rows) * e2.out), h(size_t(rows) * r1.out),
            res(size_t(rows) * r2.out);
        e1.apply(x, rows, cols, a.data());
        silu(a.data(), a.size());
        e2.apply(a.data(), rows, e1.out, lat.data());
        silu(lat.data(), lat.size());
        r1.apply(lat.data(), rows, e2.out, h.data());
        silu(h.data(), h.size());
        r2.apply(h.data(), rows, r1.out, res.data());
        for (size_t i = 0; i < lat.size(); ++i) lat[i] += res[i];
        return lat;
    }
    // memory rows: 1 (broadcast) or rows
    std::vector<float> gru_raw(const float* lat, const float* mem, int rows, int mem_rows) const {
        const int H = hidden;
        std::vector<float> gi(size_t(rows) * 3 * H), gh(size_t(mem_rows) * 3 * H), out(size_t(rows) * H);
        ih.apply(lat, rows, ih.in, gi.data());
        hh.apply(mem, mem_rows, H, gh.data());
        for (int r = 0; r < rows; ++r) {
            const float* i_ = gi.data() + size_t(r) * 3 * H;
            const int mr = mem_rows == 1 ? 0 : r;
            const float* h_ = gh.data() + size_t(mr) * 3 * H;
            const float* m = mem + size_t(mr) * H;
            float* o = out.data() + size_t(r) * H;
            for (int j = 0; j < H; ++j) {
                const float rr = sigmoid(i_[j] + h_[j]), z = sigmoid(i_[H + j] + h_[H + j]);
                const float n = std::tanh(i_[2 * H + j] + rr * h_[2 * H + j]);
                o[j] = (1.0f - z) * n + z * m[j];
            }
        }
        return out;
    }

    Array encode(const Array& x) const {
        const int rows = int(x.shape(0));
        auto lat = encode_raw(x.data(), rows, int(x.shape(1)));
        Array out = make(rows, e2.out);
        std::copy(lat.begin(), lat.end(), out.mutable_data());
        return out;
    }
    Array memory_cell(const Array& lat, const Array& mem) const {
        const int rows = int(lat.shape(0)), mem_rows = int(mem.shape(0));
        auto o = gru_raw(lat.data(), mem.data(), rows, mem_rows);
        Array out = make(rows, hidden);
        std::copy(o.begin(), o.end(), out.mutable_data());
        return out;
    }
    std::pair<Array, Array> candidates(const Array& x, const Array& mem) const {
        const int rows = int(x.shape(0)), mem_rows = int(mem.shape(0));
        if (mem_rows != 1 && mem_rows != rows) throw std::invalid_argument("candidates: memory rows");
        auto lat = encode_raw(x.data(), rows, int(x.shape(1)));
        auto after = gru_raw(lat.data(), mem.data(), rows, mem_rows);
        const int L = e2.out, H = hidden;
        std::vector<float> cat(size_t(rows) * (L + H));
        for (int r = 0; r < rows; ++r) {
            std::copy(lat.begin() + size_t(r) * L, lat.begin() + size_t(r + 1) * L, cat.begin() + size_t(r) * (L + H));
            std::copy(after.begin() + size_t(r) * H, after.begin() + size_t(r + 1) * H,
                      cat.begin() + size_t(r) * (L + H) + L);
        }
        Array context = make(rows, fusion.out), mem_out = make(rows, H);
        fusion.apply(cat.data(), rows, L + H, context.mutable_data());
        silu(context.mutable_data(), size_t(rows) * fusion.out);
        std::copy(after.begin(), after.end(), mem_out.mutable_data());
        return {context, mem_out};
    }
    std::pair<Array, Array> survival(const Array& c) const {
        const int rows = int(c.shape(0)), E = int(s2.size()), Hs = s1.out / std::max(1, E);
        std::vector<float> h(size_t(rows) * s1.out);
        s1.apply(c.data(), rows, s1.in, h.data());
        silu(h.data(), h.size());
        Array alive = make(rows, E), steps = make(rows, E);
        float* a = alive.mutable_data();
        float* s = steps.mutable_data();
        float two[2];
        for (int r = 0; r < rows; ++r)
            for (int e = 0; e < E; ++e) {
                s2[e].apply(h.data() + size_t(r) * s1.out + size_t(e) * Hs, 1, Hs, two);
                a[r * E + e] = sigmoid(two[0]);
                s[r * E + e] = std::min(2.0f, std::max(-8.0f, two[1]));
            }
        return {alive, steps};
    }
    py::array_t<float> predict_logits(const Array& c) const {
        const int rows = int(c.shape(0));
        std::vector<float> h(size_t(rows) * p1.out);
        p1.apply(c.data(), rows, p1.in, h.data());
        silu(h.data(), h.size());
        py::array_t<float> out(rows);
        float* o = out.mutable_data();
        for (int r = 0; r < rows; ++r) p2.apply(h.data() + size_t(r) * p1.out, 1, p1.out, o + r);
        return out;
    }
};

struct King {                            // 782 -> h -> h -> 1, SiLU, sigmoid
    Dense a, b, c;
    King(const std::vector<Array>& w) {
        if (w.size() != 6) throw std::invalid_argument("king: six arrays");
        a = Dense(w[0], w[1]); b = Dense(w[2], w[3]); c = Dense(w[4], w[5]);
    }
    py::array_t<float> rows(const Array& x) const {
        const int n = int(x.shape(0));
        std::vector<float> h1(size_t(n) * a.out), h2(size_t(n) * b.out);
        a.apply(x.data(), n, int(x.shape(1)), h1.data());
        silu(h1.data(), h1.size());
        b.apply(h1.data(), n, a.out, h2.data());
        silu(h2.data(), h2.size());
        py::array_t<float> out(n);
        float* o = out.mutable_data();
        for (int r = 0; r < n; ++r) {
            c.apply(h2.data() + size_t(r) * b.out, 1, b.out, o + r);
            o[r] = sigmoid(o[r]);
        }
        return out;
    }
};

}  // namespace

PYBIND11_MODULE(rede_fast, m) {
    py::class_<Net>(m, "Net")
        .def(py::init<const std::vector<Array>&>())
        .def("encode", &Net::encode)
        .def("memory_cell", &Net::memory_cell)
        .def("candidates", &Net::candidates)
        .def("survival", &Net::survival)
        .def("predict_logits", &Net::predict_logits);
    py::class_<King>(m, "King").def(py::init<const std::vector<Array>&>()).def("rows", &King::rows);
}
