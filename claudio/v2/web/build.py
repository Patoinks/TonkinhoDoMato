"""Build the pinned G16 v2 C++ rules for browsers, changing only the bindings.

python claudio/v2/web/build.py --emsdk <sdk> --out <assets>
The other agent's native source is read, never modified. No training runs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def remove_method(source, marker):
    start = source.index(marker)
    opening = source.index('{', start)
    level, end = 1, opening + 1
    while level:
        level += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[:start] + source[end:]


def generate():
    path = ROOT / 'claudio/fast/claudio_fast.cpp'
    core = path.read_text(encoding='utf-8').split('PYBIND11_MODULE(')[0]
    core = re.sub(r'#include <pybind11/[^>]+>\n', '', core)
    core = core.replace('namespace py = pybind11;', '')
    a, b = core.index('// ------------------------------------------------------------------ pieces'), core.index('// ------------------------------------------------------------------ encoding')
    core = core[:a] + core[b:]
    for marker in ('    py::dict king_detail()', '    std::pair<double, double> piece_risks_py(',
                   '    py::array_t<std::uint8_t> capture_targets(', '    py::dict activity(',
                   '    py::tuple expand2(', '    py::array_t<float> encode_here(', '    py::array_t<float> encode_after('):
        core = remove_method(core, marker)
    core = core.replace('class Board {', 'Pos parse_position(const std::string& fen);\nclass Board {')
    core = core.replace('public:\n    Board(', 'public:\n    explicit Board(const std::string& fen) : pos(parse_position(fen)) {}\n    Board(', 1)
    # Raw encodings are unchanged: same encode_row, floats and EP fields.
    core = core.replace('    // (square, piece type, colour)', '''    std::vector<float> encode_here(bool white) const {
        std::vector<float> out(782, 0); encode_row(pos, white ? WHITE : BLACK, out.data()); return out;
    }
    std::vector<std::vector<float>> encode_after(const std::vector<int>& codes, bool white) const {
        std::vector<std::vector<float>> out;
        for (int code : codes) { Pos next = pos; next.make(code); std::vector<float> row(782, 0);
            encode_row(next, white ? WHITE : BLACK, row.data()); out.push_back(std::move(row)); }
        return out;
    }
    // (square, piece type, colour)''')
    extra_path = ROOT / 'claudio/v2/native/extra.cpp'
    extra = extra_path.read_text(encoding='utf-8')
    extra = extra[extra.index('namespace {'):extra.index('py::dict mating_plan(')]
    extra = extra.replace('BoardV2', 'Board')
    extra += '}\n'
    bindings = (Path(__file__).parent / 'bindings.cpp').read_text(encoding='utf-8')
    return '#include <emscripten/bind.h>\n#include <unordered_set>\n#include <unordered_map>\n#include <sstream>\n#include <memory>\n' + core + extra + bindings, {
        'native_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'v2_native_sha256': hashlib.sha256(extra_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emsdk', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code, provenance = generate()
    generated = args.out / 'g16-rules.cpp'
    generated.write_text(code, encoding='utf-8')
    compiler = args.emsdk / 'upstream/emscripten/em++.py'
    subprocess.run([sys.executable, str(compiler), str(generated), '-o', str(args.out / 'g16-rules.mjs'),
                    '-std=c++20', '-O3', '-ffp-contract=off', '--bind', '-sMODULARIZE=1',
                    '-sEXPORT_ES6=1', '-sALLOW_MEMORY_GROWTH=1', '-sENVIRONMENT=web,node',
                    '-sSTACK_SIZE=8388608', '-sINITIAL_MEMORY=67108864', '-sDISABLE_EXCEPTION_CATCHING=0'], check=True)
    provenance['wasm_sha256'] = hashlib.sha256((args.out / 'g16-rules.wasm').read_bytes()).hexdigest()
    provenance['loader_sha256'] = hashlib.sha256((args.out / 'g16-rules.mjs').read_bytes()).hexdigest()
    (args.out / 'native-provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
