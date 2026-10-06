#!/usr/bin/env bash
# Everything Claudio needs on a fresh Linux machine (e.g. a cloud session), from the repo root:
#     bash claudio/setup_cloud.sh
# Python packages, the three C++ modules (TinyOpponent, Claudio's frozen tonkai_fast, claudio_fast)
# and the two engines, built from source at the same versions as on Martim's PC:
# Stockfish 11 (judge / opponent) and Fairy-Stockfish 14 (opponent at a nominal Elo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -q -r requirements.txt

# TinyOpponent: built from the repo's source, used from Claudio's frozen base.
python setup.py build_ext --inplace >/dev/null
cp tiny_opponent*.so claudio/base/
(cd claudio/base && python setup_claudio.py build_ext --inplace >/dev/null)
(cd claudio/fast && python setup.py build_ext --inplace >/dev/null)

mkdir -p engines/linux
build_engine() {   # url, tag (grep -E pattern), output name
    local work tag
    work="$(mktemp -d)"
    tag="$(git ls-remote --tags --refs "$1" | awk '{print $2}' | sed 's#refs/tags/##' | grep -E "$2" | sort -V | tail -1 || true)"
    git clone -q --depth 1 ${tag:+--branch "$tag"} "$1" "$work"
    (cd "$work/src" && make -j"$(nproc)" build ARCH=x86-64-modern >/dev/null)
    cp "$(find "$work/src" -maxdepth 1 -type f -perm -u+x -name '*stockfish*' | head -1)" "engines/linux/$3"
    rm -rf "$work"
    echo "$3: ${tag:-master}"
}
[ -x engines/linux/stockfish-11 ] || build_engine https://github.com/official-stockfish/Stockfish.git '^sf_11$' stockfish-11
[ -x engines/linux/fairy-stockfish ] || build_engine https://github.com/fairy-stockfish/Fairy-Stockfish.git '^fairy_sf_14$' fairy-stockfish

python - <<'EOF'
import sys
sys.path.insert(0, "claudio")
import paths
import chess.engine
for name, path in (("Stockfish", paths.STOCKFISH), ("Fairy", paths.FAIRY)):
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    print(name, "->", engine.id.get("name"))
    engine.quit()
import claudio_fast, tonkai_fast, tiny_opponent  # noqa: F401
print("modulos C++ ok")
EOF
