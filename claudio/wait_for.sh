# usage: bash claudio/wait_for.sh <file> [timeout_seconds]  - returns when the file exists
for i in $(seq 1 ${2:-50}); do [ -s "$1" ] && exit 0; sleep 1; done; exit 1
