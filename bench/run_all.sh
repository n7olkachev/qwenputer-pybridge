#!/bin/bash
# The whole matrix on the local Ollama model: every task x {direct, python-ref, python-host}.
#   bench/run_all.sh [N=50] [MODEL=qwen3.5:4b]
cd "$(dirname "$0")/.."
N=${1:-50}; MODEL=${2:-qwen3.5:4b}
RUN="uv run --project qwenputer/experimental/riscv --no-sync python bench/run.py"
for task in count_char word_sorting dyck_languages boolean_expressions reverse nth_char date_understanding multistep_arithmetic_two swap_chars object_counting; do
  for mode in python-ref direct python-host; do
    echo "=== $task $mode $(date +%H:%M)"
    $RUN --task $task --mode $mode --model $MODEL --n $N 2>&1 | tail -1
  done
done
