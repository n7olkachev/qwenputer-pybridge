#!/bin/bash
# the two python columns on the local Ollama model (direct results are kept)
cd "$(dirname "$0")/.."
N=${1:-50}; MODEL=${2:-qwen3.5:4b}
RUN="uv run --project qwenputer/experimental/riscv --no-sync python bench/run.py"
for task in reverse word_sorting swap_chars count_char dyck_languages boolean_expressions nth_char date_understanding multistep_arithmetic_two object_counting; do
  for mode in python-ref python-host; do
    echo "=== $task $mode $(date +%H:%M)"
    $RUN --task $task --mode $mode --model $MODEL --n $N 2>&1 | tail -1
  done
done
python3 bench/summarize.py > build/summarize.log 2>&1
