#!/bin/bash
# the real columns, on the pod: weights (checkpoint + python in-weights, Triton executor), then base (stock bf16)
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/venv-riscv}
RUN="uv run --project qwenputer/experimental/riscv python"
N=${N:-50}
TASKS="count_char reverse word_sorting nth_char multistep_arithmetic_two boolean_expressions date_understanding swap_chars object_counting dyck_languages"
for task in $TASKS; do
  echo "=== $task weights $(date +%H:%M)"
  $RUN bench/run_gpu.py --mode weights --model qwenputer-rv --task $task --n $N --executor 2>&1 | grep -v "^  [0-9]* tokens" | tail -3
done
echo "=== all done $(date)"
