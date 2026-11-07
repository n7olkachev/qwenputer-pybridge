#!/bin/bash
# the base column: stock Qwen/Qwen3.5-4B in bf16, no tool
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/venv-riscv}
RUN="uv run --project qwenputer/experimental/riscv python"
for task in count_char reverse word_sorting nth_char multistep_arithmetic_two boolean_expressions date_understanding swap_chars object_counting dyck_languages; do
  echo "=== $task base $(date +%H:%M)"
  $RUN bench/run_gpu.py --mode base --model Qwen/Qwen3.5-4B --task $task --n ${N:-50} 2>&1 | tail -2
done
echo "=== base done $(date)"
