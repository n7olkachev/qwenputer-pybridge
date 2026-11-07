#!/bin/bash
# One-shot setup + demo + benchmark on a rented CUDA box (L4 / A10 / 4090, 24 GB; Ubuntu).
# Expects this repo checked out (or rsync'd) as ~/qwenputer-pybridge with build/mpy-bridge.elf.
#   bash gpu/setup_l4.sh            # setup, download the checkpoint, the strawberry demo
#   bash gpu/setup_l4.sh bench      # ... then the benchmark matrix on the real checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d qwenputer ] || git clone --depth 1 https://github.com/divagr18/qwenputer
(cd qwenputer/experimental/riscv && uv sync --extra hf)
RUN="uv run --project qwenputer/experimental/riscv python"
[ -d qwenputer-rv ] || $RUN -c "from huggingface_hub import snapshot_download; snapshot_download('divagr1925/Qwen3.5-4B-Qwenputer', local_dir='qwenputer-rv')"
$RUN -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3.5-4B', 'tokenizer.json')"

# 1. the author's own calc path still works on this box (bf16 exactness check)
$RUN -m chipformer.rv.chat qwenputer-rv --firmware qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf \
    --ask "Use calc: what is 123456789 * 987654321?" 2>&1 | tail -3
# 2. the python tool, in the weights, with the IR replay check
$RUN tools/chat_py.py qwenputer-rv --ask "Use python: how many r's are in 'strawberry'?" --check --out build/demo_strawberry.json 2>&1 | tail -12
$RUN tools/chat_py.py qwenputer-rv --ask "Use python: sort these words alphabetically: pear apple fig banana" --out build/demo_sort.json 2>&1 | tail -8

if [ "${1:-}" = "bench" ]; then
  for task in count_char word_sorting dyck_languages boolean_expressions reverse nth_char date_understanding multistep_arithmetic_two swap_chars object_counting; do
    $RUN bench/run_gpu.py --mode weights --model qwenputer-rv --task $task --n ${N:-50} 2>&1 | tail -1
    $RUN bench/run_gpu.py --mode base --model Qwen/Qwen3.5-4B --task $task --n ${N:-50} 2>&1 | tail -1
  done
  python3 bench/summarize.py
fi
