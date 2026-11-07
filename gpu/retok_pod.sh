#!/bin/bash
# the re-tokenized-response variant (tools/retok.py) on the string-answer tasks
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/venv-riscv}
RUN="uv run --project qwenputer/experimental/riscv python"
echo "=== retok demo $(date +%H:%M)"
$RUN tools/chat_py.py qwenputer-rv --executor --retokenize --ask "Use python: reverse the string 'davenportwalkie'." --out build/demo_retok.json 2>&1 | grep -v "^  [0-9]* tokens" | tail -12
for task in reverse swap_chars word_sorting dyck_languages; do
  echo "=== $task weights-retok $(date +%H:%M)"
  $RUN bench/run_gpu.py --mode weights-retok --model qwenputer-rv --task $task --n ${N:-50} --executor 2>&1 | grep -v "^  [0-9]* tokens" | tail -2
done
echo "=== retok done $(date)"
