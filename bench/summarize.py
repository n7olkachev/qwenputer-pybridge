"""bench/results/*.summary.json -> bench/RESULTS.md: accuracy per task x mode, with the tool's
cost (mean calls, instructions and trace tokens per item, L4 seconds at 4K instructions/s).

usage: python bench/summarize.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
MODES = ["direct", "python-ref", "python-host", "base", "weights", "weights-retok"]
ORDER = ["count_char", "reverse", "nth_char", "swap_chars", "word_sorting", "dyck_languages",
         "boolean_expressions", "date_understanding", "multistep_arithmetic_two", "object_counting"]
LABEL = {"direct": "Base (no tool, Ollama Q4)", "python-ref": "python tool, in-weights env (ref machine, Ollama Q4)",
         "python-host": "python tool, host CPython (Ollama Q4)",
         "base": "Base bf16, no tool (GPU)", "weights": "Qwenputer + python in-weights (GPU)",
         "weights-retok": "same, responses re-tokenized (GPU)"}


def main() -> None:
    rows = {}
    for p in sorted((ROOT / "results").glob("*.summary.json")):
        s = json.loads(p.read_text())
        rows.setdefault(s["task"], {})[s["mode"]] = s
    models = {s["model"] for r in rows.values() for s in r.values()}
    ns = {s["n"] for r in rows.values() for s in r.values()}
    out = ["# Results", "",
           f"Model: {', '.join(sorted(models))} via Ollama (Q4_K_M, non-thinking, greedy). "
           f"Items per cell: {', '.join(str(n) for n in sorted(ns))}. "
           "Accuracy is exact match after task-specific normalization (`bench/tasks.py`). "
           "GPU columns (`base`, `weights`) appear once bench/run_gpu.py has been run.", "",
           "| Task | " + " | ".join(LABEL[m] for m in MODES) + " | tool calls / item | instructions / item | trace tokens / item | ≈ L4 s / item |",
           "| --- | " + " | ".join("---" for _ in MODES) + " | --- | --- | --- | --- |"]
    for task in ORDER + sorted(set(rows) - set(ORDER)):
        if task not in rows:
            continue
        r = rows[task]
        cells = []
        for m in MODES:
            cells.append(f"{100 * r[m]['accuracy']:.0f}%" if m in r else "–")
        ref = r.get("python-ref")
        if ref:
            cost = [f"{ref['mean_calls']:.2f}", f"{ref['mean_instructions']:,}",
                    f"{ref['mean_trace_tokens']:,}", f"{ref['mean_instructions'] / 4000:.0f}"]
        else:
            cost = ["–"] * 4
        out.append(f"| {task} | " + " | ".join(cells) + " | " + " | ".join(cost) + " |")
    out += ["", "Cost columns are for the in-weights environment: the model's calls executed on the "
            "reference machine with the real firmware; seconds assume the author's executor "
            "(~4K instructions/s on an L4)."]
    (ROOT / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
