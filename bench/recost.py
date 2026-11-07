"""Replay the calls recorded in python-ref result files on a given firmware and rewrite their
cost fields (instructions, trace tokens) and outputs. The model's code is recorded per item, and
the machine is deterministic, so costs for a new firmware build need no model calls.

usage: uv run --project qwenputer/experimental/riscv python bench/recost.py [--elf build/mpy-bridge.elf]
           [bench/results/*.python-ref.*.jsonl]
Rewrites the files in place (keeping a .bak) and their .summary.json; reports any output that
changed (none expected from a parser-only change).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from run import RefExecutor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--elf", default=str(ROOT / "build/mpy-bridge.elf"))
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted((ROOT / "bench/results").glob("*.python-ref.*.jsonl"))
    ex = RefExecutor(a.elf)
    for f in files:
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        changed = 0
        for r in rows:
            ses = ex.new_session()
            for c in r["calls"]:
                out = ex.run(ses, c["code"], c.get("data"))
                if out["output"] != c["output"]:
                    changed += 1
                    print(f"  {r['id']}: output changed: {c['output']!r} -> {out['output']!r}")
                c.update(out)
            r["instructions"] = sum(c["instructions"] or 0 for c in r["calls"])
            r["trace_tokens"] = sum(c["trace_tokens"] or 0 for c in r["calls"])
        f.with_suffix(".jsonl.bak").write_text(f.read_text())
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        sp = Path(str(f).replace(".jsonl", ".summary.json"))
        if sp.exists():
            s = json.loads(sp.read_text())
            n = len(rows)
            s["mean_instructions"] = round(sum(r["instructions"] for r in rows) / n)
            s["mean_trace_tokens"] = round(sum(r["trace_tokens"] for r in rows) / n)
            s["firmware"] = Path(a.elf).name
            sp.write_text(json.dumps(s, indent=1))
        print(f"{f.name}: {len(rows)} items re-costed on {Path(a.elf).name}, "
              f"mean {sum(r['instructions'] for r in rows) / len(rows):,.0f} instructions, "
              f"{changed} outputs changed")


if __name__ == "__main__":
    main()
