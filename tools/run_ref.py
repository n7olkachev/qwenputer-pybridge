"""Drive the P4 MicroPython bridge firmware on Qwenputer's reference RV32 machine (the executable
spec the circuit is verified against) through the P3 bridge Session: Qwen-side tool calls in,
the machine's tool responses out, with instruction and token counts per call.

usage: uv run --project qwenputer/experimental/riscv python tools/run_ref.py [--elf build/mpy-bridge.elf]
           [--trace build/trace.json] [CODE ...]       # each CODE is one python(code=...) call
       ... --script tests/calls.py                      # calls separated by lines of "# ---"
Without CODE, runs a built-in smoke sequence (persistent globals, exceptions, the step budget).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

SMOKE = [
    "print(2**100)",
    "x = 41\ndef f(n):\n    return n + 1",
    "f(x)",                       # REPL semantics: a bare expression prints; globals persisted
    "words = 'pear apple fig banana'.split()\nprint(sorted(words))",
    "print('strawberry'.count('r'))",
    "1/0",                        # a traceback, then the next call still works
    "import math\nprint(math.gcd(84, 18), math.sqrt(2))",
    "s = 0\nwhile True:\n    s += 1",  # the step budget
    "print('alive', f(1))",
]


def call_tokens(T: BR.Text, code: str, name: str = "python", data: str | None = None) -> list[int]:
    extra = f"<parameter=data>\n{data}\n</parameter>\n" if data is not None else ""
    return T.encode(f"<tool_call>\n<function={name}>\n<parameter=code>\n{code}\n"
                    f"</parameter>\n{extra}</function>\n</tool_call>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="*")
    ap.add_argument("--elf", default=str(ROOT / "build/mpy-bridge.elf"))
    ap.add_argument("--script", help="file of calls separated by '# ---' lines")
    ap.add_argument("--trace", help="write the session's token stream + ownership as JSON")
    ap.add_argument("--max-records", type=int, default=5_000_000)
    a = ap.parse_args()
    calls = a.code or SMOKE
    if a.script:
        calls = [c.strip("\n") for c in Path(a.script).read_text().split("# ---\n") if c.strip()]

    T = BR.Text(BR.default_tokenizer())
    m = ref.Machine.from_program(elf.program(a.elf))
    t0 = time.time()
    boot = BR.park(m)
    print(f"boot to park: {m.steps} instructions, {len(boot)} tokens, {time.time() - t0:.1f}s "
          f"(this lands in the firmware snapshot, not in a conversation)")
    ses = BR.Session(T, m)
    rows = []
    for code in calls:
        q = call_tokens(T, code)
        steps0, toks0, t0 = m.steps, len(ses.toks), time.time()
        ses.qwen(q, max_records=a.max_records)
        echo = ses.runs[-1]
        resp = T.decode_bytes(echo).decode("utf-8", "replace")
        d_steps, d_toks, dt = m.steps - steps0, len(ses.toks) - toks0, time.time() - t0
        row = {"code": code, "call_tokens": len(q), "response": resp, "instructions": d_steps,
               "trace_tokens": d_toks, "response_tokens": len(echo), "fault": m.fault,
               "seconds": round(dt, 2)}
        rows.append(row)
        print("=" * 78)
        print(">>> " + code.replace("\n", "\n... "))
        print(resp.rstrip("\n"))
        print(f"-- {d_steps:,} instructions, {d_toks:,} trace tokens, {len(echo)} response "
              f"tokens, call {len(q)} tokens, {dt:.1f}s ref time; "
              f"~{d_toks / 4000:.0f}s on an L4 at 4K tok/s")
        if m.fault:
            print("FAULT", S.FAULTS[m.fault[0]], hex(m.fault[1]))
            break
    if a.trace:
        Path(a.trace).write_text(json.dumps({"calls": rows, "toks": ses.toks, "own": ses.own}))
        print("wrote", a.trace)


if __name__ == "__main__":
    main()
