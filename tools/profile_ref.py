"""Instruction profile of one python(code=...) call on the reference machine: instructions per
function (by PC, symbols from riscv64-elf-nm), to see where a call's cost goes.

usage: uv run --project qwenputer/experimental/riscv python tools/profile_ref.py "print(2**100)" [--top 25]
"""
import argparse
import bisect
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from run_ref import call_tokens  # noqa: E402


def symbols(path: str):
    out = subprocess.run(["riscv64-elf-nm", "-n", "--defined-only", path], capture_output=True,
                         text=True, check=True).stdout
    addrs, names = [], []
    for line in out.splitlines():
        a, t, n = line.split()[:3]
        if t.lower() in "tw":
            addrs.append(int(a, 16))
            names.append(n)
    return addrs, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?", default="print(2**100)")
    ap.add_argument("--elf", default=str(ROOT / "build/mpy-bridge.elf"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--warm", action="store_true", help="profile a second, identical call")
    a = ap.parse_args()
    addrs, names = symbols(a.elf)
    T = BR.Text(BR.default_tokenizer())
    m = ref.Machine.from_program(elf.program(a.elf))
    BR.park(m)
    ses = BR.Session(T, m)
    if a.warm:
        ses.qwen(call_tokens(T, a.code))
    # the run, instruction by instruction
    q = call_tokens(T, a.code)
    ses.qwen(q, run=False)
    import chipformer.rv.spec as S
    m.halted = False
    prof = Counter()
    n = 0
    while True:
        pc = m.pc
        rec = m.record()
        n += 1
        i = bisect.bisect_right(addrs, pc) - 1
        prof[names[i] if i >= 0 else hex(pc)] += 1
        if rec[-1] == S.tok("HALT"):
            break
    print(f"{n:,} instructions; output: {bytes(m.out).decode('utf-8', 'replace')!r}")
    for name, c in prof.most_common(a.top):
        print(f"{c:9,}  {100 * c / n:5.1f}%  {name}")


if __name__ == "__main__":
    main()
