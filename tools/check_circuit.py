"""Check the compiled circuit (the chip layers merged into the Qwenputer checkpoint, with the P3
text capture) against the reference machine on a session of python(code=...) calls, the way the
author's tests/test_rv_bridge.py does: the IR evaluator runs the circuit over the session's token
stream, teacher-forced, and must own exactly the reference's positions and predict every next
token where it owns one.

usage: uv run --project qwenputer/experimental/riscv python tools/check_circuit.py [CODE ...]
           [--elf build/mpy-bridge.elf] [--chunk 2048] [--out build/circuit_check.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.q35 import ir  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import core, elf, ref  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from run_ref import call_tokens  # noqa: E402

DEFAULT = ["pass", "x = 6 * 7", "print(x)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="*")
    ap.add_argument("--elf", default=str(ROOT / "build/mpy-bridge.elf"))
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default=str(ROOT / "build/circuit_check.json"))
    ap.add_argument("--data", default=None,
                    help="also make a call print(sorted(data.split())) with this data parameter")
    a = ap.parse_args()
    calls = a.code or DEFAULT

    T = BR.Text(BR.default_tokenizer())
    t0 = time.time()
    c = core.build(text=T)
    print(f"circuit built: {len(c.dims)} dims, {time.time() - t0:.0f}s", flush=True)

    m = ref.Machine.from_program(elf.program(a.elf))
    BR.park(m)
    m0 = ref.Machine.from_program(elf.program(a.elf))  # the snapshot the host injects: parked
    BR.park(m0)
    ses = BR.Session(T, m)
    calls = [(c, None) for c in calls] + ([("print(sorted(data.split()))", a.data)] if a.data else [])
    for code, data in calls:
        ses.qwen(call_tokens(T, code, data=data))
        print(f"ref: {code!r} data={data!r} -> {T.decode_bytes(ses.runs[-1]).decode('utf-8', 'replace')!r}",
              flush=True)
    toks, own_ref = ses.toks, ses.own
    print(f"session: {len(toks):,} tokens ({sum(own_ref):,} circuit-owned)", flush=True)

    core.snapshot(c, m0)
    carry, mats = {}, []
    t0 = time.time()
    for s in range(0, len(toks), a.chunk):
        mats.append(ir.evaluate(c, toks[s:s + a.chunk], carry=carry))
        done = min(s + a.chunk, len(toks))
        el = time.time() - t0
        print(f"  {done:,}/{len(toks):,} tokens, {done / el:.0f} tok/s", flush=True)
    mat = np.concatenate(mats)
    pred = core.predictions(c, mat)
    own = np.rint(ir.ev(c.owner, mat)).astype(int)
    own_bad = [i for i in range(len(toks)) if own[i] != int(own_ref[i])]
    pred_bad = [i for i in range(len(toks) - 1) if own_ref[i] and pred[i] != toks[i + 1]]
    info = {"calls": [c for c, _ in calls], "data": a.data, "tokens": len(toks), "owned": int(sum(own_ref)),
            "owner_mismatches": len(own_bad), "prediction_mismatches": len(pred_bad),
            "first_bad": [(i, toks[i], pred[i], toks[i + 1]) for i in pred_bad[:5]],
            "seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    ok = not own_bad and not pred_bad
    print("CIRCUIT == REFERENCE" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
