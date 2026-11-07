"""Chat with the Qwenputer checkpoint using the P4 MicroPython bridge: Qwen writes Python in its
own tool-call syntax, the CPU in its weights runs it, and the model prints the tool response
itself. A GPU job (the merged model is 14.65 GB in bf16; the author's numbers are on an L4).

    hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir qwenputer-rv
    uv run --project qwenputer/experimental/riscv python tools/chat_py.py qwenputer-rv \\
        --ask "Use python: how many r's are in 'strawberry'?" [--check] [--executor]

Mirrors chipformer.rv.chat's main with the python tool, the mpy-bridge firmware and a system
prompt about the tool's limits. --check replays the generated tokens through the circuit's IR.
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
from chipformer.rv.chat import RvChat, replay  # noqa: E402

PY_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python code and return what it prints. The session persists between "
                   "calls: variables and functions defined in one call are available in the next. "
                   "A bare expression on its own line prints its value. HARD LIMIT: 80 tokens of "
                   "code per call, so write compact code: no comments, no blank lines, short "
                   "names; pass data as one string and split it, e.g. w = 'a b c'.split(); do "
                   "more work in a second call if needed. Available: builtins, math, sys, random, "
                   "re, collections, heapq, hashlib, datetime (date, datetime, timedelta, "
                   "strftime; no now()). Not available: json, os, time, files, network.",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python source code, e.g. print(sorted(data.split()))"},
        "data": {"type": "string", "description": "Long input as plain text (no quotes, brackets or "
                 "commas): the words, numbers or sequence from the question. Available in the code "
                 "as the string variable data. Always put the question's data here, not in the code."}},
        "required": ["code", "data"]}}}

SYSTEM = ("You have a python tool that runs code exactly. Use it for anything that needs exact "
          "computation, counting, sorting or string manipulation, then answer from its output. "
          "The tool takes at most 80 tokens per call, so put the question's data in the data "
          "parameter as plain text and keep the code tiny, e.g. "
          "python(code=\"print(sorted(data.split()))\", data=\"pear apple fig\"). "
          "The session keeps your variables between calls.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default=str(ROOT / "build/mpy-bridge.elf"))
    ap.add_argument("--ask", action="append", default=None, help="user turn(s), in order")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--max-new", type=int, default=800, help="text tokens per turn")
    ap.add_argument("--max-chip", type=int, default=3_000_000, help="chip tokens per turn")
    ap.add_argument("--check", action="store_true", help="replay the run through the IR")
    ap.add_argument("--executor", action="store_true", help="Triton executor for the chip layers")
    ap.add_argument("--retokenize", action="store_true",
                    help="Qwen sees tool responses re-tokenized (tools/retok.py; needs --executor)")
    ap.add_argument("--slow", action="store_true", help="every layer through the HF modules")
    ap.add_argument("--out", default=None, help="write the transcript + stats as JSON")
    args = ap.parse_args()
    asks = args.ask or ["Use python: how many r's are in 'strawberry'?"]

    chat = RvChat(args.model, fast=not args.slow, executor=args.executor)
    if args.retokenize:
        import retok
        retok.install(chat)
    # one reference machine for the whole conversation: the lock runs every tool call on it as
    # Qwen writes them, so each turn's snapshot (its state after the previous turns' calls) is
    # what the circuit continues from: the Python session persists across turns
    m = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m)
    lock = BR.Session(chat.T, m)
    # generate() calls lock.qwen with the default record limit (1M); a budget-exceeded call is
    # ~0.9M instructions, heavy loop bodies can pass that before the budget trips
    lock.qwen = lambda qids, max_records=5_000_000, run=True: BR.Session.qwen(lock, qids, max_records, run)

    messages = [{"role": "system", "content": args.system}]
    turns = []
    history: list[int] = []
    for ask in asks:
        messages.append({"role": "user", "content": ask})
        ids = chat.prompt_ids(messages, tools=(PY_TOOL,))
        new = ids[len(history):]
        lock.history(new)
        t0 = time.time()
        r = chat.generate(ids, lock.m, args.max_new, max_chip=args.max_chip, check_head=True,
                          verbose=True, lock=lock)
        text = chat.text_of(r["tokens"])
        print("=" * 78)
        print(f"USER: {ask}")
        print(text)
        print(f"[{r['text_tokens']} text + {r['chip_tokens']} chip tokens in {r['seconds']:.1f}s, "
              f"head mismatches {r['head_mismatch']}, lock error {r['lock_error']}]")
        turn = {"ask": ask, "text": text, "text_tokens": r["text_tokens"],
                "chip_tokens": r["chip_tokens"], "seconds": r["seconds"],
                "head_mismatch": r["head_mismatch"], "lock_error": r["lock_error"]}
        if args.check:
            turn["replay"] = replay(chat, ids, r["tokens"], args.firmware)
            print("replay:", turn["replay"])
        turns.append(turn)
        # the assistant's turn, as Qwen saw it (tool responses were written by the model itself)
        messages.append({"role": "assistant", "content": text})
        history = ids + r["tokens"]
        print(f"({time.time() - t0:.0f}s)")
    if args.out:
        Path(args.out).write_text(json.dumps(turns, indent=1))


if __name__ == "__main__":
    main()
