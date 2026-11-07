"""Run a task in one mode and write per-item results (JSONL) plus a summary.

modes:
  direct       the model answers on its own (no tools)
  python-ref   the model has the `python` tool; its code runs on Qwenputer's reference machine
               with the real mpy-bridge firmware (the in-weights environment, simulated), with
               instruction and trace-token counts per call
  python-host  the same tool, but the code runs on host CPython (the ceiling for model-written
               code; no MicroPython limits)

backend: an Ollama model (default qwen3.5:4b-q8, non-thinking, greedy). The GPU backend for the
real checkpoint lives in tools/chat_py.py; its results use the same JSONL format.

usage: uv run --project qwenputer/experimental/riscv python bench/run.py --task word_sorting \\
           --mode python-ref [--n 50] [--model qwen3.5:4b-q8] [--out bench/results/...jsonl]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tools"))
import tasks as TK  # noqa: E402

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

SYSTEM_DIRECT = ("Answer the question. Think briefly if needed, then end your reply with a line "
                 "of the form 'Answer: <final answer>'.")
SYSTEM_TOOL = ("You have a python tool that runs code exactly. Use it for the computation, "
               "counting, sorting or string manipulation the question needs, read its output, "
               "then end your reply with a line of the form 'Answer: <final answer>'. The tool "
               "takes at most 80 tokens per call, so put the question's data in the data "
               "parameter as plain text and keep the code tiny, e.g. "
               "python(code=\"print(sorted(data.split()))\", data=\"pear apple fig\"). "
               "The session keeps your variables between calls.")


def ollama_chat(model: str, messages: list, tools: list | None, num_predict: int = 1024) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 8192, "seed": 0}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]


class RefExecutor:
    """The model's code on the reference RV32 machine with the real bridge firmware."""

    def __init__(self, elf_path: str) -> None:
        sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
        from chipformer.rv import bridge as BR
        from chipformer.rv import elf, ref

        self.BR, self.ref = BR, ref
        self.prog = elf.program(elf_path)
        self.T = BR.Text(BR.default_tokenizer())
        from run_ref import call_tokens
        self.call_tokens = call_tokens

    def new_session(self):
        m = self.ref.Machine.from_program(self.prog)
        self.BR.park(m)
        return self.BR.Session(self.T, m)

    def run(self, ses, code: str, data: str | None = None) -> dict:
        q = self.call_tokens(self.T, code, data=data)
        steps0, toks0 = ses.m.steps, len(ses.toks)
        try:
            ses.qwen(q, max_records=5_000_000)
        except RuntimeError as e:
            return {"output": f"error: {e}", "instructions": ses.m.steps - steps0,
                    "trace_tokens": len(ses.toks) - toks0, "call_tokens": len(q), "fault": True}
        resp = self.T.decode_bytes(ses.runs[-1]).decode("utf-8", "replace")
        out = resp.split("<tool_response>\n", 1)[-1].split("</tool_response>", 1)[0]
        return {"output": out.rstrip("\n"), "instructions": ses.m.steps - steps0,
                "trace_tokens": len(ses.toks) - toks0, "call_tokens": len(q),
                "fault": ses.m.fault is not None}


class HostExecutor:
    """The model's code on host CPython; persistence by re-running the item's earlier calls."""

    def new_session(self):
        return []

    def run(self, ses, code: str, data: str | None = None) -> dict:
        if data is not None:
            ses.append("data = " + repr(data))
        ses.append(code)
        src = "\n".join(ses)
        try:
            p = subprocess.run([sys.executable, "-I", "-c", src], capture_output=True, text=True,
                               timeout=20)
            out = p.stdout
            if p.returncode != 0:
                out += p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "error"
        except subprocess.TimeoutExpired:
            out = "error: timeout"
        # only the new call's output: strip what the earlier calls printed (re-run above)
        if len(ses) > 1:
            try:
                prev = subprocess.run([sys.executable, "-I", "-c", "\n".join(ses[:-1])],
                                      capture_output=True, text=True, timeout=20).stdout
            except subprocess.TimeoutExpired:
                prev = ""
            if out.startswith(prev):
                out = out[len(prev):]
        return {"output": out.rstrip("\n"), "instructions": None, "trace_tokens": None,
                "call_tokens": None, "fault": False}


def run_item(item: TK.Item, mode: str, model: str, ex, max_calls: int = 4) -> dict:
    t0 = time.time()
    tools = None if mode == "direct" else [PY_TOOL]
    messages = [{"role": "system", "content": SYSTEM_DIRECT if mode == "direct" else SYSTEM_TOOL},
                {"role": "user", "content": item.prompt}]
    calls, ses = [], (ex.new_session() if ex else None)
    reply = ""
    for turn in range(max_calls + 1):
        if turn == max_calls:  # out of calls: ask for the answer (without tools the model goes silent)
            messages.append({"role": "user", "content": "Do not call any tool now. Give your final "
                             "answer in the form 'Answer: <answer>'."})
        msg = ollama_chat(model, messages, tools)
        reply = msg.get("content") or ""
        tcs = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": reply, "tool_calls": tcs} if tcs
                        else {"role": "assistant", "content": reply})
        if not tcs or ex is None:
            break
        for tc in tcs[:1]:  # one call at a time, as the bridge runs them
            args = tc.get("function", {}).get("arguments", {})
            code = args.get("code") if isinstance(args, dict) else None
            if not isinstance(code, str):
                code = json.dumps(args)
            data = args.get("data") if isinstance(args, dict) else None
            data = data if isinstance(data, str) else (json.dumps(data) if data is not None else None)
            r = ex.run(ses, code, data)
            calls.append({"code": code, "data": data, **r})
            messages.append({"role": "tool", "content": r["output"] or "(no output)",
                             "tool_name": "python"})
    pred = TK.final_answer(reply) if reply else None
    if pred is None and reply.strip():
        pred = reply.strip().splitlines()[-1]
    task = item.id.rsplit("-", 1)[0]
    ok = TK.score(task, pred, item.target)
    return {"id": item.id, "mode": mode, "model": model, "prediction": pred, "target": item.target,
            "correct": ok, "calls": calls, "n_calls": len(calls),
            "instructions": sum(c["instructions"] or 0 for c in calls),
            "trace_tokens": sum(c["trace_tokens"] or 0 for c in calls),
            "seconds": round(time.time() - t0, 1), "reply": reply}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True, choices=["direct", "python-ref", "python-host"])
    ap.add_argument("--model", default="qwen3.5:4b-q8")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--elf", default=str(ROOT / "build/mpy-bridge.elf"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = TK.load(a.task)[a.offset:a.offset + a.n]
    ex = RefExecutor(a.elf) if a.mode == "python-ref" else HostExecutor() if a.mode == "python-host" else None
    out = Path(a.out or ROOT / "bench/results" / f"{a.task}.{a.mode}.{a.model.replace(':', '_')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    results = []
    with out.open("a") as f:
        for i, item in enumerate(items):
            if item.id in done:
                results.append(done[item.id])
                continue
            r = run_item(item, a.mode, a.model, ex)
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            acc = sum(x["correct"] for x in results) / len(results)
            print(f"[{i + 1}/{len(items)}] {item.id}: {'ok ' if r['correct'] else 'BAD'} "
                  f"pred={r['prediction']!r} target={item.target!r} calls={r['n_calls']} "
                  f"instr={r['instructions']:,} ({r['seconds']}s)  acc so far {100 * acc:.1f}%",
                  flush=True)
    n = len(results)
    acc = sum(x["correct"] for x in results) / n
    calls = [c for x in results for c in x["calls"]]
    summary = {"task": a.task, "mode": a.mode, "model": a.model, "n": n, "accuracy": round(acc, 4),
               "mean_calls": round(sum(x["n_calls"] for x in results) / n, 2),
               "mean_instructions": round(sum(x["instructions"] for x in results) / n),
               "mean_trace_tokens": round(sum(x["trace_tokens"] for x in results) / n),
               "errors_in_calls": sum(1 for c in calls if "Error" in (c["output"] or "") or c["fault"]),
               "overflow_calls": sum(1 for c in calls if (c["output"] or "").startswith("error: call too long")
                                     or "too long for the 80-token" in (c["output"] or ""))}
    print(json.dumps(summary))
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
