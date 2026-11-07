"""The real columns, on a GPU: the Qwenputer checkpoint with the python tool running in its
weights (`weights`), and stock Qwen3.5-4B bf16 with no tool (`base`). Same JSONL format as
bench/run.py, so bench/summarize.py picks the results up (modes `weights`, `base`).

    hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir qwenputer-rv
    uv run --project qwenputer/experimental/riscv python bench/run_gpu.py --mode weights \\
        --model qwenputer-rv --task count_char --n 50 [--executor]
    uv run --project qwenputer/experimental/riscv python bench/run_gpu.py --mode base \\
        --model Qwen/Qwen3.5-4B --task count_char --n 50

In `weights` mode one generate() call is the whole episode: Qwen writes tool calls, the circuit
runs them and prints the responses, Qwen answers. The reference machine runs alongside (the
lock) and the run stops at the first token where the circuit and the reference differ, which is
recorded as lock_error (none expected). Greedy, non-thinking, as in the author's benchmarks.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
import tasks as TK  # noqa: E402
from run import PY_TOOL, SYSTEM_DIRECT, SYSTEM_TOOL  # noqa: E402


class Weights:
    def __init__(self, model: str, firmware: str, executor: bool, max_new: int, max_chip: int,
                 retokenize: bool = False):
        from chipformer.rv import bridge as BR
        from chipformer.rv import elf, ref
        from chipformer.rv.chat import RvChat

        self.BR, self.ref, self.elf = BR, ref, elf
        self.chat = RvChat(model, executor=executor)
        if retokenize:
            import retok
            retok.install(self.chat)
        self.prog = elf.program(firmware)
        self.max_new, self.max_chip = max_new, max_chip

    def run(self, prompt: str) -> dict:
        m = self.ref.Machine.from_program(self.prog)
        self.BR.park(m)
        lock = self.BR.Session(self.chat.T, m)
        lock.qwen = lambda qids, max_records=5_000_000, run=True: \
            self.BR.Session.qwen(lock, qids, max_records, run)
        ids = self.chat.prompt_ids([{"role": "system", "content": SYSTEM_TOOL},
                                    {"role": "user", "content": prompt}], tools=(PY_TOOL,))
        lock.history(ids)
        r = self.chat.generate(ids, lock.m, self.max_new, max_chip=self.max_chip,
                               check_head=True, lock=lock)
        text = self.chat.text_of(r["tokens"])
        text = re.sub(r"<\|[a-z_]+\|>", "", text)  # <|im_end|> etc. would read as words to the scorer
        calls = []
        for i, run in enumerate(lock.runs):
            out = self.chat.T.decode_bytes(run).decode("utf-8", "replace")
            out = out.split("<tool_response>\n", 1)[-1].split("</tool_response>", 1)[0]
            calls.append({"code": None, "output": out.rstrip("\n"), "fault": False})
        # the calls' code, from the text
        codes = [s.split("</parameter>", 1)[0].strip("\n")
                 for s in text.split("<parameter=code>")[1:]]
        for c, code in zip(calls, codes):
            c["code"] = code
        n = max(len(calls), 1)
        for c in calls:  # the chip cost, split evenly (generate reports the total)
            c["instructions"] = None
            c["trace_tokens"] = r["chip_tokens"] // n
        return {"reply": text, "calls": calls, "chip_tokens": r["chip_tokens"],
                "text_tokens": r["text_tokens"], "seconds": r["seconds"],
                "head_mismatch": r["head_mismatch"], "lock_error": r["lock_error"]}


class Base:
    def __init__(self, model: str, max_new: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
        self.max_new = max_new

    def run(self, prompt: str) -> dict:
        out = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_DIRECT},
                                            {"role": "user", "content": prompt}],
                                           add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True)
        if not isinstance(out, list):
            out = out["input_ids"]
        ids = self.torch.tensor([[int(x) for x in out]], device="cuda")
        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new, do_sample=False)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return {"reply": text, "calls": [], "chip_tokens": 0, "text_tokens": int(out.shape[1] - ids.shape[1]),
                "seconds": time.time() - t0, "head_mismatch": None, "lock_error": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True, choices=["weights", "weights-retok", "base"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--firmware", default=str(ROOT / "build/mpy-bridge.elf"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=800)
    ap.add_argument("--max-chip", type=int, default=3_000_000)
    ap.add_argument("--executor", action="store_true", help="Triton executor for the chip layers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = TK.load(a.task)[a.offset:a.offset + a.n]
    tag = Path(a.model).name.replace(":", "_")
    out = Path(a.out or ROOT / "bench/results" / f"{a.task}.{a.mode}.{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"]: json.loads(l) for l in out.read_text().splitlines()} if out.exists() else {}
    runner = Base(a.model, a.max_new) if a.mode == "base" else \
        Weights(a.model, a.firmware, a.executor, a.max_new, a.max_chip, retokenize=a.mode == "weights-retok")
    results = []
    with out.open("a") as f:
        for i, item in enumerate(items):
            if item.id in done:
                results.append(done[item.id])
                continue
            r = runner.run(item.prompt)
            pred = TK.final_answer(r["reply"])
            if pred is None and r["reply"].strip():
                pred = r["reply"].strip().splitlines()[-1]
            task = item.id.rsplit("-", 1)[0]
            rec = {"id": item.id, "mode": a.mode, "model": tag, "prediction": pred,
                   "target": item.target, "correct": TK.score(task, pred, item.target),
                   "calls": r["calls"], "n_calls": len(r["calls"]), "instructions": 0,
                   "trace_tokens": r["chip_tokens"], "text_tokens": r["text_tokens"],
                   "seconds": round(r["seconds"], 1), "head_mismatch": r["head_mismatch"],
                   "lock_error": r["lock_error"], "reply": r["reply"]}
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            acc = sum(x["correct"] for x in results) / len(results)
            print(f"[{i + 1}/{len(items)}] {item.id}: {'ok ' if rec['correct'] else 'BAD'} "
                  f"pred={pred!r} target={item.target!r} calls={rec['n_calls']} "
                  f"chip={rec['trace_tokens']:,} ({rec['seconds']}s) lock={rec['lock_error'] is not None}"
                  f"  acc so far {100 * acc:.1f}%", flush=True)
    n = len(results)
    summary = {"task": a.task, "mode": a.mode, "model": tag, "n": n,
               "accuracy": round(sum(x["correct"] for x in results) / n, 4),
               "mean_calls": round(sum(x["n_calls"] for x in results) / n, 2),
               "mean_instructions": 0,
               "mean_trace_tokens": round(sum(x["trace_tokens"] for x in results) / n),
               "mean_seconds": round(sum(x["seconds"] for x in results) / n, 1),
               "lock_errors": sum(1 for x in results if x.get("lock_error"))}
    print(json.dumps(summary))
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
