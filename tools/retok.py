"""Re-tokenized tool responses: a variant of RvChat.generate for the executor path.

The circuit prints a tool response one byte per token (each putchar is echoed as the Qwen token
of that character), so a long string sits in Qwen's context as one token per letter, and a 4B
model transcribing it into its answer garbles it (reverse: the tool printed the right string in
100% of items, the final answer was right in 20%). This variant keeps the generated stream
exactly as it is (the chip layers, the lock and the replay see the byte-level echoes) but
withholds those echo tokens from Qwen's layers; when the machine parks it feeds Qwen the same
response text tokenized normally. The model still prints the response itself; only its
representation in Qwen's KV cache changes.

Usage: chat = RvChat(path, executor=True); install(chat); then generate() as usual.
"""
import time

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import spec as S


def install(chat) -> None:
    chat.generate = lambda *a, **k: generate(chat, *a, **k)
    chat.retokenize = True


@torch.no_grad()
def generate(self, ids, machine, max_new: int = 4000, max_chip: int = 60000,
             check_head: bool = False, verbose: bool = False, lock=None, on_token=None) -> dict:
    assert self.executor and not self.verify, "retok.generate supports the executor path"
    from chipformer.rv import core

    x = torch.tensor([ids], device=self.dev)
    out = self.model(input_ids=x, use_cache=True)
    cache = out.past_key_values
    core.snapshot(self.c, machine)
    self._exec_start(ids)
    logits = out.logits[0, -1]
    pos = len(ids)
    gen, n_text, n_chip_tok = [], 0, 0
    nxt = int(torch.argmax(logits[: self.v0]))
    t0 = time.time()
    mismatch = 0
    expect, lock_error = [], None
    in_run, echo = False, []  # a run: GO .. the HALT echo; echo: the Qwen ids the circuit printed
    n_retok = 0
    while n_text < max_new and n_chip_tok < max_chip:
        gen.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if lock is not None:
            st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
            if expect:
                e = expect.pop(0)
                if st != e:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                  else self.tok.decode([nxt]),
                                  "want": S.name(e) if e < S.VOCAB
                                  else self.tok.decode([e - S.TEXT_TOKEN])}
                    break
            elif st < S.VOCAB:
                lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                break
            else:
                k = len(lock.toks)
                try:
                    lock.qwen([nxt])
                except RuntimeError as e:
                    lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                    break
                expect = list(lock.toks[k + 1:])
        if nxt in self.eos:
            break
        if verbose and len(gen) % 500 == 0:
            print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) {time.time() - t0:.0f}s",
                  flush=True)
        chip_in = nxt >= self.v0
        keep = self._exec_step(nxt)
        if chip_in:
            n_chip_tok += 1
            if not in_run:
                in_run, echo = True, []
            nxt = self.code_token(keep[: len(self.code_dims)])
            if nxt is None:
                raise RuntimeError("the circuit gave up a chip position")
            continue
        # a text token
        hc = self._exec_hc(nxt, keep)
        ct = self.circuit_token(hc)
        if in_run:
            # an echo the circuit printed: keep it out of Qwen's layers, remember its text
            echo.append(nxt)
            n_text += 1
            if ct is None:
                # the run is over (this was the HALT echo): Qwen sees the response re-tokenized
                text = self.tok.decode(echo, skip_special_tokens=False)
                toks = self.tok.encode(text, add_special_tokens=False)
                n_retok += len(toks)
                for t in toks:
                    k2 = self._exec_step(t)  # parked: the chip layers pass the token through
                    h2 = self._exec_hc(t, k2)
                    logits = self._qwen_step(h2, cache, pos)
                    pos += 1
                in_run, echo = False, []
                nxt = int(torch.argmax(logits[: self.v0]))
            else:
                nxt = ct
            continue
        n_text += 1
        logits = self._qwen_step(hc, cache, pos)
        pos += 1
        q = int(torch.argmax(logits))
        if ct is not None and check_head and q != ct:
            mismatch += 1
        nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
    dt = time.time() - t0
    return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok, "seconds": dt,
            "head_mismatch": mismatch, "lock_error": lock_error, "verify_mismatch": None,
            "retokenized_tokens": n_retok}
