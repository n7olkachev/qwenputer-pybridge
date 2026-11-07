"""Record the exact token stream of the demo call from the reference machine (the checkpoint
matched it token for token: 87,644 chip tokens, 0 mismatches) -> promo/stream.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv")); sys.path.insert(0, str(ROOT / "tools"))
from chipformer.rv import bridge as BR, elf, ref
from chipformer.rv import spec as S
from run_ref import call_tokens
T = BR.Text(BR.default_tokenizer())
m = ref.Machine.from_program(elf.program(str(ROOT / "build/mpy-bridge.elf"))); BR.park(m)
ses = BR.Session(T, m)
code, data = "data = 'strawberry'\nprint(data.count('r'))", "strawberry"
ses.qwen(call_tokens(T, code, data=data))
names = [("text", T.tok.decode([t - S.TEXT_TOKEN])) if S.is_text(t) else ("chip", S.name(t)) for t in ses.toks]
json.dump({"code": code, "data": data, "stream": names, "instructions": m.steps}, open(ROOT / "promo/stream.json", "w"))
print(len(ses.toks), "tokens,", sum(1 for k, _ in names if k == "chip"), "chip,", m.steps, "instructions")
