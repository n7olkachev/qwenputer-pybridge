"""Promo video + card from the real token stream of the demo call (promo/stream.json: the exact
tokens the checkpoint generated for "how many r's are in 'strawberry'", recorded from the
reference machine, which the checkpoint matched token for token).

    uv run --project qwenputer/experimental/riscv python promo/make_video.py
    -> promo/qwenputer-python.mp4 (1280x720, 30 fps, ~26 s) and promo/card.png (1600x900)
"""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1280, 720, 30
FONT = "/System/Library/Fonts/Menlo.ttc"
BG, PANEL, DIM, TEXT, USER, CALL, TRACE, RESP, ANS, ACCENT = (
    (11, 15, 20), (16, 22, 30), (90, 100, 115), (225, 228, 232), (170, 180, 195),
    (255, 183, 77), (82, 120, 170), (120, 220, 140), (255, 255, 255), (255, 183, 77))

stream = json.load(open(HERE / "stream.json"))
STREAM = stream["stream"]
N_CHIP = sum(1 for k, _ in STREAM if k == "chip")
N_INSTR = stream["instructions"]
USER_PROMPT = "Use python: how many r's are in 'strawberry'?"
CALL_TEXT = ("<tool_call>\n<function=python>\n<parameter=code>\n" + stream["code"] +
             "\n</parameter>\n<parameter=data>\n" + stream["data"] + "\n</parameter>\n</function>\n</tool_call>")
ANSWER = "There are **3** r's in 'strawberry'."
STATS = f"{N_CHIP:,} chip tokens · {N_INSTR:,} RISC-V instructions · 0 mismatches vs the reference machine · 31 s on an RTX 4090"

# the run: the stream from the first chip token (GO) to the end
RUN_START = next(i for i, (k, _) in enumerate(STREAM) if k == "chip")
RUN = STREAM[RUN_START:]
# the echo text (the response the circuit prints) appears as text tokens inside the run
ECHO_AT = [(i, s) for i, (k, s) in enumerate(RUN) if k == "text"]

# phases (seconds)
T_USER, T_CALL, T_RUN, T_ANS, T_END = 0.6, 5.2, 12.0, 3.2, 4.5
TOTAL = T_USER + T_CALL + T_RUN + T_ANS + T_END


def font(size):
    return ImageFont.truetype(FONT, size, index=0)


F_TITLE, F_BODY, F_TRACE, F_BIG, F_SMALL, F_PHASE = font(17), font(21), font(12), font(44), font(14), font(18)


def ease(x):
    return x * x * (3 - 2 * x)


def wrap_lines(text, width_chars):
    out = []
    for para in text.split("\n"):
        while len(para) > width_chars:
            out.append(para[:width_chars])
            para = para[width_chars:]
        out.append(para)
    return out


def draw_frame(t):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((28, 18), "Qwen3.5-4B with a Python interpreter in its weights  ·  one generate() call, no tools, no runtime",
           font=F_TITLE, fill=DIM)
    # panels
    d.rounded_rectangle((24, 52, 816, 636), radius=10, fill=PANEL)
    d.rounded_rectangle((836, 52, 1256, 636), radius=10, fill=PANEL)
    d.text((40, 62), "the model's output stream (what Qwen sees)", font=F_SMALL, fill=DIM)
    d.text((852, 62), "chip tokens: the compiled layers running the CPU", font=F_SMALL, fill=DIM)

    # --- transcript ---
    y = 92
    lines = [("user", USER_PROMPT)]
    phase = "1/4  Qwen writes a tool call"
    n_call_chars = 0
    run_pos = 0  # index into RUN
    answer_chars = 0
    if t < T_USER:
        lines = [("user", USER_PROMPT[: int(len(USER_PROMPT) * t / T_USER)])]
    elif t < T_USER + T_CALL:
        n_call_chars = int(len(CALL_TEXT) * ease((t - T_USER) / T_CALL))
    elif t < T_USER + T_CALL + T_RUN:
        n_call_chars = len(CALL_TEXT)
        phase = "2/4  the compiled layers run MicroPython: every CPU step is a token the model generates"
        run_pos = int(len(RUN) * ease((t - T_USER - T_CALL) / T_RUN))
    else:
        n_call_chars = len(CALL_TEXT)
        run_pos = len(RUN)
        phase = "4/4  Qwen reads the response it printed itself, and answers"
        answer_chars = min(len(ANSWER), int(len(ANSWER) * (t - T_USER - T_CALL - T_RUN) / T_ANS))
    echo = "".join(s for i, s in ECHO_AT if i < run_pos)
    if run_pos and run_pos < len(RUN) and echo.strip().startswith("<tool_response>") and len(echo.strip()) > len("<tool_response>"):
        phase = "3/4  the model prints the tool response itself, byte by byte"
    if n_call_chars:
        for ln in wrap_lines(CALL_TEXT[:n_call_chars], 62):
            lines.append(("call", ln))
    if echo:
        for ln in wrap_lines(echo.strip("\n"), 62):
            lines.append(("resp", ln))
    if answer_chars:
        lines.append(("ans", ANSWER[:answer_chars]))
    colors = {"user": USER, "call": CALL, "resp": RESP, "ans": ANS}
    for kind, ln in lines[-23:]:
        prefix = "user › " if kind == "user" else "       "
        d.text((40, y), prefix + ln, font=F_BODY, fill=colors[kind])
        y += 23
    if run_pos and run_pos < len(RUN):
        d.text((40, y + 6), "… " + f"{run_pos:,}".rjust(6) + " chip tokens so far (transparent to Qwen's layers)",
               font=F_SMALL, fill=TRACE)

    # --- trace panel ---
    if run_pos:
        shown = RUN[max(0, run_pos - 160): run_pos]
        x0, y0, line_h, col_w = 852, 86, 15, 100
        row = col = 0
        for kind, name in shown:
            if kind == "text":
                continue
            d.text((x0 + col * col_w, y0 + row * line_h), name[:12], font=F_TRACE, fill=TRACE)
            col += 1
            if col == 4:
                col, row = 0, row + 1
            if row > 24:
                break
        chip_so_far = sum(1 for k, _ in RUN[:run_pos] if k == "chip")
        d.text((852, 478), f"{chip_so_far:,}", font=F_BIG, fill=ACCENT)
        d.text((852, 528), "chip tokens generated", font=F_SMALL, fill=DIM)
        instr = int(N_INSTR * chip_so_far / N_CHIP)
        d.text((852, 556), f"{instr:,} RISC-V instructions", font=F_PHASE, fill=TEXT)
        d.text((852, 584), "mismatches vs the reference machine: 0", font=F_PHASE, fill=RESP)
    else:
        d.text((852, 300), "(parked: the firmware waits for a tool call)", font=F_SMALL, fill=DIM)

    # --- bottom ---
    d.text((28, 652), phase, font=F_PHASE, fill=TEXT)
    d.text((28, 684), "github.com/n7olkachev/qwenputer-pybridge  ·  built on Qwenputer by Divyansh Agrawal",
           font=F_SMALL, fill=DIM)

    # --- end card ---
    if t > T_USER + T_CALL + T_RUN + T_ANS:
        a = min(1.0, (t - (T_USER + T_CALL + T_RUN + T_ANS)) / 0.6)
        overlay = Image.new("RGB", (W, H), BG)
        od = ImageDraw.Draw(overlay)
        od.text((60, 170), "A Python interpreter in the weights", font=font(46), fill=ANS)
        od.text((60, 240), "Qwen3.5-4B wrote the call, its own layers ran MicroPython and printed the answer.",
                font=font(22), fill=TEXT)
        od.text((60, 300), STATS, font=font(17), fill=ACCENT)
        od.text((60, 360), "On 10 exact-answer tasks: 70.8% → 83.0%.  reverse 0 → 100, swap chars 38 → 100, count letters 78 → 100.",
                font=font(19), fill=TEXT)
        od.text((60, 400), "97 M chip tokens generated by the real checkpoint on an RTX 4090 — zero mismatches.",
                font=font(19), fill=TEXT)
        od.text((60, 470), "github.com/n7olkachev/qwenputer-pybridge", font=font(24), fill=RESP)
        od.text((60, 510), "built on Qwenputer (github.com/divagr18/qwenputer) by Divyansh Agrawal", font=font(17), fill=DIM)
        img = Image.blend(img, overlay, a)
    return img


def make_card():
    img = Image.new("RGB", (1600, 900), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 60), "A Python interpreter in the weights", font=font(56), fill=ANS)
    d.text((70, 136), "Qwen3.5-4B writes a tool call → its own layers run MicroPython → it prints the answer. One generate(), no runtime.",
           font=font(20), fill=TEXT)
    d.rounded_rectangle((70, 196, 1000, 720), radius=12, fill=PANEL)
    y = 214
    d.text((90, y), "user › " + USER_PROMPT, font=font(22), fill=USER); y += 40
    for ln in CALL_TEXT.split("\n"):
        d.text((90, y), "       " + ln, font=font(22), fill=CALL); y += 27
    y += 6
    for ln in ["<tool_response>", "3", "</tool_response>"]:
        d.text((90, y), "       " + ln, font=font(22), fill=RESP); y += 27
    y += 6
    d.text((90, y), "       " + ANSWER, font=font(22), fill=ANS); y += 40
    d.text((90, y), "       ↑ 87,644 chip tokens between the call and the response: the CPU, as generation",
           font=font(17), fill=TRACE)
    d.rounded_rectangle((1030, 196, 1530, 720), radius=12, fill=PANEL)
    x = 1052
    d.text((x, 214), "on the real checkpoint (RTX 4090, bf16)", font=font(16), fill=DIM)
    d.text((x, 246), "97,276,073", font=font(44), fill=ACCENT)
    d.text((x, 298), "chip tokens generated, 0 mismatches", font=font(17), fill=TEXT)
    rows = [("base → with python", ""), ("count letters", "78 → 100"), ("reverse a string", "0 → 100"),
            ("swap characters", "38 → 100"), ("sort words", "72 → 96"), ("10-task mean", "70.8 → 83.0")]
    yy = 350
    for a, b in rows:
        d.text((x, yy), a, font=font(19), fill=TEXT if b else DIM)
        if b:
            d.text((x + 300, yy), b, font=font(19), fill=RESP)
        yy += 34
    d.text((x, 586), "the response is printed one byte per token;", font=font(15), fill=DIM)
    d.text((x, 608), "re-tokenizing it for Qwen fixed reverse 20 → 100", font=font(15), fill=DIM)
    d.text((x, 646), "paper + code", font=font(15), fill=DIM)
    d.text((x, 666), "github.com/n7olkachev/qwenputer-pybridge", font=font(16), fill=RESP)
    d.text((x, 690), "built on Qwenputer by Divyansh Agrawal", font=font(15), fill=DIM)
    d.text((70, 760), STATS, font=font(18), fill=DIM)
    d.text((70, 800), "In-weights Python equals host Python wherever the call fits the 80-token window; it hurts where it doesn't (dates 84 → 54).",
           font=font(16), fill=DIM)
    img.save(HERE / "card.png")


def main():
    import sys
    if "--card" in sys.argv:
        make_card()
        print("wrote promo/card.png")
        return
    frames_dir = HERE / "frames"
    frames_dir.mkdir(exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    n = int(TOTAL * FPS)
    for i in range(n):
        draw_frame(i / FPS).save(frames_dir / f"f{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    str(HERE / "qwenputer-python.mp4")], check=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()
    make_card()
    print("wrote promo/qwenputer-python.mp4 and promo/card.png")


if __name__ == "__main__":
    main()
