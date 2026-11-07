"""Benchmark tasks with exact algorithmic solutions: BIG-Bench-Hard subsets (public, 250 items
each) and synthetic character-level tasks in the style of CUTE. Each item: id, prompt, target;
score(task, prediction, target) -> bool after task-specific normalization."""
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data"

BBH = {
    "word_sorting": "Sort the words alphabetically. Give the final answer as the words in order, "
                    "separated by single spaces.",
    "date_understanding": "Choose the correct option. Give the final answer as the option letter "
                          "in parentheses, e.g. (B).",
    "dyck_languages": "Complete the sequence so that the brackets are closed properly. Give the "
                      "final answer as the closing brackets only, separated by single spaces.",
    "boolean_expressions": "Evaluate the expression. Give the final answer as True or False.",
    "multistep_arithmetic_two": "Compute the value. Give the final answer as an integer.",
    "object_counting": "Give the final answer as an integer.",
}

SYNTHETIC = ["count_char", "reverse", "nth_char", "swap_chars"]


@dataclass
class Item:
    id: str
    prompt: str
    target: str


def words_pool() -> list[str]:
    d = json.loads((DATA / "bbh_word_sorting.json").read_text())
    ws = sorted({w for ex in d["examples"] for w in ex["target"].split()})
    return [w for w in ws if w.isalpha() and 6 <= len(w) <= 12]


def synthetic(task: str, n: int = 250, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    pool = words_pool()
    items = []
    for i in range(n):
        if task == "count_char":
            w = "".join(rng.sample(pool, 3))  # a long nonsense string: no memorized answers
            c = rng.choice(sorted(set(w)))
            items.append(Item(f"{task}-{i}", f"How many times does the letter '{c}' appear in "
                              f"the string '{w}'? Give the final answer as an integer.",
                              str(w.count(c))))
        elif task == "reverse":
            w = "".join(rng.sample(pool, 2))
            items.append(Item(f"{task}-{i}", f"Reverse the string '{w}'. Give the final answer "
                              "as the reversed string only.", w[::-1]))
        elif task == "nth_char":
            w = "".join(rng.sample(pool, 2))
            k = rng.randint(1, len(w))
            items.append(Item(f"{task}-{i}", f"What is the {k}th character (1-based) of the "
                              f"string '{w}'? Give the final answer as that single character.",
                              w[k - 1]))
        elif task == "swap_chars":
            w = "".join(rng.sample(pool, 2))
            a, b = rng.sample(range(len(w)), 2)
            a, b = min(a, b), max(a, b)
            s = list(w)
            s[a], s[b] = s[b], s[a]
            items.append(Item(f"{task}-{i}", f"In the string '{w}', swap the characters at "
                              f"positions {a + 1} and {b + 1} (1-based). Give the final answer as "
                              "the resulting string only.", "".join(s)))
        else:
            raise ValueError(task)
    return items


def load(task: str) -> list[Item]:
    if task in BBH:
        d = json.loads((DATA / f"bbh_{task}.json").read_text())
        return [Item(f"{task}-{i}", f"{ex['input']}\n\n{BBH[task]}", ex["target"])
                for i, ex in enumerate(d["examples"])]
    if task in SYNTHETIC:
        return synthetic(task)
    raise ValueError(f"unknown task {task}; known: {list(BBH) + SYNTHETIC}")


def final_answer(text: str) -> str | None:
    """The last 'Answer: ...' line of a reply (the prompt asks for it)."""
    hits = re.findall(r"(?im)^\s*\**\s*(?:final\s+)?answer\s*\**\s*[:：]\s*\**\s*(.+?)\s*\**\s*$", text)
    if hits:
        return hits[-1].strip()
    return None


def norm(task: str, s: str) -> str:
    s = s.strip().strip("`'\"*. ").strip()
    if task == "date_understanding":
        m = re.search(r"\(([A-F])\)", s) or re.search(r"\b([A-F])\b", s)
        return f"({m.group(1)})" if m else s
    if task == "boolean_expressions":
        return "True" if re.search(r"\btrue\b", s, re.I) else "False" if re.search(r"\bfalse\b", s, re.I) else s
    if task in ("multistep_arithmetic_two", "object_counting", "count_char"):
        m = re.search(r"-?\d+", s.replace(",", ""))
        return m.group(0) if m else s
    if task == "dyck_languages":
        return " ".join(re.findall(r"[\]\)\}>]", s))
    if task == "word_sorting":
        return " ".join(re.findall(r"[A-Za-z]+", s)).lower()
    if task in ("reverse", "swap_chars", "nth_char"):
        m = re.findall(r"[A-Za-z]+", s)
        return (m[-1] if m else s).lower()
    return s


def score(task: str, prediction: str | None, target: str) -> bool:
    if prediction is None:
        return False
    return norm(task, prediction) == norm(task, target)
