"""
Reproducible blind pairwise eval — the source of truth for "did it actually improve?".

Compares two answer sets on the SAME questions, judged by an LLM, with **position-bias
control**: every pair is judged in both orders and a win is only counted if the verdict
is consistent across orders (otherwise tie). No vibes — just numbers.

Answer file = JSON list of {"id","q","a", optional "gold","cat"}.

  from apprentice import evalharness as ev
  ev.compare("base.json", "candidate.json")     # -> tallies + win-rate
"""
import json
from collections import defaultdict
from . import llm

JUDGE = ("You are an impartial judge. Given a question, an optional reference answer, and two "
         "candidate answers A and B, decide which is better — correctness first, then clarity. "
         "Penalize answers that are vague, repetitive, truncated, or fabricated. "
         "Reply with exactly one token: A, B, or tie.")


def _judge_once(q, gold, a, b, model):
    ref = f"\n[reference]\n{gold}" if gold else ""
    msg = [{"role": "system", "content": JUDGE},
           {"role": "user", "content": f"[question]\n{q}{ref}\n\n[A]\n{a}\n\n[B]\n{b}\n\nBetter (A/B/tie):"}]
    out = llm.chat(msg, model=model, temperature=0.0, max_tokens=4).strip().lower()
    return "A" if out.startswith("a") else "B" if out.startswith("b") else "tie"


def compare(file_a, file_b, judge_model=None, label_a="A", label_b="B"):
    A = {o["id"]: o for o in json.load(open(file_a, encoding="utf-8"))}
    B = {o["id"]: o for o in json.load(open(file_b, encoding="utf-8"))}
    tot = {"a": 0, "b": 0, "tie": 0}
    bycat = defaultdict(lambda: {"a": 0, "b": 0, "tie": 0})
    rows = []
    for i in A:
        if i not in B:
            continue
        q, gold, cat = A[i].get("q", ""), A[i].get("gold", ""), A[i].get("cat", "-")
        v1 = _judge_once(q, gold, A[i]["a"], B[i]["a"], judge_model)          # order 1
        v2 = _judge_once(q, gold, B[i]["a"], A[i]["a"], judge_model)          # order 2 (swapped)
        w1 = "a" if v1 == "A" else "b" if v1 == "B" else "tie"
        w2 = "b" if v2 == "A" else "a" if v2 == "B" else "tie"               # decode swapped
        win = w1 if w1 == w2 else "tie"                                       # consistent or tie
        tot[win] += 1
        bycat[cat][win] += 1
        rows.append({"id": i, "win": win})
    dec = tot["a"] + tot["b"]
    print(f"=== {label_a} vs {label_b} ({sum(tot.values())} items) ===")
    print(f"{label_a} {tot['a']} / {label_b} {tot['b']} / tie {tot['tie']}")
    if dec:
        print(f"{label_b} win-rate (excl. tie): {100*tot['b']/dec:.1f}%")
    for c in sorted(bycat):
        b = bycat[c]
        print(f"  {c:12s}: {label_a} {b['a']:2d} / {label_b} {b['b']:2d} / tie {b['tie']:2d}")
    return tot, rows
