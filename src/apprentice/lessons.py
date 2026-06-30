"""
Self-improvement without retraining — the apprentice learns by accumulating *lessons*,
not by fine-tuning weights. The brain stays a frontier model (see `llm`); what grows is a
searchable store of corrected mistakes that gets injected back into the prompt.

  correction  --distill-->  lesson (trigger + guidance)  --embed-->  store
  new question --retrieve relevant lessons--> inject into the answer prompt

This is the loop that actually works for a frontier-brained assistant (cf. docs/FINDINGS.md:
local models can't be the brain). No GPU, no adapters. Reuses the `rag` embedder for search.

  from apprentice import lessons
  lessons.correct(task, wrong_answer, right_answer)   # learn from a mistake
  lessons.retrieve("how do I ...")                     # top relevant lessons
"""
import os
import json
import time

from . import rag, llm

STORE = rag.STORE                                          # .apprentice/ (shared with code index)
LOG = os.path.join(STORE, "lessons.jsonl")                # durable record of every lesson
IDX = os.path.join(STORE, "lessons_index.npz")            # embeddings of lesson triggers
META = os.path.join(STORE, "lessons_meta.json")           # lesson rows aligned to IDX
PENDING = os.path.join(STORE, "corrections.jsonl")        # raw corrections captured (e.g. by the panel)
# bge-m3 baseline similarity runs high (~0.45 even for unrelated coding text), so the gate is
# deliberately strict — only inject a lesson that's clearly on-topic, never pollute every answer.
MIN_SIM = float(os.environ.get("APPRENTICE_LESSON_MIN_SIM", "0.60"))

_DISTILL = (
    "You turn a single coding Q&A correction into ONE short, reusable lesson for a coding "
    "assistant. Generalize beyond the specific question so it helps on similar future ones. "
    "Output EXACTLY two lines, nothing else:\n"
    "TRIGGER: <the kind of question/situation this applies to, one short line>\n"
    "LESSON: <the concrete thing to remember, one or two sentences>")


def _load_meta():
    return json.load(open(META, encoding="utf-8")) if os.path.exists(META) else []


def distill(task, wrong, right):
    """Compress a correction into a generalized {trigger, lesson} via the frontier brain."""
    user = f"[question]\n{task}\n\n[wrong answer]\n{wrong or '(none)'}\n\n[correct answer]\n{right}"
    out = llm.chat([{"role": "system", "content": _DISTILL},
                    {"role": "user", "content": user}], temperature=0.0, max_tokens=200)
    trigger, lesson = task.strip()[:160], right.strip()
    for line in out.splitlines():
        s = line.strip()
        if s.upper().startswith("TRIGGER:"):
            trigger = s.split(":", 1)[1].strip() or trigger
        elif s.upper().startswith("LESSON:"):
            lesson = s.split(":", 1)[1].strip() or lesson
    return {"trigger": trigger, "lesson": lesson}


def add_lesson(trigger, lesson):
    """Persist a lesson and (incrementally) index its trigger for retrieval."""
    import numpy as np
    os.makedirs(STORE, exist_ok=True)
    rec = {"id": f"L{int(time.time()*1000)}", "ts": int(time.time()),
           "trigger": trigger, "lesson": lesson}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = _load_meta()
    meta.append(rec)
    vec = rag._embedder().encode([trigger], normalize_embeddings=True).astype("float32")
    emb = np.vstack([np.load(IDX)["emb"], vec]) if os.path.exists(IDX) else vec
    np.savez(IDX, emb=emb)
    json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[lessons] learned ({len(meta)} total): {lesson[:80]}")
    return rec


def correct(task, wrong, right):
    """Learn from one mistake: distill the correction, then store it as a lesson."""
    d = distill(task, wrong, right)
    return add_lesson(d["trigger"], d["lesson"])


def record_correction(task, answer, correction):
    """Append a raw correction for later digestion. Cheap, no model — safe to call from the
    panel (Node) which can't run the embedder. Turn these into lessons later with `digest`."""
    os.makedirs(STORE, exist_ok=True)
    rec = {"ts": int(time.time()), "task": task, "answer": answer, "correction": correction}
    with open(PENDING, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def digest(path=None):
    """Drain captured corrections into distilled lessons. The embedder/brain run here, not
    in the panel — capture is cheap, digestion is batched."""
    path = path or PENDING
    if not os.path.exists(path):
        print("[lessons] no pending corrections")
        return 0
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    for r in rows:
        correct(r["task"], r.get("answer", ""), r["correction"])
    open(path, "w", encoding="utf-8").close()              # clear only after all are digested
    print(f"[lessons] digested {len(rows)} correction(s) into lessons")
    return len(rows)


def retrieve(query, k=3, min_sim=MIN_SIM):
    """Top-k stored lessons whose trigger is semantically close to `query` (>= min_sim)."""
    if not (os.path.exists(IDX) and os.path.exists(META)):
        return []
    import numpy as np
    emb = np.load(IDX)["emb"]
    meta = _load_meta()
    q = rag._embedder().encode([query], normalize_embeddings=True)[0]
    sims = emb @ q
    hits = [(float(sims[i]), meta[i]) for i in np.argsort(-sims)[:k]]
    return [(s, m) for s, m in hits if s >= min_sim]


def as_prompt(query, k=3):
    """Relevant lessons as an authoritative [project knowledge] block for the *user* message, or ''.
    Injecting them in the user message (not a soft system note) is what makes the model actually
    apply them — a weak system-prompt mention gets ignored next to the code context."""
    hits = retrieve(query, k)
    if not hits:
        return ""
    body = "\n".join(f"- {m['lesson']}" for _, m in hits)
    return ("[project knowledge -- learned from past corrections; authoritative for conventions and "
            "facts not in the code below]\n" + body)
