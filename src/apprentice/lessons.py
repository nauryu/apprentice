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
# two triggers this close are treated as the same lesson (skip on add, collapse on prune)
DEDUP_SIM = float(os.environ.get("APPRENTICE_LESSON_DEDUP_SIM", "0.93"))

_DISTILL = (
    "You turn a single coding Q&A correction into ONE short, reusable lesson for a coding "
    "assistant. Generalize beyond the specific question so it helps on similar future ones. "
    "Output EXACTLY two lines, nothing else:\n"
    "TRIGGER: <the kind of question/situation this applies to, one short line>\n"
    "LESSON: <the concrete thing to remember, one or two sentences>")

# self-critique runs on a frontier brain (defaults to your Claude account, no API key)
CRITIC_BACKEND = os.environ.get("APPRENTICE_CRITIC_BACKEND", "claude-cli")
_CRITIC = ("You review a coding assistant's answer for factual errors or missing key facts. "
           "If it is correct and complete, reply with exactly: PASS\n"
           "Otherwise reply with 'FIX:' followed by the corrected answer itself — state the correct "
           "answer directly, as you would answer the question, NOT a critique of the old answer "
           "(no 'the answer is wrong', no meta-commentary). Flag only real errors, not style.")
_VERIFY = ("You are a strict fact-checker guarding a knowledge base from bad entries. Given a "
           "question, the original answer, and a proposed correction, decide whether the correction "
           "is BOTH factually correct AND a genuine improvement over the original. Reply with exactly "
           "one token: yes or no. If you are not confident, reply no.")
_SELF_LESSON = ("Given a question, a wrong answer, and the correct information, write ONE reusable "
                "lesson (one or two sentences) that states the correct fact or rule so a coding "
                "assistant won't repeat the mistake. Output only the lesson sentence — no preamble, "
                "no 'the answer was wrong', no meta-commentary.")


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


def add_lesson(trigger, lesson, source="user"):
    """Persist a lesson and (incrementally) index its trigger for retrieval. Near-duplicates are
    skipped (hygiene). `source` ('user' | 'self') marks who proposed it, for audit/rollback."""
    import numpy as np
    os.makedirs(STORE, exist_ok=True)
    vec = rag._embedder().encode([trigger], normalize_embeddings=True).astype("float32")
    meta = _load_meta()
    emb = np.load(IDX)["emb"] if os.path.exists(IDX) else None
    if emb is not None and emb.shape[0] and float((emb @ vec[0]).max()) >= DEDUP_SIM:
        print(f"[lessons] near-duplicate -- skipped: {lesson[:70]}")
        return None
    rec = {"id": f"L{int(time.time()*1000)}", "ts": int(time.time()),
           "trigger": trigger, "lesson": lesson, "source": source}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta.append(rec)
    emb = np.vstack([emb, vec]) if emb is not None else vec
    np.savez(IDX, emb=emb)
    json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[lessons] learned ({len(meta)} total): {lesson[:80]}")
    return rec


def prune(max_age_days=None):
    """Hygiene: collapse near-duplicate lessons (keep the earliest) and optionally drop ones older
    than `max_age_days`. Rewrites the store. User-invoked and reports exactly what it removes."""
    import numpy as np
    meta = _load_meta()
    if not meta or not os.path.exists(IDX):
        print("[lessons] nothing to prune")
        return 0
    emb = np.load(IDX)["emb"]
    now = int(time.time())
    keep = []
    for i, m in enumerate(meta):
        if max_age_days and (now - m.get("ts", now)) > max_age_days * 86400:
            continue
        if any(float(emb[i] @ emb[j]) >= DEDUP_SIM for j in keep):
            continue
        keep.append(i)
    removed = len(meta) - len(keep)
    meta2 = [meta[i] for i in keep]
    emb2 = emb[keep] if keep else np.zeros((0, emb.shape[1]), dtype="float32")
    np.savez(IDX, emb=emb2)
    json.dump(meta2, open(META, "w", encoding="utf-8"), ensure_ascii=False)
    with open(LOG, "w", encoding="utf-8") as f:
        for m in meta2:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[lessons] pruned {removed} (kept {len(meta2)})")
    return removed


def audit(sim=0.75):
    """Read-only: flag pairs of lessons with similar triggers — candidates for overlap/conflict to
    review. Deliberately does not auto-resolve; contradictions are for a human to judge."""
    import numpy as np
    meta = _load_meta()
    if not meta or not os.path.exists(IDX):
        print("[lessons] no lessons")
        return []
    emb = np.load(IDX)["emb"]
    pairs = sorted(((float(emb[i] @ emb[j]), i, j)
                    for i in range(len(meta)) for j in range(i + 1, len(meta))
                    if float(emb[i] @ emb[j]) >= sim), reverse=True)
    print(f"[lessons] {len(pairs)} overlapping pair(s) (trigger sim >= {sim}) to review:")
    for s, i, j in pairs:
        print(f"  {s:.2f}  ({meta[i].get('source', '?')}) {meta[i]['lesson'][:55]}")
        print(f"        ({meta[j].get('source', '?')}) {meta[j]['lesson'][:55]}")
    return pairs


def correct(task, wrong, right, source="user"):
    """Learn from one mistake: distill the correction, then store it as a lesson."""
    d = distill(task, wrong, right)
    return add_lesson(d["trigger"], d["lesson"], source=source)


def reflect(task, answer=None, k=6):
    """Self-critique: the assistant reviews its own answer and, only if an INDEPENDENT verifier
    confirms the fix is correct and better, stores it as a self-lesson. The verify gate is what
    keeps a hallucinated 'correction' from poisoning the store — no gate, no self-improvement."""
    from . import rag
    if answer is None:
        answer, _ = rag.ask(task, k=k, use_lessons=True)
    crit = llm.chat([{"role": "system", "content": _CRITIC},
                     {"role": "user", "content": f"[question]\n{task}\n\n[answer]\n{answer}"}],
                    temperature=0.0, max_tokens=700, backend=CRITIC_BACKEND).strip()
    if crit.upper().startswith("PASS"):
        print("[reflect] PASS -- answer looks correct, no self-lesson")
        return {"verdict": "pass", "task": task}
    proposed = crit[4:].strip() if crit[:4].upper() == "FIX:" else crit.strip()
    ok = llm.chat([{"role": "system", "content": _VERIFY},
                   {"role": "user", "content": f"[question]\n{task}\n\n[original answer]\n{answer}\n\n"
                                               f"[proposed correction]\n{proposed}\n\nCorrect and better (yes/no):"}],
                  temperature=0.0, max_tokens=4, backend=CRITIC_BACKEND).strip().lower().startswith("y")
    if not ok:
        print("[reflect] proposed correction NOT verified -- discarded (conservative)")
        return {"verdict": "rejected", "task": task, "proposed": proposed}
    lesson = llm.chat([{"role": "system", "content": _SELF_LESSON},
                       {"role": "user", "content": f"[question]\n{task}\n\n[wrong answer]\n{answer}\n\n"
                                                   f"[correct information]\n{proposed}"}],
                      temperature=0.0, max_tokens=150, backend=CRITIC_BACKEND).strip()
    rec = add_lesson(task.strip()[:160], lesson, source="self")
    print("[reflect] verified self-lesson stored" if rec else "[reflect] verified, but duplicate -- skipped")
    return rec


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
