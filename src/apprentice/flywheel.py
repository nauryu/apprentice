"""
The self-improvement flywheel — turn supervisor corrections into training data.

When the supervisor (frontier model) corrects the apprentice (local model), log the pair.
Those pairs become SFT (learn the right answer) or DPO (prefer right over wrong) data.
The train step itself needs a GPU and the [train] extra; it is intentionally kept separate,
and a new adapter should only be promoted if it WINS `evalharness.compare` on a held-out set.

  from apprentice import flywheel as fw
  fw.log_correction(task, apprentice_answer, supervisor_answer)
  fw.build_dpo("dpo.jsonl"); fw.build_sft("sft.jsonl")
"""
import os
import json
import time

LOG = os.environ.get("APPRENTICE_CORRECTIONS", os.path.join(".apprentice", "corrections.jsonl"))


def log_correction(task, apprentice_answer, supervisor_answer, verdict="corrected"):
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    rec = {"ts": int(time.time()), "task": task,
           "apprentice": apprentice_answer, "supervisor": supervisor_answer, "verdict": verdict}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _load():
    if not os.path.exists(LOG):
        return []
    return [json.loads(x) for x in open(LOG, encoding="utf-8") if x.strip()]


def build_dpo(out="dpo.jsonl"):
    """Preference pairs: supervisor=chosen, apprentice=rejected (only corrected items)."""
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in _load():
            if r.get("verdict") != "corrected" or not r.get("supervisor"):
                continue
            f.write(json.dumps({"prompt": r["task"], "chosen": r["supervisor"],
                                "rejected": r.get("apprentice", "")}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[flywheel] wrote {n} DPO pairs -> {out}")
    return n


def build_sft(out="sft.jsonl"):
    """Instruction/output pairs from the corrected (gold) answers."""
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in _load():
            if not r.get("supervisor"):
                continue
            f.write(json.dumps({"instruction": r["task"], "output": r["supervisor"]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"[flywheel] wrote {n} SFT examples -> {out}")
    return n
