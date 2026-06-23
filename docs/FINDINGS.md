# Findings — can a local LLM be your coding brain?

These are results from a one-week experiment that motivated `apprentice`. The point of
publishing them — including what *didn't* work — is that the eval harness in this repo lets
**anyone reproduce the methodology** on their own models and domain. Exact figures depend on
the run; the *shape* of the result is what matters.

## Setup

- **Models compared**
  - *Model-A* — a ~30B coder-specialized MoE (≈3B active params), served via `llama.cpp`.
  - *Model-B* — a ~72B general instruct model (Q4), same server.
  - *Supervisor* — a frontier model (the brain in the hybrid).
- **Method** — domain-targeted **held-out** question sets, **blind pairwise** judging with
  position-bias control (judge both orders; count a win only if consistent). For fine-tunes,
  base-vs-candidate was compared on the *same* server by toggling the LoRA at runtime.

## Result 1 — fine-tuning shapes *style*, not *reasoning*

| Fine-tune (LoRA on Model-A) | Domain | Blind result vs base |
|---|---|---|
| Style / orthography (Korean) | language/writing | **~81% win** (significant) |
| Reasoning traces (debug/algorithm) | code/math reasoning | ~68% but **not significant** (≈75% ties) |

Take-away: SFT/DPO reliably move *how* a model writes (tone, format, language conventions),
but the base model's *reasoning ceiling* barely moves. Don't expect fine-tuning to make a small
model "smarter" — expect it to make it more *on-style*.

## Result 2 — bigger isn't uniformly better; specialization matters

*Model-B (72B)* vs *Model-A (30B coder)*, held-out 195 items, blind pairwise:

| Domain | Winner |
|---|---|
| General / language / instruction-following / practical | **72B (~66%)** |
| Pure code & math reasoning (debug, output-prediction, algorithms) | **30B coder (~67%)** |
| Overall | 72B ~56% |

Take-away: a larger general model wins comprehension and instruction-following; a smaller
*coder-specialized* model holds or wins on pure code reasoning. "Use the biggest model" is wrong;
**match the model to the task** (which is what the routing/hybrid layer is for).

## Result 3 — the local model is not a reliable standalone brain

Across the agentic panel sessions, the local model repeatedly: mis-classified intent, gave up
early, answered with generic filler instead of investigating, and even reported an existing file
as "not found". A larger local model reduced — but did not remove — this. Frontier-class
reasoning remained out of reach locally.

Take-away: for serious agentic coding, the local model should be a **verified sub-agent**, not
the brain.

## Conclusion → the apprentice architecture

1. **Frontier model = brain** (plans, verifies, corrects).
2. **Local model = cheap/private hands** for simple, verifiable subtasks — always verified.
3. **code-RAG = grounding** in the actual codebase (model-agnostic; closes the "fabrication" gap).
4. **Eval harness = source of truth** — nothing is "better" until it wins a blind eval.
5. **Flywheel** — supervisor corrections become training data, promoted only when eval improves.

Hardware note: approximating frontier *locally* currently needs the ~400–671B class (e.g. a
512GB unified-memory box, or a multi-GPU server). A 70B box caps out well below frontier. Unless
privacy/offline is mandatory, a frontier API is usually the cheaper brain.
