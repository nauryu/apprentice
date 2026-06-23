# apprentice

**A self-improving local coding apprentice — supervised by a frontier model, kept honest by a reproducible eval harness.**

> Most "local LLM coding assistant" projects stop at retrieval. `apprentice` adds the part that's missing: a closed feedback loop where a strong *supervisor* model corrects a small *local* model, and those corrections are turned into training data — gated by a reproducible, blind eval so the model only ships when it measurably improves.

It is **model-agnostic** and **bring-your-own-key**: point it at any OpenAI-compatible endpoint (OpenAI, a local `llama.cpp`/vLLM server, OpenRouter, …) for both the supervisor and the apprentice.

**It lives where you code.** The primary interface is a **VS Code side panel** — an agentic coding assistant that reads your files, searches your codebase semantically, edits with diff-approval, and runs commands — driven by *your* model of choice. The CLI and Python library are there too, but the panel is the face.

---

## Why this exists

We ran a one-week experiment to answer a blunt question: *can a local LLM be your primary coding brain?* We built the fine-tuning pipeline, a code-RAG layer, a hybrid orchestration bridge, and a domain eval harness — then measured honestly.

The honest finding (with numbers, see [`docs/FINDINGS.md`](docs/FINDINGS.md)):

- **Fine-tuning shaped *style* well** (Korean orthography/writing: ~81% blind win vs base) **but did not lift reasoning** (a reasoning-trace fine-tune tied 75% of head-to-head comparisons — no significant gain).
- A local 30–72B model **cannot reliably be the brain** for serious agentic coding. It *can* be a useful, verified sub-agent.
- The architecture that actually works is a **hybrid**: frontier model as the brain, local model as cheap/private hands, retrieval for grounding, and an eval harness as the source of truth.

`apprentice` is the distilled, generalized, secrets-free reference implementation of that result — built so anyone can reproduce it on their own codebase and their own models.

## Architecture

```
                         ┌──────────────────────────┐
   your question  ─────▶ │  supervisor (frontier)   │  plans · verifies · corrects
                         └─────────────┬────────────┘
                                       │ delegates simple/verifiable subtasks
                                       ▼
   code-RAG  ◀── grounds ──   ┌──────────────────┐
   (your repo, semantic)      │ apprentice(local)│  executes
                              └─────────┬────────┘
                                        │ result
                         ┌──────────────▼───────────┐
                         │ verify (supervisor)      │  correct? log (chosen/rejected)
                         └──────────────┬───────────┘
                                        │ corrections accumulate
                         ┌──────────────▼───────────┐
                         │ self-improve (SFT/DPO)   │  train apprentice on its own
                         │   ── gated by eval ──    │  corrected failures
                         └──────────────────────────┘
```

Four composable pieces, each usable on its own:

| Module | What it does | Runs anywhere? |
|--------|--------------|----------------|
| `rag` | AST + generic chunking → embeddings → semantic code search → grounded answer | ✅ BYO-key |
| `bridge` | delegate → execute → **verify** loop between supervisor & apprentice | ✅ BYO-key |
| `eval` | reproducible **blind pairwise** judging on *your* held-out set; base-vs-candidate | ✅ BYO-key |
| `flywheel` | turn logged corrections into SFT/DPO data; retrain; **ship only if eval improves** | ⚙️ needs a GPU for the train step |

## Quickstart (5 minutes, BYO-key)

```bash
pip install -e .

# point at any OpenAI-compatible endpoint
export APPRENTICE_LLM_BASE_URL="https://api.openai.com/v1"     # or http://localhost:8080/v1 (llama.cpp)
export APPRENTICE_LLM_API_KEY="sk-..."
export APPRENTICE_LLM_MODEL="gpt-4o-mini"

# dogfood: index THIS repo (no private code, fully reproducible)
apprentice index .

# semantic code search
apprentice search "where is the chunking logic"

# grounded answer
apprentice ask "how does the eval harness avoid position bias?"
```

No private code ships with this repo — the demo indexes the project itself.

## What makes it different (honest positioning)

The idea space is crowded (RAG frameworks, model routers, hybrid orchestration). `apprentice` does **not** claim a novel primitive. Its value is in the **combination and the rigor**:

1. The **self-improvement loop is closed and eval-gated** — corrections become training data, but a model is only promoted if it wins a blind eval on your domain. No vibes-based "it feels better."
2. **Everything is reproducible and BYO-key** — including the negative results in [`docs/FINDINGS.md`](docs/FINDINGS.md).
3. **Honest methodology over hype** — we publish what *didn't* work and why.

## Status

Early / reference implementation. See [`ROADMAP.md`](ROADMAP.md). Issues and discussion welcome.

## License

MIT — see [`LICENSE`](LICENSE).
