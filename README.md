# apprentice

**A self-improving coding apprentice with a frontier brain — it learns from corrections by *remembering* them (not retraining), and proves the gain with a reproducible eval.**

> Most "local LLM coding assistant" projects stop at retrieval. `apprentice` adds the part that's missing: a closed feedback loop where mistakes are distilled into **lessons**, stored, and pulled back into the prompt on similar questions later — so the assistant measurably improves over time **without ever touching model weights**. Whether a lesson actually helps is decided by a blind, reproducible eval, not by vibes.

The brain is a **frontier model** — your logged-in Claude account via the Claude Code CLI (**no API key**), or any OpenAI-compatible endpoint. That's a deliberate result, not laziness: see [`docs/FINDINGS.md`](docs/FINDINGS.md) — a local model can't reliably *be* the brain, so `apprentice` keeps the brain frontier and makes the **learning** local.

**It lives where you code.** The primary interface is a **VS Code side panel** — an agentic coding assistant that reads your files, searches your codebase semantically, edits with diff-approval, and runs commands — driven by *your* model of choice. The CLI and Python library are there too.

---

## Why this exists

We ran a one-week experiment to answer a blunt question: *can a local LLM be your primary coding brain?* We built a fine-tuning pipeline, a code-RAG layer, a hybrid orchestration bridge, and a domain eval harness — then measured honestly.

The honest finding (with numbers, see [`docs/FINDINGS.md`](docs/FINDINGS.md)):

- **Fine-tuning shaped *style* well** (Korean orthography/writing: ~81% blind win vs base) **but did not lift reasoning** (a reasoning-trace fine-tune tied 75% of head-to-head comparisons — no significant gain).
- A local 30–72B model **cannot reliably be the brain** for serious agentic coding. It *can* be a useful, verified sub-agent.
- The architecture that actually works keeps a **frontier model as the brain**, retrieval for grounding, and an eval harness as the source of truth.

`apprentice` is the distilled, secrets-free reference implementation of that result. Its twist: instead of fine-tuning a weak model, it lets a *strong* brain **accumulate lessons** from its own corrected mistakes and reuse them — the improvement you can actually get without a GPU.

## Architecture

```text
   your question
        │
        ▼
   ┌──────────────────────┐  retrieve relevant   ┌───────────────────────────┐
   │  lessons (self-improve)│◀───────────────────│  lessons store            │
   └──────────┬───────────┘                      │  (past corrections,       │
              │ inject                            │   distilled & embedded)   │
              ▼                                   └─────────────▲─────────────┘
   ┌──────────────────────┐   grounded by                       │ distill + embed
   │  frontier brain      │◀──────────────  code-RAG (your repo)│
   │  (Claude · no key)   │                                     │
   └──────────┬───────────┘                                     │
              │ answer                                          │
              ▼                  if it's wrong: correction ─────┘
            you  ──────────────────────────────────────────────
```

Composable pieces, each usable on its own:

| Module | What it does | Runs anywhere? |
|--------|--------------|----------------|
| `rag` | AST + generic chunking → embeddings → semantic code search → grounded answer | ✅ frontier brain / BYO-key |
| `lessons` | distill a correction into a reusable lesson, embed it, and inject the relevant ones into future answers — self-improvement with **no GPU, no retraining** | ✅ |
| `eval` | reproducible **blind pairwise** judging on *your* held-out set; e.g. lessons-on vs lessons-off | ✅ |

## Quickstart (5 minutes)

```bash
pip install -e .

# Brain option A — your logged-in Claude account, NO API key (needs the Claude Code CLI)
export APPRENTICE_LLM_BACKEND="claude-cli"

# Brain option B — any OpenAI-compatible endpoint (bring your own key)
export APPRENTICE_LLM_BASE_URL="https://api.openai.com/v1"     # or http://localhost:8080/v1 (llama.cpp)
export APPRENTICE_LLM_API_KEY="sk-..."
export APPRENTICE_LLM_MODEL="gpt-4o-mini"

# dogfood: index THIS repo (no private code, fully reproducible)
apprentice index .

# grounded answer (relevant lessons, if any, are injected automatically)
apprentice ask "how does the eval harness avoid position bias?"

# teach it from a mistake — it distills a reusable lesson and remembers it
apprentice learn handling file paths in this project :: always build paths from the workspace root with abs()/os.path.join, never os.getcwd()

# what has it learned?
apprentice lessons
```

**In VS Code:** build the panel (`cd extension && npm install && npm run compile`), open the
folder in VS Code, press `F5` (Extension Development Host), and set `apprentice.backend` in
Settings (use `claude-cli` for a frontier brain with no API key). The panel reads files,
searches your code, and edits with diff-approval.

**Panel self-improvement** (so the panel actually uses lessons): run the local retrieval service
and point the panel at it —

```bash
apprentice serve                 # http://127.0.0.1:8799  (/lessons/search, /code/search)
```

then set `apprentice.lessonsUrl` = `http://127.0.0.1:8799/lessons/search` (and optionally
`apprentice.codeRagUrl` = `http://127.0.0.1:8799/code/search`). Now every panel question pulls
relevant lessons and injects them; the "👎 wrong? teach it" button captures corrections that
`apprentice digest` turns into new lessons. If the service isn't running, the panel just skips it.

No private code ships with this repo — the demo indexes the project itself.

## Does it actually improve? (the eval)

The point isn't "it feels smarter" — it's a number you can reproduce on your own questions:

```bash
# generate answers WITHOUT lessons, then WITH lessons, over the same held-out questions
apprentice gen questions.json no_lessons.json --no-lessons
apprentice gen questions.json with_lessons.json

# blind pairwise judge — does the lessons-on set win overall?
apprentice eval no_lessons.json with_lessons.json

# fact accuracy (when questions carry a "gold" fact) — did each answer get it right?
apprentice acc no_lessons.json      # e.g. correct 0/4
apprentice acc with_lessons.json    # e.g. correct 4/4
```

`questions.json` is a list of `{"id","q", optional "gold","cat"}`. `eval` judges each pair in
**both orders** and only counts a win if the verdict is consistent — no position bias. For
convention/knowledge questions, `acc` is the cleaner signal: pairwise judging underweights
factual specificity, so a correct-but-specific answer often reads as a "tie". The numbers *are*
the demo — reproduce them on your own questions and lessons.

## What makes it different (honest positioning)

The idea space is crowded (RAG frameworks, model routers, memory layers). `apprentice` does **not** claim a novel primitive. Its value is in the **combination and the rigor**:

1. **Self-improvement without retraining, and eval-gated** — corrections become retrievable lessons injected back into the prompt; a lesson set is trusted only if it wins a blind eval on your domain. No GPU, no vibes.
2. **Frontier brain by design** — driven by the negative result in [`docs/FINDINGS.md`](docs/FINDINGS.md), not hand-waving. `claude-cli` gives you that brain with no API key.
3. **Honest methodology over hype** — we publish what *didn't* work (local-as-brain) and why.

## Status

Early / reference implementation. See [`ROADMAP.md`](ROADMAP.md). Issues and discussion welcome.

## License

MIT — see [`LICENSE`](LICENSE).
