# Roadmap

`apprentice` is an early reference implementation. Current state and direction:

## Working

- **VS Code panel** — agentic loop (read/search/edit/run/finish), BYO-key, model-agnostic. Builds.
- **code-RAG** — AST + multi-language chunking, embeddings, semantic search, grounded answers.
- **Eval harness** — blind pairwise with position-bias control.
- **Lessons (self-improvement, no GPU)** — distill a correction into a reusable lesson, embed it,
  and inject the relevant ones into future answers. `learn` to teach, `lessons` to inspect.

## Next

- [x] `gen` + eval recipe for **lessons-on vs lessons-off**, plus `acc` fact-accuracy (measured
  0/4 → 4/4 on a controlled knowledge set after fixing lesson injection).
- [x] Panel hook: thumbs-down a wrong answer → capture the correction → `digest` into a lesson.
- [x] Local retrieval service (`apprentice serve`) for `/lessons/search` + `/code/search`; the panel
  injects retrieved lessons at query time via `apprentice.lessonsUrl`.
- [x] Self-critique source for lessons (`apprentice reflect`): critique → independent verify gate
  → store only verified fixes, tagged `source=self`. No gate, no self-improvement.
- [ ] Incremental re-indexing on file change.
- [ ] Retrieval-quality eval (recall@k) for both code and lessons.
- [ ] Lesson hygiene at scale: dedup, conflict resolution, staleness/decay.

## Non-goals

- Being "another Copilot". The differentiator is the **eval-gated, no-retraining self-improvement
  loop** and **honest, reproducible methodology**, not feature count.
- Making a local model *be* the brain — [`docs/FINDINGS.md`](docs/FINDINGS.md) shows that doesn't work.

Contributions and issues welcome.
