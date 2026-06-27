# Roadmap

`apprentice` is an early reference implementation. Current state and direction:

## Working

- **VS Code panel** — agentic loop (read/search/edit/run/finish), BYO-key, model-agnostic. Builds.
- **code-RAG** — AST + multi-language chunking, embeddings, semantic search, grounded answers.
- **Eval harness** — blind pairwise with position-bias control.
- **Lessons (self-improvement, no GPU)** — distill a correction into a reusable lesson, embed it,
  and inject the relevant ones into future answers. `learn` to teach, `lessons` to inspect.

## Next

- [ ] `gen` + eval recipe for **lessons-on vs lessons-off** win-rate (the "did it improve?" number).
- [ ] Panel hook: thumbs-down a wrong answer → capture the correction → distill into a lesson.
- [ ] Serve lessons retrieval to the panel (a small local service the panel can query, alongside
  `code_search`).
- [ ] Self-critique source for lessons (a verify pass proposes corrections, not just the user).
- [ ] Incremental re-indexing on file change.
- [ ] Retrieval-quality eval (recall@k) for both code and lessons.

## Non-goals

- Being "another Copilot". The differentiator is the **eval-gated, no-retraining self-improvement
  loop** and **honest, reproducible methodology**, not feature count.
- Making a local model *be* the brain — [`docs/FINDINGS.md`](docs/FINDINGS.md) shows that doesn't work.

Contributions and issues welcome.
