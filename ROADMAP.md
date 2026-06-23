# Roadmap

`apprentice` is an early reference implementation. Current state and direction:

## Working
- **VS Code panel** — agentic loop (read/search/edit/run/finish), BYO-key, model-agnostic. Builds.
- **code-RAG** — AST + multi-language chunking, embeddings, semantic search, grounded answers.
- **Eval harness** — blind pairwise with position-bias control.
- **Flywheel (data side)** — correction logging → SFT/DPO dataset builders.

## Next
- [ ] Wire the panel's `code_search` to a bundled local code-RAG service (one-command launch).
- [ ] Supervisor↔apprentice bridge in the panel (delegate simple subtasks, auto-verify).
- [ ] `apprentice train` reference (LoRA SFT/DPO) behind the `[train]` extra + GPU docs.
- [ ] "promote only if eval wins" CI gate for new adapters.
- [ ] Incremental re-indexing on file change.
- [ ] Retrieval-quality eval (recall@k) alongside the answer-quality eval.

## Non-goals
- Being "another Copilot". The differentiator is the **eval-gated self-improvement loop** and
  **honest, reproducible methodology**, not feature count.

Contributions and issues welcome.
