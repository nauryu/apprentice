"""
code-RAG — semantic search over a codebase, grounded answers.

Multi-language: Python via AST (functions/classes/module), others via line windows.
Embeddings: any sentence-transformers model (default BAAI/bge-m3, multilingual).
Vector store: numpy cosine (plenty for repo-scale). Index in ./.apprentice/.

  from apprentice import rag
  rag.build_index(["."])          # index repos
  rag.search("chunking logic")    # top-k chunks
  rag.ask("how does X work?")     # grounded answer via apprentice.llm
"""
import os
import ast
import json
import numpy as np

from . import llm

EMB_MODEL = os.environ.get("APPRENTICE_EMBED_MODEL", "BAAI/bge-m3")
STORE = os.environ.get("APPRENTICE_INDEX_DIR", ".apprentice")
IDX = os.path.join(STORE, "code_index.npz")
META = os.path.join(STORE, "code_meta.json")

CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".cs", ".rs", ".go", ".java",
             ".cpp", ".cc", ".c", ".h", ".hpp", ".kt", ".swift", ".rb", ".php",
             ".scala", ".lua", ".sh", ".sql", ".vue"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "out", "dist",
             "build", "target", "bin", "obj", ".next", ".apprentice", ".idea"}
MAX_BYTES, WIN, OVL = 200_000, 50, 8

_model = None


def _embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[apprentice] loading embedder {EMB_MODEL} ...")
        _model = SentenceTransformer(EMB_MODEL)
    return _model


def _chunk_py(src):
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _chunk_generic(src)
    lines, covered = src.splitlines(), set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(src, node) or ""
            kind = "class" if isinstance(node, ast.ClassDef) else "func"
            out.append({"name": node.name, "type": kind, "lineno": node.lineno, "code": seg})
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                covered.add(ln)
    top = [lines[i] for i in range(len(lines)) if (i + 1) not in covered and lines[i].strip()]
    if top:
        out.append({"name": "(module)", "type": "module", "lineno": 1, "code": "\n".join(top)[:1500]})
    return out


def _chunk_generic(src):
    lines, out, i = src.splitlines(), [], 0
    while i < len(lines):
        seg = "\n".join(lines[i:i + WIN])
        if seg.strip():
            out.append({"name": f"L{i+1}", "type": "block", "lineno": i + 1, "code": seg})
        i += WIN - OVL
    return out


def _collect(roots):
    if isinstance(roots, str):
        roots = [roots]
    meta = []
    for root in roots:
        root = os.path.abspath(root)
        repo = os.path.basename(root.rstrip("/\\"))
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in fns:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in CODE_EXTS:
                    continue
                fp = os.path.join(dp, fn)
                try:
                    if os.path.getsize(fp) > MAX_BYTES:
                        continue
                    src = open(fp, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                rel = repo + "/" + os.path.relpath(fp, root).replace("\\", "/")
                for ch in (_chunk_py(src) if ext == ".py" else _chunk_generic(src)):
                    if ch["code"].strip():
                        ch["file"] = rel
                        meta.append(ch)
    return meta


def build_index(roots):
    meta = _collect(roots)
    os.makedirs(STORE, exist_ok=True)
    texts = [f"# {m['file']} :: {m['name']}\n{m['code']}" for m in meta]
    emb = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    np.savez(IDX, emb=emb.astype(np.float32))
    json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[apprentice] indexed {len(meta)} chunks -> {STORE}/")
    return len(meta)


def search(query, k=6):
    emb = np.load(IDX)["emb"]
    meta = json.load(open(META, encoding="utf-8"))
    q = _embedder().encode([query], normalize_embeddings=True)[0]
    sims = emb @ q
    return [(float(sims[i]), meta[i]) for i in np.argsort(-sims)[:k]]


def ask(query, k=6, model=None, use_lessons=True):
    hits = search(query, k)
    ctx = "\n\n".join(f"[{m['file']}:{m['lineno']} — {m['name']}]\n{m['code']}" for _, m in hits)
    know = ""
    if use_lessons:                       # self-improvement: inject relevant past corrections
        from . import lessons             # lazy import (lessons imports rag)
        block = lessons.as_prompt(query)
        if block:
            know = block + "\n\n"
    sys_p = ("You are a coding assistant. Use the [codebase] snippets for how the code works and any "
             "[project knowledge] for conventions or facts not in the snippets — project knowledge is "
             "authoritative and overrides generic assumptions. Cite file:line. If neither covers it, "
             "say so. Reply in the user's language.")
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": f"{know}[codebase]\n{ctx}\n\n[question]\n{query}"}]
    return llm.chat(msgs, model=model, temperature=0.2, max_tokens=700), hits
