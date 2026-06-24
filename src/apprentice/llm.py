"""Model-agnostic LLM client — bring-your-own brain, two backends:

1. OpenAI-compatible HTTP (default) — any /v1/chat/completions endpoint, BYO key.
     APPRENTICE_LLM_BASE_URL  (default https://api.openai.com/v1)
     APPRENTICE_LLM_API_KEY   (Bearer; empty for local servers)
     APPRENTICE_LLM_MODEL     (default gpt-4o-mini)

2. Claude CLI (no API key — uses your logged-in Claude account):
     APPRENTICE_LLM_BACKEND=claude-cli
   Shells out to `claude -p` (Claude Code in print mode). Great when you want a
   frontier brain without managing an API key.

Stdlib only.
"""
import os
import json
import shutil
import tempfile
import subprocess
import urllib.request


def _env(k: str, d: str = "") -> str:
    return os.environ.get(k, d)


_NO_TOOLS = "Bash Edit Write Read Glob Grep WebSearch WebFetch Task TodoWrite NotebookEdit"


def _claude_cli(messages, timeout) -> str:
    """Use `claude -p` (Claude Code, account auth — no API key) as a clean chat function.
    Tamed into a stateless responder: system prompt replaced, tools disabled, and user/project
    memory (CLAUDE.md) not loaded — otherwise it runs as a full agent instead of answering."""
    sys_p, parts = "", []
    for m in messages:
        if m["role"] == "system":
            sys_p = (sys_p + "\n\n" + m["content"]).strip()
        elif m["role"] == "user":
            parts.append(m["content"])
        else:
            parts.append(f"[assistant]\n{m['content']}")
    exe = shutil.which("claude") or "claude"
    cmd = [exe, "-p", "--setting-sources", "", "--disallowedTools", _NO_TOOLS]
    if sys_p:
        cmd += ["--system-prompt", sys_p]
    # prompt via stdin (avoids OS command-line length limits on large contexts);
    # run from a neutral dir so Claude Code doesn't inject the caller's git/project context.
    r = subprocess.run(cmd, input="\n\n".join(parts), cwd=tempfile.gettempdir(),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    out = (r.stdout or "").strip()
    if not out and r.stderr:
        raise RuntimeError(r.stderr.strip()[:300])
    return out


def _openai(messages, model, temperature, max_tokens, timeout) -> str:
    base = _env("APPRENTICE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = _env("APPRENTICE_LLM_API_KEY", "")
    model = model or _env("APPRENTICE_LLM_MODEL", "gpt-4o-mini")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r["choices"][0]["message"]["content"].strip()


def chat(messages, model=None, temperature=0.2, max_tokens=800, timeout=240):
    backend = _env("APPRENTICE_LLM_BACKEND", "openai").lower()
    if backend in ("claude-cli", "claude"):
        return _claude_cli(messages, timeout)
    return _openai(messages, model, temperature, max_tokens, timeout)
