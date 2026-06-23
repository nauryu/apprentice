"""Model-agnostic LLM client — any OpenAI-compatible endpoint, bring-your-own-key.

Config via env:
  APPRENTICE_LLM_BASE_URL   (default https://api.openai.com/v1)
  APPRENTICE_LLM_API_KEY    (Bearer; empty for local servers)
  APPRENTICE_LLM_MODEL      (default gpt-4o-mini)

Zero hard dependencies (stdlib only) so the core stays portable.
"""
import os
import json
import urllib.request


def _env(k: str, d: str = "") -> str:
    return os.environ.get(k, d)


def chat(messages, model=None, temperature=0.2, max_tokens=800, timeout=180):
    base = _env("APPRENTICE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = base + "/chat/completions"
    key = _env("APPRENTICE_LLM_API_KEY", "")
    model = model or _env("APPRENTICE_LLM_MODEL", "gpt-4o-mini")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r["choices"][0]["message"]["content"].strip()
