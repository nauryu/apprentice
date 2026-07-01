"""
Tiny local retrieval service so the VS Code panel (Node) can reach the Python embedder.
Stdlib only, bound to localhost. Two POST endpoints, JSON body {"query": ..., "k": ...}:

  /lessons/search -> {"knowledge": "<[project knowledge] block, or ''>", "hits": [...]}
  /code/search    -> {"results": "<formatted code snippets>"}

  apprentice serve 8799

The panel injects the returned `knowledge` as an authoritative [project knowledge] block — the
same injection that measured 0/4 -> 4/4 fact accuracy — so it self-improves at query time.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _handler():
    from . import lessons, rag

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):        # keep the console quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                q, k = req.get("query", ""), int(req.get("k", 6))
                path = self.path.rstrip("/")
                if path == "/lessons/search":
                    hits = lessons.retrieve(q, k=k)
                    self._send(200, {"knowledge": lessons.as_prompt(q, k=k),
                                     "hits": [{"score": s, "lesson": m["lesson"]} for s, m in hits]})
                elif path == "/code/search":
                    hits = rag.search(q, k)
                    res = "\n\n".join(f"[{m['file']}:{m['lineno']} {m['name']}]\n{m['code']}" for _, m in hits)
                    self._send(200, {"results": res})
                else:
                    self._send(404, {"error": f"unknown path {path}"})
            except Exception as e:
                self._send(500, {"error": str(e)})

    return H


def serve(port=8799):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler())
    print(f"[apprentice] retrieval service on http://127.0.0.1:{port}  (/lessons/search, /code/search)")
    print("[apprentice] set apprentice.lessonsUrl and apprentice.codeRagUrl in VS Code to these paths.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[apprentice] stopped")
