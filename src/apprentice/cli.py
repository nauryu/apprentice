"""apprentice CLI — index / search / ask / eval / flywheel."""
import sys


def main():
    try:                       # robust UTF-8 output on Windows consoles (cp949 etc.)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a = sys.argv[1:]
    if not a:
        print("usage: apprentice [index <roots...> | search <q> | ask <q> | "
              "eval <base.json> <cand.json> | dpo | sft]")
        return
    cmd = a[0]
    if cmd == "index":
        from . import rag
        rag.build_index(a[1:] or ["."])
    elif cmd == "search":
        from . import rag
        for s, m in rag.search(" ".join(a[1:])):
            print(f"[{s:.3f}] {m['file']}:{m['lineno']} — {m['name']}")
    elif cmd == "ask":
        from . import rag
        ans, hits = rag.ask(" ".join(a[1:]))
        print("--- sources ---")
        for s, m in hits:
            print(f"  [{s:.3f}] {m['file']}:{m['lineno']} — {m['name']}")
        print("\n--- answer ---\n" + ans)
    elif cmd == "eval":
        from . import evalharness
        evalharness.compare(a[1], a[2], label_a="base", label_b="candidate")
    elif cmd == "dpo":
        from . import flywheel
        flywheel.build_dpo()
    elif cmd == "sft":
        from . import flywheel
        flywheel.build_sft()
    else:
        print("unknown command:", cmd)


if __name__ == "__main__":
    main()
