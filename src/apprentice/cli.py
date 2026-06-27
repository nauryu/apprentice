"""apprentice CLI — index / search / ask / gen / learn / lessons / eval."""
import sys


def main():
    try:                       # robust UTF-8 output on Windows consoles (cp949 etc.)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a = sys.argv[1:]
    if not a:
        print("usage: apprentice [index <roots...> | search <q> | ask <q> | "
              "gen <questions.json> <out.json> [--no-lessons] | "
              "learn <task> :: <correction> | lessons | digest | eval <base.json> <cand.json>]")
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
    elif cmd == "learn":
        from . import lessons
        rest = " ".join(a[1:])
        if "::" not in rest:
            print("usage: apprentice learn <task> :: <correct answer/guidance>")
            return
        task, right = (s.strip() for s in rest.split("::", 1))
        lessons.correct(task, wrong="", right=right)
    elif cmd == "lessons":
        from . import lessons
        meta = lessons._load_meta()
        print(f"{len(meta)} lesson(s):")
        for m in meta:
            print(f"  - [{m['trigger'][:50]}] {m['lesson']}")
    elif cmd == "digest":
        from . import lessons
        lessons.digest()        # turn panel-captured corrections into lessons
    elif cmd == "gen":
        # generate an answer set over a question file, for the lessons-on vs -off eval
        import json
        from . import rag
        use_lessons = "--no-lessons" not in a
        qs = json.load(open(a[1], encoding="utf-8"))
        out = []
        for o in qs:
            ans, _ = rag.ask(o["q"], use_lessons=use_lessons)
            out.append({**o, "a": ans})
            print(f"  [{o.get('id', '?')}] generated")
        json.dump(out, open(a[2], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[gen] wrote {len(out)} answers (lessons={'on' if use_lessons else 'off'}) -> {a[2]}")
    elif cmd == "eval":
        from . import evalharness
        evalharness.compare(a[1], a[2], label_a="base", label_b="candidate")
    else:
        print("unknown command:", cmd)


if __name__ == "__main__":
    main()
