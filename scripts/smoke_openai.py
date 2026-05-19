#!/usr/bin/env python3
import argparse, json, time, urllib.request, sys


def post_json(url, payload, timeout=180):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="nvidia/nemotron-3-super")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = [
        ("arithmetic", "Answer with only the number: 17 + 25 = ?", "42"),
        ("exact", "Return exactly this word and nothing else: pineapple", "pineapple"),
    ]
    results = []
    total_tokens = 0
    t0 = time.time()
    for name, prompt, expect in prompts:
        req = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"force_nonempty_content": True},
        }
        st = time.time()
        try:
            resp = post_json(args.base_url + "/chat/completions", req)
            elapsed = time.time() - st
            content = resp["choices"][0]["message"].get("content") or ""
            usage = resp.get("usage", {})
            ctok = usage.get("completion_tokens") or 0
            total_tokens += ctok
            ok = expect.lower() in content.lower()
            results.append({"name": name, "ok": ok, "elapsed_sec": elapsed, "completion_tokens": ctok, "content": content[:500], "usage": usage})
            print(f"{name}: ok={ok} tokens={ctok} elapsed={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - st
            results.append({"name": name, "ok": False, "elapsed_sec": elapsed, "error": repr(e)})
            print(f"{name}: ERROR {e!r}", file=sys.stderr)
    wall = time.time() - t0
    summary = {"wall_sec": wall, "total_completion_tokens": total_tokens, "aggregate_tok_s": total_tokens / wall if wall else 0, "results": results}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    if not all(r.get("ok") for r in results):
        sys.exit(2)

if __name__ == "__main__":
    main()
