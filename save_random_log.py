import json, os

p = os.path.expanduser("~/.config/colab-cli/history/only-eval.jsonl")
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

if os.path.exists(p):
    lines_raw = open(p, encoding='utf-8', errors='ignore').readlines()
    target_text = None
    for l in reversed(lines_raw):
        try:
            r = json.loads(l)
            if r.get('event_type') == 'execution' and 'outputs' in r:
                out = r['outputs']
                full_s = ""
                if isinstance(out, str): full_s = out
                elif isinstance(out, list):
                    chunks = []
                    for item in out:
                        if isinstance(item, dict): chunks.append(item.get('text', ''))
                        elif isinstance(item, str): chunks.append(item)
                    full_s = "".join(chunks)
                if "acc: 81.63" in full_s or "acc: 81.6" in full_s:
                    target_text = full_s
                    break
        except Exception:
            pass

    if target_text:
        txt_path = os.path.join(outdir, "pope_random_recovered.txt")
        tsv_path = os.path.join(outdir, "pope_random_recovered.tsv")
        with open(txt_path, "w") as f:
            f.write(target_text)
        
        lines = target_text.splitlines()
        rows = []
        i, n = 0, len(lines)
        while i < n:
            if lines[i].lstrip().startswith("V: ["):
                v = lines[i].lstrip()[3:]
                q = a = gt = ""
                j = i + 1
                while j < n and j < i + 8:
                    t = lines[j]
                    if t.startswith("Q: "): q = t[3:]
                    elif t.startswith("A: "): a = t[3:]
                    elif t.startswith("GT: "): gt = t[4:]
                    if q and a and gt: break
                    j += 1
                rows.append((v, q, a, gt))
                i = j + 1
            else:
                i += 1
                
        with open(tsv_path, "w") as f:
            for v, q, a, gt in rows:
                f.write(f"{v}\t{q}\t{a}\t{gt}\n")
        print(f"Successfully recovered {len(rows)} questions for POPE random!")
        print(f"  Txt log: {txt_path}")
        print(f"  TSV log: {tsv_path}")
    else:
        print("Could not find acc: 81.63 in execution history.")
