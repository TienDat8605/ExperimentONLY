#!/usr/bin/env python3
"""Recover POPE eval logs for ANY/all setups from colab-cli's exec history.

WHY THIS EXISTS
  The `colab exec` streamed output is the ONLY complete local copy of every
  per-question answer.  The Colab-side `results.tar.gz` download can silently
  fail or return a STALE tarball from a previous run (that is exactly what
  bit us on the adversarial run: the auto-download returned nothing usable and
  the friendly session name was lost, yet the full 3000-Q stream survived in
  ~/.config/colab-cli/history/).  This tool is the safety net that ALWAYS
  works as long as `colab exec` ran at all: it reads the JSONL history the CLI
  keeps locally and reconstructs the streamed logs.

WHAT IT WRITES (into ./logs/)
  pope_<setup>_recovered.txt   the per-setup streamed block (V/Q/A/GT/acc...)
  pope_<setup>_recovered.tsv   tidy V/Q/A/GT extract, one row per question
  pope_recovered_summary.txt   one line per setup: final acc line + question cnt
                               (printed to stdout too)

USAGE
  python3 recover_pope_log.py                  # auto: pick latest only-eval*.jsonl
  python3 recover_pope_log.py --hist <file>   # explicit history JSONL
  python3 recover_pope_log.py --setup random  # only one setup (else all found)
  python3 recover_pope_log.py --all           # scan EVERY only-eval*.jsonl, last run each
"""
import argparse, glob, json, os, re, sys

HOME = os.path.expanduser("~")
HIST_DIR = os.path.join(HOME, ".config/colab-cli/history/")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Markers emitted by colab.sh run (run_all.sh prints none of these; only the
# in-VM colab.sh does).  They delimit each POPE setup inside one streamed run.
SETUP_HDR = ">>> Setup:"                # e.g. "[05:xx] [run/4] >>> Setup: random ..."
RUN_HDR = "=== colab_run.py started ==="  # start of each `colab exec` run

SETUP_RE = re.compile(r">>>\s*Setup:\s*([A-Za-z0-9_]+)")
ACC_RE = re.compile(r"acc:\s*([0-9.]+),\s*precision:.*yes_ratio:\s*([0-9.]+)")


def newest_hist_files(pattern):
    """All history JSONLs matching `only-eval*`, newest by mtime first."""
    files = glob.glob(os.path.join(HIST_DIR, pattern))
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files


def extract_full_stream(path):
    """Concatenate every `execution` record's outputs into one big string."""
    chunks = []
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event_type") != "execution" or "outputs" not in r:
            continue
        out = r["outputs"]
        if isinstance(out, str):
            chunks.append(out)
        elif isinstance(out, list):
            for c in out:
                if isinstance(c, dict):
                    chunks.append(c.get("text", ""))
                elif isinstance(c, str):
                    chunks.append(c)
        elif isinstance(out, dict):
            chunks.append(json.dumps(out))
    return "".join(chunks) if chunks else None


def split_runs(stream):
    """Split the combined stream into per-run chunks (one per `colab exec`)."""
    # Keep the header line on each chunk.
    parts = re.split(r"(?=" + re.escape(RUN_HDR) + r")", stream)
    return [p for p in parts if p.strip()]


def split_setups(run):
    """Within one run, split into {setup: block_text} by the Setup header.

    Headers look like:  '[05:00] [run/4] >>> Setup: random  (Questions: ...)'
    A block runs from its Setup header up to the next Setup header (or to the
    end / the '[run/5] Packaging' packing line).
    """
    # Find all setup-header match positions.
    idx = [m.start() for m in re.finditer(r".*" + SETUP_HDR + r".*", run)]
    if not idx:
        # Legacy single-setup run: no Setup markers.  Attribute by --type in args.
        m = re.search(r"'type':\s*'([A-Za-z0-9_]+)'", run)
        setup = m.group(1) if m else "unknown"
        return {setup: run}
    out = {}
    for i, pos in enumerate(idx):
        hdr_end = run.index("\n", pos) + 1 if "\n" in run[pos:] else len(run)
        m = SETUP_RE.search(run[pos:hdr_end])
        setup = m.group(1) if m else f"setup{i}"
        end = idx[i + 1] if i + 1 < len(idx) else len(run)
        # Trim trailing packaging/summary lines that belong to run/5, not the eval.
        block = run[pos:end]
        pack = block.find("[run/5] Packaging")
        if pack != -1:
            block = block[:pack].rstrip() + "\n"
        out[setup] = block
    return out


def extract_qa(block):
    """V/Q/A/GT rows from a single setup's streamed block."""
    rows = []
    lines = block.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].lstrip().startswith("V: ["):
            v = lines[i].lstrip()[3:]
            q = a = gt = ""
            j = i + 1
            while j < n and j < i + 8:
                t = lines[j]
                if t.startswith("Q: "):
                    q = t[3:]
                elif t.startswith("A: "):
                    a = t[3:]
                elif t.startswith("GT: "):
                    gt = t[4:]
                if q and a and gt:
                    break
                j += 1
            rows.append((v, q, a, gt))
            i = j + 1
        else:
            i += 1
    return rows


def final_acc(block):
    """The LAST acc: line (overall summary, not the running per-question one)."""
    hits = ACC_RE.findall(block)
    return f"acc: {hits[-1][0]}, yes_ratio: {hits[-1][1]}" if hits else "(no acc line found)"


def write_setup(setup, block, outdir, want=None):
    if want and setup != want:
        return None
    full = os.path.join(outdir, f"pope_{setup}_recovered.txt")
    qa_p = os.path.join(outdir, f"pope_{setup}_recovered.tsv")
    with open(full, "w") as f:
        f.write(block)
    rows = extract_qa(block)
    with open(qa_p, "w") as f:
        for v, q, a, gt in rows:
            f.write(f"{v}\t{q}\t{a}\t{gt}\n")
    return (setup, full, qa_p, len(rows), final_acc(block))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", help="explicit history JSONL file")
    ap.add_argument("--setup", help="only recover this setup (else all found)")
    ap.add_argument("--all", action="store_true",
                    help="scan every only-eval*.jsonl; use the last run in each")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    if args.hist:
        targets = [args.hist]
    elif args.all:
        targets = newest_hist_files("only-eval*.jsonl")
    else:
        # Default: the stable session file if present, else newest only-eval*.
        stable = os.path.join(HIST_DIR, "only-eval.jsonl")
        if os.path.exists(stable):
            targets = [stable]
        else:
            targets = newest_hist_files("only-eval*.jsonl")[:1]

    if not targets or not any(os.path.exists(t) for t in targets):
        sys.exit("No history JSONL found in %s" % HIST_DIR)

    summary = []
    recovered_setups = set()
    for hist in targets:
        if not os.path.exists(hist):
            continue
        stream = extract_full_stream(hist)
        if not stream:
            continue
        runs = split_runs(stream)
        run = runs[-1] if runs else stream   # latest run = the one we want
        setups = split_setups(run)
        for setup, block in setups.items():
            res = write_setup(setup, block, OUTDIR, want=args.setup)
            if res:
                setup, full, qa_p, nq, acc = res
                if setup in recovered_setups:
                    continue
                recovered_setups.add(setup)
                summary.append((setup, nq, acc, os.path.basename(full)))

    if not summary:
        sys.exit("No setup blocks recovered (no `>>> Setup:` markers and no "
                 "`'type':` arg dump found in %s)" % targets[0])

    lines = ["# POPE recovered-log summary"]
    for setup, nq, acc, full in summary:
        lines.append(f"  {setup:12s} | {nq:5d} questions | {acc} | {full}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUTDIR, "pope_recovered_summary.txt"), "w") as f:
        f.write(text)
    print(text)
    print("Recovered files written to: %s" % OUTDIR)


if __name__ == "__main__":
    main()