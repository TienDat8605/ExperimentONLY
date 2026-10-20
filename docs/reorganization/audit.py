"""Regenerate the read-only inventory and proposed move audit. Never moves files."""

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PEER = ROOT.parent / "experiment"
OUT = Path(__file__).resolve().parent
EXCLUDED = {".git", ".agents", ".codex", ".aws", "__pycache__", ".venv", ".venv-proposal4", "venv"}
MOVES = [
    ("paper/ACOMPA26_P02", "paper/versions/initial", "product: earlier manuscript"),
    ("write-paper/ACOMPA26_P02", "paper/versions/july-rewrite", "product: earlier rewrite"),
    ("ACOMPA26_P02", "paper/ACOMPA26_P02", "product: active manuscript, preserve all variants"),
    ("write-paper/main_proposal6_rewrite_with_mme.md", "docs/method/main_proposal6_rewrite_with_mme.md", "product: earlier writing draft"),
    ("ONLY_report.md", "docs/method/ONLY_report.md", "factory: method reference"),
    ("CMD_abstract_intro_draft.md", "docs/method/CMD_abstract_intro_draft.md", "product: earlier writing draft"),
    ("CMD_method_draft.md", "docs/method/CMD_method_draft.md", "product: earlier writing draft"),
    ("CMD_experiments_draft.md", "docs/method/CMD_experiments_draft.md", "product: earlier writing draft"),
]


def walk(root):
    for directory, folders, names in os.walk(root, followlinks=False):
        folders[:] = sorted(n for n in folders if n not in EXCLUDED)
        for name in sorted(names + [n for n in folders if (Path(directory) / n).is_symlink()]):
            yield Path(directory) / name


def write_csv(name, columns, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    if (OUT / "migration-result.json").exists():
        print("Original inventory is a preserved pre-migration snapshot; verifying current destinations.", flush=True)
        raise SystemExit(subprocess.call([sys.executable, str(OUT / "verify.py")]))
    inventory, links, texts, counts = [], [], [], {}
    for root in (ROOT, PEER):
        if not root.is_dir():
            continue
        tracked = set(subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0"))
        paths = list(walk(root))
        counts[root.name] = len(paths)
        for path in paths:
            relative = path.relative_to(root)
            if root == ROOT and relative.parts[:2] == ("docs", "reorganization"):
                continue  # Audit output must not recursively inventory or cite itself.
            stat = path.lstat()
            role = "factory: source/runtime"
            if relative.parts[0] in {"logs", "paper", "write-paper", "ACOMPA26_P02"}:
                role = "product"
            elif relative.parts[0] in {"data", "models"}:
                role = "factory: assets"
            elif path.name == "README.md":
                role = "catalog"
            elif path.suffix == ".md":
                role = "reference/draft: inspect migration role"
            inventory.append([root.name, str(relative), role, stat.st_size,
                              datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                              str(relative) in tracked, path.is_symlink()])
            if path.is_symlink():
                links.append([str(path), os.readlink(path), str(path.resolve()), path.exists()])
                continue
            # Search bounded source/documentation surfaces in both repositories.
            if relative.parts[0] in {"data", "models", "logs"} or stat.st_size > 2_000_000:
                continue
            if path.suffix in {".py", ".sh", ".md", ".tex", ".bib", ".toml", ".yaml", ".yml", ".json"}:
                texts.append((path, path.read_text(errors="replace").splitlines()))
    write_csv("inventory.csv", ["repository", "path", "role", "bytes", "mtime_utc", "git_tracked", "symlink"], inventory)
    write_csv("symlinks.csv", ["path", "target", "resolved", "exists"], links)
    manifests, references, collisions = [], [], []
    existing = {str(p.relative_to(ROOT)).casefold(): str(p.relative_to(ROOT)) for p in walk(ROOT)}
    for order, (source, destination, role) in enumerate(MOVES, 1):
        src = ROOT / source
        for path in sorted(walk(src)) if src.is_dir() else [src]:
            suffix = path.relative_to(src) if src.is_dir() else Path()
            dest = Path(destination) / suffix if src.is_dir() else Path(destination)
            manifests.append([order, str(path.relative_to(ROOT)), str(dest), role,
                              path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()])
            resolved_source = path.resolve()
            if str(dest).casefold() in existing:
                collisions.append([str(dest), existing[str(dest).casefold()], "vacate paper/ACOMPA26_P02 in step 1 before step 3" if order == 3 else "REVIEW"])
            for referrer, lines in texts:
                for number, line in enumerate(lines, 1):
                    latex_targets = re.findall(r"\\(?:input|include|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}", line)
                    resolved_latex = [(referrer.parent / (name if Path(name).suffix else name + ".tex")).resolve() for name in latex_targets]
                    if str(path.relative_to(ROOT)) in line or path.name in line or resolved_source in resolved_latex:
                        references.append([str(path.relative_to(ROOT)), str(referrer), number, line.strip()])
    write_csv("migration.csv", ["order", "source", "destination", "role", "bytes", "sha256"], manifests)
    write_csv("references.csv", ["source", "referrer", "line", "text"], references)
    write_csv("collisions.csv", ["destination", "existing_casefold_match", "resolution"], collisions)
    main_tex = ROOT / "ACOMPA26_P02/main.tex"
    entry = main_tex.read_text()
    includes = re.findall(r"\\input\{([^}]+)\}", entry)
    content = "\n".join((main_tex.parent / (name + ".tex")).read_text() for name in includes)
    labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", content))
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", content))
    cites = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", content) for key in group.split(",")}
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", content)
    summary = {
        "repository_file_counts_including_audit": counts,
        "proposed_move_files": len(manifests),
        "source_hashes_recorded": len(manifests),
        "existing_destination_matches": len(collisions),
        "unexpected_collisions": [row for row in collisions if row[2] == "REVIEW"],
        "duplicate_proposed_destinations": len(manifests) - len({row[2].casefold() for row in manifests}),
        "symlinks": len(links),
        "active_latex_inputs": includes,
        "missing_inputs": [name for name in includes if not (main_tex.parent / (name + ".tex")).exists()],
        "undefined_latex_refs": sorted(refs - labels),
        "undefined_citation_keys": sorted(cites - bib),
        "missing_figure_files": [name for name in figures if not (main_tex.parent / name).exists()],
        "moves_executed": False,
    }
    (OUT / "audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
