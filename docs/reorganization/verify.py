"""Read-only verification of the completed organization against its source manifest."""

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def latex_graph(entry):
    pending, seen, missing = [entry], set(), []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text()
        text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
        for target in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            child = path.parent / (target if Path(target).suffix else target + ".tex")
            if child.exists():
                pending.append(child)
            else:
                missing.append(str(child.relative_to(ROOT)))
        for target in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            figure = path.parent / target
            if not any(p.exists() for p in [figure] + [figure.with_suffix(ext) for ext in (".pdf", ".png", ".jpg")]):
                missing.append(str(figure.relative_to(ROOT)))
    return seen, missing


def main():
    errors = []
    rows = list(csv.DictReader((HERE / "migration.csv").open()))
    for row in rows:
        destination = ROOT / row["destination"]
        if not destination.is_file():
            errors.append(f"Missing destination: {destination}")
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
            errors.append(f"Content differs from pre-migration source: {destination}")
    alias = ROOT / "ACOMPA26_P02"
    if not alias.is_symlink() or alias.resolve() != ROOT / "paper/ACOMPA26_P02":
        errors.append("Current-paper compatibility symlink is incorrect")
    entries = [ROOT / name for name in (
        "paper/ACOMPA26_P02/main.tex", "paper/versions/initial/main.tex",
        "paper/versions/july-rewrite/main.tex", "paper/versions/july-rewrite/main-old.tex",
    )]
    graphs = {}
    for entry in entries:
        files, missing = latex_graph(entry)
        errors.extend(f"Missing LaTeX dependency: {item}" for item in missing)
        graphs[str(entry.relative_to(ROOT))] = len(files)
    markdown = [ROOT / "AGENTS.md", ROOT / "CONTEXT.md", ROOT / "README.md"]
    markdown += list(ROOT.rglob("CONTEXT.md"))
    markdown += [HERE / name for name in ("PLAN.md", "paper-readout.md", "VALIDATION.md") if (HERE / name).exists()]
    for path in set(markdown):
        for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                errors.append(f"Broken Markdown link in {path.relative_to(ROOT)}: {target}")
    active, _ = latex_graph(entries[0])
    text = "\n".join(p.read_text() for p in active)
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    citation_keys = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    result = {
        "verified_migrated_files": len(rows),
        "latex_input_graph_file_counts": graphs,
        "markdown_files_checked": len(set(markdown)),
        "preexisting_undefined_latex_labels": sorted(refs - labels),
        "undefined_citation_keys": sorted(citation_keys - bibitems),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
