"""Check the built report (ROADMAP 9.1): page count, references, citations, leftovers.

Run after ``latexmk -pdf main.tex`` in ``report/``. Exit code 1 if a check fails.

    python scripts/check_report.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_PAGES, MAX_PAGES = 12, 15
FLOAT_PREFIXES = ("fig:", "tab:", "alg:", "eq:")
BANNED = ("novel", "state-of-the-art", "state of the art", "best-in-class")


def sources(report: Path) -> dict[str, str]:
    """``{relative path: text}`` of the hand-written LaTeX sources (comments removed)."""
    files = [report / "main.tex", *sorted((report / "chapters").glob("*.tex")),
             report / "figures" / "algorithm.tex"]  # fmt: skip
    return {
        str(f.relative_to(report)): re.sub(r"(?<!\\)%.*", "", f.read_text())
        for f in files if f.exists()
    }  # fmt: skip


def labels(text: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", text))


def references(text: str) -> set[str]:
    out: set[str] = set()
    for group in re.findall(r"\\[Cc]ref\{([^}]+)\}", text):
        out.update(k.strip() for k in group.split(","))
    return out


def citations(text: str) -> set[str]:
    out: set[str] = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", text):
        out.update(k.strip() for k in group.split(","))
    return out


def bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,\s]+),", bib, flags=re.M))


def static_problems(report: Path) -> list[str]:
    """Problems that can be found without compiling."""
    src = sources(report)
    text = "\n".join(src.values())
    problems = []
    unreferenced = {k for k in labels(text) if k.startswith(FLOAT_PREFIXES)} - references(text)
    problems += [f"label never referenced: {k}" for k in sorted(unreferenced)]
    unknown = references(text) - labels(text)
    problems += [f"reference to unknown label: {k}" for k in sorted(unknown)]
    missing = citations(text) - bib_keys((report / "references.bib").read_text())
    problems += [f"citation not in references.bib: {k}" for k in sorted(missing)]
    for name, body in src.items():
        if "TODO" in body:
            problems.append(f"TODO left in {name}")
        low = body.lower()
        problems += [f"banned wording '{w}' in {name}" for w in BANNED if w in low]
    return problems


def placeholders(report: Path) -> list[str]:
    return sorted(set(re.findall(r"\[(?:Author Name|Roll No\.|E-mail|Department)\]",
                                 (report / "main.tex").read_text())))  # fmt: skip


def build_problems(report: Path) -> tuple[list[str], int | None]:
    pdf, log = report / "build" / "main.pdf", report / "build" / "main.log"
    if not pdf.exists() or not log.exists():
        return [f"no build found ({pdf}); run `latexmk -pdf main.tex` in report/"], None
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    problems = []
    if not MIN_PAGES <= pages <= MAX_PAGES:
        problems.append(f"{pages} pages (allowed {MIN_PAGES}-{MAX_PAGES})")
    text = log.read_text(errors="replace")
    checks = (
        (r"Reference `[^']+' on page \d+ undefined", "undefined reference"),
        (r"Citation `[^']+' on page \d+ undefined", "undefined citation"),
        (r"multiply defined", "multiply defined label"),
        (r"Overfull \\hbox \((?:[1-9]\d|[5-9])\.", "overfull line (> 5 pt)"),
    )
    for pattern, what in checks:
        n = len(re.findall(pattern, text))
        if n:
            problems.append(f"{n} x {what} in build/main.log")
    return problems, pages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "report")
    args = ap.parse_args(argv)
    problems = static_problems(args.report)
    built, pages = build_problems(args.report)
    problems += built
    if pages is not None:
        print(f"pages: {pages}")
    for p in placeholders(args.report):
        print(f"reminder: title-page placeholder still to fill: {p}")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("report check: " + ("FAILED" if problems else "ok"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
