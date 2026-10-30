#!/usr/bin/env python3
"""Render the source-code copy (소스코드 복제물) for the program copyright
registration as a single PDF.

Scope is exactly the code this project wrote: every git-tracked .py/.sh file
under src/, plugins/ and scripts/, *excluding* any plugins/**/vendor/**
directory and the upstream training dirs in UPSTREAM_DIRS (third-party code,
see NOTICE.md) and anything outside git (raw
data, results, caches). Files are concatenated in path order, each under a
header with its path; pages are numbered and carry the program name and the
git revision the copy was taken from, so the PDF can be matched to a tag.

    .venv/bin/pip install -e '.[registration]'      # once (reportlab)
    .venv/bin/python scripts/export_source_pdf.py --out /path/to/source.pdf
    .venv/bin/python scripts/export_source_pdf.py --out excerpt.pdf --excerpt 25

--excerpt N keeps only the first N and last N pages (the usual form when the
full source isn't to be deposited). Writes outside the repo by default
(DATA_ROOT/registration/), since the PDF is a derived artifact, not source.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
PROGRAM_NAME = "OmniLogue-Benchmark"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE = 7.5
LINE_H = 9.2
MARGIN = 40
WRAP = 118  # characters per printed line at this font size on A4


# Upstream training code that predates the vendor/ convention and still lives
# outside vendor/ (each is listed in NOTICE.md). Third-party, so not ours to
# register -- excluded exactly like vendor/.
UPSTREAM_DIRS = (
    "plugins/methods/bob/training/",
    "plugins/methods/lmedr/training/",
    "plugins/methods/orig/training/",
    "plugins/methods/paa/training/",
)


def source_files() -> list:
    out = subprocess.run(
        ["git", "ls-files", "src", "plugins", "scripts"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.split()
    return sorted(
        p
        for p in out
        if p.endswith((".py", ".sh")) and "/vendor/" not in p and not p.startswith(UPSTREAM_DIRS)
    )


def revision() -> str:
    def _git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    tag = _git("describe", "--tags", "--exact-match")
    return tag or _git("rev-parse", "--short", "HEAD")


def build_lines(files: list) -> list:
    """[(text, is_header)] with long lines hard-wrapped."""
    lines = []
    for path in files:
        lines.append((f"===== {path} =====", True))
        text = (ROOT / path).read_text(encoding="utf-8", errors="replace").expandtabs(4)
        for n, raw in enumerate(text.splitlines(), start=1):
            prefix = f"{n:>5} "
            body = raw or ""
            first = True
            while first or body:
                chunk, body = body[: WRAP - len(prefix)], body[WRAP - len(prefix) :]
                lines.append(((prefix if first else " " * len(prefix)) + chunk, False))
                first = False
        lines.append(("", False))
    return lines


def paginate(lines: list) -> list:
    per_page = int((A4[1] - 2 * MARGIN - 2 * LINE_H) // LINE_H)
    return [lines[i : i + per_page] for i in range(0, len(lines), per_page)]


def render(pages: list, out: Path, rev: str, total: int, page_numbers: list) -> None:
    pdfmetrics.registerFont(TTFont("Mono", FONT_PATH))
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setTitle(f"{PROGRAM_NAME} source ({rev})")
    width, height = A4
    for page, number in zip(pages, page_numbers):
        c.setFont("Mono", FONT_SIZE)
        c.drawString(MARGIN, height - MARGIN + 12, f"{PROGRAM_NAME}  [{rev}]")
        c.drawRightString(width - MARGIN, MARGIN - 18, f"{number} / {total}")
        y = height - MARGIN - LINE_H
        for text, is_header in page:
            c.setFillGray(0.0 if not is_header else 0.15)
            c.drawString(MARGIN, y, text)
            y -= LINE_H
        c.showPage()
    c.save()


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from dialogue_benchmarks._paths import DATA_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--excerpt", type=int, default=None, help="keep only the first and last N pages")
    args = parser.parse_args()

    rev = revision()
    out = args.out or DATA_ROOT / "registration" / f"{PROGRAM_NAME}_{rev}_source.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    files = source_files()
    lines = build_lines(files)
    pages = paginate(lines)
    total = len(pages)
    numbers = list(range(1, total + 1))
    if args.excerpt and total > 2 * args.excerpt:
        pages = pages[: args.excerpt] + pages[-args.excerpt :]
        numbers = numbers[: args.excerpt] + numbers[-args.excerpt :]
    render(pages, out, rev, total, numbers)

    n_code = sum(1 for _, h in lines if not h)
    print(f"{out}\n  revision {rev}: {len(files)} files, ~{n_code} printed lines, {len(pages)} page(s) written")


if __name__ == "__main__":
    main()
