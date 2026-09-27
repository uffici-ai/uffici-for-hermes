"""This project's Markdown link guard.

🔴 It exists because the platform's own doc-link guard cannot reach here.
that repository's own doc-link guard scans its instruction set and the
root ``CLAUDE.md`` only, so every relative link in this child's Markdown is
checked by nothing on the platform side, and by nothing at all once the child is
mirrored out. The check therefore lives in this project's own toolchain, inside
its own ``check`` entry.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The target of a Markdown inline link, ``[text](target)``.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return sorted(p for p in PROJECT_ROOT.rglob("*.md") if ".git" not in p.parts)


def test_every_relative_markdown_link_resolves() -> None:
    files = markdown_files()
    # 🔴 Non-vacuity: an empty file set would make the loop below assert
    # nothing, forever (defect-ledger class 1).
    assert len(files) >= 2, (
        f"found {len(files)} Markdown file(s) under {PROJECT_ROOT} — README.md and "
        "CLAUDE.md are both expected, so the walk is broken"
    )

    checked = 0
    broken: list[str] = []
    for file in files:
        for target in MARKDOWN_LINK.findall(file.read_text(encoding="utf-8")):
            target = target.strip()
            if not target or "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            if not (file.parent / target).exists():
                broken.append(f"{file.relative_to(PROJECT_ROOT)} links {target!r}")

    assert checked > 0, "no relative Markdown links were checked — this guard asserts nothing"
    assert broken == [], "dangling relative link(s): " + ", ".join(broken)
