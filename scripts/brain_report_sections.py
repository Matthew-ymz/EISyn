"""Update bounded generated sections of the consolidated Brain report."""
from __future__ import annotations

from pathlib import Path
import re

BRAIN_REPORT = Path(__file__).resolve().parents[1] / "docs/reports/brain.md"


def section(text: str, heading: str) -> str:
    """Return a heading and its contents, including lower-level subsections."""
    lines = text.splitlines()
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        match = re.match(r"^(#{1,6}) ", lines[i])
        if match and len(match[1]) <= level:
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def demote(text: str, levels: int = 2) -> str:
    return re.sub(r"^(#{1,6})(?= )", lambda m: "#" * min(6, len(m[1]) + levels),
                  text.strip(), flags=re.MULTILINE)


def write_section(section_id: str, body: str, path: Path = BRAIN_REPORT) -> None:
    """Replace only an existing owned block; never overwrite the whole report."""
    start = f"<!-- report-section:{section_id}:start -->"
    end = f"<!-- report-section:{section_id}:end -->"
    text = path.read_text()
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Cannot uniquely locate Brain report section: {section_id}")
    a, b = text.index(start), text.index(end)
    if b < a:
        raise ValueError(f"Reversed Brain report section: {section_id}")
    path.write_text(text[:a] + start + "\n" + body.strip() + "\n" + text[b:])
