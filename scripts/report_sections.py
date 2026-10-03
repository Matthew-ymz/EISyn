"""Update one generated section without replacing a consolidated report."""
from __future__ import annotations

from pathlib import Path
import re


SECTIONS = {
    "dmf-subject-curves": ("brain.md", "J", "个体指标曲线、动力学转折与形状比较"),
    "dmf-ei-components": ("brain.md", "K", "整体 EI、部分 EI 之和与整合峰"),
    "dmf-response-pilot": ("brain.md", "M.1", "刺激响应 A–C 预实验负结果"),
    "dmf-joint-pilot": ("brain.md", "M.2", "联合符号读取预实验负结果"),
    "dmf-subject-dense": ("brain.md", "P", "93 人 DMF 细扫描"),
    "sine-frequency": ("exploration.md", "B", "固定干预支持下的 Sine 校准"),
}


def format_section(content: str, prefix: str, title: str, anchor: str) -> str:
    """Nest Markdown headings, leaving code blocks and their contents intact."""
    nested = "." in prefix
    level_offset = 2 if nested else 1
    first_heading = True
    in_fence = False
    lines = []
    for line in content.strip().splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else re.match(r"^(#{1,6}) (.+)$", line)
        if match:
            level, heading = len(match[1]), match[2]
            if first_heading and level == 1:
                first_heading = False
                heading = f"{prefix} {title}" if nested else f"附录 {prefix}：{title}"
            else:
                heading = re.sub(r"^(\d+(?:\.\d+)*)[.、]?\s+", rf"{prefix}.\1 ", heading)
            line = "#" * min(6, level + level_offset) + " " + heading
        lines.append(line)
    return f'<a id="{anchor}"></a>\n\n' + "\n".join(lines) + "\n"


def section_block(key: str, content: str) -> str:
    _, prefix, title = SECTIONS[key]
    body = format_section(content, prefix, title, key)
    return f"<!-- report-section:{key}:start -->\n{body}<!-- report-section:{key}:end -->"


def write_report_section(path: Path, key: str, content: str) -> Path:
    """Preserve custom standalone output paths; update canonical files by marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name != SECTIONS[key][0]:
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    start, end = (f"<!-- report-section:{key}:{edge} -->" for edge in ("start", "end"))
    if existing.count(start) != existing.count(end) or existing.count(start) > 1:
        raise ValueError(f"Invalid report section markers for {key} in {path}")
    block = section_block(key, content)
    if start in existing:
        first, last = existing.index(start), existing.index(end) + len(end)
        if last < first:
            raise ValueError(f"Reversed report section markers for {key} in {path}")
        updated = existing[:first] + block + existing[last:]
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")
    return path
