from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "doc" / "研究框架.md"
DEFAULT_EXPORT_MD = REPO_ROOT / "doc" / "研究框架_pdf.md"
DEFAULT_OUTPUT_PDF = REPO_ROOT / "doc" / "研究框架.pdf"
DEFAULT_XELATEX = Path("/Library/TeX/texbin/xelatex")
DEFAULT_PANDOC = shutil.which("pandoc") or "/usr/local/bin/pandoc"

HTML_IMG_PATTERN = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+\.svg)(")', re.IGNORECASE)
MARKDOWN_IMG_PATTERN = re.compile(r'(!\[[^\]]*\]\()([^)]+\.svg)(\))', re.IGNORECASE)
HTML_CENTER_BLOCK_PATTERN = re.compile(
    r"<p\s+align=\"center\"\s*>(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
HTML_IMG_TAG_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
HTML_ATTR_PATTERN = re.compile(r'(\w+)="([^"]*)"')


def _svg_to_png(path: str) -> str:
    if path.lower().endswith(".svg"):
        return path[:-4] + ".png"
    return path


def rewrite_markdown_for_pdf_export(markdown_text: str) -> str:
    rewritten = HTML_IMG_PATTERN.sub(
        lambda match: f'{match.group(1)}{_svg_to_png(match.group(2))}{match.group(3)}',
        markdown_text,
    )
    rewritten = MARKDOWN_IMG_PATTERN.sub(
        lambda match: f"{match.group(1)}{_svg_to_png(match.group(2))}{match.group(3)}",
        rewritten,
    )
    rewritten = convert_centered_html_image_blocks(rewritten)
    return rewritten


def _img_tag_to_markdown(tag: str) -> str:
    attrs = dict(HTML_ATTR_PATTERN.findall(tag))
    src = attrs.get("src", "").strip()
    alt = attrs.get("alt", "").strip()
    width = attrs.get("width", "").strip()
    src = _svg_to_png(src)
    markdown = f"![{alt}]({src})"
    if width:
        markdown += f"{{ width={width} }}"
    return markdown


def convert_centered_html_image_blocks(markdown_text: str) -> str:
    def replace_block(match: re.Match[str]) -> str:
        images = [_img_tag_to_markdown(tag) for tag in HTML_IMG_TAG_PATTERN.findall(match.group(1))]
        if not images:
            return match.group(0)
        return "\n\n" + " ".join(images) + "\n\n"

    return HTML_CENTER_BLOCK_PATTERN.sub(replace_block, markdown_text)


def collect_svg_image_references(markdown_text: str) -> list[str]:
    refs: list[str] = []
    for pattern in (HTML_IMG_PATTERN, MARKDOWN_IMG_PATTERN):
        refs.extend(match.group(2) for match in pattern.finditer(markdown_text))
    return list(dict.fromkeys(refs))


def ensure_pdf_assets(
    svg_references: list[str],
    *,
    markdown_dir: Path,
) -> list[Path]:
    qlmanage = shutil.which("qlmanage")
    if qlmanage is None:
        raise RuntimeError("qlmanage not found; macOS Quick Look is required for SVG conversion.")

    generated: list[Path] = []
    for reference in svg_references:
        svg_path = (markdown_dir / reference).resolve()
        png_path = svg_path.with_suffix(".png")
        if png_path.exists() and png_path.stat().st_mtime >= svg_path.stat().st_mtime:
            continue
        temp_dir = Path("/tmp")
        subprocess.run(
            [qlmanage, "-t", "-s", "2000", "-o", str(temp_dir), str(svg_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        temp_png_path = temp_dir / f"{svg_path.name}.png"
        if not temp_png_path.exists():
            raise RuntimeError(f"Quick Look did not create PNG for {svg_path}")
        png_path.write_bytes(temp_png_path.read_bytes())
        try:
            temp_png_path.unlink()
        except OSError:
            pass
        generated.append(png_path)
    return generated


def run_pandoc(
    *,
    export_markdown: Path,
    output_pdf: Path,
    pandoc_path: str,
    xelatex_path: Path,
    cjk_font: str,
    margin: str,
) -> None:
    if not Path(pandoc_path).exists():
        raise RuntimeError(f"pandoc not found at {pandoc_path}")
    if not xelatex_path.exists():
        raise RuntimeError(f"xelatex not found at {xelatex_path}")

    command = [
        pandoc_path,
        export_markdown.name,
        "-o",
        str(output_pdf),
        f"--pdf-engine={xelatex_path}",
        "-V",
        f"CJKmainfont={cjk_font}",
        "-V",
        f"geometry:margin={margin}",
    ]
    subprocess.run(command, cwd=export_markdown.parent, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PDF-friendly Markdown copy and optionally compile it to PDF."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--export-md", type=Path, default=DEFAULT_EXPORT_MD)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--pandoc", default=DEFAULT_PANDOC)
    parser.add_argument("--xelatex", type=Path, default=DEFAULT_XELATEX)
    parser.add_argument("--cjk-font", default="Songti SC")
    parser.add_argument("--margin", default="1in")
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Only generate the export Markdown and converted PDF assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source.resolve()
    export_md_path = args.export_md.resolve()
    output_pdf_path = args.output_pdf.resolve()

    source_text = source_path.read_text(encoding="utf-8")
    svg_refs = collect_svg_image_references(source_text)
    ensure_pdf_assets(svg_refs, markdown_dir=source_path.parent)

    export_md_path.write_text(
        rewrite_markdown_for_pdf_export(source_text),
        encoding="utf-8",
    )

    if not args.skip_pdf:
        run_pandoc(
            export_markdown=export_md_path,
            output_pdf=output_pdf_path,
            pandoc_path=args.pandoc,
            xelatex_path=args.xelatex,
            cjk_font=args.cjk_font,
            margin=args.margin,
        )

    print(f"Source kept unchanged: {source_path}")
    print(f"Export Markdown written: {export_md_path}")
    if args.skip_pdf:
        print("Skipped PDF compilation.")
    else:
        print(f"PDF written: {output_pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
