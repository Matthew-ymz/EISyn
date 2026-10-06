"""Export the current EI framework's two reference tables and recursion diagram.

The Markdown document is the source of truth for both tables. No fitted data,
EI estimator, channel optimizer, or numerical Syn projection is used here.
The diagram follows Sections 4.2--4.6: source SPT budgets, conditional MI/KL
recursion, fixed KL references, and terminal-only accounting.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_ei_decomposition_xor_copy import CHINESE, INK, MUTED, PURPLE, RULE, TEAL


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/ref/1对n的EI分解_最小耦合方案.md"
FIG_DIR = ROOT / "fig"
STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stixsans",
    "font.size": 15,
    "text.color": INK,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
}


def document_table(marker: str) -> list[list[str]]:
    """Read the first Markdown table after a unique caption marker."""
    source = DOCUMENT.read_text(encoding="utf-8")
    if source.count(marker) != 1:
        raise ValueError(f"Expected one document caption: {marker}")
    lines = source.split(marker, 1)[1].splitlines()
    table = []
    for line in lines:
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                table.append(cells)
        elif table:
            break
    if not table or any(len(row) != len(table[0]) for row in table):
        raise ValueError(f"Malformed table after {marker}")
    return table


def plot_math(content: str) -> str:
    # Mathtext uses the same bold lower-case vector convention as the document.
    content = re.sub(r"\\boldsymbol\s+([A-Za-z])", r"\\mathbf{\1}", content)
    content = (content.replace(r"\boldsymbol", r"\mathbf")
               .replace(r"\land", r"\wedge")
               .replace(r"\lvert", r"\vert")
               .replace(r"\rvert", r"\vert"))
    content = re.sub(r"\\(mathbf|mathcal|mathbb)\s+([A-Za-z])", r"\\\1{\2}", content)
    return re.sub(r"\\(ge|le)(?![A-Za-z])", lambda match: "\\" + match[1] + "q", content)


def canvas(width: int, height: int):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, content, *, size=16, color=INK, ha="left", va="center"):
    return ax.text(x, y, plot_math(content), fontsize=size, color=color,
                   fontproperties=CHINESE, ha=ha, va=va)


def wrapped_lines(content, width, probe, renderer):
    """Wrap prose by measured width; keep each inline formula intact."""
    lines = []
    for paragraph in content.split("<br>"):
        tokens = re.findall(r"\$[^$]+\$|[A-Za-z0-9]+(?:\.[0-9]+)?|.", plot_math(paragraph.strip()))
        line = ""
        for token in tokens:
            candidate = line + token
            probe.set_text(candidate)
            if probe.get_window_extent(renderer).width <= width:
                line = candidate
                continue
            if line:
                lines.append(line.rstrip())
                line = token.lstrip()
                probe.set_text(line)
            if probe.get_window_extent(renderer).width > width:
                raise ValueError(f"Unbreakable formula exceeds its table cell: {token}")
        if line:
            lines.append(line.rstrip())
    return lines or [""]


def save_verified(fig, artists, stem):
    """Reject clipped text before exporting; visually inspect final PNGs too."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in artists:
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < -0.5 or bounds.y0 < -0.5
                or bounds.x1 > fig.bbox.width + 0.5
                or bounds.y1 > fig.bbox.height + 0.5):
            raise ValueError(f"Text outside figure: {artist.get_text()}")
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(FIG_DIR / (stem + suffix), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved fig/{stem}.png and .svg")


def table_figure(table, widths, stem, title, subtitle, notes, *, body_size=16):
    width = sum(widths) + 100
    measure_fig, measure_ax = canvas(width, 100)
    measure_fig.canvas.draw()
    probe = text(measure_ax, 0, 0, "", size=body_size)
    renderer = measure_fig.canvas.get_renderer()
    wrapped = [
        [wrapped_lines(cell, cell_width - 26, probe, renderer)
         for cell, cell_width in zip(row, widths)]
        for row in table[1:]
    ]
    plt.close(measure_fig)
    line_height = 32
    row_heights = [max(len(cell) for cell in row) * line_height + 20 for row in wrapped]
    footer = 38 * len(notes) + 28
    height = 172 + sum(row_heights) + footer
    fig, ax = canvas(width, height)
    artists = [
        text(ax, 50, height - 44, title, size=25),
        text(ax, 50, height - 92, subtitle, size=16, color=MUTED),
    ]
    y = height - 138
    x = 50
    for index, (heading, cell_width) in enumerate(zip(table[0], widths)):
        centered = stem.endswith("mechanisms") and index in (1, 2, 3, 4)
        artists.append(text(ax, x + (cell_width / 2 if centered else 12), y,
                            heading, size=17, ha="center" if centered else "left"))
        x += cell_width
    y -= 27
    ax.plot((50, width - 50), (y, y), color=INK, lw=1)
    for row, row_height in zip(wrapped, row_heights):
        center = y - row_height / 2
        x = 50
        for index, (lines, cell_width) in enumerate(zip(row, widths)):
            centered = stem.endswith("mechanisms") and index in (1, 2, 3, 4)
            for line_index, line in enumerate(lines):
                line_y = center + (len(lines) - 1) * line_height / 2 - line_index * line_height
                artists.append(text(ax, x + (cell_width / 2 if centered else 12), line_y,
                                    line, size=body_size,
                                    ha="center" if centered else "left"))
            x += cell_width
        y -= row_height
        ax.plot((50, width - 50), (y, y), color=RULE, lw=0.8)
    for index, note in enumerate(notes):
        artists.append(text(ax, 62, y - 32 - index * 38, note, size=14, color=MUTED))
    save_verified(fig, artists, stem)


def recursive_flow():
    fig, ax = canvas(1600, 1140)
    artists = []

    def label(x, y, content, **kwargs):
        artist = text(ax, x, y, content, **kwargs)
        artists.append(artist)
        return artist

    def box(x, y, width, height, accent, background="white"):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0,rounding_size=8",
                                   linewidth=1.25, edgecolor=accent, facecolor=background))

    def arrow(start, end, color=MUTED):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                    linewidth=1.5, color=color, shrinkA=3, shrinkB=3))

    label(80, 1090, "从二元定义到 N→N：固定父预算，逐行二分", size=25)
    label(80, 1048, "源相互独立 · 边缘无需均匀 · 有限离散 · bit；目标条件分布任意", size=15, color=MUTED)
    box(80, 905, 1395, 125, INK, "#F5F7F9")
    label(777, 992, "源侧 SPT：二分独立源块，得到 n 个单源行与 n−1 个协同行", size=18, ha="center")
    label(777, 946,
          r"$I(\mathbf{u};\mathbf{v})=\sum_i I(U_i;\mathbf{v})+\sum_v s_v$", size=20, ha="center")
    arrow((400, 905), (400, 855), TEAL)
    arrow((1175, 905), (1175, 855), PURPLE)

    box(80, 715, 640, 140, TEAL, "#F4FAFA")
    label(400, 816, "单源行：MI 父项", size=19, color=TEAL, ha="center")
    label(400, 768, r"$I(U_i;\mathbf{v})$", size=21, ha="center")
    label(400, 735, "始终使用同一联合分布", size=14, color=MUTED, ha="center")
    box(875, 715, 600, 140, PURPLE, "#F8F5FB")
    label(1175, 816, "源协同行：固定一份参照", size=19, color=PURPLE, ha="center")
    label(1175, 768, r"$s_v=D_{\mathrm{KL}}(P_v\Vert Q_v)=b_v+F_{v,\mathbf{u}_v}(J)$",
          size=18, ha="center")
    label(1175, 735, "根基线单独保留；子节点沿用同一 P、Q", size=14, color=MUTED, ha="center")
    arrow((400, 715), (400, 635), TEAL)
    arrow((1175, 715), (1175, 635), PURPLE)

    # The root baseline bypasses all target recursion and enters the sum once.
    ax.plot((1475, 1535, 1535), (785, 785, 120), color=MUTED, lw=1.5)
    arrow((1535, 120), (1475, 120))
    label(1535, 814, r"$b_v$", size=19, ha="center", color=PURPLE)

    box(80, 485, 640, 150, TEAL)
    label(400, 601, "继续细化：按条件 MI 链二分递归", size=18, color=TEAL, ha="center")
    label(400, 558, r"$I(U_i;\mathbf{v}_J\mid\mathbf{c})$", size=18, ha="center")
    label(400, 522,
          r"$=I(U_i;\mathbf{v}_G\mid\mathbf{c})+I(U_i;\mathbf{v}_H\mid\mathbf{c},\mathbf{v}_G)$",
          size=17, ha="center")
    box(875, 485, 600, 150, PURPLE)
    label(1175, 601, "继续细化：按条件 KL 链二分递归", size=18, color=PURPLE, ha="center")
    label(1175, 558, r"$F_{v,\mathbf{c}}(J)$", size=18, ha="center")
    label(1175, 522, r"$=F_{v,\mathbf{c}}(G)+F_{v,(\mathbf{c},\mathbf{v}_G)}(H)$",
          size=17, ha="center")
    arrow((400, 485), (400, 405), TEAL)
    arrow((1175, 485), (1175, 405), PURPLE)
    label(447, 445, "更新已读上下文", size=14, color=MUTED)
    label(1222, 445, "更新已读上下文", size=14, color=MUTED)

    box(80, 245, 640, 160, TEAL, "#F4FAFA")
    label(400, 372, "二块终端：复用原 1→2", size=18, color=TEAL, ha="center")
    label(400, 332, r"$C_{\mathrm{ext}}(\mathbf{v}_G,\mathbf{v}_H;U_i)$",
          size=20, ha="center")
    label(400, 294, "逐条件只处理源；输出冗余、特有、协同", size=15, ha="center")
    label(400, 264, "单目标终端直接输出条件 MI 预算", size=14, color=MUTED, ha="center")
    box(875, 245, 600, 160, PURPLE, "#F8F5FB")
    label(1175, 372, "二块终端：固定 KL 共同读出", size=18, color=PURPLE, ha="center")
    label(1175, 332, r"$r_F(G,H\mid\mathbf{c})$", size=20, ha="center")
    label(1175, 294, "同核推送 P、Q；输出共同读出及余额", size=15, ha="center")
    label(1175, 264, "单目标终端直接输出条件 KL 预算", size=14, color=MUTED, ha="center")
    arrow((400, 245), (400, 210), TEAL)
    arrow((1175, 245), (1175, 210), PURPLE)

    box(80, 60, 1395, 150, INK, "#F5F7F9")
    label(777, 173, r"最终汇总：各行终端 $t_\ell$＋协同行根基线", size=18, ha="center")
    label(777, 113, r"$\sum_\ell t_\ell+\sum_v b_v=I(\mathbf{u};\mathbf{v})$",
          size=21, ha="center")
    label(80, 26, "内部四项仅用于择路，继续递归时由条件链替换父预算；二块节点可提前停止。", size=14, color=MUTED)
    save_verified(fig, artists, "ei_decomposition_recursive_flow")


def main():
    with plt.rc_context(STYLE):
        table_figure(
            document_table("**表 1　二元性质概览。**"),
            [325, 1125, 175],
            "ei_decomposition_properties",
            "二元分解的性质与适用条件",
            r"$R(U,V;W)=C_{\mathrm{ext}}(U,V;W)$：只处理指定端 W；完整证明见文档第 1 节",
            ["噪声操作保留原信号并附加分量；不包括翻转、遮蔽或信号替换。",
             "这里列出二元性质；KL 共同读出及双侧树的性质须分别证明。"],
            body_size=16,
        )
        table_figure(
            document_table("**表 2　九种输出机制。**"),
            [880, 130, 135, 135, 95, 425],
            "ei_decomposition_mechanisms",
            "1→2：九种机制的非负分解",
            r"$R=C_{\mathrm{ext}}(U,V;W)$：只处理源 W · 均匀干预 · 单位 bit",
            [r"$U_0,V_0,C,T,N$ 独立公平；翻转噪声独立于基本比特，且两个 $F_i$ 彼此独立。",
             r"$k=1-h_2(0.1)\simeq0.531004,\quad d=1-h_2(0.18)\simeq0.319923$；有限状态编码无损。",
             "各行值由解析最优性确定；显示小数取六位。最后一行的证明见第 1.10 节。"],
            body_size=17,
        )
        recursive_flow()


if __name__ == "__main__":
    main()
