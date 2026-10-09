#!/usr/bin/env python
"""Schematic of potential forecast impacts, with the existing May 2024 curves.

Figure contract: a schematic-led composite, Chinese presentation labels, actual
one-minute IMF magnitude (nT) and flow speed (km/s), no smoothing/interpolation.
The physical drivers are upstream descriptors, not forecast-impact measurements.
Solid arrows encode established mechanisms or system dependencies, not proof
that the May 2024 event activated every pathway. The dashed final arrow denotes
an event-specific forecast effect that has not been estimated in this project.

Sources, accessed for the accompanying conversation on 2026-10-09:
https://omniweb.gsfc.nasa.gov/html/omni_min_data.html
https://www.metoffice.gov.uk/research/weather/space-weather
https://amt.copernicus.org/articles/18/843/2025/
https://www.ecmwf.int/en/about/media-centre/news/2026/satellite-data-weather-forecasts

The diagram is drawn with Matplotlib because draw.io desktop is unavailable.
The same layout also produces an editable draw.io source, with embedded curve
panels. PNG and SVG contain numerical curves; draw.io embeds their PNG renders.
"""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
import pandas as pd

from reproduce_omni_curves import parse_listing


HERE = Path(__file__).resolve().parent
STEM = "geomagnetic_forecast_cascade"
W, H = 2400, 1080
NAVY, MUTED, BORDER = "#182C43", "#64758A", "#DDE5ED"
BLUE, RED, TEAL, GOLD = "#2F6FB0", "#C75555", "#267C80", "#AB7530"
CJK_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
CJK = FontProperties(fname=str(CJK_PATH)) if CJK_PATH.exists() else FontProperties(family="sans-serif")


class Diagram:
    """Shared logical coordinates for the rendered and editable artifacts."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=180, facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1], zorder=1)
        self.ax.set(xlim=(0, W), ylim=(H, 0))
        self.ax.axis("off")
        self.file = ET.Element("mxfile", host="app.diagrams.net", version="26.0.0")
        page = ET.SubElement(self.file, "diagram", name="地磁暴与天气预报")
        model = ET.SubElement(page, "mxGraphModel", page="1", pageScale="1", pageWidth=str(W), pageHeight=str(H), background="#FFFFFF", grid="0")
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.n = 2
        self.boxes = {}
        self.labels = []

    def cell(self, style, **attrs):
        cell = ET.SubElement(self.root, "mxCell", id=str(self.n), style=style, **attrs)
        self.n += 1
        return cell

    def box(self, name, x, y, w, h, fill="white", border=BORDER):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=15", linewidth=1.05, facecolor=fill, edgecolor=border, zorder=2))
        cell = self.cell(f"rounded=1;arcSize=10;html=1;container=1;pointerEvents=0;fillColor={fill};strokeColor={border};strokeWidth=1;", vertex="1", parent="1", value="")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        self.boxes[name] = (cell.attrib["id"], x, y, w, h)

    def text(self, x, y, value, size=18, color=NAVY, bold=False, align="left", parent=None, width=None):
        label = self.ax.text(x, y, value, fontsize=size, fontproperties=CJK, fontweight="bold" if bold else "normal", color=color, ha=align, va="center", linespacing=1.55, zorder=6)
        self.labels.append(label)
        px = size * 100 / 72
        width = width or max(180, max(len(s) for s in value.splitlines()) * px + 15)
        height = px * (1.5 * len(value.splitlines()) + .25)
        lx = x if align == "left" else x - width if align == "right" else x - width / 2
        ly = y - height / 2
        pid = "1"
        if parent:
            pid, ox, oy, _, _ = self.boxes[parent]
            lx, ly = lx - ox, ly - oy
        cell = self.cell(f"text;html=1;whiteSpace=wrap;overflow=visible;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;fontFamily=Hiragino Sans GB;fontSize={px:.2f};fontColor={color};fontStyle={1 if bold else 0};spacing=0;", vertex="1", parent=pid, value=html.escape(value).replace("\n", "<br>"))
        ET.SubElement(cell, "mxGeometry", x=f"{lx:.2f}", y=f"{ly:.2f}", width=f"{width:.2f}", height=f"{height:.2f}", **{"as": "geometry"})

    def arrow(self, points, source=None, target=None, color=MUTED, dashed=False, head=True, lw=1.8):
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
        self.ax.add_patch(FancyArrowPatch(path=MplPath(points, codes), arrowstyle="-|>" if head else "-", mutation_scale=17, color=color, linewidth=lw, linestyle=(0, (5, 4)) if dashed else "solid", capstyle="round", joinstyle="round", zorder=4))
        attrs = dict(edge="1", parent="1", value="")
        anchors = ""
        for name, endpoint, prefix in [(source, points[0], "exit"), (target, points[-1], "entry")]:
            if name:
                pid, x, y, w, h = self.boxes[name]
                attrs["source" if prefix == "exit" else "target"] = pid
                anchors += f"{prefix}X={(endpoint[0]-x)/w:.4f};{prefix}Y={(endpoint[1]-y)/h:.4f};{prefix}Dx=0;{prefix}Dy=0;"
        cell = self.cell(f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth={lw};endArrow={'block' if head else 'none'};endFill=1;dashed={int(dashed)};dashPattern=6 5;{anchors}", **attrs)
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if not source:
            ET.SubElement(geom, "mxPoint", x=str(points[0][0]), y=str(points[0][1]), **{"as": "sourcePoint"})
        if not target:
            ET.SubElement(geom, "mxPoint", x=str(points[-1][0]), y=str(points[-1][1]), **{"as": "targetPoint"})
        if len(points) > 2:
            mids = ET.SubElement(geom, "Array", **{"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(mids, "mxPoint", x=str(x), y=str(y))

    def curve(self, data, col, rect, parent, color, ylim, yticks, unit):
        x, y, w, h = rect
        ax = self.fig.add_axes([x/W, (H-y-h)/H, w/W, h/H], zorder=3)
        style_curve(ax, data, col, color, ylim, yticks, unit)
        # Identical numerical panel for the editable file, with margins kept
        # inside its image bounds so ticks and unit labels remain visible.
        mini = plt.figure(figsize=((w+90)/100, (h+65)/100), facecolor="white")
        ma = mini.add_axes([75/(w+90), 50/(h+65), w/(w+90), h/(h+65)])
        style_curve(ma, data, col, color, ylim, yticks, unit)
        buf = BytesIO()
        mini.savefig(buf, format="png", dpi=180)
        plt.close(mini)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        pid, px, py, _, _ = self.boxes[parent]
        cell = self.cell(f"shape=image;html=1;imageAspect=0;aspect=fixed;image=data:image/png,{b64};", vertex="1", parent=pid, value="")
        ET.SubElement(cell, "mxGeometry", x=str(x-75-px), y=str(y-15-py), width=str(w+90), height=str(h+65), **{"as": "geometry"})

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        box = self.fig.bbox
        for label in self.labels:
            b = label.get_window_extent(renderer)
            if b.x0 < box.x0 or b.x1 > box.x1 or b.y0 < box.y0 or b.y1 > box.y1:
                raise RuntimeError(f"Label outside canvas: {label.get_text()}")
        self.fig.savefig(HERE/f"{STEM}.png", dpi=180, bbox_inches=None, facecolor="white")
        self.fig.savefig(HERE/f"{STEM}.svg", bbox_inches=None, facecolor="white", metadata={"Description": __doc__})
        ET.indent(self.file)
        ET.ElementTree(self.file).write(HERE/f"{STEM}.drawio", encoding="utf-8", xml_declaration=True)
        plt.close(self.fig)


def style_curve(ax, data, col, color, ylim, yticks, unit):
    ax.plot(data.time, data[col], color=color, lw=.8, rasterized=False)
    ax.set_xlim(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-01"))
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(pd.date_range("2024-05-01", "2024-05-31", freq="10D"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d May"))
    ax.grid(axis="y", color="#E6ECF1", linewidth=.65)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set(color="#ACBAC7", linewidth=.7)
    ax.tick_params(labelsize=13, colors=MUTED, length=3, pad=8)
    ax.set_ylabel(unit, fontsize=14, color=MUTED, labelpad=10)
    # No legends or peak labels overlay the observations; labels are in headers.


def main():
    data = parse_listing((HERE/"omni_min_202405_imf_speed.txt").read_text())
    with plt.rc_context({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"], "svg.fonttype":"path", "axes.unicode_minus":False}):
        d = Diagram()
        d.text(70, 82, "从地磁暴到天气预报", size=39, bold=True)
        d.arrow([(1590, 83), (1645, 83)], color=MUTED)
        d.text(1660, 83, "机制路径", size=17, color=MUTED)
        d.arrow([(1980, 83), (2035, 83)], color=GOLD, dashed=True)
        d.text(2050, 83, "效应待验证", size=17, color=GOLD)
        d.arrow([(70, 147), (2320, 147)], color=BORDER, head=False, lw=.8)

        d.box("solar", 70, 250, 290, 170, "#F3F7FC", "#C8D9EB")
        d.text(95, 291, "太阳风扰动", 23, bold=True, parent="solar")
        d.text(95, 337, "|B|：磁场幅度", 18, color=BLUE, parent="solar")
        d.text(95, 382, "V：太阳风流速", 18, color=RED, parent="solar")

        d.box("storm", 435, 250, 300, 170, "#F4F3F9", "#D5D0E5")
        d.text(460, 291, "地磁暴", 23, bold=True, parent="storm")
        d.text(460, 355, "Dst / Kp", 19, color=MUTED, parent="storm")

        d.box("iono", 815, 210, 320, 130, "#F0F7F7", "#C6DEDD")
        d.text(840, 250, "电离层扰动", 22, bold=True, parent="iono")
        d.text(840, 299, "GNSS 信号受扰", 18, color=MUTED, parent="iono")
        d.box("infra", 815, 390, 320, 130, "#F7F6F2", "#E2DCCC")
        d.text(840, 430, "基础设施受扰", 22, bold=True, parent="infra")
        d.text(840, 479, "卫星 / 通信 / 供电", 18, color=MUTED, parent="infra")

        d.box("obs", 1220, 250, 300, 170)
        d.text(1245, 291, "气象观测变化", 22, bold=True, parent="obs")
        d.text(1245, 355, "偏差 / 缺测 / 延迟", 18, color=TEAL, parent="obs")
        d.box("assim", 1600, 250, 300, 170)
        d.text(1625, 291, "数据同化与初值", 22, bold=True, parent="assim")
        d.text(1625, 355, "质控与初值变化", 18, color=TEAL, parent="assim")
        d.box("forecast", 1980, 250, 340, 170, "#FAF6EE", "#E6D5B7")
        d.text(2005, 291, "天气预报影响", 23, bold=True, parent="forecast")
        d.text(2005, 355, "准确率 / 时效 / 可用性", 18, color=GOLD, parent="forecast")

        d.arrow([(360,335),(435,335)], "solar", "storm")
        d.arrow([(735,295),(775,295),(775,275),(815,275)], "storm", "iono")
        d.arrow([(735,375),(775,375),(775,455),(815,455)], "storm", "infra")
        d.arrow([(1135,275),(1175,275),(1175,295),(1220,295)], "iono", "obs")
        d.arrow([(1135,455),(1175,455),(1175,375),(1220,375)], "infra", "obs")
        d.arrow([(1520,335),(1600,335)], "obs", "assim")
        d.arrow([(1900,335),(1980,335)], "assim", "forecast", color=GOLD, dashed=True)

        d.arrow([(70,575),(2320,575)], color=BORDER, head=False, lw=.8)
        d.text(70, 631, "两条上游驱动曲线", 25, bold=True)
        d.text(2320, 631, "NASA OMNIWeb · 2024 年 5 月 · UTC", 16, color=MUTED, align="right")
        d.box("panel_b", 70, 680, 1090, 355)
        d.box("panel_v", 1230, 680, 1090, 355)
        d.text(100, 723, "IMF 磁场强度 |B|", 23, color=BLUE, bold=True, parent="panel_b")
        d.text(100, 768, "磁场扰动幅度 · 结合 Bz 判断耦合", 18, color=MUTED, parent="panel_b")
        d.text(1260, 723, "太阳风速度 V", 23, color=RED, bold=True, parent="panel_v")
        d.text(1260, 768, "流速变化 · 与密度共同决定动压", 18, color=MUTED, parent="panel_v")
        d.curve(data, "imf_magnitude", (180, 810, 930, 165), "panel_b", BLUE, (0, 76), [0,35,70], "nT")
        d.curve(data, "speed", (1340, 810, 930, 165), "panel_v", RED, (250, 1100), [300,700,1100], "km/s")
        d.save()
    print(f"PNG: {HERE / (STEM+'.png')}")
    print(f"SVG: {HERE / (STEM+'.svg')}")
    print(f"Editable: {HERE / (STEM+'.drawio')}")
    print(f"Native records: {len(data):,}; IMF missing: {data.imf_magnitude.isna().sum():,}; speed missing: {data.speed.isna().sum():,}")


if __name__ == "__main__":
    main()
