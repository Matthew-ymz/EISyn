"""Draw the cached A/C generator networks, with explicit triadic hyperedges.

PNG uses Matplotlib because draw.io Desktop is unavailable on this host.
The same coordinates and incidences are saved as an editable .drawio diagram.
No estimated tree, learned coefficient, or recovery score enters the figure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto

FOLDER = ROOT / 'docs/reports/assets/kuramoto_hierarchy'
CACHE = ROOT / 'results/spt_local_kuramoto_tm/simplified_ac_light.json'
COLORS = {1.: '#334653', .5: '#0072B2', .25: '#D55E00'}
TRIAD = '#7654A3'
INK = '#24333E'
POSITIONS = {
    'A': {1: (.1, .85), 2: (.1, -.85), 3: (1.5, 0.),
          4: (2.8, .6), 5: (3.8, -.2), 6: (5.1, .6),
          7: (6.1, -.2), 8: (7.4, .6), 9: (8.4, -.2)},
    'C': {1: (.1, .85), 2: (.1, -.85), 3: (1.5, 0.),
          4: (3.5, .85), 5: (3.5, -.85), 6: (4.9, 0.),
          7: (6.9, .85), 8: (6.9, -.85), 9: (8.3, 0.)},
}


def edges():
    payload = json.loads(CACHE.read_text())
    result = {}
    for case in ('A', 'C'):
        record = next(r for r in payload['records'] if r['case'] == case)
        network = {tuple(r['edge']): r['weight'] for r in record['network']}
        if network != SimplifiedKuramoto(case).weights:
            raise ValueError(f'{case}: cached network differs from current generator')
        result[case] = network
    return result


def center(positions, edge):
    return np.mean([positions[i] for i in edge], axis=0)


def draw_panel(ax, case, network):
    positions = POSITIONS[case]
    for edge, weight in network.items():
        points = np.array([positions[i] for i in edge])
        if len(edge) == 3:
            hub = center(positions, edge)
            for point in points:
                ax.plot([hub[0], point[0]], [hub[1], point[1]], color=TRIAD,
                        ls=(0, (3, 2)), lw=1.6, zorder=1)
            ax.scatter(*hub, s=330, marker='D', color='#EEE7F5',
                       edgecolor=TRIAD, linewidth=1.2, zorder=3)
            ax.text(*hub, r'$H_3$', color=TRIAD, ha='center', va='center', fontsize=9, zorder=4)
            ax.text(hub[0], -1.27, r'$K=1$', color=TRIAD, ha='center', va='center', fontsize=9)
        else:
            ax.plot(points[:, 0], points[:, 1], color=COLORS[weight],
                    lw=2.5 if weight == 1. else 1.9, zorder=1)
            midpoint = points.mean(axis=0)
            # All pair labels sit in free space above their corresponding line.
            offset = .27 if weight == 1 else .30
            ax.text(midpoint[0], midpoint[1] + offset, f'{weight:g}',
                    color=COLORS[weight], ha='center', va='center', fontsize=9,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .4}, zorder=5)
    for node, point in positions.items():
        ax.scatter(*point, s=370, color='white', edgecolor=INK, linewidth=1.15, zorder=6)
        ax.text(*point, str(node), ha='center', va='center', color=INK, fontsize=11, zorder=7)
    pair_count = sum(len(e) == 2 for e in network)
    triad_count = sum(len(e) == 3 for e in network)
    title = 'Mixed modules' if case == 'A' else 'Three triadic modules'
    ax.text(-.35, 1.65, f'{case}   {title}', fontsize=12, weight='bold', color=INK, va='center')
    ax.text(8.8, 1.65, f'9 nodes · {pair_count} pair edges · {triad_count} triadic hyperedge'
            + ('s' if triad_count > 1 else ''), ha='right', va='center', fontsize=9, color='#596873')
    ax.set(xlim=(-.45, 8.9), ylim=(-1.55, 1.95), aspect='equal')
    ax.axis('off')


def write_drawio(networks, path):
    document = ET.Element('mxfile', host='drawio', version='26.0.0')
    page = ET.SubElement(document, 'diagram', name='Simplified A and C')
    model = ET.SubElement(page, 'mxGraphModel', page='1', pageWidth='1240', pageHeight='1020')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')
    next_id = 2
    def vertex(value, x, y, width, height, style):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        cell = ET.SubElement(root, 'mxCell', id=key, value=value,
                            style=style + ';html=1;', vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(width),
                      height=str(height), attrib={'as': 'geometry'})
        return key
    def link(source, target, value, color, dashed=False, width=2):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        style = ('edgeStyle=none;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;'
                 'startArrow=none;endArrow=none;labelBackgroundColor=#FFFFFF;'
                 f'strokeColor={color};fontColor={color};strokeWidth={width};fontSize=15;')
        if dashed:
            style += 'dashed=1;dashPattern=3 2;'
        cell = ET.SubElement(root, 'mxCell', id=key, value=value, style=style,
                            edge='1', parent='1', source=source, target=target)
        ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
    text_style = f'text;strokeColor=none;fillColor=none;fontColor={INK};fontFamily=Arial;align=left;'
    for row, case in enumerate(('A', 'C')):
        offset = 210 + row * 380
        def point(p):
            return 85 + 120 * p[0], offset - 120 * p[1]
        vertex(f'{case} — Simplified Kuramoto network', 40, offset - 185, 700, 35,
               text_style + 'fontSize=23;fontStyle=1;')
        ids = {}
        for node, location in POSITIONS[case].items():
            x, y = point(location)
            ids[node] = vertex(str(node), x - 20, y - 20, 40, 40,
                f'ellipse;fillColor=#FFFFFF;strokeColor={INK};strokeWidth=1.5;'
                f'fontColor={INK};fontSize=19;fontFamily=Arial;')
        for edge, weight in networks[case].items():
            if len(edge) == 3:
                hub = center(POSITIONS[case], edge)
                x, y = point(hub)
                factor = vertex('H₃', x - 21, y - 21, 42, 42,
                    f'rhombus;fillColor=#EEE7F5;strokeColor={TRIAD};fontColor={TRIAD};fontSize=15;')
                for node in edge:
                    link(factor, ids[node], '', TRIAD, dashed=True)
                x, y = point((hub[0], -1.27))
                vertex('K = 1', x - 40, y - 12, 80, 24,
                       text_style + f'align=center;fontSize=16;fontColor={TRIAD};')
            else:
                link(ids[edge[0]], ids[edge[1]], f'{weight:g}', COLORS[weight], width=3 if weight == 1 else 2)
    vertex('Circles: phase oscillators. H₃: one three-body interaction; dashed spokes denote membership.',
           40, 820, 1120, 30, text_style + 'fontSize=17;')
    vertex('Solid edges: pairwise Kuramoto interactions. Edge labels are coupling coefficients.',
           40, 855, 1120, 30, text_style + 'fontSize=17;')
    vertex('Physical generator topology only; not an inferred SPT.',
           40, 890, 1120, 30, text_style + 'fontSize=17;')
    ET.indent(document, space='  ')
    ET.ElementTree(document).write(path, encoding='utf-8', xml_declaration=True)


def main():
    networks = edges()
    FOLDER.mkdir(parents=True, exist_ok=True)
    image_path = FOLDER / 'spt_kuramoto_simplified_networks.png'
    editable_path = FOLDER / 'spt_kuramoto_simplified_networks.drawio'
    write_drawio(networks, editable_path)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.2), layout='constrained')
        for ax, case in zip(axes, ('A', 'C')):
            draw_panel(ax, case, networks[case])
        handles = [Line2D([], [], color=COLORS[w], lw=2, label=f'Pairwise, K={w:g}')
                   for w in (1., .5, .25)]
        handles.append(Line2D([], [], color=TRIAD, ls='--', marker='D',
            markerfacecolor='#EEE7F5', lw=1.5, label='Triadic hyperedge, K=1'))
        fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
        fig.get_layout_engine().set(rect=(0., .07, 1., 1.))
        fig.text(.5, .065, 'Circles: oscillators · Diamonds: 3-body interactions · Dashed spokes: membership',
                 ha='center', fontsize=9, color='#596873')
        fig.text(.5, .035, 'Generator topology only; not an inferred Synergy Partition Tree',
                 ha='center', fontsize=9, color='#596873')
        fig.savefig(image_path, dpi=240, bbox_inches='tight', facecolor='white', metadata={
            'Description': 'Cached simplified A/C generator topologies; hyperedge incidences and coefficients.',
            'Source': str(CACHE.relative_to(ROOT)),
            'SourceSHA256': hashlib.sha256(CACHE.read_bytes()).hexdigest()})
        plt.close(fig)
    print(image_path)
    print(editable_path)


if __name__ == '__main__':
    main()
