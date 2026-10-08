#!/usr/bin/env python3
"""C1/C2/C3 membership on the existing Schaefer-100 fsaverage5 four views.

Reads only selected complete memberships and the existing anatomical assets.
Colors encode candidate identity, not Syn or recurrence; no simulation or
information estimation. Exact parcel names are matched, while the inherited
structural-connectome row order remains inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgb
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.brain_surface_plot import _draw_surface, _face_colors
from scripts.plot_dmf_schaefer100_summary import load_schaefer100_surface_map
from scripts.run_dmf_subject_consistency import atomic_json, digest

BASE = ROOT / 'results/dmf_schaefer100'
COLORS = ['#236A91', '#BA5636', '#717A38']
BACKGROUND = '#D7D7D7'


def plot(output: Path):
    summary_path = BASE / 'critical_coalitions/all93_summary.json'
    contract_path = BASE / 'critical_coalitions/contract.json'
    labels_path = BASE / 'schaefer100_labels.txt'
    surface_path = BASE / 'schaefer100_fsaverage5_surface.npz'
    summary = json.loads(summary_path.read_text())
    if summary['status'] != 'complete' or summary['subject_count'] != 93:
        raise ValueError('Expected the complete all93 candidate selection')
    if summary['scientific_contract_sha256'] != digest(contract_path):
        raise ValueError('Candidate summary scientific provenance mismatch')
    selected = summary['selected'][:3]
    if [r['display_id'] for r in selected] != ['C1', 'C2', 'C3']:
        raise ValueError('Unexpected candidate display identities')
    labels = labels_path.read_text().splitlines()
    if len(labels) != 100 or len(set(labels)) != 100:
        raise ValueError('Expected 100 unique Schaefer parcel names')
    owner = np.full(100, -1, dtype=int)
    for ci, row in enumerate(selected):
        members = np.asarray(row['members'], dtype=int)
        if (len(set(members)) != row['order'] or (members < 0).any()
                or (members >= 100).any()
                or (members + 1).tolist() != row['members_one_based']):
            raise ValueError('Invalid complete candidate membership')
        if (owner[members] != -1).any():
            raise ValueError('Overlapping candidates require an explicit overlap encoding')
        owner[members] = ci

    left, right, lv, rv = load_schaefer100_surface_map(
        surface_path, np.asarray(labels), np.arange(100, dtype=float))
    cmap = ListedColormap(COLORS)
    norm = Normalize(0, 2)
    cache = []
    face_counts = {}
    mapped_members = set()
    for side, mesh, vertex_roi in [('left', left, lv), ('right', right, rv)]:
        sampled = vertex_roi[mesh.faces]
        # Categorical parcels are never averaged across a triangle boundary.
        uniform = np.isfinite(sampled).all(1) & (sampled == sampled[:, :1]).all(1)
        face_roi = np.full(len(mesh.faces), -1, dtype=int)
        face_roi[uniform] = sampled[uniform, 0].astype(int)
        background = _face_colors(mesh, np.full(len(vertex_roi), np.nan),
            cmap=cmap, norm=norm, background_color=BACKGROUND)
        shade = background[:, 0] / to_rgb(BACKGROUND)[0]
        rgba = background.copy()
        for ci, row in enumerate(selected):
            mask = uniform & np.isin(face_roi, row['members'])
            rgba[mask, :3] = np.clip(np.asarray(to_rgb(COLORS[ci])) * shade[mask, None], 0, 1)
            face_counts[f'{side}_{row["display_id"]}'] = int(mask.sum())
            mapped_members.update(face_roi[mask].tolist())
        cache.append((mesh, vertex_roi, rgba))
    expected_members = {i for r in selected for i in r['members']}
    if mapped_members != expected_members:
        raise ValueError(f'Unrendered candidate parcels: {expected_members - mapped_members}')

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'savefig.facecolor': 'white', 'text.color': '#303030'}):
        fig = plt.figure(figsize=(10.8, 6.5), facecolor='white')
        specs = [(0, 180, 'Left lateral', .005, .48),
                 (1, 0, 'Right lateral', .365, .48),
                 (0, 0, 'Left medial', .005, .025),
                 (1, 180, 'Right medial', .365, .025)]
        for hemisphere, azim, title, x, y in specs:
            mesh, vertex_roi, rgba = cache[hemisphere]
            ax = fig.add_axes([x, y, .36, .50], projection='3d')
            _draw_surface(ax, mesh, np.full(len(vertex_roi), np.nan),
                cmap=cmap, norm=norm, elev=0, azim=azim,
                background_color=BACKGROUND, zoom=1.52)
            ax.collections[-1].set_facecolor(rgba)
            ax.text2D(.5, .93, title, transform=ax.transAxes, ha='center', fontsize=11)
        handles = [Patch(facecolor=color, label=row['display_id'] + ': ROI\n'
                   + ', '.join(map(str, row['members_one_based'])))
                   for color, row in zip(COLORS, selected)]
        handles.append(Patch(facecolor=BACKGROUND, label='Other parcels /\nmedial wall'))
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.765, .54),
                   frameon=False, title='Complete memberships', labelspacing=1.25,
                   handlelength=1.5, handleheight=1.2)
        fig.text(.765, .22, 'Schaefer-100\nfsaverage5 inflated surface',
                 fontsize=9, color='#666666', linespacing=1.5)
        fig.text(.035, .022, 'Colors mark coalition members; SC row-to-parcel order is inherited and inferred.',
                 fontsize=8.5, color='#666666')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches=None)
        plt.close(fig)

    metadata = dict(output=str(output), figure_sha256=digest(output),
        plotting_sha256=digest(Path(__file__)), summary_sha256=digest(summary_path),
        scientific_contract_sha256=digest(contract_path), surface_sha256=digest(surface_path),
        labels_sha256=digest(labels_path), encoding='Categorical complete membership, no Syn values consumed',
        selected=[dict(display_id=r['display_id'], members_one_based=r['members_one_based'],
                       color=COLORS[ci], parcels=[labels[i] for i in r['members']])
                  for ci, r in enumerate(selected)],
        overlap_count=0, unique_marked_parcels=len(expected_members), face_counts=face_counts,
        anatomical_mapping='Exact parcel-name match onto existing fsaverage5 asset; SC row order inferred',
        categorical_boundaries='Only faces with three vertices in the same ROI are colored; boundary faces remain gray',
        shading='Existing uniform sulcal/geometry shading applied equally to all colors',
        views='Left/right columns; lateral above, medial below; azimuth 180/0/0/180, elevation 0',
        legend_placement='Outside all four brain viewports',
        manuscript=dict(checked_date='2026-10-08', parent_key='P6UJCVG8',
            title='Emergent hierarchical organization of causal interactions in complex systems',
            main_attachment='DXGC7JEA', supplement_attachment='MWIWKSVG',
            version='No explicit draft version/date; attachment metadata does not resolve version',
            relevant_locations='Main Brain/Fig.2 pp6–7 and SPT Eqs11–12 pp16–17; supplement S5 pp8–9'),
        visual_review='pending')
    atomic_json(BASE / 'critical_coalitions/all93_surface_figure_metadata.json', metadata)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'fig/dmf_schaefer100/critical_coalition_all93_surface.png')
    plot(parser.parse_args().output)
