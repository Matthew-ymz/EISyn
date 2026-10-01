#!/usr/bin/env python3
"""Exploratory held-out-seed audit of frozen C10 versus equal-size random sets.

Contract: three quantitative validation panels; native Xi bits; compare fixed
seed-4 reference sets with 256 size-matched random sets on seven other seeds,
and repeat with G=0 background. A second control also matches hemisphere and
Yeo7 composition, using the inherited inferred labels. Neither matches spatial
distance or structural degree. This checks informativeness
under the existing Gaussian estimator, not causal necessity or exact PEID.
Added after observing the primary experiment; not a preregistered analysis.
"""
from collections import Counter
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_unconstrained_spt import ROIXiOracle, TOL
from scripts.analyze_dmf_spt_membership import membership_data, exact_paired_p, bh_q


def main():
    root = ROOT/'results/dmf_schaefer100/unconstrained_spt_wide'
    gs, seeds, records, selected, present = membership_data(root)
    with np.load(root/'membership.npz') as a:
        labels = a['labels']
    strata = [tuple(str(label).split('_')[1:3]) for label in labels]
    if not present.all():
        raise RuntimeError('Full membership scan required')
    heldout = [int(s) for s in seeds if s != 4]
    all_values = []; pvalues = []; zeros = 0; minimum = 0.
    for ref in (.8, 1.3, 2.):
        members = records[4, ref]['cores']['10']['members']
        rng = np.random.default_rng(np.random.SeedSequence([45081, int(ref*100)]))
        controls = set()
        while len(controls) < 256:
            draw = tuple(sorted(rng.choice(100, len(members), replace=False).tolist()))
            if draw != tuple(members):
                controls.add(draw)
        controls = sorted(controls)
        counts = Counter(strata[i] for i in members)
        matched = set()
        while len(matched) < 256:
            draw = []
            for key, count in sorted(counts.items()):
                pool = [i for i in range(100) if strata[i] == key]
                draw.extend(rng.choice(pool, count, replace=False).tolist())
            draw = tuple(sorted(draw))
            if draw != tuple(members):
                matched.add(draw)
        matched = sorted(matched)
        values = np.empty((len(heldout), 5))
        for si, seed in enumerate(heldout):
            for gi, g in enumerate((0., ref)):
                with np.load(records[seed, g]['covariance_path']) as a:
                    oracle = ROIXiOracle(a['conditional_covariance'])
                if gi == 0:
                    values[si, 0] = np.mean([oracle.xi(c) for c in controls])
                    values[si, 1] = oracle.xi(members)
                else:
                    values[si, 2] = np.mean([oracle.xi(c) for c in controls])
                    values[si, 3] = np.mean([oracle.xi(c) for c in matched])
                    values[si, 4] = oracle.xi(members)
                zeros += oracle.tolerance_zero_count
                minimum = min(minimum, oracle.minimum_xi)
        p = exact_paired_p(values[:, 4]-values[:, 2])
        pm = exact_paired_p(values[:, 4]-values[:, 3]); pvalues.extend([p, pm])
        all_values.append(values)
        print(f'G={ref}, size={len(members)}, means [random0, core0, randomG, matchedG, coreG]='
              f'{values.mean(0).tolist()}, p size={p}, p matched={pm}')
    qvalues = bh_q(pvalues)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(12.8, 4.5))
    for i, (ref, values) in enumerate(zip((.8, 1.3, 2.), all_values)):
        ax = fig.add_axes([.065+i*.322, .27, .26, .57])
        for si, v in enumerate(values):
            offset = (si-3)*.025
            ax.plot(np.arange(2)+offset, v[:2], color='#bac4cb', lw=.8)
            ax.plot(np.arange(2,5)+offset, v[2:], color='#bac4cb', lw=.8)
            ax.scatter(np.arange(5)+offset, v, s=16,
                       c=['#bac4cb', '#6c8597', '#bac4cb', '#6c8597', '#234f77'], zorder=3)
        ax.scatter(np.arange(5), values.mean(0), marker='D', color='#234f77', s=28, zorder=4)
        ax.set(xticks=np.arange(5), xticklabels=['Random\nG=0', 'Core\nG=0',
                                                f'Random\nG={ref:g}', f'Matched\nG={ref:g}', f'Core\nG={ref:g}'],
               xlim=(-.3, 4.3), ylim=(-.06, 2.2), ylabel='Cross-ROI Xi (bits)')
        size = len(records[4, ref]['cores']['10']['members'])
        ax.text(0, 1.16, f'{"abc"[i]}  Frozen G={ref:g} core ({size} ROI)',
                transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.text(.5, 1.035, f'Core vs random: BH q={qvalues[2*i]:.4g}; vs matched: q={qvalues[2*i+1]:.4g}',
                transform=ax.transAxes, ha='center', fontsize=7.4)
        ax.tick_params(axis='x', labelsize=8)
    fig.text(.52, .10, 'Core selected using seed 4; validation uses the other 7 seeds. '
             'Random / matched = mean of 256 fixed control sets.', ha='center', fontsize=9)
    fig.text(.52, .045, 'Circles: seeds; diamonds: means. Random: size only. '
             'Matched: same size, hemisphere and Yeo7 composition. Exploratory Gaussian-estimator audit.',
             ha='center', fontsize=8, color='#56636d')
    output = ROOT/'fig/brain_dmf_spt_core_size_control.png'
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'BH q={qvalues.tolist()}; tolerance={TOL} bits; tolerance-negative count={zeros}; minimum Xi={minimum}')
    print(output)


if __name__ == '__main__':
    main()
