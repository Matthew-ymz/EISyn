"""Read-only numerical-feasibility figure, including the retained coarse grid."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    directory = ROOT / 'results/spt_matched_three_case/glv_simplified_A'
    pilot = json.loads((directory / 'pilot.json').read_text())
    refined = json.loads((directory / 'proposal_refinement_polished.json').read_text())
    if refined['status'] != 'feasibility_passed':
        raise ValueError('A passed refined pilot is required for this figure')
    masks = pilot['masks']
    summary = pilot['precision_result']['summary']
    sem = np.array([[r['ei_sem_nats'][str(m)] for m in masks] for r in summary])
    sem[-1] = [refined['information_sem_nats'][str(m)] for m in masks]
    means = np.array([[r['mean_ei_nats'][str(m)] for m in masks] for r in summary])
    means[-1] = np.mean([[row[str(m)] for m in masks]
                        for row in refined['updated_finest_information_nats']], axis=0)
    changes = abs(np.diff(means, axis=0))
    figure = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_glv_simplified_a_pilot.png'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.right': False, 'axes.spines.top': False,
                         'axes.linewidth': .8, 'xtick.labelsize': 8, 'ytick.labelsize': 8}):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25), layout='constrained')
        labels = [['256 / 64', '256 / 256', '1024 / 256', '4096 / 256'],
                  ['Inner\n64→256', 'Outer\n256→1024', 'Outer\n1024→4096']]
        for panel, (ax, values) in enumerate(zip(axes, (sem, changes))):
            # Deterministic horizontal offsets separate queries without altering values.
            for x, row in enumerate(values):
                ax.scatter(x + np.linspace(-.12, .12, len(row)), row, s=17,
                           color='#9aa6ad', alpha=.65, linewidths=0,
                           label='16 EI queries' if x == 0 else None, zorder=2)
            ax.plot(np.arange(len(values)), values.max(1), '-D', color='#167663',
                    lw=1.2, markersize=4, label='Maximum', zorder=3)
            ax.axhline(.01, color='#aa6429', linestyle='--', lw=1., label='0.01 nats')
            ax.set_xticks(range(len(values)), labels[panel])
            ax.set_ylim(0, max(.0125, float(values.max()) * 1.17))
            ax.set_xlim(-.35, len(values)-.65)
            ax.tick_params(length=3)
            ax.text(-.14, 1.04, 'ab'[panel], transform=ax.transAxes,
                    fontweight='bold', fontsize=11, va='bottom')
        axes[0].set_ylabel('EI numerical SEM (nats)')
        axes[0].set_xlabel('Outer / inner samples')
        axes[1].set_ylabel('|Change in mean EI| (nats)')
        axes[1].set_xlabel('Precision comparison')
        axes[1].legend(loc='center left', bbox_to_anchor=(1.02, .5), frameon=False, fontsize=8)
        fig.savefig(figure, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(figure)


if __name__ == '__main__':
    main()
