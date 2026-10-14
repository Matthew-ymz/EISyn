"""Plot measured phase-benchmark recovery and paired RQMC diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
METHODS = ("spt", "block_phi_r", "pair_phi_r_hac", "pair_xi_hac", "future_mi_hac")
LABELS = ("SPT", r"Block $\Phi_R$", r"Pair $\Phi_R$ + HAC", r"Pair $\Xi$ + HAC", "Future MI + HAC")
STYLE = {"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
         "font.size":10, "axes.labelsize":10, "axes.spines.top":False,
         "axes.spines.right":False, "xtick.labelsize":9, "ytick.labelsize":9}


def normalized_phase_controls(data):
    """Recompute the added control on B/C with the original relabeling schedule.

    No sampling or expensive validation rerun. The original cached methods
    remain untouched; the new method uses the same known-model information
    queries, including the permuted local parameters, not just planted edges.
    """
    from scripts.spt_continuous_phase_benchmark import (
        SOURCES, TOLERANCE_NATS, InformationOracle, PhaseKernel, block_tree, counterfactuals, score,
    )
    architectures = counterfactuals()
    oracles = {}
    records = []
    for original in data['records']:
        if original['case'] not in 'BC':
            continue
        key = (original['case'],original['variant'])
        if key not in oracles:
            oracles[key] = InformationOracle(PhaseKernel(architectures[key[0]][key[1]]),data['protocol']['grid'])
        oracle = oracles[key]
        schedule = data['protocol']['permutations'][original['permutation_id']]
        to_original = dict(zip(SOURCES,schedule))
        def info(a,b=SOURCES):
            return oracle.info(tuple(to_original[i] for i in a),tuple(to_original[i] for i in b))
        before = oracle.query_count
        root,clades,values = block_tree(info,SOURCES,normalize=True)
        if min(values)<-TOLERANCE_NATS:
            raise ArithmeticError('Normalized phase control has an invalid raw PhiR candidate')
        record = {**original,'methods':dict(original['methods'])}
        record['methods']['normalized_block_phi_r'] = score(
            {'root_split':root,'clades':clades,'status':'resolved'},original['hyperedges'])
        record['normalized_block_query_count'] = oracle.query_count-before
        records.append(record)
    return records


def plot_recovery(data, path, replacement_a=None):
    records = data['records']
    methods, labels = METHODS, LABELS
    if replacement_a is not None:
        records = replacement_a['records']+normalized_phase_controls(data)
        # Preserve compatibility with previously saved equal-gain A records.
        for record in records[:len(replacement_a['records'])]:
            if 'normalized_block_phi_r' not in record['methods']:
                record['methods'] = {**record['methods'],'normalized_block_phi_r':record['normalized_block_control']}
        methods = (METHODS[0],METHODS[1],'normalized_block_phi_r',*METHODS[2:])
        labels = (LABELS[0],LABELS[1],r'Normalized Block $\Phi_R$',*LABELS[2:])
    if any(r['audit']['significant_violation_count'] for r in records):
        raise ArithmeticError('Recovery plot received invalid Syn results')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1,3,figsize=(10.9,4.4 if replacement_a else 4.0),layout="constrained")
        metrics = (("clade_f1","Support F1"), ("upper_support_recall","Upper-support recall"),
                   ("correct_root","Root recovery rate"))
        for letter, ax, (metric,label) in zip("abc",axes,metrics):
            values = np.empty((len(methods),3))
            for i,method in enumerate(methods):
                for j,case in enumerate('ABC'):
                    selected = [r['methods'][method][metric] for r in records
                                if r['case']==case and r['methods'][method][metric] is not None]
                    values[i,j] = np.mean(selected) if selected else np.nan
            cmap = plt.get_cmap('Blues').copy()
            cmap.set_bad('#EEEEEE')
            ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap=cmap,aspect="auto")
            case_labels = list('ABC') if replacement_a else ["A (60)","B (240)","C (60)"]
            ax.set_xticks(range(3),case_labels)
            ax.set_yticks(range(len(methods)),labels if ax is axes[0] else [""]*len(methods))
            ax.tick_params(length=0,pad=7)
            ax.set_xlabel("Case" if replacement_a else "Architecture (labelings × variants)")
            ax.set_title(f"{letter}   {label}",loc="left",fontweight="bold",fontsize=11,pad=12)
            for i,j in np.ndindex(values.shape):
                text = f"{values[i,j]:.2f}" if np.isfinite(values[i,j]) else 'N/A'
                if j==2 and methods[i] in METHODS[2:]:
                    text += "*"
                ax.text(j,i,text,ha="center",va="center",color="white" if values[i,j]>.65 else "#17212B",fontsize=10)
        fig.get_layout_engine().set(rect=(0,.14,1,.86))
        if replacement_a:
            unavailable = 'N/A: no upper support. ' if any(r['methods']['spt']['upper_support_recall'] is None for r in records) else ''
            note = ("Cases use different model families.\n"
                    +r"Normalized: $\Phi_R/\min(|L|,|R|)$. "+unavailable+"*C: zero pair affinity; no reported hierarchy.")
        else:
            note = "Descriptive means over 20 common labelings.  *C: zero pair affinity; methods report no hierarchy."
        fig.text(.5,.025,note,
                 ha="center",fontsize=9,color="#333333")
        fig.savefig(path,dpi=260,bbox_inches="tight",facecolor="white")
        plt.close(fig)


def plot_precision(data,path):
    colors = {"raw":"#0072B2","jackknife":"#D55E00"}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2,3,figsize=(10.6,6.3),sharey="row",layout="constrained")
        for col,case in enumerate("ABC"):
            runs = [r for r in data["runs"] if r["case"]==case]
            for row in range(2):
                ax = axes[row,col]
                settings = [(1024,p) for p in (1024,4096,16384)] if row==0 else [(n,16384) for n in (1024,4096)]
                x = np.array([p if row==0 else n for n,p in settings])
                for kind in colors:
                    values = np.array([[next(r for r in run["records"] if (r["evaluations"],r["particles"])==(n,p))
                                        ["residual_nats"][kind]["previous_worst"] for n,p in settings] for run in runs])
                    marker = "o" if kind=="raw" else "s"
                    ax.plot(x,values.mean(0),marker=marker,color=colors[kind],lw=1.5,ms=5,
                            linestyle="-" if kind=="raw" else "--")
                    for replicate in values:
                        ax.scatter(x,replicate,color=colors[kind],s=14,alpha=.55,marker=marker,zorder=3)
                ax.axhline(0,color="#777777",lw=.7)
                ax.axhline(-.03,color="#555555",lw=.8,ls=":")
                ax.set_xscale("log",base=2)
                ax.set_xticks(x,[f"{v//1024}k" for v in x])
                ax.set_xlabel("Integration particles P" if row==0 else "Evaluation samples N")
                ax.set_title(f"{'abcdef'[row*3+col]}   {case} · "+("N = 1024" if row==0 else "P = 16384"),
                             loc="left",fontweight="bold",fontsize=11,pad=9)
                if col==0:
                    ax.set_ylabel("Previously worst split\nSyn residual (nats)")
                ax.grid(axis="y",alpha=.18,lw=.5)
        handles = [Line2D([],[],color=colors[k],marker="o" if k=="raw" else "s",
                          ls="-" if k=="raw" else "--",label=label)
                   for k,label in (("raw","Raw log mixture"),("jackknife","Two-half jackknife"))]
        handles += [Line2D([],[],color="#555555",ls=":",label="Declared threshold −0.03 nats")]
        fig.get_layout_engine().set(rect=(0,.065,1,.935))
        fig.legend(handles=handles,loc="outside upper center",ncol=3,frameon=False,fontsize=9)
        fig.text(.5,.018,"Dots: 4 independent scrambles; lines: means.  Partial queries do not certify a full EI table or tree.",
                 ha="center",fontsize=9,color="#333333")
        fig.savefig(path,dpi=240,bbox_inches="tight",facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",type=Path,default=ROOT/"results/spt_continuous_phase_benchmark/summary.json")
    parser.add_argument("--precision",type=Path,default=ROOT/"results/spt_continuous_precision_diagnostic/summary.json")
    parser.add_argument("--output",type=Path,default=ROOT/"docs/reports/assets/kuramoto_hierarchy")
    parser.add_argument('--replacement-a',type=Path,help='Use an alternative Gaussian A for the current-case plot')
    parser.add_argument('--current-only',action='store_true',help='Only render current A/B/C; preserve historical phase and precision plots')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.replacement_a:
        plot_recovery(json.loads(args.phase.read_text()),args.output/'spt_continuous_current_comparison.png',
                      replacement_a=json.loads(args.replacement_a.read_text()))
    if args.current_only:
        if not args.replacement_a:
            parser.error('--current-only requires --replacement-a')
        return
    plot_recovery(json.loads(args.phase.read_text()),args.output/"spt_continuous_phase_recovery.png")
    plot_precision(json.loads(args.precision.read_text()),args.output/"spt_continuous_precision_diagnostic.png")


if __name__=="__main__":
    main()
