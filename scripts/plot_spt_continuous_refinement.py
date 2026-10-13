"""Plot measured phase-benchmark recovery and paired RQMC diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("spt", "block_phi_r", "pair_phi_r_hac", "pair_xi_hac", "future_mi_hac")
LABELS = ("SPT", r"Block $\Phi_R$", r"Pair $\Phi_R$ + HAC", r"Pair $\Xi$ + HAC", "Future MI + HAC")
STYLE = {"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
         "font.size":10, "axes.labelsize":10, "axes.spines.top":False,
         "axes.spines.right":False, "xtick.labelsize":9, "ytick.labelsize":9}


def plot_recovery(data, path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1,3,figsize=(9.8,3.65),layout="constrained")
        metrics = (("clade_f1","Support F1"), ("upper_support_recall","Upper-support recall"),
                   ("correct_root","Root recovery rate"))
        for letter, ax, (metric,label) in zip("abc",axes,metrics):
            values = np.array([[np.mean([r["methods"][method][metric] for r in data["records"] if r["case"]==case])
                                for case in "ABC"] for method in METHODS])
            ax.imshow(values,vmin=0,vmax=1,cmap="Blues",aspect="auto")
            ax.set_xticks(range(3),["A (60)","B (240)","C (60)"])
            ax.set_yticks(range(5),LABELS if ax is axes[0] else [""]*5)
            ax.tick_params(length=0,pad=7)
            ax.set_xlabel("Architecture (labelings × variants)")
            ax.set_title(f"{letter}   {label}",loc="left",fontweight="bold",fontsize=11,pad=12)
            for i,j in np.ndindex(values.shape):
                text = f"{values[i,j]:.2f}"
                if j==2 and i>=2:
                    text += "*"
                ax.text(j,i,text,ha="center",va="center",color="white" if values[i,j]>.65 else "#17212B",fontsize=10)
        fig.get_layout_engine().set(rect=(0,.11,1,.89))
        fig.text(.5,.027,"Descriptive means over 20 common labelings.  *C: zero pair affinity; methods report no hierarchy.",
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
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plot_recovery(json.loads(args.phase.read_text()),args.output/"spt_continuous_phase_recovery.png")
    plot_precision(json.loads(args.precision.read_text()),args.output/"spt_continuous_precision_diagnostic.png")


if __name__=="__main__":
    main()
