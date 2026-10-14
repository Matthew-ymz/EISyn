"""Show accepted finite-data A/B/C; --phase-bc reproduces historical phase B/C.

These are generator supports, not inferred SPTs.  Recurrent trajectories are
illustrative natural dynamics after one model-specific initialization. The separate
one-step response panels use fresh iid full-state interventions, as in
the relevant information benchmark (Gaussian A; uniform phase B/C). Historical
nine-node phase A can be reproduced with --legacy-phase-a. No EI/Syn is
estimated here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0e, i1e
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_continuous_phase_benchmark import (
    GAIN, HARMONICS, KAPPA, OMEGA, TAU, TWO_PI, PhaseKernel, counterfactuals,
)

LABELS = {"A":"Mixed branches", "B":"Deep cascade", "C":"Triadic modules"}
CASE_COLORS = {"A":"#0072B2", "B":"#D55E00", "C":"#009E73"}
STYLE = {"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
         "font.size":10, "axes.labelsize":10, "axes.spines.top":False,
         "axes.spines.right":False, "xtick.labelsize":9, "ytick.labelsize":9}


def wrap(angle):
    return (np.asarray(angle)+np.pi) % TWO_PI-np.pi


def edge_label(edge):
    return "{"+",".join(map(str,edge))+"}"


def simulate_originals(steps=80, samples=4096, seed=20261014, *, legacy_phase_a=False, symmetric_a=False, four_node_a=False):
    sequence = np.random.SeedSequence(seed).spawn(steps+3)
    initial = np.random.default_rng(sequence[0]).uniform(0,TWO_PI,(1,9,9))
    # Common original source draws for the separate intervention panels.
    intervention = np.random.default_rng(sequence[-2]).uniform(0,TWO_PI,(samples,9,9))
    records = {}
    for case, variants in counterfactuals().items():
        if case=='A' and not legacy_phase_a:
            continue
        supports = variants[0]
        kernel = PhaseKernel(supports)
        state = initial.copy()
        states, drives, joint = [state[0].copy()], [], []
        for step in range(steps):
            f = np.array([kernel.edge_mean(state,j,e)[0] for j,e in enumerate(supports)])
            # Reset to the same independent step stream across architectures.
            # Local updates precede variable-length edge updates in the kernel,
            # so this pairs the actual local noises, not just the seed integer.
            future = kernel.step(state,np.random.default_rng(sequence[step+1]))
            z = np.array([wrap(np.sum(future[0,np.asarray(e)-1,j+1]-OMEGA[np.asarray(e)-1]*TAU))
                          for j,e in enumerate(supports)])
            assert np.all(np.isfinite(future)) and np.all((future>=0)&(future<TWO_PI))
            drives.append(f)
            joint.append(z)
            states.append(future[0].copy())
            state = future
        # Largest proper support, selected by mechanism size before drawing.
        slot = max((j for j,e in enumerate(supports) if len(e)<9), key=lambda j:len(supports[j]))
        selected = supports[slot]
        future = kernel.step(intervention,np.random.default_rng(sequence[-1]))
        idx = np.asarray(selected)-1
        # edge_mean is K sin(s); reconstruct s with the same harmonic profile.
        source_sum = wrap(np.sum(intervention[:,idx,slot+1]*HARMONICS[idx],axis=1)+kernel.shifts[slot])
        response = wrap(np.sum(future[:,idx,slot+1]-OMEGA[idx]*TAU,axis=1))
        residual = wrap(response-GAIN*np.sin(source_sum))
        noise_moment = np.mean(np.exp(1j*residual))
        expected_moment = i1e(KAPPA)/i0e(KAPPA)
        # Control the actual sampled joint response against the declared VM
        # noise, using a loose six-SEM iid integration check (no Syn involved).
        real_sem = np.std(np.cos(residual),ddof=1)/np.sqrt(samples)
        imag_sem = np.std(np.sin(residual),ddof=1)/np.sqrt(samples)
        if abs(noise_moment.real-expected_moment)>6*real_sem or abs(noise_moment.imag)>6*imag_sem:
            raise ArithmeticError("Sampled joint response does not match the original kernel")
        records[case] = {"supports":supports, "states":np.asarray(states),
                         "drive":np.asarray(drives), "joint":np.asarray(joint),
                         "response_edge":selected, "source_sum":source_sum, "response":response,
                         "residual_circular_mean":float(np.angle(noise_moment)),
                         "residual_resultant":float(abs(noise_moment))}
    np.testing.assert_array_equal(records['B']['states'][:,:,0],records['C']['states'][:,:,0])
    if legacy_phase_a:
        np.testing.assert_array_equal(records['A']['states'][:,:,0],records['B']['states'][:,:,0])
    elif symmetric_a or four_node_a:
        from scripts.spt_continuous_broadcast_case import simulate, WEIGHTED_GAINS
        records['A'] = simulate(steps,samples,seed,gains=(1.,1.,1.,1.) if symmetric_a else WEIGHTED_GAINS)
    else:
        from scripts.spt_nested_broadcast_case import simulate
        records['A'] = simulate(steps,samples,seed)
    return records


def plot_topologies(records,path):
    # Reuse the original reference-style renderer, including exact membership,
    # full-boundary nesting, and rendered-label clearance checks. H_m labels
    # denote continuous m-body channels, not the discrete parity/XOR mechanism
    # for which this renderer uses its default oplus symbol.
    from scripts.plot_spt_heterogeneous_hierarchy import network, verify_export_clearance
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16.8,2.75),facecolor="white",dpi=300)
        grid = fig.add_gridspec(1,3,left=.015,right=.985,bottom=.02,top=.90,wspace=.08)
        axes = []
        for col,case in enumerate("ABC"):
            ax = fig.add_subplot(grid[col])
            supports = records[case]['supports']
            primitive = tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
            record = {"case":case,"hyperedges":supports,"planted_bottom":primitive}
            if records[case].get('model')=='gaussian_public_readout':
                positions = {1:(-1.7,.65),2:(-1.7,-.10),3:(1.0,.65),4:(1.0,-.10)}
                anchors = {(1,2):(-1.63,.28),(3,4):(1.07,.28)}
                network(ax,record,operator_symbol='H',node_size=360,positions=positions,operator_positions=anchors)
                box = FancyBboxPatch((-2.4,-1.12),4.9,.35,boxstyle='round,pad=.02,rounding_size=.10',
                                     facecolor='#F0EDF7',edgecolor='#17106F',lw=.9)
                ax.add_patch(box)
                ax.text(.05,-.945,r'Noisy reads of $Z_1,Z_2$ at every node',ha='center',va='center',fontsize=8,color='#27343D')
                for x,g in ((-1.7,1),(1.,2)):
                    ax.annotate('',xy=(x,-.77),xytext=(x,-.35),arrowprops={'arrowstyle':'->','color':'#17106F','lw':.85})
                    ax.text(x+.22,-.54,rf'$Z_{g}$',ha='left',va='center',fontsize=9,color='#54208D')
                    weak = records[case]['driver_gains'][2*g-1]
                    coefficient = '' if weak==1 else r'\frac{1}{2}'
                    formula = rf'$D_{2*g-1}+{coefficient}D_{2*g}+\varepsilon_{g}$'
                    ax.text(x+.54,.28,formula,ha='left',va='center',fontsize=9,color='#27343D')
            else:
                network(ax,record,operator_symbol='H',node_size=360)
            label = 'Mixed branches' if case=='A' and records[case].get('model')!='gaussian_public_readout' else LABELS[case]
            if case=='A' and records[case].get('driver_gains')==(1.,1.,1.,1.):
                label = 'Noisy public readout'
            elif case=='A' and records[case].get('model')=='gaussian_public_readout':
                label = 'Unequal-input modules'
            ax.text(.5,1.01,f"{case}  {label}",ha="center",va="bottom",transform=ax.transAxes,
                    fontsize=12,fontweight="bold",color="#27343D")
            axes.append(ax)
        fig.canvas.draw()
        for ax in axes:
            verify_export_clearance(ax)
        fig.savefig(path,dpi=300,bbox_inches="tight",pad_inches=.04,facecolor="white")
        plt.close(fig)


def plot_dynamics(records,path,steps,samples):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.2,9.4),layout="constrained")
        grid = fig.add_gridspec(3,3,width_ratios=(1.12,1.12,1.0))
        phase_axes, joint_axes, response_axes = [], [], []
        angle_grid = np.linspace(-np.pi,np.pi,256)
        for row,case in enumerate("ABC"):
            record = records[case]
            gaussian = record.get('model') in ('gaussian_public_readout','nested_gaussian_public_readout')
            local_ax,joint_ax,response_ax = [fig.add_subplot(grid[row,col]) for col in range(3)]
            response_axes.append(response_ax)
            if not gaussian:
                phase_axes.append(local_ax); joint_axes.append(joint_ax)
            node_count = record['states'].shape[1]
            limit = max(np.max(abs(record['states'][:,:,0])),np.max(abs(record['joint']))) if gaussian else np.pi
            local_values = record['states'][:,:,0] if gaussian else wrap(record['states'][:,:,0])
            cmap = 'RdBu_r' if gaussian else 'twilight'
            local_ax.imshow(local_values.T,origin="upper",aspect="auto",cmap=cmap,
                            vmin=-limit,vmax=limit,interpolation="nearest",extent=(-.5,steps+.5,node_count+.5,.5))
            local_ax.set_yticks(range(1,node_count+1))
            local_ax.set_ylabel(f"{case} · Node",fontweight="bold")
            local_ax.set_xticks([0,20,40,60,80] if steps==80 else np.linspace(0,steps,5).astype(int))
            n = len(record['supports'])
            image = joint_ax.imshow(record['joint'].T,origin="upper",aspect="auto",cmap=cmap,
                            vmin=-limit,vmax=limit,interpolation="nearest",extent=(.5,steps+.5,n+.5,.5))
            joint_ax.set_yticks(range(1,n+1),["".join(map(str,e)) for e in record['supports']])
            joint_ax.set_ylabel("Public-readout mean" if gaussian else "Hyperedge members")
            joint_ax.set_xticks([1,20,40,60,80] if steps==80 else np.linspace(1,steps,5).astype(int))
            # All iid interventions are displayed; transparency encodes density
            # visually, with equal N, markers, limits and opacity in every case.
            response_ax.scatter(record['source_sum'],record['response'],s=2.2,alpha=.14,
                                color=CASE_COLORS[case],edgecolors="none",rasterized=True)
            if gaussian:
                extent = 1.03*max(np.max(abs(record['source_sum'])),np.max(abs(record['response'])))
                response_ax.plot([-extent,extent],[-extent,extent],color='#17212B',lw=1.6)
                response_ax.set(xlim=(-extent,extent),ylim=(-extent,extent))
                if record.get('model')=='nested_gaussian_public_readout':
                    response_ax.set_xlabel(r'Source drive $s_e(t)$')
                    response_ax.set_ylabel(r'Mean future readout $\overline{Y}_{e}$')
                else:
                    response_ax.set_xlabel(r'Source drive $D_1+\frac{1}{2}D_2$' if record['driver_gains'][1]==.5
                                           else r'Source sum $D_1+D_2$')
                    response_ax.set_ylabel(r'Mean future readout $\overline{Y}_{1}$')
                real_bar = fig.colorbar(image,ax=[local_ax,joint_ax],location='bottom',fraction=.035,shrink=.5,pad=.02,aspect=50)
                real_bar.set_label('A: real-valued state')
            else:
                response_ax.plot(angle_grid,GAIN*np.sin(angle_grid),color="#17212B",lw=1.6,zorder=4)
                response_ax.set(xlim=(-np.pi,np.pi),ylim=(-np.pi,np.pi),
                                xticks=(-np.pi,0,np.pi),yticks=(-np.pi,0,np.pi))
                response_ax.set_xticklabels((r"$-\pi$","0",r"$\pi$"))
                response_ax.set_yticklabels((r"$-\pi$","0",r"$\pi$"))
                response_ax.set_ylabel(r"Joint response $Z_e(t+1)$")
            response_ax.set_title("Support "+edge_label(record['response_edge']),loc="left",fontsize=10,pad=5)
            if row==2:
                local_ax.set_xlabel(r"Time step $t$")
                joint_ax.set_xlabel(r"Future step $t+1$")
                response_ax.set_xlabel(r"Source phase sum $s_e(t)$")
        titles = ("a   Node driver / local phase","b   Public / joint readout", "c   One-step causal channel")
        for ax,title in zip((fig.axes[0],fig.axes[1],response_axes[0]),titles):
            ax.text(0,1.18,title,transform=ax.transAxes,ha="left",va="bottom",fontweight="bold",fontsize=11)
        bar = fig.colorbar(image,ax=phase_axes+joint_axes,location="bottom",fraction=.035,shrink=.55,pad=.035,aspect=50)
        bar.set_ticks((-np.pi,0,np.pi),labels=(r"$-\pi$","0",r"$\pi$"))
        current_a = records['A'].get('model') in ('gaussian_public_readout','nested_gaussian_public_readout')
        bar.set_label("B/C: wrapped phase (rad)" if current_a else "Wrapped phase (rad)")
        fig.get_layout_engine().set(rect=(0,.055,1,.945))
        distribution = 'Gaussian A; uniform B/C' if current_a else 'uniform phases'
        mechanism = ('A: Gaussian public readout; B/C: circular multi-channel dynamics.' if current_a
                     else 'All cases: circular multi-channel dynamics.')
        fig.text(.5,.015,f"Left / middle: {steps} recurrent steps. Right: {samples:,} one-step interventions per case ({distribution}).\n"
                 +"Black curves: conditional centres. "+mechanism,
                 ha="center",fontsize=9,color="#40464C")
        fig.savefig(path,dpi=260,bbox_inches="tight",facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/reports/assets/kuramoto_hierarchy")
    parser.add_argument("--steps",type=int,default=80)
    parser.add_argument("--samples",type=int,default=4096)
    parser.add_argument("--seed",type=int,default=20261014)
    parser.add_argument("--networks-only",action="store_true",help="Redraw structural style without rerunning dynamics")
    version = parser.add_mutually_exclusive_group()
    version.add_argument('--phase-bc',action='store_true',help='Reproduce historical phase B/C')
    version.add_argument("--legacy-phase-a",action="store_true",help="Reproduce the historical nine-node phase A")
    version.add_argument('--symmetric-a',action='store_true',help='Reproduce the earlier equal-gain Gaussian A')
    version.add_argument('--four-node-a',action='store_true',help='Reproduce the earlier four-node weighted Gaussian A')
    args = parser.parse_args()
    if args.steps<1 or args.samples<128:
        parser.error("Use at least one time step and 128 intervention samples")
    if not (args.phase_bc or args.legacy_phase_a or args.symmetric_a or args.four_node_a):
        import json
        accepted=ROOT/'results/spt_finite_replacement_bc/summary.json'
        if accepted.exists() and json.loads(accepted.read_text()).get('acceptance',{}).get('replace_main_figure'):
            from scripts.plot_spt_finite_replacement_bc import simulate_originals as simulate_finite, dynamics
            if args.networks_only:
                from scripts.spt_nested_broadcast_case import SUPPORTS as A_SUPPORTS
                from scripts.spt_finite_replacement_bc import SUPPORTS as BC_SUPPORTS
                records={c:{'supports':A_SUPPORTS if c=='A' else BC_SUPPORTS[c]} for c in 'ABC'}
            else:
                records=simulate_finite(args.steps,args.samples,args.seed)
            args.output.mkdir(parents=True,exist_ok=True)
            plot_topologies(records,args.output/'spt_continuous_original_networks.png')
            if not args.networks_only:
                dynamics(records,args.output/'spt_continuous_original_dynamics.png')
            print('Accepted continuous finite-data A/B/C rendered from original generators.')
            return
    if args.networks_only:
        records = {case:{"supports":variants[0]} for case,variants in counterfactuals().items()}
        if args.symmetric_a or args.four_node_a:
            from scripts.spt_continuous_broadcast_case import SUPPORTS, WEIGHTED_GAINS
            records['A'] = {'supports':SUPPORTS,'model':'gaussian_public_readout',
                            'driver_gains':(1.,1.,1.,1.) if args.symmetric_a else WEIGHTED_GAINS}
        elif not args.legacy_phase_a:
            from scripts.spt_nested_broadcast_case import SUPPORTS
            records['A'] = {'supports':SUPPORTS,'model':'nested_gaussian_public_readout'}
    else:
        records = simulate_originals(args.steps,args.samples,args.seed,legacy_phase_a=args.legacy_phase_a,
                                     symmetric_a=args.symmetric_a,four_node_a=args.four_node_a)
    args.output.mkdir(parents=True,exist_ok=True)
    suffix='_phase_bc' if args.phase_bc else ''
    plot_topologies(records,args.output/f"spt_continuous_original_networks{suffix}.png")
    if args.networks_only:
        print("Original supports rendered in the reference style; membership, nesting and label clearance verified.")
        return
    plot_dynamics(records,args.output/f"spt_continuous_original_dynamics{suffix}.png",args.steps,args.samples)
    for case,record in records.items():
        if record.get('model') in ('gaussian_public_readout','nested_gaussian_public_readout'):
            print(f"A: {record['states'].shape[1]}-node Gaussian public readout; real-valued simulation")
            continue
        print(f"{case}: supports={len(record['supports'])}, response_edge={record['response_edge']}, "
              f"residual circular mean={record['residual_circular_mean']:.5f}, "
              f"resultant={record['residual_resultant']:.5f}")
    print("Verified identical B/C local trajectories under common local noise.")


if __name__=="__main__":
    main()
