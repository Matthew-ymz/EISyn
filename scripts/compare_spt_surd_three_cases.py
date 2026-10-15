"""Lightweight SURD comparison on the screenshot's identity-label A/B/C.

All nine node blocks and the complete one-step future target are retained.
Exact known-model specific information is integrated over the target with
scrambled Sobol quadrature. Allocation reuses the existing all-order SURD
implementation, NOT an allocation of already-averaged mutual information.
Native atom supports and a declared hypergraph-cut tree are scored separately.

Authority rechecked 2026-10-15: Zotero P6UJCVG8, main DXGC7JEA Methods
(5)-(12), SI MWIWKSVG S1.2/S5/S9.3. Neither has a manuscript date/version.
SURD: K69ZCNZG, SI KNU3AJTZ S1.2 (S12)-(S16), pp3-5, and authors' code
https://github.com/ALD-Lab/SURD/blob/main/utils/surd.py .
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import nontrivial_bipartitions
from scripts.spt_continuous_phase_benchmark import (
    SOURCES, LOCAL_KAPPA, InformationOracle, PhaseKernel, channel_integral,
    noise_entropy, score,
)
from scripts.spt_heterogeneous_hierarchy_case import quality
from scripts.spt_nested_broadcast_case import NestedGaussianBroadcast
from scripts.unicm_observational_information import assign_surd_specific

N = len(SOURCES)
TOLERANCE_NATS = 1e-8
MEMBERSHIP = ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1).astype(float)
FIGURE = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_surd_three_cases.png'


def subset_mask(group):
    return sum(1 << (i-1) for i in group)


def source_group(mask):
    return tuple(i for i in SOURCES if mask >> (i-1) & 1)


def gaussian_specific(weights, standardized_targets):
    """KL(q(x_S|y)||q(x_S)) for independent Gaussian broadcast channels.

    The nine-observation channel mean is sufficient: its residual noise
    variance is 1+1/9. For r=sum_{i in S} w_i^2/Var(Y), the posterior KL is
    [-log(1-r)+r*(z^2-1)]/2. Independent channel KLs add before SURD sorting.
    Unused past readouts and independent future drivers cancel exactly.
    """
    variance = np.sum(weights**2, axis=1) + 1 + 1/N
    ratios = MEMBERSHIP @ weights.T**2 / variance
    constant = -.5*np.log1p(-ratios).sum(axis=1)
    return constant[None, :] + .5*(standardized_targets**2-1) @ ratios.T


def phase_specific_grid(grid):
    """Exact one-dimensional periodic quadrature of target-specific KL."""
    channel = channel_integral(grid)
    theta = channel['angles']
    from scripts.spt_continuous_phase_benchmark import KAPPA, GAIN, log_i0
    density = np.exp(KAPPA*np.cos(theta[:, None]-GAIN*np.sin(theta)[None, :])
                     -log_i0(KAPPA))/(2*np.pi)
    marginal = channel['marginal']
    specific = np.mean(density*np.log(density/marginal[:, None]), axis=1)/marginal
    mass = marginal*(2*np.pi/grid)
    error = abs(float(mass @ specific)-channel['weight_nats'])
    if error > 1e-11 or specific.min() < -TOLERANCE_NATS:
        raise ArithmeticError(f'Circular specific-information quadrature failed: {error}')
    return mass, specific, error


def phase_specific(supports, uniforms, grid=512):
    """Full-target KL: local-channel constants + complete-edge channel KLs.

    Different hyperedges use independent source coordinates. A strict subset
    of an edge leaves its total input phase uniform, so supplies zero about
    that edge output. Uniform sharing masks add no source information.
    """
    mass, lookup, error = phase_specific_grid(grid)
    indices = np.searchsorted(np.cumsum(mass), uniforms, side='right')
    if np.any(indices >= grid):
        raise ArithmeticError('Target quadrature exceeded its support')
    edge_specific = lookup[indices]
    edge_masks = np.array([subset_mask(e) for e in supports])
    completed = ((np.arange(1 << N)[:, None] & edge_masks) == edge_masks).astype(float)
    local = MEMBERSHIP @ (np.log(2*np.pi)-noise_entropy(LOCAL_KAPPA))
    return local[None, :] + edge_specific @ completed.T, error


def allocate_specific(specific, checkpoints):
    """Allocate at every target realization; average atoms only afterwards."""
    minimum = float(specific.min())
    negative = int(np.count_nonzero(specific < 0))
    violations = int(np.count_nonzero(specific < -TOLERANCE_NATS))
    if violations:
        raise ArithmeticError(f'Specific KL: minimum={minimum} nats; '
                              f'threshold={-TOLERANCE_NATS}; affected_count={violations}')
    red_sum = np.zeros(1 << N)
    syn_sum = np.zeros(1 << N)
    maximum_closure = 0.
    records = {}
    for i, row in enumerate(specific, 1):
        red, syn = assign_surd_specific(row, N)
        maximum_closure = max(maximum_closure, abs(float(red.sum()+syn.sum()-row[-1])))
        red_sum += red
        syn_sum += syn
        if i in checkpoints:
            records[i] = {'redundancy_unique': red_sum/i, 'synergy': syn_sum/i,
                          'mi_table': specific[:i].mean(axis=0)}
    if maximum_closure > TOLERANCE_NATS:
        raise ArithmeticError(f'SURD closure failed: {maximum_closure} nats')
    return records, {'minimum_specific_nats':minimum,
                     'specific_tolerance_negative_count':negative,
                     'specific_significant_violation_count':violations,
                     'maximum_statewise_closure_error_nats':maximum_closure}


def hypergraph_tree(synergy, *, normalize=False):
    """Experimental adaptation: recursively minimize crossing atom weight.

    F(A)=sum_{E subset A} S(E). At node A maximize F(L)+F(R), equivalently
    minimize F(A)-F(L)-F(R). Edges not contained in A stay assigned above A.
    Full enumeration, raw weights, lexicographic ties, singleton termination.
    This is not an algorithm supplied by the original SURD publication.
    """
    if not np.isfinite(synergy).all() or np.min(synergy) < -TOLERANCE_NATS:
        raise ArithmeticError('Invalid SURD hypergraph weights')
    retained = np.asarray(synergy, float).copy()
    for bit in range(N):
        for mask in range(1 << N):
            if mask >> bit & 1:
                retained[mask] += retained[mask ^ (1 << bit)]
    clades, residuals = [], []
    gaps = []
    def visit(group):
        if len(group) == 1:
            return ()
        candidates = []
        for l, r in nontrivial_bipartitions(group):
            cut = float(retained[subset_mask(group)]-retained[subset_mask(l)]-retained[subset_mask(r)])
            residuals.append(cut)
            objective = (cut/min(len(l),len(r)) if normalize
                         else -float(retained[subset_mask(l)]+retained[subset_mask(r)]))
            candidates.append((objective,l,r))
        candidates.sort()
        objective, left, right = candidates[0]
        gaps.append(float(candidates[1][0]-objective) if len(candidates)>1 else None)
        for child in (left, right):
            if len(child)>1:
                clades.append(child)
            visit(child)
        return left, right
    root = visit(SOURCES)
    violations = sum(r < -TOLERANCE_NATS for r in residuals)
    if violations:
        raise ArithmeticError(f'SURD cut: minimum={min(residuals)} nats; '
                              f'threshold={-TOLERANCE_NATS}; affected_count={violations}')
    return {'root_split':root, 'clades':clades,
            'status':'posthoc_normalized_hypergraph_cut' if normalize else 'experimental_hypergraph_cut',
            'root_score_gap_nats':gaps[0], 'minimum_cut_nats':min(residuals),
            'cut_tolerance_negative_count':sum(-TOLERANCE_NATS<=r<0 for r in residuals)}


def maximum_weight_tree(synergy, sources=SOURCES):
    """Posthoc sensitivity: globally maximize sum of selected clade atom weights.

    Dynamic programming over all 511 subsets. No planted supports, support
    counts, size targets, or thresholds enter this alternative tree objective.
    It is retained alongside the predeclared cut, not selected as a winner.
    """
    choices = {}
    @lru_cache(None)
    def optimum(group):
        if len(group)==1:
            return 0.
        neg, left, right = min((-optimum(l)-optimum(r),l,r)
                               for l,r in nontrivial_bipartitions(group))
        choices[group] = left,right
        return float(synergy[subset_mask(group)]-neg)
    sources = tuple(sources)
    optimum(sources)
    clades = []
    def visit(group):
        if len(group)==1:
            return
        for child in choices[group]:
            if len(child)>1:
                clades.append(child)
            visit(child)
    visit(sources)
    return {'root_split':choices[sources], 'clades':clades,
            'status':'posthoc_maximum_weight_laminar_tree',
            'selected_atom_weight_nats':optimum(sources)}


def summaries(atoms, supports):
    synergy = atoms['synergy']
    minimum = float(synergy.min())
    violations = int(np.count_nonzero(synergy < -TOLERANCE_NATS))
    if violations:
        raise ArithmeticError(f'SURD atom: minimum={minimum} nats; '
                              f'threshold={-TOLERANCE_NATS}; affected_count={violations}')
    # The threshold is numerical zero in nats, not a fitted significance level.
    detected = [source_group(m) for m in range(1, (1 << N)-1)
                if m.bit_count()>=2 and synergy[m] > TOLERANCE_NATS]
    primitive = tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
    native = quality((), detected, supports, primitive)
    native = {k:v for k,v in native.items() if k not in ('root_split','correct_root')}
    truth = {frozenset(e) for e in supports if len(e)<N}
    counted = {frozenset(e) for e in detected if not any(set(e)<set(g) for g in primitive)}
    native.update(detected_count=len(counted), true_positive_count=len(counted & truth),
                  false_positive_count=len(counted-truth), false_negative_count=len(truth-counted),
                  supports_are_laminar=all(not (a&b) or a<=b or b<=a
                      for a in counted for b in counted))
    native['threshold_sensitivity'] = []
    for threshold in (1e-8,1e-6,1e-4,1e-3,1e-2):
        selected = {frozenset(source_group(m)) for m in range(1,(1 << N)-1)
                    if m.bit_count()>=2 and synergy[m]>threshold
                    and not any(set(source_group(m))<set(g) for g in primitive)}
        common = len(selected & truth)
        native['threshold_sensitivity'].append({'threshold_nats':threshold,
            'detected_count':len(selected),'true_positive_count':common,
            'f1':2*common/(len(selected)+len(truth))})
    raw_tree = hypergraph_tree(synergy)
    tree = {**raw_tree, **score(raw_tree, supports)}
    return native, tree


def load_representatives():
    nested = json.loads((ROOT/'results/spt_nested_broadcast_case/summary.json').read_text())
    phase = json.loads((ROOT/'results/spt_continuous_phase_benchmark/summary.json').read_text())
    records = nested['records'] + [r for r in phase['records'] if r['case'] in 'BC']
    selected = []
    for case in 'ABC':
        matches = [r for r in records if r['case']==case and r['variant']==0 and r['permutation_id']==0]
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one representative for {case}')
        r = matches[0]
        a = r['audit']
        if a['significant_violation_count'] or a['minimum_syn_nats'] < -a['tolerance_nats'] or a['clipping']:
            raise ArithmeticError(f'Cached PEID audit failed: {a}')
        if abs(a['closure_error_nats']) > a['tolerance_nats']:
            raise ArithmeticError('Cached PEID closure failed')
        selected.append(r)
    return selected, phase['protocol']['grid']


def run(power=13, seed=20261015):
    start = time.monotonic()
    representatives, grid = load_representatives()
    gaussian = NestedGaussianBroadcast()
    results = []
    for original in representatives:
        case, supports = original['case'], original['hyperedges']
        dimensions = gaussian.channels if case=='A' else len(supports)
        exact_mi = np.zeros(1 << N)
        oracle = gaussian if case=='A' else InformationOracle(PhaseKernel(supports), grid)
        for m in range(1, 1 << N):
            exact_mi[m] = oracle.info(source_group(m))
        checks = []
        for scramble in (seed, seed+1):
            uniforms = qmc.Sobol(dimensions, scramble=True, seed=scramble).random_base2(power)
            if case=='A':
                specific = gaussian_specific(gaussian.weights, ndtri(uniforms))
                quadrature_error = None
            else:
                specific, quadrature_error = phase_specific(supports, uniforms, grid)
            checkpoints = (1 << (power-2), 1 << power) if scramble==seed else (1 << power,)
            allocated, audit = allocate_specific(specific, checkpoints)
            for count, atoms in allocated.items():
                native, tree = summaries(atoms, supports)
                checks.append({'seed':scramble, 'samples':count,
                    'native_support_f1':native['clade_f1'], 'tree_f1':tree['clade_f1'],
                    'native_detected_count':native['detected_count'], 'root_split':tree['root_split'],
                    'clades':tree['clades'],
                    'maximum_subset_mi_integration_error_nats':float(np.max(abs(atoms['mi_table']-exact_mi)))})
                for method, constructor in (
                    ('surd_normalized_cut',lambda w:hypergraph_tree(w,normalize=True)),
                    ('surd_maximum_weight',maximum_weight_tree),
                ):
                    output = constructor(atoms['synergy'])
                    output = {**output, **score(output,supports)}
                    checks[-1][method+'_f1'] = output['clade_f1']
                    checks[-1][method+'_clades'] = output['clades']
                    if scramble==seed and count==1 << power:
                        if method=='surd_normalized_cut':
                            normalized_tree = output
                        else:
                            max_weight_tree = output
                if scramble==seed and count==1 << power:
                    primary = {'case':case, 'hyperedges':supports,
                        'methods':{'spt':original['methods']['spt'], 'surd_tree':tree,
                                   'surd_normalized_cut':normalized_tree,
                                   'surd_maximum_weight':max_weight_tree},
                        'native_supports':native,
                        'synergy_atoms_nats':[{ 'sources':source_group(m), 'value':float(atoms['synergy'][m])}
                                              for m in range(1, 1 << N) if m.bit_count()>=2],
                        'total_mi_nats':float(exact_mi[-1]),
                        'integrated_total_surd_nats':float(atoms['synergy'].sum()+atoms['redundancy_unique'].sum()),
                        'surd_synergy_sum_nats':float(atoms['synergy'].sum()),
                        'surd_redundancy_unique_sum_nats':float(atoms['redundancy_unique'].sum()),
                        'periodic_mean_specific_check_nats':quadrature_error,
                        'peid_audit':original['audit'], 'surd_audit':audit}
        primary['integration_checks'] = checks
        results.append(primary)
        print(json.dumps({'case':case, 'native_f1':primary['native_supports']['clade_f1'],
            'tree_f1':primary['methods']['surd_tree']['clade_f1'], 'integration_checks':checks}), flush=True)
    return {'protocol':{'scope':'screenshot version; identity labels; variant 0; three conditions',
        'source':'all nine original node blocks', 'target':'complete one-step future state, jointly',
        'distributions':'A iid N(0,1)^126; B/C iid Uniform(0,2*pi)^81; unchanged',
        'units':'nats', 'specific_estimator':'exact known-model posterior KL / periodic quadrature',
        'allocation':'existing assign_surd_specific; allocate before expectation; no Gaussian MI-only shortcut',
        'allocation_ties':'ascending order then source mask, existing implementation',
        'target_integration':{'method':'scrambled Sobol', 'primary_samples':1 << power,
                              'seeds':[seed,seed+1], 'periodic_grid':grid},
        'native_support_threshold_nats':TOLERANCE_NATS,
        'threshold_interpretation':'numerical zero; no significance test or ground-truth-based selection',
        'tree':'experimental recursive unnormalized minimum crossing SURD atom weight; lexicographic ties',
        'posthoc_sensitivities':['crossing weight / min(child sizes)',
                                'global maximum sum of selected clade atom weights',
                                'native support thresholds 1e-8,1e-6,1e-4,1e-3,1e-2 nats'],
        'sensitivity_disclosure':'added after observing the primary cut result; all outputs retained; no winner selected',
        'score':'same nonroot supports; exclude leaves/root and proper refinements inside primitive supports',
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','si':'MWIWKSVG',
                      'checked':'2026-10-15','version':'no explicit manuscript date/version; sole current attachment pair'},
        'surd_reference':{'parent':'K69ZCNZG','si':'KNU3AJTZ','locations':'S1.2 Eqs S12-S16, pp3-5'},
        'new_training':False, 'clipping':False},
        'elapsed_seconds':time.monotonic()-start, 'records':results}


def plot(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scripts.compose_spt_framework_comparison import current_tree
    records = data['records']
    style = {'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
             'font.size':10, 'axes.spines.top':False,'axes.spines.right':False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.6,7.0),layout='constrained')
        gs = fig.add_gridspec(3,3,height_ratios=(1.6,1.25,1.25))
        ax = fig.add_subplot(gs[0,:])
        labels = ('SPT hierarchy','SURD native supports','SURD + cut (primary)',
                  'SURD + normalized cut*','SURD + max-weight tree*')
        values = np.array([[r['methods']['spt']['clade_f1'] for r in records],
                           [r['native_supports']['clade_f1'] for r in records],
                           *[[r['methods'][m]['clade_f1'] for r in records]
                             for m in ('surd_tree','surd_normalized_cut','surd_maximum_weight')]])
        ax.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_xticks(range(3),['A  Mixed branches','B  Deep cascade','C  Triadic modules'])
        ax.set_yticks(range(len(labels)),labels)
        ax.tick_params(length=0,pad=7)
        ax.set_title('a   Nonroot support F1 · one identity-label case per architecture',
                     loc='left',fontsize=11,weight='bold',pad=10)
        for i,j in np.ndindex(values.shape):
            ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',
                    color='white' if values[i,j]>.55 else '#24323C')
        for row, method in enumerate(('spt','surd_maximum_weight'),1):
            for col, record in enumerate(records):
                ax = fig.add_subplot(gs[row,col])
                current_tree(ax,record,method)
                label = 'SPT' if method=='spt' else 'SURD + max-weight tree*'
                ax.set_title(f'{"bcdefg"[(row-1)*3+col]}   {record["case"]} · {label}',
                             loc='left',fontsize=10.5,weight='bold',pad=5)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec='#217B68',label='True support'),
                            Patch(fc='#FCEFEA',ec='#B74738',label='Wrong clade'),
                            Patch(fc='white',ec='#B7C0C6',label='Leaf / unscored refinement')],
                   loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        fig.suptitle('SURD tree rules are experimental adaptations.  *Posthoc sensitivity checks; all rules retained.',
                     fontsize=10.5,color='#46545E')
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=250,bbox_inches='tight',facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power',type=int,default=13)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--output',type=Path,default=Path('/tmp/eisyn_spt_surd_three_cases.json'))
    parser.add_argument('--figure',type=Path,default=FIGURE)
    args = parser.parse_args()
    if args.power<3:
        parser.error('--power must be at least 3')
    result = run(args.power,args.seed)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    plot(result,args.figure)
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'figure':str(args.figure)}))
