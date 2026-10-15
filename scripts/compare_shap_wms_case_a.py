"""Natural-trajectory WMS and SHAP explanation-profile trees, case A only.

WMS uses an unshuffled continuous natural trajectory, not factorized inputs.
The complete natural cross-time Gaussian joint is fitted with an affine TM;
the exact stationary Gaussian counterpart is checked independently. SHAP
explains the full conditional mean, with natural population background, and
clusters exact mean-absolute attribution profiles by cosine/single linkage.
No claim that SHAP itself defines a causal hierarchy or that WMS is a new
objective relative to the fixed-target SPT residual.

Authority rechecked 2026-10-15: P6UJCVG8, DXGC7JEA Methods (5)-(12), SI
MWIWKSVG S9.1/S9.4, S1.2/S5. Sole attachment pair has no explicit version/date.
SHAP definitions: official LinearExplainer and utils.hclust documentation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.linalg import solve_triangular
from scipy.spatial.distance import pdist,squareform

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.spt import all_nonempty_subsets,nontrivial_bipartitions
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_nested_broadcast_case import NestedGaussianBroadcast,SOURCES,SUPPORTS

TOLERANCE_NATS = 1e-8
FIGURE = ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_case_a_shap_wms.png'


def coordinates(group):
    return np.array([j for i in group for j in range(14*(i-1),14*i)])


def natural_trajectory(model,samples,seed,burn=128):
    """Exact vectorized continuous Markov trajectory for this T^2=0 model.

    State t uses the freshly drawn drivers at t and readouts generated from
    drivers t-1. Adjacent pairs genuinely share a state. Every readout retains
    its common signal and independent observation noise; no node is shuffled.
    """
    rng = np.random.default_rng(seed)
    count = samples+burn+1
    drivers = rng.normal(size=(count+1,9,7))
    signal = np.einsum('nig,gi->ng',drivers[:-1],model.weights)+rng.normal(size=(count,7))
    readout = signal[:,None,:]+rng.normal(size=(count,9,7))
    return np.concatenate((drivers[1:],readout),axis=2)[burn:].reshape(samples+1,126)


def stationary_joint(model):
    transition,noise = model.transition,model.noise_covariance
    if np.max(abs(transition @ transition))>1e-14:
        raise ValueError('The finite-memory stationary formula does not apply')
    sigma = noise+transition @ noise @ transition.T
    if np.max(abs(sigma-transition @ sigma @ transition.T-noise))>1e-12:
        raise ArithmeticError('Natural covariance violates the stationary equation')
    return np.block([[sigma,sigma @ transition.T],[transition @ sigma,sigma]])


def gaussian_mi_table(joint):
    """Full-block MI from one positive Gaussian joint via Cholesky TM factors."""
    if joint.shape!=(252,252) or not np.isfinite(joint).all():
        raise ValueError('Expected the complete source/future joint covariance')
    np.linalg.cholesky(joint)
    source = joint[:126,:126]
    white_cross = solve_triangular(np.linalg.cholesky(joint[126:,126:]),joint[126:,:126],lower=True)
    posterior = source-white_cross.T @ white_cross
    np.linalg.cholesky(posterior)
    table = {():0.}
    for group in all_nonempty_subsets(SOURCES):
        idx = coordinates(group)
        prior_factor = np.linalg.cholesky(source[np.ix_(idx,idx)])
        posterior_factor = np.linalg.cholesky(posterior[np.ix_(idx,idx)])
        table[group] = float(np.log(np.diag(prior_factor)).sum()-np.log(np.diag(posterior_factor)).sum())
    return table


def wms_tree(table):
    clades = []
    node_values = []
    def visit(group):
        if len(group)==1:
            return ()
        value,left,right = min((table[group]-table[l]-table[r],l,r)
                              for l,r in nontrivial_bipartitions(group))
        node_values.append({'sources':group,'wms_nats':value})
        for child in (left,right):
            if len(child)>1:
                clades.append(child)
            visit(child)
        return left,right
    root = visit(SOURCES)
    whole_minus_singletons = table[SOURCES]-sum(table[(i,)] for i in SOURCES)
    closure = whole_minus_singletons-sum(v['wms_nats'] for v in node_values)
    if abs(closure)>1e-10:
        raise ArithmeticError(f'WMS tree telescoping check failed: {closure} nats')
    return {'root_split':root,'clades':clades,'status':'natural_wms_minimum_split',
            'nodes':node_values,'wms_whole_minus_singletons_nats':whole_minus_singletons,
            'closure_error_nats':closure}


def shap_profiles(transition):
    """E|phi_i,o| = sqrt(2/pi)*||T[o, block i]|| for used iid N(0,1) drivers.

    The natural background's correlated old readouts have zero coefficients
    and are independent of the fresh drivers, so they do not alter this exact
    conditional or interventional SHAP expression. Full 126 outputs retained.
    """
    return np.stack([np.sqrt(2/np.pi)*np.sqrt(np.sum(transition[:,coordinates((i,))]**2,axis=1))
                     for i in SOURCES])


def profile_tree(profiles):
    distances = pdist(profiles,metric='cosine')
    if not np.isfinite(distances).all():
        raise ArithmeticError('Undefined attribution-profile cosine distance')
    hierarchy = linkage(distances,method='single')
    groups = {i:(i+1,) for i in range(9)}
    clades = []
    for j,row in enumerate(hierarchy):
        left,right = (groups[int(k)] for k in row[:2])
        group = tuple(sorted(left+right))
        groups[9+j] = group
        if len(group)<9:
            clades.append(group)
    return {'root_split':(left,right),'clades':clades,'status':'shap_output_profile_clustering',
            'linkage':hierarchy.tolist(),'cosine_distances':squareform(distances).tolist()}


def sample_shap_checks(model,states,profiles):
    # Verify exact Shapley values against the full 512-coalition game for one
    # real natural state, rather than against another linear-SHAP shortcut.
    x = states[0]
    game = {():np.zeros(126)}
    for group in all_nonempty_subsets(SOURCES):
        idx = coordinates(group)
        game[group] = model.transition[:,idx] @ x[idx]
    errors = []
    efficiency = np.zeros(126)
    for i in SOURCES:
        explicit = np.zeros(126)
        other = tuple(j for j in SOURCES if j!=i)
        for size in range(9):
            weight = math.factorial(size)*math.factorial(8-size)/math.factorial(9)
            for group in itertools.combinations(other,size):
                combined = tuple(sorted((*group,i)))
                explicit += weight*(game[combined]-game[group])
        idx = coordinates((i,))
        expected = model.transition[:,idx] @ x[idx]
        errors.append(float(np.max(abs(explicit-expected))))
        efficiency += expected
    interaction = 0.
    for i,j in itertools.combinations(SOURCES,2):
        other = tuple(k for k in SOURCES if k not in (i,j))
        for size in range(8):
            for group in itertools.combinations(other,size):
                delta = (game[tuple(sorted((*group,i,j)))]-game[tuple(sorted((*group,i)))]
                         -game[tuple(sorted((*group,j)))]+game[group])
                interaction = max(interaction,float(np.max(abs(delta))))
    empirical_profiles = np.zeros_like(profiles)
    for i in SOURCES:
        idx = coordinates((i,))
        # Only seven coefficients are nonzero for each output, so avoid a large
        # sample x player x output tensor while retaining every output field.
        contribution = states[:-1,idx] @ model.transition[:,idx].T
        empirical_profiles[i-1] = np.abs(contribution).mean(axis=0)
    return {'explicit_512_coalition_shap_max_error_output_units':max(errors),
            'shap_efficiency_error_output_units':float(np.max(abs(efficiency-model.transition @ x))),
            'maximum_second_difference_output_units':interaction,
            'pair_interactions_exact_population':0.,
            'empirical_profile_max_error_output_units':float(np.max(abs(empirical_profiles-profiles))),
            'empirical_profiles':empirical_profiles}


def run(samples=65536,seed=20261015):
    start = time.monotonic()
    model = NestedGaussianBroadcast()
    source = json.loads((ROOT/'results/spt_nested_broadcast_case/summary.json').read_text())
    prior_surd = json.loads((ROOT/'results/spt_surd_full_benchmark/summary.json').read_text())
    surd = {r['permutation_id']:r for r in prior_surd['records'] if r['case']=='A'}
    if len(source['records'])!=20 or len(surd)!=20:
        raise ValueError('Expected the original twenty A labelings')
    states = natural_trajectory(model,samples,seed)
    joint_samples = np.concatenate((states[:-1],states[1:]),axis=1)
    empirical_joint = np.cov(joint_samples,rowvar=False)
    del joint_samples
    natural_exact_joint = stationary_joint(model)
    natural = gaussian_mi_table(empirical_joint)
    exact = gaussian_mi_table(natural_exact_joint)
    intervention = gaussian_mi_table(model.joint_covariance)
    exact_equality = max(abs(exact[g]-intervention[g]) for g in exact)
    if exact_equality>1e-11:
        raise ArithmeticError('This model no longer has equivalent natural/interventional effective inputs')
    profiles = shap_profiles(model.transition)
    shap_check = sample_shap_checks(model,states,profiles)
    empirical_profiles = shap_check.pop('empirical_profiles')
    records = []
    all_residuals = [natural[a]-natural[l]-natural[r] for a in natural if len(a)>1
                     for l,r in nontrivial_bipartitions(a)]
    exact_residuals = [exact[a]-exact[l]-exact[r] for a in exact if len(a)>1
                      for l,r in nontrivial_bipartitions(a)]
    for original in source['records']:
        permutation_id,permutation = original['permutation_id'],original['permutation']
        to_physical = dict(zip(SOURCES,permutation))
        def labeled(table):
            return {g:table[tuple(sorted(to_physical[i] for i in g))] for g in table}
        wms = wms_tree(labeled(natural))
        oracle = wms_tree(labeled(exact))
        target_indices = coordinates(permutation)
        shap = profile_tree(profiles[np.array(permutation)-1][:,target_indices])
        empirical_shap = profile_tree(empirical_profiles[np.array(permutation)-1][:,target_indices])
        methods = {'spt':original['methods']['spt'],
                   'wms_natural':{**wms,**score(wms,original['hyperedges'])},
                   'wms_natural_exact':{**oracle,**score(oracle,original['hyperedges'])},
                   'shap_profiles':{**shap,**score(shap,original['hyperedges'])},
                   'shap_profiles_empirical':{**empirical_shap,**score(empirical_shap,original['hyperedges'])},
                   'surd_maximum_weight':surd[permutation_id]['methods']['surd_maximum_weight']}
        records.append({'case':'A','variant':0,'permutation_id':permutation_id,'permutation':permutation,
                        'hyperedges':original['hyperedges'],'methods':methods})
    readout_a,readout_b = 7,21
    def correlation(cov,a,b):
        return float(cov[a,b]/np.sqrt(cov[a,a]*cov[b,b]))
    validation = {**shap_check,'maximum_exact_natural_intervention_mi_difference_nats':exact_equality,
        'maximum_empirical_natural_mi_error_nats':max(abs(natural[g]-exact[g]) for g in exact),
        'maximum_empirical_joint_covariance_error':float(np.max(abs(empirical_joint-natural_exact_joint))),
        'natural_old_readout_correlation':correlation(empirical_joint,readout_a,readout_b),
        'exact_natural_old_readout_correlation':correlation(natural_exact_joint,readout_a,readout_b),
        'natural_fresh_driver_correlation':correlation(empirical_joint,0,14),
        'wms_minimum_all_split_nats':min(all_residuals),
        'wms_negative_split_count':sum(v<0 for v in all_residuals),
        'wms_below_minus_tolerance_count':sum(v < -TOLERANCE_NATS for v in all_residuals),
        'wms_in_negative_tolerance_band_count':sum(-TOLERANCE_NATS <= v < 0 for v in all_residuals),
        'wms_exact_minimum_all_split_nats':min(exact_residuals),
        'wms_exact_negative_split_count':sum(v<0 for v in exact_residuals),
        'wms_partition_count':len(all_residuals),'wms_clipping':False}
    return {'protocol':{'case':'A','labelings':20,'samples':samples,'trajectory_seed':seed,'burn':128,
        'wms_distribution':'unshuffled continuous natural trajectory; full 252-dimensional cross-time joint',
        'wms_estimator':'empirical full Gaussian covariance / affine Cholesky TM, no ridge or projection',
        'wms_split':'recursive minimum I(X_A;Y_full)-I(X_L;Y_full)-I(X_R;Y_full), signed',
        'wms_natural_oracle':'exact stationary covariance, used only for the separately reported oracle check',
        'spt_reference':'existing exact factorized Gaussian intervention; finite estimation differs from empirical WMS',
        'target':'same full 126-dimensional one-step future state',
        'shap_model':'exact full conditional mean T x, natural population background',
        'shap_tree':'exact mean absolute output profiles, all 126 outputs, cosine distance / single linkage',
        'shap_scope':'output-effect similarity; profile construction is an explicit experimental adaptation',
        'shap_package':'not installed; exact analytic SHAP and SciPy implementation of the documented clustering setting',
        'wms_units':'nats','shap_units':'original model output units; cosine distances dimensionless',
        'spt_reference_syn_tolerance_nats':source['protocol']['syn_tolerance_nats'],
        'wms_signed_audit_threshold_nats':TOLERANCE_NATS,
        'wms_negative_handling':'retain raw signed values; no nonnegative projection or numerical zeroing',
        'shap_zero_tolerance_output_units':1e-12,
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','si':'MWIWKSVG','checked':'2026-10-15',
                      'version':'sole main/SI, no explicit draft version/date'},
        'new_BC_computations':False,'new_training':False},
        'elapsed_seconds':time.monotonic()-start,'validation':validation,
        'shap_profiles_full_outputs':profiles.tolist(),
        'summary':{m:float(np.mean([r['methods'][m]['clade_f1'] for r in records]))
                   for m in records[0]['methods']},'records':records}


def plot(data,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scripts.compose_spt_framework_comparison import current_tree,GREEN,INK
    from scripts.plot_spt_heterogeneous_hierarchy import network,verify_export_clearance
    representative = data['records'][0]
    methods = ('spt','wms_natural','shap_profiles','surd_maximum_weight')
    names = ('SPT','WMS · natural','SHAP profiles\n+ hclust','SURD\n+ max-weight tree')
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':INK}):
        fig = plt.figure(figsize=(15,7.6),facecolor='white',dpi=250)
        ax = fig.add_axes((.02,.64,.39,.29))
        primitive = [e for e in representative['hyperedges'] if not any(set(g)<set(e) for g in representative['hyperedges'])]
        network(ax,{'case':'A','hyperedges':representative['hyperedges'],'planted_bottom':primitive},
                operator_symbol='H',node_size=220)
        ax.set_title('a   A · Mixed branches',loc='left',weight='bold',fontsize=11,pad=8)
        network_axis = ax
        profile = np.array(data['shap_profiles_full_outputs'])
        display = np.stack([profile[:,g+7::14].mean(axis=1) for g in range(7)],axis=1)
        ax = fig.add_axes((.47,.64,.25,.29))
        plotted = ax.imshow(display,aspect='auto',cmap='Blues',vmin=0)
        ax.set_xticks(range(7),[str(i+1) for i in range(7)])
        ax.set_yticks(range(9),[str(i) for i in SOURCES])
        ax.set(xlabel='Readout type',ylabel='Source node')
        ax.set_title('b   Mean |SHAP| profiles',loc='left',weight='bold',fontsize=11,pad=8)
        bar = fig.add_axes((.73,.64,.009,.29))
        fig.colorbar(plotted,cax=bar)
        ax = fig.add_axes((.923,.665,.057,.24))
        values = np.array([[data['summary'][m]] for m in methods])
        ax.imshow(values,cmap='Greens',vmin=0,vmax=1,aspect='auto')
        ax.set_xticks([0],['A (20)']);ax.xaxis.tick_top()
        ax.set_yticks(range(4),names);ax.tick_params(length=0,pad=6,labelsize=9)
        for i,v in enumerate(values[:,0]):
            ax.text(0,i,f'{v:.2f}',ha='center',va='center',color='white' if v>.7 else INK)
        ax.set_xlabel('Support F1',labelpad=8)
        ax.set_title('c',loc='left',weight='bold',pad=28)
        for spine in ax.spines.values():spine.set_visible(False)
        grid = fig.add_gridspec(1,4,left=.04,right=.98,top=.50,bottom=.12,wspace=.19)
        trees=[]
        for j,(method,name) in enumerate(zip(methods,names)):
            ax=fig.add_subplot(grid[j]);current_tree(ax,representative,method);trees.append(ax)
            ax.set_title(f'{"defg"[j]}   {name}',loc='left',fontsize=11,weight='bold',pad=9)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec=GREEN,label='True support'),
                            Patch(fc='#FCEFEA',ec='#B74738',label='Wrong clade'),
                            Patch(fc='white',ec='#B7C0C6',label='Leaf / unscored refinement')],
                   loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=9)
        fig.text(.5,.035,'WMS: continuous natural trajectory. SHAP tree: output-profile similarity; pairwise SHAP interactions are zero.',
                 ha='center',fontsize=9,color='#57636D')
        fig.text(.5,.014,'All scores: 20 common labelings. Trees: identity labels. Readout profiles condensed for display; all 126 outputs used.',
                 ha='center',fontsize=9,color='#57636D')
        fig.canvas.draw();verify_export_clearance(network_axis)
        renderer=fig.canvas.get_renderer()
        for ax in trees:
            boxes=[t.get_bbox_patch().get_window_extent(renderer) for t in ax._spt_node_labels]
            if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
                raise ValueError('Tree labels overlap')
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=250,bbox_inches='tight',facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=65536)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--output',type=Path,default=Path('/tmp/eisyn_shap_wms_case_a.json'))
    parser.add_argument('--figure',type=Path,default=FIGURE)
    args=parser.parse_args()
    result=run(args.samples,args.seed)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    plot(result,args.figure)
    print(json.dumps({'summary':result['summary'],'validation':result['validation'],
                      'elapsed_seconds':result['elapsed_seconds'],'figure':str(args.figure)},indent=2))
