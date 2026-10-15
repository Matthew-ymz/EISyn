"""Complete the screenshot's SURD maximum-weight-tree comparison: 320 records.

All 511 subset specific KLs / 502 synergy atoms, 16 physical architectures,
20 common labelings each. Every labeling is allocated afresh, including ties.
Known-joint density and target conventions match compare_spt_surd_three_cases.
Primary: 16384 Sobol targets, seed 20261015; checks: its first 4096 and an
independent 16384-target scramble. Cache all integrated R/U/S/MI tables in NPZ.

Authority freshly read 2026-10-15: P6UJCVG8, DXGC7JEA Methods (5)-(12),
MWIWKSVG S1.2/S5/S9.3; no explicit draft date/version. SURD SI KNU3AJTZ,
S1.2 (S12)-(S16), pp3-5. This is joint-target interventional SURD plus our
fixed maximum-clade-weight tree, not the manuscript's observational baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.compare_spt_surd_three_cases import (
    N, SOURCES, TOLERANCE_NATS, gaussian_specific, maximum_weight_tree,
    phase_specific, source_group,
)
from scripts.spt_continuous_phase_benchmark import InformationOracle, PhaseKernel, score
from scripts.spt_heterogeneous_hierarchy_case import quality
from scripts.spt_nested_broadcast_case import NestedGaussianBroadcast
from scripts.surd_batch_allocation import allocate_batch

OUTPUT = ROOT/'results/spt_surd_full_benchmark'


def load_cases():
    a = json.loads((ROOT/'results/spt_nested_broadcast_case/summary.json').read_text())
    bc = json.loads((ROOT/'results/spt_continuous_phase_benchmark/summary.json').read_text())
    schedules = bc['protocol']['permutations']
    if len(schedules)!=20 or [r['permutation'] for r in a['records']]!=schedules:
        raise ValueError('Expected the same twenty registered labelings for all methods')
    records = a['records']+[r for r in bc['records'] if r['case'] in 'BC']
    groups = defaultdict(list)
    for r in records:
        audit = r['audit']
        if (audit['significant_violation_count'] or audit['clipping']
            or audit['minimum_syn_nats'] < -audit['tolerance_nats']
            or abs(audit['closure_error_nats'])>audit['tolerance_nats']):
            raise ArithmeticError(f'Invalid source PEID record: {r["case"]}/{r["variant"]}')
        groups[r['case'],r['variant']].append(r)
    if len(records)!=320 or {c:sum(r['case']==c for r in records) for c in 'ABC'}!={'A':20,'B':240,'C':60}:
        raise ValueError('Incomplete screenshot protocol')
    for rows in groups.values():
        rows.sort(key=lambda r:r['permutation_id'])
        if [r['permutation_id'] for r in rows]!=list(range(20)):
            raise ValueError('Incomplete labeling group')
    return groups,schedules,bc['protocol']['grid']


def permutation_columns(permutation):
    """Columns in physical coordinates for each coalition of the new labels."""
    if sorted(permutation)!=list(SOURCES):
        raise ValueError('Expected a source-label bijection')
    return np.array([sum(1 << (original-1) for label,original in enumerate(permutation)
                         if mask >> label & 1) for mask in range(1 << N)])


def native_quality(synergy,supports):
    primitive = tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
    detected = [source_group(m) for m in range(1,(1 << N)-1)
                if m.bit_count()>=2 and synergy[m]>TOLERANCE_NATS]
    result = quality((),detected,supports,primitive)
    return {k:v for k,v in result.items() if k not in ('clades','root_split','correct_root')}


def input_signature(power,seed,grid,batch):
    digest = hashlib.sha256()
    paths = ('scripts/compare_spt_surd_three_cases.py','scripts/surd_batch_allocation.py',
             'scripts/spt_continuous_phase_benchmark.py','scripts/spt_nested_broadcast_case.py',
             'scripts/unicm_observational_information.py',
             'results/spt_continuous_phase_benchmark/summary.json',
             'results/spt_nested_broadcast_case/summary.json')
    for path in paths:
        digest.update(path.encode())
        digest.update((ROOT/path).read_bytes())
    digest.update(json.dumps({'power':power,'seed':seed,'grid':grid,'batch':batch,
                             'allocation':'SURD S13-S16, stable mask ties','schema':1},sort_keys=True).encode())
    return digest.hexdigest()


def validate_integrated(arrays):
    """Audit cached means as well as the statewise allocation diagnostics."""
    for name in ('redundancy_unique','synergy','mi'):
        values = arrays[name]
        if values.shape!=(3,20,512) or not np.isfinite(values).all():
            raise ArithmeticError(f'Invalid integrated {name} shape or values')
        count = int(np.count_nonzero(values < -TOLERANCE_NATS))
        if count:
            raise ArithmeticError(f'{name}: minimum={values.min()} nats; '
                                  f'threshold={-TOLERANCE_NATS}; affected_count={count}')
    closure = np.abs(arrays['synergy'].sum(axis=2)+arrays['redundancy_unique'].sum(axis=2)
                     -arrays['mi'][:,:,-1])
    if closure.max()>TOLERANCE_NATS or arrays['maximum_statewise_closure_error_nats'].max()>TOLERANCE_NATS:
        raise ArithmeticError(f'Integrated SURD closure failed: {closure.max()} nats')
    return {'minimum_synergy_nats':float(arrays['synergy'].min()),
            'synergy_tolerance_negative_count':int(np.count_nonzero(arrays['synergy']<0)),
            'minimum_redundancy_unique_nats':float(arrays['redundancy_unique'].min()),
            'maximum_integrated_closure_error_nats':float(closure.max()),
            'significant_violation_count':0}


def integrate_group(case,supports,schedules,power,seed,grid,batch):
    count = 1 << power
    low_count = 1 << (power-2)
    dimensions = 7 if case=='A' else len(supports)
    model = NestedGaussianBroadcast() if case=='A' else None
    shape = (3,len(schedules),1 << N)
    arrays = {name:np.zeros(shape) for name in ('redundancy_unique','synergy','mi')}
    closure = np.zeros((3,len(schedules)))
    minimum = np.zeros((3,len(schedules)))
    negative = np.zeros((3,len(schedules)),dtype=np.int64)
    for scramble_index,scramble in enumerate((seed,seed+1)):
        uniforms = qmc.Sobol(dimensions,scramble=True,seed=scramble).random_base2(power)
        physical = (gaussian_specific(model.weights,ndtri(uniforms)) if case=='A'
                    else phase_specific(supports,uniforms,grid)[0])
        for permutation_id,permutation in enumerate(schedules):
            columns = permutation_columns(permutation)
            red = np.zeros(1 << N); syn = np.zeros(1 << N); mi = np.zeros(1 << N)
            max_closure,min_specific,negative_count = 0.,0.,0
            for start in range(0,count,batch):
                end = min(start+batch,count)
                result = allocate_batch(physical[start:end][:,columns],TOLERANCE_NATS)
                red += result['redundancy_sum']; syn += result['synergy_sum']; mi += result['mi_sum']
                audit = result['audit']
                max_closure = max(max_closure,audit['maximum_statewise_closure_error_nats'])
                min_specific = min(min_specific,audit['minimum_specific_nats'])
                negative_count += audit['specific_tolerance_negative_count']
                setting = 0 if scramble_index==0 and end==low_count else (
                    1 if scramble_index==0 and end==count else 2 if scramble_index==1 and end==count else None)
                if setting is not None:
                    arrays['redundancy_unique'][setting,permutation_id] = red/end
                    arrays['synergy'][setting,permutation_id] = syn/end
                    arrays['mi'][setting,permutation_id] = mi/end
                    closure[setting,permutation_id] = max_closure
                    minimum[setting,permutation_id] = min_specific
                    negative[setting,permutation_id] = negative_count
    arrays.update(maximum_statewise_closure_error_nats=closure,
                  minimum_specific_nats=minimum,specific_tolerance_negative_count=negative)
    return arrays


def run(output=OUTPUT,power=14,seed=20261015,batch=2048):
    start = time.monotonic()
    if power<3 or batch<1 or (1 << (power-2))%batch:
        raise ValueError('Batch size must divide the low-resolution checkpoint')
    groups,schedules,grid = load_cases()
    output.mkdir(parents=True,exist_ok=True)
    signature = input_signature(power,seed,grid,batch)
    settings = ({'samples':1 << (power-2),'seed':seed},
                {'samples':1 << power,'seed':seed},
                {'samples':1 << power,'seed':seed+1})
    records = []
    for (case,variant),rows in sorted(groups.items()):
        group_start = time.monotonic()
        supports = rows[0]['hyperedges']
        cache = output/f'{case}_variant{variant:02d}.npz'
        arrays = None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as previous:
                if str(previous['signature'])==signature:
                    arrays = {k:previous[k] for k in previous.files if k!='signature'}
        reused = arrays is not None
        if arrays is None:
            arrays = integrate_group(case,supports,schedules,power,seed,grid,batch)
            temporary = cache.with_name(cache.stem+'.tmp.npz')
            np.savez_compressed(temporary,signature=np.array(signature),**arrays)
            temporary.replace(cache)
        validate_integrated(arrays)
        oracle = NestedGaussianBroadcast() if case=='A' else InformationOracle(PhaseKernel(supports),grid)
        exact = np.array([0.,*[oracle.info(source_group(m)) for m in range(1,1 << N)]])
        group_records = []
        for permutation_id,original in enumerate(rows):
            permutation = schedules[permutation_id]
            to_label = {v:i+1 for i,v in enumerate(permutation)}
            expected = {tuple(sorted(to_label[i] for i in e)) for e in supports}
            if expected!={tuple(e) for e in original['hyperedges']}:
                raise ValueError('Source permutation/architecture mismatch')
            checks = []
            methods = None
            for setting,configuration in enumerate(settings):
                syn = arrays['synergy'][setting,permutation_id]
                raw_tree = maximum_weight_tree(syn)
                tree = {**raw_tree,**score(raw_tree,original['hyperedges'])}
                checks.append({**configuration,'clade_f1':tree['clade_f1'],
                    'correct_root':tree['correct_root'],'clades':tree['clades'],
                    'root_split':tree['root_split'],
                    'maximum_subset_mi_integration_error_nats':float(np.max(abs(
                        arrays['mi'][setting,permutation_id]-exact[permutation_columns(permutation)]))),
                    'maximum_statewise_closure_error_nats':float(arrays['maximum_statewise_closure_error_nats'][setting,permutation_id])})
                if setting==1:
                    methods = {'spt':original['methods']['spt'],'surd_maximum_weight':tree}
            native = native_quality(arrays['synergy'][1,permutation_id],original['hyperedges'])
            audit = {'minimum_specific_nats':float(arrays['minimum_specific_nats'].min()),
                     'specific_tolerance_negative_count':int(arrays['specific_tolerance_negative_count'][1,permutation_id]),
                     'specific_significant_violation_count':0,
                     'maximum_statewise_closure_error_nats':float(arrays['maximum_statewise_closure_error_nats'].max())}
            record = {'case':case,'variant':variant,'permutation_id':permutation_id,
                      'permutation':permutation,'hyperedges':original['hyperedges'],
                      'methods':methods,'native_supports':native,'integration_checks':checks,
                      'surd_audit':audit,'peid_audit':original['audit'],
                      'atom_cache':cache.name,'cache_permutation_index':permutation_id,
                      'total_mi_nats':float(exact[-1])}
            group_records.append(record)
        records.extend(group_records)
        print(json.dumps({'case':case,'variant':variant,'records':len(group_records),'cache_reused':reused,
            'mean_f1':float(np.mean([r['methods']['surd_maximum_weight']['clade_f1'] for r in group_records])),
            'seconds':time.monotonic()-group_start}),flush=True)
    counts = {c:sum(r['case']==c for r in records) for c in 'ABC'}
    if counts!={'A':20,'B':240,'C':60}:
        raise ValueError(f'Incomplete final result: {counts}')
    summaries = {}
    for case in 'ABC':
        rows = [r for r in records if r['case']==case]
        f1 = np.array([r['methods']['surd_maximum_weight']['clade_f1'] for r in rows])
        summaries[case] = {'records':len(rows),'mean_f1':float(f1.mean()),
            'minimum_f1':float(f1.min()),'maximum_f1':float(f1.max()),
            'correct_root_rate':float(np.mean([r['methods']['surd_maximum_weight']['correct_root'] for r in rows])),
            'integration_mean_f1':[float(np.mean([r['integration_checks'][s]['clade_f1'] for r in rows])) for s in range(3)],
            'f1_changed_low_to_primary':sum(r['integration_checks'][0]['clade_f1']!=r['integration_checks'][1]['clade_f1'] for r in rows),
            'f1_changed_between_scrambles':sum(r['integration_checks'][1]['clade_f1']!=r['integration_checks'][2]['clade_f1'] for r in rows)}
    result = {'status':'complete','protocol':{'scope':'all screenshot labelings and variants',
        'records_by_case':counts,'physical_architectures':len(groups),'permutations':schedules,
        'source':'all nine original node blocks','target':'complete one-step future state, jointly',
        'distributions':'A iid N(0,1)^126; B/C iid Uniform(0,2*pi)^81; unchanged',
        'units':'nats','clipping':False,'syn_tolerance_nats':TOLERANCE_NATS,
        'all_nonempty_source_subsets':511,'synergy_atoms':502,
        'target_integration':{'method':'scrambled Sobol','primary_samples':1 << power,
                              'primary_setting_index':1,'settings':settings,'periodic_grid':grid},
        'allocation':'SURD at each target state, then expectation; fresh allocation for each source permutation',
        'allocation_ties':'ascending information, order, source mask; stable orderwise batched implementation',
        'tree':'global maximum sum of the SURD atom weight at each selected internal clade; exact subset DP',
        'tree_ties':'source tuple lexicographic; no ground-truth input or tuning',
        'score':'same nonroot support F1, leaves/root and proper refinements inside primitive supports excluded',
        'replication':'label/structure sensitivity and quadrature randomizations, not independent physical repetitions',
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','si':'MWIWKSVG','checked':'2026-10-15',
                      'version':'sole current main/SI; no explicit manuscript date/version'},
        'surd_reference':{'parent':'K69ZCNZG','si':'KNU3AJTZ','locations':'S1.2 S12-S16 pp3-5'},
        'input_signature':signature,'cache_format':'NPZ: three settings x twenty labelings x 512 source masks',
        'new_training':False},'elapsed_seconds':time.monotonic()-start,
        'summary':summaries,'records':records}
    temporary = output/'summary.tmp.json'
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False))
    temporary.replace(output/'summary.json')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--power',type=int,default=14)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--batch',type=int,default=2048)
    args = parser.parse_args()
    result = run(args.output,args.power,args.seed,args.batch)
    print(json.dumps({'summary':result['summary'],'elapsed_seconds':result['elapsed_seconds']}))
