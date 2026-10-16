"""Reuse frozen candidate data/models/trees; evaluate fixed clustering rules only.

No simulation, training, EI estimation, seed selection or historical cache write.
The Hénon root node-pair F1 is not the nine-node hierarchy's module-clade F1.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_hierarchical_clustering_reduced import song_auv2
from scripts.spt_hierarchical_clustering_screen import ksg, save
from scripts.henon_fixed_lambda_module_recovery import partition_f1, MODULES
from scripts.henon_high_order_comparison import sha

OUT = ROOT / 'results/spt_hierarchical_clustering_screen/existing_candidate_audit.json'


def root_score(z, truth):
    root = to_tree(z)
    parts = [sorted(i+1 for i in root.left.pre_order()), sorted(i+1 for i in root.right.pre_order())]
    return partition_f1(parts, truth)


def fit(x, method):
    if method == 'song2012_auv2':
        d, audit = song_auv2(x)
        return linkage(squareform(d), method='average'), audit
    return linkage(pdist(x.T), method=method), {}


def main():
    started = time.monotonic()
    result = dict(scope='Read-only paired baseline evaluation on BOTH frozen existing Henon-style cases; no new dynamics, fitting or EI',
                  case_selection='Both cases from existing fixed-lambda run; no new seed or model selection',
                  metric='Existing K=2 root same-module node-pair F1, distinct from nine-node hierarchy F1', records=[])
    for case in 'AB':
        folder = ROOT / 'results/henon_fixed_lambda_module_recovery' / f'{case}_lambda0.05'
        read = lambda name: json.loads((folder/name).read_text())
        manifest, sf, ct, model = [read(n) for n in ('data_manifest.json', 'module_f1.json', 'conditional_tm.json', 'mlp.json')]
        with np.load(folder/'natural.npz') as pool:
            states, latent, order = (pool[k] for k in ('states', 'latent', 'chain_order'))
        x, y = (a[:, order[:48]].reshape(-1, 6) for a in (states, latent))
        if sha(x, y) != manifest['partitions']['train']['sha256'] or model['training_pairs_sha256'] != sha(x, y):
            raise ValueError('Clustering and frozen MLP do not share the same training data')
        encoded = json.dumps(model, sort_keys=True).encode()
        if hashlib.sha256(encoded).hexdigest() != sf['model_sha256'] or sf['model_sha256'] != ct['model_sha256']:
            raise ValueError('SPT is not associated with this frozen MLP')
        if sf['status'] != 'complete' or ct['status'] != 'complete':
            raise ValueError('Completed native SPT output required for read-only reuse')
        root = [child['members'] for child in ct['spt']['tree']['children']]
        if partition_f1(root, MODULES[case])['f1'] != sf['f1']:
            raise ValueError('Native SPT tree and stored score disagree')
        row = dict(case=case, model_sha256=sf['model_sha256'], training_pairs_sha256=sha(x, y),
                   training_pairs=len(x), true_modules=MODULES[case], spt=sf,
                   spt_native_sha256=hashlib.sha256((folder/'conditional_tm.json').read_bytes()).hexdigest(),
                   model_diagnostics=read('metrics.json')['diagnostics'], methods={})
        for method in ('ward', 'average', 'song2012_auv2'):
            z, audit = fit(x, method)
            subsets = [root_score(fit(states[:, order[i*12:(i+1)*12]].reshape(-1, 6), method)[0], MODULES[case]) for i in range(4)]
            row['methods'][method] = dict(full_train=root_score(z, MODULES[case]), audit=audit, chain_subset_sensitivity=subsets)
        # Estimator sensitivity only: this previously disclosed adaptation is
        # not relabeled as a fourth published baseline or Song reproduction.
        sim = np.zeros((6, 6))
        for i in range(6):
            for j in range(i):
                sim[i, j] = sim[j, i] = ksg(x[:, i], x[:, j], False, k=5)
        d = sim[np.triu_indices(6, 1)].max()-sim
        np.fill_diagonal(d, 0)
        sensitivity = root_score(linkage(squareform(d), method='average'), MODULES[case])
        row['estimator_diagnostic_not_formal_baseline'] = dict(
            description='Existing KSG/copula C-MI adaptation, full training data, k5; sensitivity diagnostic, not new published baseline',
            score=sensitivity, raw_mi_nats=sim.tolist())
        result['records'].append(row)
        print(case, 'SPT', sf['f1'], {m: v['full_train']['f1'] for m, v in row['methods'].items()}, 'KSG sensitivity', sensitivity['f1'])
    existing = ROOT / 'results/spt_hierarchical_clustering_screen/paper_summary.json'
    source = json.loads(existing.read_text())['records'][1]['batches'][0]['methods']['song2012_auv2']
    a = np.array(source['audit']['adjacency'])
    mean = lambda l, r: float(a[np.ix_(np.array(l)-1, np.array(r)-1)].mean())
    wrong, correct = mean([1,2,3,4,5], [6,7]), mean([6,7], [8,9])
    result['ho_lv_wrong_merge'] = dict(
        source_summary_sha256=hashlib.sha256(existing.read_bytes()).hexdigest(),
        wrong_mean_normalized_mi=wrong, correct_mean_normalized_mi=correct,
        difference_dimensionless=wrong-correct,
        interpretation='Observed UPGMA merge preference; not a causal attribution to dynamics or a statistical significance result')
    result['elapsed_seconds'] = time.monotonic()-started
    save(OUT, result)
    print(OUT)


if __name__ == '__main__':
    main()
