"""Lightweight, natural-data node hierarchy screen; no new MLP or EI estimates.

Same 48-chain training pairs as the natural-MLP contract, fixed before scoring.
Display Network B is native simplified C. No ground truth enters clustering.
Run with PYTHONPATH=/tmp/eisyn_hc_deps for optional official Genie/HDBSCAN wheels.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, is_valid_linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma
from scipy.stats import rankdata
from sklearn.cluster import BisectingKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_natural_mlp import chain_partitions
from scripts.spt_kuramoto_finite_tm import digest
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS, wrap

OUT = ROOT / 'results/spt_hierarchical_clustering_screen'
REPORT = ROOT / 'docs/reports/spt_hierarchical_clustering_screen.md'
SEED = 20261016
LINKAGES = ('single', 'complete', 'average', 'weighted', 'ward', 'centroid', 'median')
ARBITRARY_LINKAGES = LINKAGES[:4]
POOLS = (
    ('Kuramoto A', 'A', True, 'results/spt_local_kuramoto_tm/ac_comparison/A/natural_chains.npz'),
    ('HO-LV A', 'A', False, 'results/spt_matched_three_case/full_comparison/glv_natural_chains.npz'),
    ('Kuramoto B', 'C', True, 'results/spt_local_kuramoto_tm/ac_comparison/C/natural_chains.npz'),
)
SOURCES = {
    'SciPy linkage': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html',
    'DIANA': 'https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/diana.html',
    'Bisecting k-means': 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html',
    'Normalized cuts': 'https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf',
    'Genie': 'https://genieclust.gagolewski.com/pythonapi_root.html',
    'HDBSCAN hierarchy': 'https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html',
    'KSG MI': 'https://doi.org/10.1103/PhysRevE.69.066138',
    'SHiP (NeurIPS 2025)': 'https://github.com/pasiweber/SHiP-framework/',
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def configs():
    methods = []
    for rep in ('state', 'increment'):
        methods += [dict(id=f'{rep}_{m}', representation=rep, algorithm=m,
                         label=f'{m.title()} | {rep}') for m in LINKAGES]
        methods += [dict(id=f'{rep}_{m}', representation=rep, algorithm=m,
                         label=f'{label} | {rep}') for m, label in
                    [('diana', 'DIANA'), ('bisect', 'Bisecting k-means'), ('ncut', 'Recursive Ncut')]]
        methods += [dict(id=f'{rep}_genie_{g}', representation=rep, algorithm='genie',
                         gini=g, label=f'Genie g={g} | {rep}') for g in (.1, .3, .5)]
        methods += [dict(id=f'{rep}_hdb_{k}', representation=rep, algorithm='hdbscan',
                         min_samples=k, label=f'HDBSCAN m={k} | {rep}') for k in (2, 3, 5)]
    for rep, label in [('absolute', 'PLV / abs Pearson'), ('kernel', 'RBF CKA'),
                       ('mi', 'KSG MI')]:
        methods += [dict(id=f'{rep}_{m}', representation=rep, algorithm=m,
                         label=f'{m.title()} | {label}') for m in ARBITRARY_LINKAGES]
    methods += [dict(id=f'{rep}_{m}', representation=rep, algorithm=m,
                     label=f'{name} | {label}') for rep, label, m, name in
                [('kernel', 'RBF CKA', 'diana', 'DIANA'),
                 ('kernel', 'RBF CKA', 'ncut', 'Recursive Ncut'),
                 ('mi', 'KSG MI', 'ncut', 'Recursive Ncut'),
                 ('lag_mi', 'lagged KSG MI', 'average', 'Average'),
                 ('lag_mi', 'lagged KSG MI', 'ncut', 'Recursive Ncut')]]
    return methods


def hierarchy_from_linkage(z):
    z = np.asarray(z, dtype=float)
    is_valid_linkage(z, throw=True)
    if z.shape != (8, 4):
        raise ValueError('Complete nine-leaf hierarchy required')
    tree = to_tree(z)
    clades = []
    def visit(node):
        if node.is_leaf():
            return [int(node.id) + 1]
        group = sorted(visit(node.left) + visit(node.right))
        if len(group) < 9:
            clades.append(group)
        return group
    visit(tree)
    root = [sorted(i + 1 for i in tree.left.pre_order()), sorted(i + 1 for i in tree.right.pre_order())]
    return dict(clades=clades, root_split=sorted(root), linkage=z.tolist())


def divisive(split):
    clades = []
    def visit(group):
        if len(group) == 1:
            return
        left, right = (tuple(sorted(g)) for g in split(group))
        if not left or not right or set(left) & set(right) or set(left) | set(right) != set(group):
            raise ValueError('Invalid divisive split')
        for child in (left, right):
            if len(child) > 1:
                clades.append([i + 1 for i in child])
            visit(child)
        return left, right
    root = visit(tuple(range(9)))
    return dict(clades=clades, root_split=sorted([[i + 1 for i in g] for g in root]))


def diana(distance):
    # Kaufman/Rousseeuw splinter rule; deterministic source-index tie resolution.
    def split(group):
        if len(group) == 2:
            return group[:1], group[1:]
        rest = list(group)
        first = max(rest, key=lambda i: (float(distance[i, rest].sum() / (len(rest) - 1)), -i))
        new = [first]
        rest.remove(first)
        while len(rest) > 1:
            gain, chosen = max((float(distance[i, [j for j in rest if j != i]].mean()
                                     - distance[i, new].mean()), -i) for i in rest)
            if gain <= 0:
                break
            chosen = -chosen
            rest.remove(chosen)
            new.append(chosen)
        return tuple(rest), tuple(new)
    return divisive(split)


def spectral(distance, affinity=None):
    if affinity is None:
        scale = np.median(distance[np.triu_indices(9, 1)])
        affinity = np.exp(-distance ** 2 / (2 * scale ** 2))
    affinity = affinity.copy()
    np.fill_diagonal(affinity, 0.)
    def split(group):
        if len(group) == 2:
            return group[:1], group[1:]
        w = affinity[np.ix_(group, group)]
        degree = w.sum(1)
        if np.any(degree <= 0):
            raise ArithmeticError('Disconnected zero-degree spectral input')
        inv = 1 / np.sqrt(degree)
        lap = np.eye(len(group)) - inv[:, None] * w * inv[None, :]
        eig, vec = np.linalg.eigh(lap)
        order = np.argsort(vec[:, 1] * inv, kind='stable')
        candidates = []
        for k in range(1, len(group)):
            l, r = order[:k], order[k:]
            cut = w[np.ix_(l, r)].sum()
            objective = float(cut * (1 / degree[l].sum() + 1 / degree[r].sum()))
            children = sorted([tuple(sorted(group[i] for i in l)), tuple(sorted(group[i] for i in r))])
            candidates.append((objective, children))
        return min(candidates)[1]
    return divisive(split)


def euclidean_profile(data, circular=False):
    blocks = np.stack([np.sin(data), np.cos(data)], axis=2) if circular else data[:, :, None]
    blocks = blocks - blocks.mean(axis=0, keepdims=True)
    vectors = blocks.transpose(1, 0, 2).reshape(9, -1)
    norm = np.linalg.norm(vectors, axis=1)
    if np.any(norm == 0):
        raise ValueError('Constant node profile')
    vectors /= norm[:, None]
    distance = squareform(pdist(vectors))
    # Exact low-dimensional isometry for centroid, Ward and bisecting k-means.
    centered = vectors - vectors.mean(0, keepdims=True)
    gram = centered @ centered.T
    values, basis = np.linalg.eigh(gram)
    if values.min() < -1e-10:
        raise ArithmeticError('Euclidean Gram matrix is not positive semidefinite')
    coords = basis[:, values > 1e-12] * np.sqrt(values[values > 1e-12])
    if np.max(np.abs(squareform(pdist(coords)) - distance)) > 1e-9:
        raise ArithmeticError('Profile dimension reduction changed distances')
    return distance, coords


def kernel_affinity(x, circular):
    kernels = []
    bandwidth = []
    for i in range(9):
        v = np.column_stack((np.sin(x[:, i]), np.cos(x[:, i]))) if circular else x[:, i, None]
        sq = squareform(pdist(v, metric='sqeuclidean'))
        scale = float(np.median(sq[np.triu_indices(len(v), 1)]))
        if scale <= 0:
            raise ArithmeticError('Zero kernel bandwidth')
        k = np.exp(-sq / (2 * scale))
        k = k - k.mean(0)[None, :] - k.mean(1)[:, None] + k.mean()
        norm = np.linalg.norm(k)
        kernels.append((k / norm).ravel())
        bandwidth.append(scale)
    z = np.asarray(kernels)
    similarity = z @ z.T
    return similarity, bandwidth


def ksg(a, b, circular, k=5):
    n = len(a)
    if circular:
        a, b = (np.mod(v + np.pi, 2 * np.pi) / (2 * np.pi) for v in (a, b))
        period = 1.
    else:
        a, b = ((rankdata(v, method='ordinal') - .5) / n for v in (a, b))
        period = None
    points = np.column_stack((a, b))
    eps = cKDTree(points, boxsize=period).query(points, k=k + 1, p=np.inf)[0][:, -1]
    eps = np.nextafter(eps, 0)
    counts = [cKDTree(v[:, None], boxsize=period).query_ball_point(v[:, None], eps, p=np.inf,
              return_length=True) for v in (a, b)]
    if min(c.min() for c in counts) < 1:
        raise ArithmeticError('KSG self-neighbor count is missing')
    return float(digamma(k) + digamma(n) - np.mean(digamma(counts[0]) + digamma(counts[1])))


def representations(x, y, circular, seed):
    delta = wrap(y - x) if circular else y - x
    out = {}
    for name, data in [('state', x), ('increment', delta)]:
        d, coords = euclidean_profile(data, circular and name == 'state')
        out[name] = dict(distance=d, coords=coords)
    if circular:
        z = np.exp(1j * x)
        absolute = abs(z.T @ z.conj() / len(x))
    else:
        absolute = abs(np.corrcoef(x, rowvar=False))
    # These are geometric roundoff bounds, unrelated to EI/Syn tolerances.
    if absolute.max() > 1 + 1e-10:
        raise ArithmeticError('Invalid correlation/PLV magnitude')
    out['absolute'] = dict(distance=np.sqrt(2 * np.maximum(0., 1 - absolute)), affinity=absolute)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(x), min(2048, len(x)), replace=False)
    rows = np.sort(chosen)
    kernel_rows = np.sort(chosen[:1024])  # predeclared uniform subset, not a time prefix
    sim, bandwidth = kernel_affinity(x[kernel_rows], circular)
    if sim.max() > 1 + 1e-10:
        raise ArithmeticError('Invalid normalized kernel similarity')
    out['kernel'] = dict(distance=np.sqrt(2 * np.maximum(0., 1 - sim)), affinity=sim)
    mi, lag = np.zeros((9, 9)), np.zeros((9, 9))
    for i in range(9):
        for j in range(i):
            mi[i, j] = mi[j, i] = ksg(x[rows, i], x[rows, j], circular)
            lag[i, j] = lag[j, i] = .5 * (ksg(x[rows, i], y[rows, j], circular)
                                               + ksg(x[rows, j], y[rows, i], circular))
    for name, affinity in [('mi', mi), ('lag_mi', lag)]:
        vals = affinity[np.triu_indices(9, 1)]
        d = vals.max() - affinity
        scale = max(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 1e-6)
        graph = np.exp((affinity - vals.max()) / scale)
        out[name] = dict(distance=d, affinity=graph)
    asymmetry = 0.
    for rep in out.values():
        error = float(abs(rep['distance'] - rep['distance'].T).max())
        asymmetry = max(asymmetry, error)
        if error > 1e-10:
            raise ArithmeticError('Asymmetry exceeds floating point roundoff')
        rep['distance'] = .5 * (rep['distance'] + rep['distance'].T)
        np.fill_diagonal(rep['distance'], 0.)
        if np.min(rep['distance']) < 0 or not np.isfinite(rep['distance']).all():
            raise ArithmeticError('Invalid dissimilarity')
    audit = dict(mi_rows=rows.tolist(), kernel_rows=kernel_rows.tolist(), kernel_bandwidth_squared=bandwidth,
                 ksg_neighbors=5, ksg_negative_estimates=int((mi[np.triu_indices(9, 1)] < 0).sum()),
                 ksg_minimum_nats=float(mi[np.triu_indices(9, 1)].min()),
                 lag_ksg_negative_estimates=int((lag[np.triu_indices(9, 1)] < 0).sum()),
                 ksg_clipping=False, maximum_distance_roundoff_asymmetry=asymmetry)
    return out, audit


def fit_tree(config, representation):
    d = representation['distance']
    alg = config['algorithm']
    if alg in LINKAGES:
        return hierarchy_from_linkage(linkage(squareform(d, checks=True), method=alg))
    if alg == 'diana':
        return diana(d)
    if alg == 'ncut':
        return spectral(d, representation.get('affinity'))
    if alg == 'bisect':
        model = BisectingKMeans(n_clusters=9, init='k-means++', n_init=20, random_state=SEED,
                                max_iter=300).fit(representation['coords'])
        # Official sklearn's tree retains original row indices at singleton leaves.
        clades = []
        def leaf(node):
            # sklearn releases leaf.indices after fit; labels_ retains membership.
            indices = np.flatnonzero(model.labels_ == node.label)
            if len(indices) != 1:
                raise ValueError('Bisecting k-means did not reach singleton leaves')
            return [int(indices[0]) + 1]
        def visit(node):
            if node.left is None:
                return leaf(node)
            left, right = visit(node.left), visit(node.right)
            group = sorted(left + right)
            if len(group) < 9:
                clades.append(group)
            return group
        root = model._bisecting_tree
        visit(root)
        def leaves(node):
            if node.left is None:
                return leaf(node)
            return leaves(node.left) + leaves(node.right)
        return dict(clades=clades, root_split=sorted([sorted(leaves(root.left)), sorted(leaves(root.right))]))
    if alg == 'genie':
        import genieclust
        # Precomputed exact Euclidean distances avoid changing numpy in the project.
        model = genieclust.Genie(n_clusters=1, gini_threshold=config['gini'], metric='precomputed').fit(d)
        z = np.column_stack((model.children_, model.distances_, model.counts_))
        return hierarchy_from_linkage(z)
    if alg == 'hdbscan':
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=config['min_samples'],
                               metric='precomputed', algorithm='generic', approx_min_span_tree=False).fit(d)
        return hierarchy_from_linkage(model.single_linkage_tree_.to_numpy())
    raise ValueError(alg)


def score(tree, case):
    clades = {frozenset(c) for c in tree['clades']}
    root = frozenset(range(1, 10))
    if len(clades) != 7 or any(not 1 < len(c) < 9 for c in clades):
        raise ValueError('Missing binary tree clades')
    all_groups = clades | {root} | {frozenset([i]) for i in root}
    for a in all_groups:
        for b in all_groups:
            if a & b and not (a <= b or b <= a):
                raise ValueError('Non-laminar hierarchy')
    truth = {frozenset(c) for c in SUPPORTS[case]}
    primitive = {c for c in truth if not any(g < c for g in truth)}
    scored = {c for c in clades if not any(c < p for p in primitive)}
    truth.remove(root)
    hit = len(scored & truth)
    expected_root = {c for c in truth if not any(c < g for g in truth)}
    upper = truth - primitive
    return dict(**tree, module_f1=2 * hit / (len(scored) + len(truth)),
                clade_f1=2 * hit / (len(scored) + len(truth)),
                strict_clade_f1=2 * len(clades & truth) / (len(clades) + len(truth)),
                precision=hit / len(scored) if scored else 0., recall=hit / len(truth),
                correct_root={frozenset(c) for c in tree['root_split']} == expected_root,
                upper_recall=len(scored & upper) / len(upper),
                tp=hit, fp=len(scored - truth), fn=len(truth - scored),
                scored_clades=[sorted(c) for c in sorted(scored, key=lambda c: (len(c), sorted(c)))])


def protocol():
    return dict(version=1, date='2026-10-16 Asia/Shanghai', methods=configs(),
                cases=[dict(name=n, native_case=c, circular=p, pool=f) for n, c, p, f in POOLS],
                input='same natural source/future training pairs; 48 fixed training chains only',
                split_seed=SEED, split_sizes=[48, 8, 8], chain_fold_count=4,
                chain_fold_rule='contiguous quarters of the registered training-chain order; 12 chains each',
                representation='cluster nodes, not time points; centered unit-norm node trajectories; state or increment',
                circular='state sin/cos embedding; wrapped increments; PLV; toroidal KSG; circular chordal RBF',
                real='stored u=pi*log(abundance); centered unit-norm profiles; abs Pearson; copula KSG',
                nonlinear='RBF CKA median bandwidth, 1024 rows; KSG k=5, 2048 rows; all fixed before F1',
                ksg_row_rule='uniform 2048 without replacement; first 1024 UNSORTED selected rows for kernel, then sort each set',
                hdbscan='complete mutual-reachability single-linkage hierarchy before condensation/flat noise filtering',
                sources=SOURCES, hyperparameters_selected_by_truth=False, new_simulations=0, new_mlp_fits=0,
                new_ei_or_syn_estimates=0, spt_mlp_f1=None,
                score='pre-existing module reference; ignore primitive-internal strict refinements equally; also strict clade F1',
                inference='exploratory fixed-three-system screen; four chain subsets are sensitivity checks, not independent system repetitions',
                manuscript=dict(parent='P6UJCVG8', main='DXGC7JEA', supplement='MWIWKSVG',
                                checked='2026-10-16', version='sole main/SI, no explicit draft date',
                                locations='main Methods Eqs(10)-(12); SI S5.1 Eqs(S49)-(S50)'))


def main():
    started = time.monotonic()
    registered = protocol()
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != registered:
        raise ValueError('Registered screen changed; do not overwrite a scored protocol')
    save(path, registered)  # must precede all clustering and scoring
    result = dict(status='running', protocol=registered, records=[], optional_unavailable=[])
    for name in ('numpy', 'scipy', 'scikit-learn', 'genieclust', 'hdbscan', 'deadwood'):
        try:
            result.setdefault('software', {})[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    for display, native, circular, filename in POOLS:
        with np.load(ROOT / filename, allow_pickle=False) as data:
            states, meta = data['states'].copy(), json.loads(str(data['meta']))
        pairs, manifest = chain_partitions(states)
        if display == 'Kuramoto A':
            old = json.loads((ROOT / 'results/spt_natural_mlp_pilot/kuramoto_A/shared_data.json').read_text())
            if manifest['splits'] != old['splits']:
                raise ValueError('Kuramoto A data differ from the approved natural MLP')
        record = dict(case=display, native_case=native, pool=filename,
                      pool_sha256=hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
                      metadata=meta, shared_data=manifest, batches=[])
        training_chains = manifest['splits']['train']['chains']
        batches = [('full_train', training_chains)] + [(f'chain_fold_{i}', training_chains[i * 12:(i + 1) * 12]) for i in range(4)]
        for batch_index, (batch_name, chains) in enumerate(batches):
            x, y = (a.reshape(-1, 9) for a in (states[:-1, chains], states[1:, chains]))
            reps, audit = representations(x, y, circular, SEED + batch_index)
            batch = dict(name=batch_name, chains=chains, pairs=len(x), paired_arrays_sha256=digest(x, y),
                         representation_audit=audit, methods={})
            for config in registered['methods']:
                tick = time.monotonic()
                try:
                    tree = fit_tree(config, reps[config['representation']])
                    value = dict(status='complete', metrics=score(tree, native))
                except (ImportError, ModuleNotFoundError) as error:
                    value = dict(status='unavailable', error=f'{type(error).__name__}: {error}', metrics=None)
                except Exception as error:
                    value = dict(status='failed', error=f'{type(error).__name__}: {error}', metrics=None)
                value['elapsed_seconds'] = time.monotonic() - tick
                batch['methods'][config['id']] = value
            record['batches'].append(batch)
            print(display, batch_name, 'completed', sum(m['status'] == 'complete' for m in batch['methods'].values()),
                  '/', len(registered['methods']), flush=True)
        result['records'].append(record)
        save(OUT / 'summary.json', result)
    result.update(status='complete', elapsed_seconds=time.monotonic() - started)
    result['optional_unavailable'].append(dict(method='SHiP NeurIPS2025',
        reason='No compatible binary distribution; bounded source installation attempted separately; see installation audit. Not scored as zero.'))
    save(OUT / 'summary.json', result)
    print('screen seconds', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
