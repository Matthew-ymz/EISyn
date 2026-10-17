#!/usr/bin/env python3
"""Bounded SC-only community comparison against the displayed G=1.3 SPT.

No DMF simulation, EI query, density fitting or SPT reconstruction. Compare the
raw binary trees, not the display's contracted singleton runs. All choices below
are fixed independently of the amount of disagreement with SPT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import scipy
from scipy.cluster.hierarchy import cut_tree, linkage, to_tree
from scipy.spatial.distance import squareform
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_dmf_spt_peel_groups import PAIRED_SUMMARY, YEO_METADATA
from scripts.plot_dmf_schaefer100_tree_examples import audit_tree
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record

SC = ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
REPORT = ROOT/'docs/reports/brain_spt_sc_communities.md'
FIGURE = ROOT/'docs/reports/assets/dmf_spt_sc_communities.png'
N = 100
K_VALUES = list(range(2, 16))
LOUVAIN_SEEDS = list(range(10))
PERMUTATIONS = 499
NULL_SEED = 20261017
NAMES = ['Louvain', 'CNM', 'Spectral', 'Average linkage']
COLORS = ['#A64B64', '#257D98', '#CF8C23', '#7160A2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(groups):
    groups = sorted((tuple(sorted(map(int, x))) for x in groups), key=lambda x: (-len(x), x))
    flat = [i for g in groups for i in g]
    if sorted(flat) != list(range(N)):
        raise ValueError('A partition must contain all 100 ROI exactly once')
    return groups


def labels(groups):
    result = np.empty(N, int)
    for i, group in enumerate(partition(groups)):
        result[list(group)] = i
    return result


def from_labels(values):
    return partition([np.flatnonzero(values == k) for k in np.unique(values)])


def nodes(tree):
    yield tree
    for child in tree['children']:
        yield from nodes(child)


def clades(tree):
    return {tuple(sorted(n['indices'])) for n in nodes(tree) if 1 < len(n['indices']) < N}


def tree_cut(tree, k):
    """A deterministic frontier: split the largest available source set first.

    No canonical flat partition follows from SPT, so this explicit convention is
    only a coarse diagnostic, not an optimization or a grouping of peeled ROIs.
    """
    frontier = [tree]
    while len(frontier) < k:
        choices = [n for n in frontier if n['children']]
        chosen = min(choices, key=lambda n: (-len(n['indices']), tuple(n['indices'])))
        frontier = [n for n in frontier if n is not chosen] + chosen['children']
    return partition([n['indices'] for n in frontier])


def indicators(groups):
    result = np.zeros((len(groups), N), dtype=np.int16)
    for i, group in enumerate(groups):
        result[i, list(group)] = 1
    return result


def jaccards(left, right):
    intersection = left @ right.T
    union = left.sum(1)[:, None] + right.sum(1)[None] - intersection
    return intersection/union


def agreement(left, right):
    x, y = labels(left), labels(right)
    return dict(ARI=float(adjusted_rand_score(x, y)),
                AMI=float(adjusted_mutual_info_score(x, y, average_method='arithmetic')))


def permutation_orders():
    rng = np.random.default_rng(NULL_SEED)
    unrestricted = np.array([rng.permutation(N) for _ in range(PERMUTATIONS)])
    hemisphere = np.array([np.r_[rng.permutation(50), 50+rng.permutation(50)]
                           for _ in range(PERMUTATIONS)])
    return {'all ROI': unrestricted, 'within hemisphere': hemisphere}


def overlap(groups, reference_clades, orders):
    """Each SC community is matched to its best local SPT subtree (2–50 ROI).

    The same best-match search is repeated under every null permutation. All
    tree node sizes and community sizes are preserved. Hemisphere permutations
    additionally preserve each subtree's left/right composition.
    """
    candidates = sorted((x for x in reference_clades if 2 <= len(x) <= 50),
                        key=lambda x: (len(x), x))
    a, b = indicators(groups), indicators(candidates)
    matrix = jaccards(a, b)
    best = matrix.argmax(1)
    record = dict(mean_best_jaccard=float(matrix.max(1).mean()),
                  exact_match_count=int(np.count_nonzero(matrix.max(1) == 1)),
                  candidate_count=len(candidates), matches=[
                      dict(group=list(group), best_spt=list(candidates[int(j)]),
                           jaccard=float(matrix[i, j]), intersection=sorted(set(group)&set(candidates[int(j)])))
                      for i, (group, j) in enumerate(zip(groups, best))])
    nulls = {}
    for name, permutations in orders.items():
        values = np.array([jaccards(a, b[:, order]).max(1).mean() for order in permutations])
        nulls[name] = dict(mean=float(values.mean()), low=float(np.quantile(values, .025)),
                           high=float(np.quantile(values, .975)))
    record['null'] = nulls
    return record


def load_inputs():
    summary = json.loads(PAIRED_SUMMARY.read_text())
    if summary['config']['input_sha256'] != sha(YEO_METADATA):
        raise ValueError('SPT input identity changed')
    with np.load(SC) as a:
        w = a['connectivity'].copy()
    if (w.shape != (N, N) or not np.isfinite(w).all() or w.min() < 0
            or not np.array_equal(w, w.T) or np.any(np.diag(w))):
        raise ValueError('Expected the native symmetric nonnegative loop-free SC')
    graph = nx.from_numpy_array(w)
    if not nx.is_connected(graph):
        raise ValueError('Disconnected SC requires a declared baseline policy')
    with np.load(YEO_METADATA, allow_pickle=True) as a:
        region_labels = a['region_labels'].copy()
    if (not all(str(x).startswith('7Networks_LH_') for x in region_labels[:50])
            or not all(str(x).startswith('7Networks_RH_') for x in region_labels[50:])):
        raise ValueError('Hemisphere null requires the verified 50 LH / 50 RH metadata order')
    tree_records, audits, sources = {}, {}, {}
    for seed in [3, 4, 5]:
        row = next(r for r in summary['rows'] if r['coupling_g'] == 1.3 and r['seed'] == seed)
        path = Path(row['cache'])
        record = json.loads(path.read_text())
        density = next(p for p in record['config']['paired_covariance_inputs']
                       if p['simulation_config']['horizon'] == 300
                       and f'seed{seed:02d}_G1.30.npz' in p['path'])
        if (density['simulation_config']['source_sha256'] != sha(SC)
                or record['config']['input_sha256'] != sha(YEO_METADATA)
                or record['seed'] != seed or record['coupling_g'] != 1.3):
            raise ValueError('SPT and baseline connectivity/ROI/condition differ')
        tree_records[seed] = record['tree']
        native = _node_from_record(record['tree'])
        audits[seed] = audit_tree(native, native.xi_bits)
        if len(clades(record['tree'])) != N-2:
            raise ValueError('Expected 98 proper non-singleton clades in a binary tree')
        sources[str(path.relative_to(ROOT))] = sha(path)
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in [SC, YEO_METADATA, PAIRED_SUMMARY]})
    return w, graph, region_labels, tree_records, audits, sources


def run():
    started = time.monotonic()
    w, graph, region_labels, trees, audits, sources = load_inputs()
    # SC-only selection: choose greatest modularity, then smallest RNG seed.
    louvain_runs = []
    for seed in LOUVAIN_SEEDS:
        groups = partition(nx.community.louvain_communities(graph, weight='weight',
                            resolution=1., threshold=1e-7, seed=seed))
        louvain_runs.append(dict(seed=seed, groups=groups,
            Q=float(nx.community.modularity(graph, groups, weight='weight', resolution=1.))))
    selected = min(louvain_runs, key=lambda r: (-r['Q'], r['seed']))
    reference_k = len(selected['groups'])
    native_cnm = partition(nx.community.greedy_modularity_communities(graph, weight='weight', resolution=1.))
    distances = 1-w/w.max()
    np.fill_diagonal(distances, 0.)
    z = linkage(squareform(distances), method='average', optimal_ordering=True)
    avg_clades = set()
    def collect(node):
        if node.is_leaf():
            return (node.id,)
        members = tuple(sorted(collect(node.left)+collect(node.right)))
        if len(members) < N:
            avg_clades.add(members)
        return members
    collect(to_tree(z))
    # Public forced-K CNM output reconstructs its nested merge history. This
    # continues past its natural modularity maximum; it is not a new optimum.
    cnm_cuts = {}
    cnm_clades = set()
    for k in range(2, N):
        groups = partition(nx.community.greedy_modularity_communities(
            graph, weight='weight', resolution=1., cutoff=k, best_n=k))
        if len(groups) != k:
            raise ValueError('CNM did not return the requested K')
        cnm_cuts[k] = groups
        cnm_clades.update(g for g in groups if 1 < len(g) < N)
    if len(cnm_clades) != N-2 or len(avg_clades) != N-2:
        raise ValueError('The structural baselines did not produce full binary hierarchies')
    for family in [cnm_clades, avg_clades]:
        sets = [set(c) for c in family]
        if any(a & b and not (a <= b or b <= a) for a in sets for b in sets):
            raise ValueError('Structural merge outputs are not a nested hierarchy')
    print('SC baselines and full linkage trees computed', flush=True)
    spectral, average = {}, {}
    for k in sorted(set(K_VALUES+[reference_k, len(native_cnm)])):
        spectral[k] = from_labels(SpectralClustering(n_clusters=k, affinity='precomputed',
            assign_labels='kmeans', n_init=20, random_state=0, eigen_solver='arpack').fit_predict(w))
        average[k] = from_labels(cut_tree(z, n_clusters=k).ravel())
    baselines = {'Louvain': selected['groups'], 'CNM': native_cnm,
                 'Spectral': spectral[reference_k], 'Average linkage': average[reference_k]}
    orders = permutation_orders()
    primary, sensitivity = {}, {}
    for name, groups in baselines.items():
        primary[name] = dict(K=len(groups), sizes=[len(x) for x in groups],
            Q=float(nx.community.modularity(graph, groups, weight='weight')),
            groups=groups, **agreement(tree_cut(trees[4], len(groups)), groups),
            overlap=overlap(groups, clades(trees[4]), orders))
        sensitivity[name] = {seed: agreement(tree_cut(tree, len(groups)), groups)
                              for seed, tree in trees.items()}
    curves = {name: {k: agreement(tree_cut(trees[4], k), lookup[k])
                    for k in K_VALUES} for name, lookup in
              [('CNM', cnm_cuts), ('Spectral', spectral), ('Average linkage', average)]}
    hierarchy = {}
    spt_clades = clades(trees[4])
    for name, candidate in [('CNM merge tree', cnm_clades), ('Average-linkage tree', avg_clades),
                            ('SPT seed 3', clades(trees[3])), ('SPT seed 5', clades(trees[5]))]:
        shared = sorted(spt_clades & candidate, key=lambda x: (len(x), x))
        hierarchy[name] = dict(shared=shared, count=len(shared), fraction=len(shared)/(N-2))
    # Scientific smoke controls: metric label invariance, exact clade identity,
    # complete ROI coverage, and independent set-based overlap calculation.
    assert agreement(baselines['Louvain'], list(reversed(baselines['Louvain'])))['ARI'] == 1.
    assert jaccards(indicators([(0, 1), (2, 3)]), indicators([(0, 1), (1, 2)])).tolist() == [[1., 1/3], [0., 1/3]]
    for record in primary.values():
        for match in record['overlap']['matches']:
            a, b = set(match['group']), set(match['best_spt'])
            assert abs(match['jaccard']-len(a&b)/len(a|b)) < 1e-14
    return dict(primary=primary, curves=curves, trees=trees, hierarchy=hierarchy,
        sensitivity=sensitivity, selected=selected, louvain_runs=louvain_runs,
        region_labels=region_labels, audits=audits, sources=sources,
        edges=graph.number_of_edges(), k=reference_k, elapsed=time.monotonic()-started)


def draw(result):
    primary, curves, hierarchy = result['primary'], result['curves'], result['hierarchy']
    style = {'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
             'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
             'axes.titlesize':10, 'axes.labelsize':9}
    with plt.rc_context(style):
        fig, axs = plt.subplots(2, 2, figsize=(11.7, 7.7), layout='constrained',
                                gridspec_kw={'width_ratios':[1.,1.08]})
        a,b,c,d = axs.flat
        parts = {'SPT':tree_cut(result['trees'][4], result['k']),
                 **{name:primary[name]['groups'] for name in NAMES}}
        for i,(name,groups) in enumerate(parts.items()):
            start=0
            for j,group in enumerate(groups):
                a.barh(i, len(group), left=start, height=.65,
                       color=plt.cm.Blues(.35+.52*(1-j/max(1,len(groups)-1))),
                       edgecolor='white', linewidth=.7)
                if len(group)>=6:
                    a.text(start+len(group)/2,i,str(len(group)),ha='center',va='center',fontsize=8)
                start+=len(group)
        a.set(yticks=range(5), yticklabels=[f'{name} (K={len(groups)})' for name,groups in parts.items()], xlim=(0,100),
              xlabel='ROI per group (sorted; color is not group identity)',
              title='a  Group sizes at the SC-selected K')
        a.invert_yaxis()
        for name,color in zip(NAMES,COLORS):
            if name in curves:
                y=[curves[name][k]['ARI'] for k in K_VALUES]
                b.plot(K_VALUES,y,color=color,lw=1.5,marker='o',ms=2.8,label=name)
            else:
                runs=result['louvain_runs']
                b.scatter([len(r['groups']) for r in runs],
                    [agreement(tree_cut(result['trees'][4],len(r['groups'])),r['groups'])['ARI'] for r in runs],
                    color=color,s=16,marker='D',label='Louvain: 10 starts',zorder=4)
        b.axhline(0,color='.6',lw=.7,ls='--')
        b.set(xlabel='Number of groups K',ylabel='Adjusted Rand index (ARI)',
              title='b  Same-K cuts: split largest SPT node first', xticks=[2,4,6,8,10,12,14])
        b.legend(loc='upper left',bbox_to_anchor=(1.02,1.),frameon=False,fontsize=8)
        for i,name in enumerate(NAMES):
            record=primary[name]['overlap']
            for offset,key,col in [(-.12,'all ROI','#B9B9B9'),(.12,'within hemisphere','#626262')]:
                row=record['null'][key]
                c.plot([row['low'],row['high']],[i+offset]*2,color=col,lw=3,solid_capstyle='butt')
                c.plot(row['mean'],i+offset,marker='|',color=col,ms=8)
            c.scatter(record['mean_best_jaccard'],i,color=COLORS[i],s=42,zorder=3)
        c.set(yticks=range(4),yticklabels=NAMES,xlim=(0,.42),xticks=[0,.1,.2,.3,.4],
              xlabel='Mean best Jaccard: SC community → SPT subtree',
              title='c  Search all SPT subtrees with 2–50 ROI')
        c.invert_yaxis()
        c.legend(handles=[Line2D([],[],marker='o',ls='',color='black',label='Observed'),
            Line2D([],[],lw=3,color='#B9B9B9',label='ROI-shuffle 95% range'),
            Line2D([],[],lw=3,color='#626262',label='Within-hemisphere shuffle')],
            loc='upper left',bbox_to_anchor=(0,-.20),ncol=1,frameon=False,fontsize=8)
        for i,(name,row) in enumerate(hierarchy.items()):
            d.barh(i,100*row['fraction'],height=.6,color='#257D98' if i<2 else '#7B8D82')
            d.text(100*row['fraction']+1,i,f'{row["count"]}/98',va='center',fontsize=8)
        d.set(yticks=range(4),yticklabels=list(hierarchy),xlim=(0,23),xticks=[0,5,10,15,20],
              xlabel='Exactly shared nontrivial subtrees (%)',
              title='d  Full hierarchies vs SPT seed 4')
        d.invert_yaxis()
        FIGURE.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(FIGURE,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)


def write_report(result):
    p=result['primary']; h=result['hierarchy']; k=result['k']
    groups=tree_cut(result['trees'][4],k)
    rows=['# G=1.3 的 SPT 与结构连接社团划分：经典算法首轮对比', '',
        '2026-10-17；平均 SC、100 ROI，主比较为文首主图 g 的原始 seed 4 树。', '',
        f'**首轮结论：整体分组和大尺度层级明显不同，但存在可辨认的局部重叠，不能写成“没有重叠”。** 四种拓扑方法与同 K SPT 前沿的 ARI 均接近 0；完整 CNM／平均连接树分别与 SPT 共享 {h["CNM merge tree"]["count"]}/98、{h["Average-linkage tree"]["count"]}/98 个精确子树，均为 2–5 ROI 小组合。Louvain 的平均最佳成员 Jaccard 为 {p["Louvain"]["overlap"]["mean_best_jaccard"]:.3f}，最高单社团匹配达到 '+f'{max(r["jaccard"] for r in p["Louvain"]["overlap"]["matches"]):.3f}；其平均重叠高于本轮半球内重标号的 97.5 百分位。', '',
        '**还须同时看到 SPT 自身的成员不稳定性。** 固定同一平均 SC、同 G，只换既有模拟 seed，seed 3/5 与 seed 4 也都仅共享 17/98 个完整子树。因此本轮能确认这些输出并不相同，尚不能把其全部差异归为稳定的动力学特异组织，更不能据此断言 SPT 优于社团算法。', '',
        '本轮检验结构连接自身的社团是否复现 SPT 的 ROI 组合。它比较实际成员与原始树，未把压缩显示的分叉当作平行社团。结果用于描述结构差别，不以“重叠越少越好”选择算法或参数。', '',
        '![同社团数划分、子树成员重叠与完整层次比较](assets/dmf_spt_sc_communities.png)', '',
        '**图说明。** a 为社团大小分布，Louvain/CNM 保留各自原生 K，SPT、谱聚类及平均连接取所选 Louvain 的 K；颜色仅区分相邻段，不跨算法对应社团身份。b 为与 seed 4 SPT 同社团数前沿的 ARI；CNM、谱聚类、平均连接遍历 K=2–15，Louvain 显示十次原生输出。c 为每个拓扑社团在 SPT 所有 2–50 ROI 子树中的最大 Jaccard，再对社团等权平均；点为实测，横线为 499 次随机重标号的 2.5–97.5 百分位范围，竖记号为随机均值，非置信区间。d 为完整二叉层次的精确共同子树比例，排除全根和单 ROI；另列 SPT seed 3/5 作为内部重复性参照。', '',
        '## 选择了哪些经典算法', '',
        '| 方法 | 仅使用 SC 的标准与参数 |', '|---|---|',
        '| Louvain | 加权模块度最大化，resolution=1；seed 0–9 十次启动，仅按 SC 模块度最高选择，平手取最小 seed |',
        '| CNM（Clauset–Newman–Moore） | 加权贪心模块度，resolution=1；主表为自然停止结果，同 K 曲线另强制停止于各 K |',
        f'| 谱聚类／normalized-cut 谱松弛 | SC 直接作为 precomputed affinity；K={k}，取自 Louvain 的原生社团数；k-means n_init=20、random_state=0 |',
        f'| 平均连接层次聚类 | 距离 dᵢⱼ=1−Wᵢⱼ/max(W)，对角为0；average linkage，在 K={k} 切树；这是层次基线，不是模块度算法 |', '',
        f'完整保留原生非负权重和零边：100 节点、{result["edges"]} 条无向非零边，无自环、无阈值化、无二值化、无额外 SC 重标度。SC 为原 DMF 模拟所用 93 人群体平均矩阵；四种基线不读取 E/I 响应、EI、Syn、Yeo 标签或 SPT 分组。', '',
        f'Louvain 选中启动 seed={result["selected"]["seed"]}，Q={result["selected"]["Q"]:.6f}。谱聚类与平均连接的 K 取自该 SC-only 结果，并未按 SPT 重叠选择。K=2–15 为预先限定的轻量多尺度查看，不扫描 resolution 寻找差异最大点。', '',
        '## 同社团数的粗尺度划分', '',
        'SPT 本身没有唯一平面社团。这里从根出发，每次拆分当前 ROI 数最多的非叶节点，平手按成员编号排序，直到达到同 K；该规则不查询模块度或本次重叠结果。', '',
        f'主工作点 K={k} 时，seed 4 SPT 前沿大小为 **'+', '.join(str(len(x)) for x in groups)+'**。其形状与传统社团的较均衡块很不相同，故低 ARI/AMI 同时受大小分布差别影响。它不能独自证明局部组合没有交集。', '',
        '| 结构方法 | K | 社团大小 | SC 模块度 Q | 与同 K SPT 的 ARI | AMI |', '|---|---:|---|---:|---:|---:|']
    for name in NAMES:
        r=p[name];rows.append(f'| {name} | {r["K"]} | '+', '.join(map(str,r['sizes']))+f' | {r["Q"]:.4f} | {r["ARI"]:.4f} | {r["AMI"]:.4f} |')
    rows += ['', 'ARI 与 AMI 均对随机一致性作校正：1 表示同一划分，接近 0 表示整体一致性接近相应随机基线，负值不是“负重叠”。模块度仅说明各方法对 SC 社团目标的拟合，不用它评判 SPT 的因果信息目标优劣。', '',
        '## 不依赖切树位置的局部成员重叠', '',
        '对每个拓扑社团 C，在原 SPT 的全部 2–50 ROI 子树 S 中搜索 max |C∩S|/|C∪S|，再等权平均社团。该搜索允许社团匹配树的任一深度，不受上面的巨型前沿块限制；排除 >50 ROI 的背景大集合和单 ROI。各社团可以匹配同一子树，因此该统计是偏宽松的覆盖诊断，不是一一社团对应率。', '',
        '| 方法 | 平均最佳 Jaccard | 完整社团精确匹配数 | 全 ROI 重标号随机范围 | 半球内重标号随机范围 |', '|---|---:|---:|---|---|']
    for name in NAMES:
        r=p[name]['overlap'];a,b=r['null']['all ROI'],r['null']['within hemisphere']
        rows.append(f'| {name} | {r["mean_best_jaccard"]:.3f} | {r["exact_match_count"]}/{p[name]["K"]} | {a["low"]:.3f}–{a["high"]:.3f} | {b["low"]:.3f}–{b["high"]:.3f} |')
    rows += ['', '随机重标号保持完整 SPT 的形状与每个子树的大小，也保持拓扑社团的大小；每次重标号后重新搜索最佳子树。半球内重标号还固定每个子树的左右半球组成。两类随机范围均为描述性参照，不代表独立被试抽样，不在本轮作多重检验或确认性显著性结论。', '',
        '## 完整层次中是否仍有相同组合', '',
        '二叉树各有 98 个非平凡子树，成员完全相同才计为共享。CNM 的合并层次通过公共接口逐 K 读取；自然模块度最大值之后的合并为强制继续的层次，不声称这些后续节点仍提高 Q。', '',
        '| 与 seed 4 SPT 比较的树 | 共同非平凡子树 | 占 98 个子树的比例 |', '|---|---:|---:|']
    for name,r in h.items():rows.append(f'| {name} | {r["count"]}/98 | {100*r["fraction"]:.1f}% |')
    rows += ['', '共享的具体小组合（ROI 名称按当前推定 Schaefer 标签）：', '']
    for name in ['CNM merge tree','Average-linkage tree']:
        rows.append(f'**{name}**：')
        for group in h[name]['shared']:
            text=', '.join(str(result['region_labels'][i]).replace('7Networks_','') for i in group)
            rows.append(f'- {len(group)} ROI：{text}。')
        if not h[name]['shared']:rows.append('- 无精确共同子树。')
        rows.append('')
    rows += ['## 三个既有模拟 seed 的描述性复核', '',
        'SC 基线固定不变，换用既有 G=1.3 的 seed 3/4/5 原始 SPT，只考察其模拟重复性；三棵树共享平均 SC，不是三个独立被试。', '',
        '| 方法 | seed 3 ARI | seed 4 ARI | seed 5 ARI |', '|---|---:|---:|---:|']
    for name in NAMES:
        r=result['sensitivity'][name];rows.append(f'| {name} | {r[3]["ARI"]:.4f} | {r[4]["ARI"]:.4f} | {r[5]["ARI"]:.4f} |')
    rows += ['', '## Louvain 社团与最接近 SPT 子树的成员明细', '',
        'ROI 使用 1-based 原始输入序号（不因图中叶序改变）；括号给出社团及子树大小。相同成员与近似成员应区分。', '',
        '| Louvain 社团 ROI | 最佳 SPT 子树 ROI | 共有 ROI | Jaccard |', '|---|---|---|---:|']
    for r in p['Louvain']['overlap']['matches']:
        fmt=lambda g: ', '.join(str(i+1) for i in g)
        rows.append(f'| {fmt(r["group"])} ({len(r["group"])}) | {fmt(r["best_spt"])} ({len(r["best_spt"])}) | {fmt(r["intersection"])} | {r["jaccard"]:.3f} |')
    rows += ['', '## 证据边界与稿件核对', '',
        '本轮是平均 SC、G=1.3 的探索性算法对比，未以不同结构 null、个体连接组或其他 G 检验泛化，也未检验结构社团的信息保留或预测效用。差异不能证明 SPT 更优，也不能证明其组织独立于 SC；DMF 的动力学本来就由该 SC 约束。未比较 Leiden、Infomap、随机块模型或重叠社团模型。', '',
        'SPT 使用完整原始二叉树，八个多 ROI 分叉不当作八个平面社团；压缩显示的单 ROI 剥离段不重新命名为社团。大节点的谱候选来自 pairwise Syn affinity，最终按 EI/Ξ 子集合目标选择，与直接在原始 SC 上谱聚类具有不同的输入及优化目标；因此这不是固定同一 affinity 后只替换优化器的单因素比较。当前 g 是历史条件协方差树，未重建为主图 b 的连续 G 约束估计家族。', '',
        '2026-10-17 重新经 Zotero 本地 API 核验父条目 **P6UJCVG8**，题名 *Emergent hierarchical organization of causal interactions in complex systems*；重列附件并读取正文 **DXGC7JEA**（19/19 页）Methods 式（10）–（12）（第16–17页）及 Brain/Fig.2（第6–7页），补充 **MWIWKSVG**（28/28 页）S5／Algorithm S1（第8–9页）。正文与补充各一份，无明确稿件日期或修订号，元数据修改不能确立版本先后。本轮不作版本依赖的方法修改。稿件 Fig.2 的 Yeo 首层约束树与当前无先验 ROI 树不同，本比较使用后者。', '',
        '原树继续按原生 Syn 非负容差 **1e-8 bits** 审计；这里只复用已有估计，不新增 EI 或 Syn 值。三棵树的显著非负违反和容差内负值均为 '+str(sum(a['significant_nonnegativity_violation_count'] for a in result['audits'].values()))+' 与 '+str(sum(a['tolerance_negative_count'] for a in result['audits'].values()))+'；没有裁剪或非负投影。结构社团的 Q、ARI、AMI、Jaccard 均为无量纲。', '',
        f'运行版本：NetworkX {nx.__version__}，scikit-learn {sklearn.__version__}，SciPy {scipy.__version__}。固定 499 次随机重标号／规则，随机种子 {NULL_SEED}。分析耗时 {result["elapsed"]:.1f} 秒。复现：`python scripts/compare_dmf_spt_sc_communities.py`，只重读现有数据并运行轻量拓扑基线。', '',
        '来源指纹：', '']
    rows += [f'- `{path}`：`{digest}`。' for path,digest in result['sources'].items()]
    rows += ['', '## 经典方法与实现依据', '',
        '- Blondel et al. (2008), [Fast unfolding of communities in large networks](https://arxiv.org/abs/0803.0476)；[NetworkX Louvain 文档](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)。',
        '- Clauset, Newman & Moore (2004), [Finding community structure in very large networks](https://arxiv.org/abs/cond-mat/0408187)；[NetworkX CNM 与强制 K 参数](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)。',
        '- Shi & Malik (2000), Normalized cuts and image segmentation；[scikit-learn SpectralClustering 文档及原论文链接](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)。这里使用其 normalized-cut 谱松弛及 k-means 指派实现，不宣称获得组合 normalized-cut 全局最优。',
        '- [SciPy average-linkage 定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)；[ARI 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)；[AMI 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)。', '']
    REPORT.write_text('\n'.join(rows))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result=run()
        draw(result)
        write_report(result)
    print(json.dumps(dict(report=str(REPORT),figure=str(FIGURE),
        primary={name:{k:v for k,v in record.items() if k not in ['groups','overlap']} for name,record in result['primary'].items()},
        overlap={name:record['overlap'] for name,record in result['primary'].items()},
        hierarchy={name:{k:v for k,v in row.items() if k!='shared'} for name,row in result['hierarchy'].items()},
        elapsed_seconds=result['elapsed']),ensure_ascii=False,indent=2))
