"""Cluster the six nodes using the original, hash-verified short transitions.

No MLP is fitted or loaded and no EI/Syn is estimated. The original SPT is only
a cached reference. Main view: observed wrapped increments. Additional view:
each node's sin/cos initial phase plus observed increment. Both use all 4800
original independent short trajectories, seed 0, at K_out=.04 and 5.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, squareform
from scipy.special import digamma
import sklearn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_mixed_order_kuramoto_kout_main as original

OUT = ROOT / "results/kuramoto_short_trajectory_clustering/summary.json"
ASSETS = ROOT / "docs/reports/assets/kuramoto_hierarchy"
REPORT = ROOT / "docs/reports/kuramoto_short_trajectory_clustering.md"
METHODS = ("kmeans", "ward", "spectral", "correlation_hac", "dcor_hac", "mi_hac", "mic")
TITLES = {"kmeans": "K-means", "ward": "Ward", "spectral": "Spectral",
          "correlation_hac": "|Correlation| HAC", "dcor_hac": "Distance-corr. HAC",
          "mi_hac": "Pairwise MI HAC", "mic": "Group MI (MIC)", "spt": "SPT (cached)"}
BLUE, ORANGE, GREEN, GRAY = "#356B91", "#B66A2D", "#247C70", "#A8B1BA"


def digest(array):
    values = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((values.shape, str(values.dtype))).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def standardize(values):
    values = np.asarray(values, dtype=float)
    scale = values.std(axis=0, keepdims=True)
    if np.any(scale < 1e-12):
        raise ValueError("Constant input coordinate; dependence is not identifiable")
    return (values - values.mean(axis=0, keepdims=True)) / scale


def ksg_mi(x, y, k=5):
    """Raw KSG-1, max norm, strict marginal radii, in nats (no clipping).

    Coordinates must already be standardized/jittered consistently, including
    when concatenated into MIC cluster variables. Query includes self at index 0.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if len(x) != len(y) or not 1 <= k < len(x):
        raise ValueError("Mismatched samples or invalid k")
    joint = np.column_stack((x, y))
    radius = cKDTree(joint).query(joint, k=k + 1, p=np.inf, workers=1)[0][:, k]
    if np.any(radius <= 0):
        raise ValueError("Duplicate samples require declared jitter")
    radius = np.nextafter(radius, 0)
    nx = cKDTree(x).query_ball_point(x, radius, p=np.inf, return_length=True, workers=1) - 1
    ny = cKDTree(y).query_ball_point(y, radius, p=np.inf, return_length=True, workers=1) - 1
    return float(digamma(k) + digamma(len(x)) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def distance_correlation_matrix(blocks, chunk=192):
    """Biased sample distance correlation, Euclidean node blocks, O(chunk*N*6) memory."""
    blocks = np.asarray(blocks, dtype=float)
    n, nodes, _ = blocks.shape
    means = np.zeros((nodes, n))
    for i in range(nodes):
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            means[i, start:stop] = cdist(blocks[start:stop, i], blocks[:, i]).mean(axis=1)
    grand = means.mean(axis=1)
    products = np.zeros((nodes, nodes))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        centered = []
        for i in range(nodes):
            d = cdist(blocks[start:stop, i], blocks[:, i])
            d -= means[i, start:stop, None]
            d -= means[i, None, :]
            d += grand[i]
            centered.append(d.ravel())
        a = np.asarray(centered)
        products += a @ a.T
    products /= n * n
    denominator = np.sqrt(np.outer(products.diagonal(), products.diagonal()))
    squared = products / denominator
    if squared.min() < -1e-12 or squared.max() > 1 + 1e-12:
        raise ArithmeticError("Distance-correlation numerical range violation")
    # Only documented roundoff correction for this non-Syn statistic.
    return np.sqrt(np.clip(squared, 0, 1))


def canonical_groups(groups):
    return sorted((sorted(map(int, group)) for group in groups), key=lambda group: group[0])


def groups_from_labels(labels):
    labels = np.asarray(labels)
    return canonical_groups([np.flatnonzero(labels == value) + 1 for value in np.unique(labels)])


def hierarchy_result(z):
    nodes = {i: {"sources": [i + 1], "children": []} for i in range(6)}
    for i, (left, right, height, count) in enumerate(z):
        a, b = nodes[int(left)], nodes[int(right)]
        nodes[6 + i] = {"sources": sorted(a["sources"] + b["sources"]),
                        "children": [a, b], "native_height": float(height)}
    root = nodes[10]
    return dict(kind="hierarchy", tree=root,
                groups=canonical_groups([child["sources"] for child in root["children"]]),
                linkage=np.asarray(z).tolist())


def clades(tree):
    if not tree:
        return []
    rows = [tree["sources"]] if 1 < len(tree["sources"]) < 6 else []
    for child in tree["children"]:
        rows.extend(clades(child))
    return rows


def annotate(result, kout):
    reference = np.array([0, 0, 0, 1, 1, 1])
    predicted = np.empty(6, dtype=int)
    for i, group in enumerate(result["groups"]):
        predicted[np.array(group) - 1] = i
    present = clades(result.get("tree")) if result["kind"] == "hierarchy" else result["groups"]
    result.update(module_123_retained=[1, 2, 3] in present,
                  module_456_retained=[4, 5, 6] in present,
                  original_partition_ari=float(adjusted_rand_score(reference, predicted)),
                  exact_original_two_groups=result["groups"] == [[1, 2, 3], [4, 5, 6]],
                  ari_interpretation="reference recovery" if kout == .04 else "original-partition retention, not accuracy",
                  native_causal_order_output=False)
    return result


class GroupMI:
    def __init__(self, blocks, k=5):
        self.blocks = blocks
        self.k = k
        self.cache = {}

    def __call__(self, left, right):
        a, b = sorted((tuple(sorted(left)), tuple(sorted(right))))
        key = (a, b)
        if key not in self.cache:
            n = len(self.blocks)
            x = self.blocks[:, np.array(a) - 1, :].reshape(n, -1)
            y = self.blocks[:, np.array(b) - 1, :].reshape(n, -1)
            self.cache[key] = ksg_mi(x, y, self.k)
        return self.cache[key]

    def records(self):
        return [{"left": a, "right": b, "mi_nats": value}
                for (a, b), value in sorted(self.cache.items())]


def mic_tree(information, node_dimension):
    """Continuous MIC: merge max I(groupA;groupB)/(dimA+dimB); recompute joint MI.

    Eq. (10), Kraskov et al., q-bio/0311037. Heights are not used: each node
    stores the raw group MI and normalized merge similarity, including negatives.
    """
    active = [{"sources": [i], "children": []} for i in range(1, 7)]
    while len(active) > 1:
        candidates = []
        for a, b in itertools.combinations(active, 2):
            value = information(a["sources"], b["sources"])
            score = value / (node_dimension * (len(a["sources"]) + len(b["sources"])))
            candidates.append((score, tuple(a["sources"]), tuple(b["sources"]), value, a, b))
        score, _, _, value, a, b = min(candidates, key=lambda row: (-row[0], row[1], row[2]))
        active = [node for node in active if node is not a and node is not b]
        active.append({"sources": sorted(a["sources"] + b["sources"]), "children": [a, b],
                       "group_mi_nats": value, "merge_similarity": score})
        active.sort(key=lambda node: tuple(node["sources"]))
    root = active[0]
    return dict(kind="hierarchy", tree=root,
                groups=canonical_groups([child["sources"] for child in root["children"]]))


def analyze_view(raw_blocks, kout, seed=0):
    blocks = standardize(raw_blocks)
    n, nodes, dimensions = blocks.shape
    # Same small fixed jitter for every query; no group-dependent reprocessing.
    jittered = blocks + np.random.default_rng(20261017).normal(0, 1e-10, blocks.shape)
    profiles = blocks.transpose(1, 0, 2).reshape(nodes, -1)
    correlation = profiles @ profiles.T / (n * dimensions)
    np.fill_diagonal(correlation, 1)
    dcor = distance_correlation_matrix(blocks)
    information = GroupMI(jittered)
    mi = np.zeros((6, 6))
    for i, j in itertools.combinations(range(6), 2):
        mi[i, j] = mi[j, i] = information([i + 1], [j + 1])
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=2, n_init=50, random_state=seed).fit_predict(profiles)
        spectral = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="cluster_qr",
                                     random_state=seed).fit_predict(np.abs(correlation))
    methods = {"kmeans": dict(kind="partition", groups=groups_from_labels(labels)),
               "ward": hierarchy_result(linkage(profiles, method="ward", optimal_ordering=True)),
               "spectral": dict(kind="partition", groups=groups_from_labels(spectral))}
    # Retain raw signed KSG values; negative MI estimates give distances >1.
    mi_scale = max(float(np.abs(mi).max()), 1e-12)
    for name, distance in (("correlation_hac", 1 - np.abs(correlation)),
                            ("dcor_hac", 1 - dcor), ("mi_hac", 1 - mi / mi_scale)):
        np.fill_diagonal(distance, 0)
        if distance.min() < -1e-12:
            raise ArithmeticError(f"Invalid distance for {name}")
        z = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
        methods[name] = hierarchy_result(z)
    methods["mic"] = mic_tree(information, dimensions)
    return dict(methods={key: annotate(methods[key], kout) for key in METHODS},
                matrices=dict(correlation=correlation.tolist(), distance_correlation=dcor.tolist(),
                              pair_mi_nats=mi.tolist()),
                group_mi_queries=information.records(),
                negative_pair_mi_estimate_count=int(np.sum(mi[np.triu_indices(6, 1)] < 0)),
                node_block_dimension=dimensions)


def original_data(kout):
    original._configure_coherent_tm()
    phases, future = original.coherent_tm.generated_transition_data(
        sample_count=4800, seed=1000, pairwise_coupling=original.WITHIN_COUPLING,
        triadic_coupling=original.WITHIN_COUPLING / math.sqrt(2), cross_coupling=kout,
        process_noise=.08, tau=.2, dt=.01)
    increment = np.angle(np.exp(1j * (future - phases)))
    slug = f"{kout:g}".replace(".", "p")
    locator = ROOT / f"results/spt_kuramoto_transition_pilot/kout_{slug}_seed_0_learned_components.json"
    reference = json.loads(locator.read_text())["shared_channel_hashes"]
    actual = dict(training_phases=digest(phases), training_increment=digest(increment))
    if any(reference[name] != value for name, value in actual.items()):
        raise RuntimeError("Reconstructed dataset differs from the original paired dataset")
    return phases, future, increment, dict(hashes=actual, expected_hashes={k: reference[k] for k in actual},
                                          exact_match=True, reference_record=str(locator.relative_to(ROOT)))


def cached_spt(kout):
    cache = ROOT / "results/mixed_order_kuramoto_kout_main/summary.json"
    payload = json.loads(cache.read_text())
    record = next(row for row in payload["conditions"] if row["k_out"] == kout)
    tolerance = float(payload["experiment_contract"]["syn_nonnegative_tolerance_bits"])
    atoms = np.array([row["value_bits"] for row in record["atoms"]])
    audit = record["marginal_diagnostics"]
    if atoms.min() < -tolerance or audit["violating_partition_syn_count"] or audit["violating_non_singleton_xi_count"]:
        raise ArithmeticError(f"Cached SPT audit failed: min={atoms.min()} bits, threshold={-tolerance} bits")
    def convert(node):
        return dict(sources=[int(name.removeprefix("theta")) for name in node.sources],
                    children=[convert(child) for child in node.children], syn_bits=node.syn_value)
    tree = convert(original.tree_from_record(record))
    return annotate(dict(kind="hierarchy", tree=tree,
                         groups=canonical_groups([child["sources"] for child in tree["children"]]),
                         root_xi_bits=record["root_xi_bits"], cached_reference=True,
                         syn_audit=dict(tolerance_bits=tolerance, minimum_saved_atom_bits=float(atoms.min()),
                                        near_zero_negative_atom_count=int(np.sum((atoms < 0) & (atoms >= -tolerance))),
                                        violating_atom_count=0, original_audit=audit, clipping=False),
                         source=str(cache.relative_to(ROOT))), kout)


def tree_drawing(ax, result):
    root = result["tree"]
    positions = {}
    cursor = [0]
    def locate(node):
        if not node["children"]:
            x, y = cursor[0], 0
            cursor[0] += 1
        else:
            child_positions = [locate(child) for child in node["children"]]
            x, y = float(np.mean([point[0] for point in child_positions])), max(point[1] for point in child_positions) + 1
        positions[tuple(node["sources"])] = (x, y)
        return x, y
    locate(root)
    def draw(node):
        x, y = positions[tuple(node["sources"])]
        if not node["children"]:
            label = node["sources"][0]
            color = BLUE if label <= 3 else ORANGE
            ax.text(x, -.38, str(label), ha="center", va="center", color=color, fontsize=11, weight="bold")
            return
        color = BLUE if node["sources"] == [1, 2, 3] else ORANGE if node["sources"] == [4, 5, 6] else GRAY
        child_x = []
        for child in node["children"]:
            cx, cy = positions[tuple(child["sources"])]
            child_x.append(cx)
            ax.plot([cx, cx], [cy, y], color=color, lw=1.6)
            draw(child)
        ax.plot([min(child_x), max(child_x)], [y, y], color=color, lw=1.6)
        if "group_mi_nats" in node:
            label = f"{node['group_mi_nats']:.3f}"
            if node["sources"] in ([1, 2, 3], [4, 5, 6]):
                label = "".join(map(str, node["sources"])) + "\n" + label
            ax.text(x, y + .10, label, color=color if color != GRAY else "#65727C",
                    fontsize=7.5, ha="center", va="bottom", linespacing=1.05)
        elif node["sources"] in ([1, 2, 3], [4, 5, 6]):
            ax.text(x, y + .13, "".join(map(str, node["sources"])), color=color,
                    fontsize=8, ha="center", va="bottom")
    draw(root)


def partition_drawing(ax, result):
    for row, group in enumerate(result["groups"]):
        xs = np.linspace(.8, 4.2, len(group)) if len(group) > 1 else [2.5]
        y = 3.6 - row * 1.8
        for x, label in zip(xs, group):
            color = BLUE if label <= 3 else ORANGE
            ax.scatter([x], [y], s=180, facecolors="white", edgecolors=color, lw=1.2)
            ax.text(x, y, str(label), ha="center", va="center", fontsize=10, color=color)
    ax.text(2.5, .25, "partition only", ha="center", fontsize=8, color="#65727C")


def group_string(groups):
    return " | ".join("{" + ",".join(map(str, group)) + "}" for group in groups)


def plot(payload, view, path):
    names = ("spt",) + METHODS
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 8, figsize=(19, 6.5), constrained_layout=True)
        for row, condition in enumerate(payload["conditions"]):
            for col, name in enumerate(names):
                ax = axes[row, col]
                result = condition["spt_reference"] if name == "spt" else condition["views"][view]["methods"][name]
                if result["kind"] == "hierarchy":
                    tree_drawing(ax, result)
                else:
                    partition_drawing(ax, result)
                ax.set(xlim=(-.35, 5.35), ylim=(-1.35, 5.75))
                ax.axis("off")
                if row == 0:
                    ax.set_title(TITLES[name] + ("\nmerge MI / nats" if name == "mic" else ""), fontsize=10.5, pad=7)
                root_text = group_string(result["groups"])
                ax.text(2.5, -1.02, root_text, ha="center", va="center", fontsize=8.5)
                if col == 0:
                    ax.text(-.20, 1.03, rf"$K_{{\mathrm{{out}}}}={condition['k_out']:g}$",
                            transform=ax.transAxes, fontsize=12, weight="bold", ha="left")
        fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Original module 123"),
                            Line2D([], [], color=ORANGE, lw=2, label="Original module 456")],
                   loc="outside lower center", ncols=2, frameon=False, fontsize=10)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_report(payload):
    lines = ["# 原短轨迹数据上的节点聚类比较", "", "2026-10-17。原图两个条件，各 4,800 条独立短轨迹，seed 0。用户已明确授权本次比较；没有重训 MLP、重算 EI/SPT、调用子代理或按结果调参。", "",
             "![相位增量上的实际聚类](assets/kuramoto_hierarchy/kuramoto_short_trajectory_clustering.png)", "",
             "主图每棵树保留实际拓扑，枝长仅表示拓扑层次，不编码可跨方法比较的距离。蓝色/橙色分支标出完整的原模块 123/456；叶子颜色仅标记原归属。K-means、谱聚类仅画原生两团划分。下方集合是根切分或平面划分。SPT 是旧缓存参照，含依赖分量及机制特征近似，不构成仅改变聚类规则的公平性能排名。", "",
             "## 数据与固定口径", "",
             "原始状态对未单独保存。本次用原生成器、相同参数与随机种子重建，两个条件的初态及 wrapped 增量摘要均与既有配对记录逐字节一致。所有对比方法使用同一完整 4,800 行；不抽取较有利的样本。它们是独立初态出发的短轨迹终点对，不是自然稳态时间序列。", "",
             "- K_out 固定为 0.04、5；tau=0.20，dt=0.01，噪声 0.08，pairwise 不对称参数 0.25。", 
             "- 主视图：每节点 4,800 个 wrapped 相位增量。范围未触及 ±pi，可按实数处理；各节点中心化并标准化。", 
             "- 补充视图：每节点 (sin 初态相位, cos 初态相位, wrapped 增量)，逐坐标标准化；相同状态对提供全部信息。", 
             "- 聚类对象始终是六个节点。K-means/Ward 接收每节点展开的响应向量；谱聚类使用绝对相关 affinity。相关 HAC 使用 1-|r|；多坐标视图 r 为对齐特征响应向量的相关。", 
             "- distance-correlation HAC 使用欧氏距离、双中心化的有偏样本距离相关，距离为 1-dCor；按行块精确累计，未抽样。仅修正 1e-12 以内的浮点范围误差。", 
             "- MI 使用 KSG-1，k=5、最大范数，严格边界计数，单位 nats；固定 1e-10 标准差的抖动消除重复距离。原始负 MI 估计保留。MI-HAC 距离为 1-MI/max|MI|，故负估计对应距离大于 1，没有投影或截断。", 
             "- 相关、距离相关及成对 MI 三种 HAC 使用 average linkage；Ward 使用欧氏特征上的 Ward linkage。MIC 每次合并后重新估计组联合 MI，按 I(A;B)/(dim A+dim B) 选择最大者；确定性平局按节点元组排序。多坐标视图各节点维数相同，其常数因子不改变合并选择。", 
             "- K-means 固定 k=2、n_init=50、random_state=0；谱聚类固定 k=2、cluster_qr。两方法获得已知两团分辨率，未强制每团三个节点。Ward/HAC/MIC 的两团为其自然根切分。", 
             "- 123/456 是否存在为层级 clade 保留指标；平面方法只检查输出两团。弱耦合 ARI 是对原模块的恢复；强耦合 ARI 只描述原分组保留程度，没有唯一正确二叉树。单数据种子，无数据重复置信区间。", "",
             "## 主视图：相位增量", ""]
    for view in ("increment", "transition"):
        if view == "transition":
            lines.extend(["", "## 补充视图：起始相位和增量", "", "![完整节点转移块](assets/kuramoto_hierarchy/kuramoto_short_transition_clustering.png)", ""])
        lines += ["| K_out | 方法 | 两团划分 | 保留123 | 保留456 | 原分组ARI |", "|---:|---|---|:---:|:---:|---:|"]
        for condition in payload["conditions"]:
            for name in ("spt",) + METHODS:
                r = condition["spt_reference"] if name == "spt" else condition["views"][view]["methods"][name]
                method_name = TITLES[name].replace("|", "\\|")
                groups_text = group_string(r['groups']).replace("|", "\\|")
                lines.append(f"| {condition['k_out']:g} | {method_name} | {groups_text} | {'是' if r['module_123_retained'] else '否'} | {'是' if r['module_456_retained'] else '否'} | {r['original_partition_ari']:.3f} |")
    lines += ["", "## 本次观察与 MIC 的组联合读数", "",
              "弱耦合增量视图中，谱聚类、绝对相关 HAC、距离相关 HAC、MIC 均恢复 123|456；加入起始相位后，成对 MI-HAC 也恢复该划分。当前案例不能用来支持只有 SPT 才能恢复模块。K-means/Ward 的欧氏响应相似性与绝对相关使用不同标准；增量中 1 与 2 的相关约为 -0.900，反相响应在前者中相距很远，不能把该结果泛化成普通聚类普遍失效。", "",
              "MIC 在弱耦合下还给出以下原生合并互信息。首个两节点合并和第三节点加入时的组 MI 分开列出，均为 nats；它们不是 PEID 的纯二阶/三阶质量。图中 MIC 的节点数字也是两子组的原始 MI，纵向位置仍仅编码拓扑。", "",
              "| 视图 | 模块 | 首次两节点合并 MI | 第三节点加入的组 MI |", "|---|---|---:|---:|"]
    weak = next(c for c in payload["conditions"] if c["k_out"] == .04)
    for view, label in (("increment", "增量"), ("transition", "初态＋增量")):
        root = weak["views"][view]["methods"]["mic"]["tree"]
        def find(node, sources):
            if node["sources"] == sources:
                return node
            for child in node["children"]:
                match = find(child, sources)
                if match is not None:
                    return match
            return None
        for group in ([1, 2, 3], [4, 5, 6]):
            node = find(root, group)
            if node is None:
                lines.append(f"| {label} | {group} | 未形成该模块 | 未形成该模块 |")
                continue
            pair = next(child for child in node["children"] if len(child["sources"]) == 2)
            lines.append(f"| {label} | {''.join(map(str, group))} | {pair['group_mi_nats']:.6f} | {node['group_mi_nats']:.6f} |")
    lines += ["", "123 的首次成对 MI 更强，456 在第三节点加入时的组 MI 明显高于其首个成对 MI；因此，本次 MIC 的统计依赖读数也反映了用户关注的差别。这只描述当前样本与估计器，不等同于纯三体因果力的识别，也未估计统计不确定性。强耦合各方法产生不同组织，原两团保留不作为正确性的唯一标准。", ""]
    lines += ["", "## 解释边界与手稿", "",
              "这些算法均没有原生的 PEID 二阶/三阶因果贡献输出。MIC 会读取已合并变量组的联合依赖，但普通树中出现 456 或更高合并节点，不能单独证明恢复了三体机制。负 KSG MI 是有限样本估计现象，与 PEID Syn 无关。", "",
              "当前源初态按构造独立。本次另存初态圆周特征的对齐相关矩阵作输入分布检查，未用这一无机制信号的视图评判方法胜负。直接增量聚类使用观测到的响应依赖；完整节点转移块增加本节点初态。两者均不同于 SPT 在模型上进行全源到固定完整目标的 EI 计算。", "",
              "本次重新通过 Zotero 状态检查及题名搜索核对父项 P6UJCVG8，重新列举并读取正文 DXGC7JEA（19/19页）Methods (10)–(12)，以及补充 MWIWKSVG（28/28页）S5/Algorithm S1 和 S7。当前唯一附件组没有明确稿件版本或日期，元数据修改时间不能消除版本歧义。全树增量 target 与正文一般完整未来 target 的差异、机制 Fourier context、弱耦合分量近似均沿用并披露；未声称解决它们。", "",
              "SPT 仅消费已有 Syn 缓存，沿用原生容差 0.1 bits，记录最小值、容差内负值数及显著违规数；显著违规停止展示，无静默裁剪。本次不新增 Syn 估计。", "",
              "复现入口：`scripts/compare_kuramoto_short_trajectory_clustering.py`。原始 MI、完整树、各距离矩阵和样本摘要保存在 `results/kuramoto_short_trajectory_clustering/summary.json`，未生成 CSV 或重复保存易重建的原始样本。", "",
              "## 方法与软件依据", "",
              "- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)；[SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)。", 
              "- [Székely et al., Measuring and testing dependence by correlation of distances](https://doi.org/10.1214/009053607000000505)。", 
              "- [Kraskov et al., Estimating mutual information](https://arxiv.org/abs/cond-mat/0305641)，KSG-1 Eq. (8)。", 
              "- [Kraskov et al., Hierarchical Clustering Using Mutual Information](https://arxiv.org/abs/q-bio/0311037)，连续 MIC Eq. (10)。", 
              "- 工作流使用 scikit-learn 与 nature-figure 技能；程序性参考：Kassis, Agarwal, He, Patel & Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。", "",
              f"新聚类计算用时 {payload['elapsed_seconds']:.2f} 秒（不含绘图）。4 项独立软件检查通过：Gaussian MI 解析基准、距离相关非线性/尺度检验、MIC 联合组查询及两条件样本摘要复现。软件检查不构成恢复率的统计验证。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run(output):
    started = time.monotonic()
    payload = dict(protocol=dict(sample_count=4800, seed=0, k_out=[.04, 5], tau=.2, dt=.01,
                                  views=["increment", "transition"], primary_view="increment", ksg_k=5,
                                  no_mlp_fit=True, no_new_syn=True, no_hyperparameter_search=True,
                                  computational_limit_seconds=900),
                   runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                sklearn=sklearn.__version__), conditions=[])
    for kout in (.04, 5.):
        phases, future, increment, provenance = original_data(kout)
        raw = dict(increment=increment[:, :, None],
                   transition=np.stack((np.sin(phases), np.cos(phases), increment), axis=2))
        phase_profiles = standardize(np.stack((np.sin(phases), np.cos(phases)), axis=2)).transpose(1, 0, 2).reshape(6, -1)
        condition = dict(k_out=kout, dataset=provenance, increment_range=[float(increment.min()), float(increment.max())],
                         source_only_profile_correlation=(phase_profiles @ phase_profiles.T / phase_profiles.shape[1]).tolist(),
                         spt_reference=cached_spt(kout), views={})
        for name, blocks in raw.items():
            if time.monotonic() - started > 900:
                raise TimeoutError("Declared 15-minute comparison budget reached")
            print(f"K_out={kout:g} {name}: all seven methods, N={len(blocks)}", flush=True)
            condition["views"][name] = analyze_view(blocks, kout)
            print({key: value['groups'] for key, value in condition['views'][name]['methods'].items()}, flush=True)
        payload["conditions"].append(condition)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    payload["elapsed_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    plot(payload, "increment", ASSETS / "kuramoto_short_trajectory_clustering.png")
    plot(payload, "transition", ASSETS / "kuramoto_short_transition_clustering.png")
    write_report(payload)
    print(f"Complete in {payload['elapsed_seconds']:.1f}s; {REPORT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments.output)
