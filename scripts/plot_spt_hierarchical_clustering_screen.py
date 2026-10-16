"""All registered clustering results and fixed-family representative trees."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.plot_spt_matched_cases_composite import draw_tree, verify_tree_text
from scripts.spt_hierarchical_clustering_screen import OUT, REPORT

ASSETS = ROOT / 'docs/reports/assets/kuramoto_hierarchy'
COLORS = ['#FCF4EE', '#C4DECE', '#1D7967']
REPRESENTATIVES = ['state_ward', 'absolute_average', 'kernel_average', 'mi_average',
                   'state_diana', 'state_bisect', 'state_genie_0.3', 'state_hdb_3']
TITLES = ['Ward\nstate profile', 'Average\nPLV / abs Pearson', 'Average\nRBF CKA',
          'Average\nKSG MI', 'DIANA\nstate profile', 'Bisecting k-means\nstate profile',
          'Genie (g=0.3)\nstate profile', 'HDBSCAN (m=3)\nstate profile']


def value(case, key, batch=0):
    return case['batches'][batch]['methods'][key].get('metrics')


def ranges(case, key):
    vals = [value(case, key, b)['module_f1'] for b in range(1, 5) if value(case, key, b)]
    return min(vals), max(vals)


def stable(case, key):
    target = {tuple(c) for c in value(case, key)['scored_clades']}
    return sum({tuple(c) for c in value(case, key, b)['scored_clades']} == target for b in range(1, 5))


def plot_scores(data, sha):
    groups = [[c for c in data['protocol']['methods'] if c['representation'] == rep] for rep in ['state', 'increment']]
    groups.append([c for c in data['protocol']['methods'] if c['representation'] not in ['state', 'increment']])
    headings = ['Node state trajectories', 'One-step increments', 'Phase / nonlinear dependence']
    cmap = LinearSegmentedColormap.from_list('recovery', COLORS)
    cmap.set_bad('#E7EAED')
    fig = plt.figure(figsize=(15.4, 8.7), dpi=250, facecolor='white')
    axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.subplots_adjust(left=.15, right=.958, bottom=.17, top=.88, wspace=1.0)
    for ax, configs, heading in zip(axes, groups, headings):
        vals = np.array([[value(r, c['id'])['module_f1'] if value(r, c['id']) else np.nan for r in data['records']] for c in configs])
        image = ax.imshow(vals, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        for i, c in enumerate(configs):
            for j, case in enumerate(data['records']):
                metric = value(case, c['id'])
                marker = '*' if metric and stable(case, c['id']) < 4 else ''
                label = f"{metric['module_f1']:.2f}{marker}" if metric else 'N/A'
                ax.text(j, i, label, ha='center', va='center', fontsize=8.2,
                        color='white' if vals[i, j] > .7 else '#26333D')
        labels = [c['label'].replace(' | state', '').replace(' | increment', '') for c in configs]
        ax.set_yticks(range(len(configs)), labels, fontsize=8)
        ax.set_xticks(range(3), ['Kuramoto\nA', 'HO-LV\nA', 'Kuramoto\nB'], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=5)
        ax.set_title(heading, fontsize=10, weight='bold', pad=42)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.976, .29, .009, .42])
    bar = fig.colorbar(image, cax=cax, ticks=[0, .5, 1])
    bar.set_label('Module F1', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.text(.50, .095, '* Scoring modules differ in at least one of four disjoint training-chain subsets.',
             ha='center', fontsize=9, color='#53616A')
    fig.text(.50, .061, 'All 49 predeclared configurations; identical natural training pairs within each case. No truth-based parameter selection.',
             ha='center', fontsize=9, color='#53616A')
    fig.text(.50, .027, 'MLP-based SPT comparison is pending; historical non-MLP SPT scores are not included. SHiP (2025): not run (build limit).',
             ha='center', fontsize=9, color='#53616A')
    dest = ASSETS / 'spt_hierarchical_clustering_screen.png'
    fig.savefig(dest, dpi=250, bbox_inches='tight', pad_inches=.10,
                metadata={'Description': f'All registered natural-data hierarchy results. Source SHA256 {sha}.'})
    plt.close(fig)
    return dest


def plot_trees(data, sha):
    fig, axes = plt.subplots(3, 8, figsize=(21.2, 8.6), dpi=250)
    fig.subplots_adjust(left=.074, right=.995, top=.84, bottom=.115, wspace=.20, hspace=.30)
    for row, case in enumerate(data['records']):
        for col, key in enumerate(REPRESENTATIVES):
            ax = axes[row, col]
            metric = value(case, key)
            record = dict(metrics=metric, status='complete' if metric else 'unavailable')
            draw_tree(ax, record, case['native_case'])
            if row == 0:
                ax.set_title(TITLES[col], fontsize=8.5, pad=19)
            if metric:
                low, high = ranges(case, key)
                ax.text(.5, -.10, f"F1 {metric['module_f1']:.2f}  |  subsets {low:.2f}–{high:.2f}",
                        ha='center', va='top', transform=ax.transAxes, fontsize=7, color='#596771')
        box = axes[row, 0].get_position()
        fig.text(.009, .5 * (box.y0 + box.y1), case['case'].replace(' ', '\n', 1), fontsize=9,
                 color='#53616A', va='center')
    fig.text(.52, .958, 'Node hierarchies from the same natural training data', ha='center', fontsize=12, weight='bold')
    fig.text(.52, .921, 'Fixed representatives by algorithm family; all 49 configurations retained in the score figure.',
             ha='center', fontsize=9, color='#53616A')
    fig.legend(handles=[Patch(fc='#E5F1EB', ec='#217B68', label='Reference module'),
                        Patch(fc='#FCEFEA', ec='#B74738', label='Wrong clade'),
                        Patch(fc='white', ec='#B7C0C6', label='Leaf / unscored primitive refinement')],
               loc='lower center', bbox_to_anchor=(.53, .015), ncol=3, frameon=False, fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        verify_tree_text(ax, renderer)
    dest = ASSETS / 'spt_hierarchical_clustering_trees.png'
    fig.savefig(dest, dpi=250, bbox_inches='tight', pad_inches=.10,
                metadata={'Description': f'Fixed 8 algorithm families, 3 cases; source SHA256 {sha}.'})
    plt.close(fig)
    return dest


def report(data, figures):
    records = data['records']
    n_methods = len(data['protocol']['methods'])
    complete = sum(v['status'] == 'complete' for r in records for b in r['batches'] for v in b['methods'].values())
    lines = ['# 三案例：自然数据层次聚类轻量筛查（历史探索归档）', '',
        '2026-10-16。按用户要求，从协同指标比较转向数据驱动的节点层次恢复；本轮只复用既有自然轨迹，不生成新动力学数据、不训练新MLP、不估计EI或Syn。', '',
        '**现有结果不支持“直接层次聚类普遍无法恢复这三个案例”。KSG互信息＋平均连接在三个完整训练集上模块F1均为1；完整连接、加权连接及基于MI的递归Ncut也达到同样结果。** 常规轨迹几何聚类在HO-LV A与Kuramoto B上明显较弱，但不能省略非线性依赖聚类的成功。', '',
        '本表不搬用旧Fourier／指数特征拟合所得的SPT=1。后续自然MLP试算见[独立报告](spt_natural_mlp_three_cases.md)，其积分尚未解决；不能宣称MLP-SPT优于这些基线。', '',
        f'预登记{n_methods}组算法／表示／参数配置，三个案例各做完整训练集与四个训练链子集，共{complete}/{3 * 5 * n_methods}棵完整树。正式重跑计算{data["elapsed_seconds"]:.2f}秒；首次实现检查19.87秒，另有限时120秒的SHiP源码安装尝试。安装与修复不是新物理实验。', '',
        '![全部配置的模块F1](assets/kuramoto_hierarchy/spt_hierarchical_clustering_screen.png)', '',
        '星号表示至少一个训练链子集的计分模块不同于完整训练集。所有配置均展示，不按F1选方法或参数。', '',
        '## 固定代表方法与观察结果', '',
        '| 方法／表示 | Kuramoto A | HO-LV A | Kuramoto B |', '|---|---:|---:|---:|']
    keys = ['state_ward', 'state_average', 'absolute_average', 'increment_ward', 'kernel_average',
            'mi_average', 'lag_mi_average', 'state_diana', 'state_bisect', 'state_genie_0.3', 'state_hdb_3']
    by_id = {c['id']: c for c in data['protocol']['methods']}
    for key in keys:
        lines.append('| ' + by_id[key]['label'] + ' | ' + ' | '.join(f'{value(c,key)["module_f1"]:.3f}' if value(c,key) else '未运行' for c in records) + ' |')
    lines += ['', '![预先按家族指定的24棵代表树](assets/kuramoto_hierarchy/spt_hierarchical_clustering_trees.png)', '',
        '代表树固定为Ward、相位锁定／绝对相关平均连接、核依赖平均连接、MI平均连接、DIANA、二分k-means、Genie g=.3和HDBSCAN m=3。选择按方法家族覆盖，未按结果好坏选树。枝长仅排版；红色节点为相对既定参考的额外clade，不代表物理相互作用为负。', '',
        '## 数据、模型与评分边界', '',
        '- 聚类对象始终是九个系统节点。每个节点的训练轨迹为其特征，未将时刻聚成状态簇后冒充节点树。',
        '- 复用每例64条链×512个相邻状态对的原自然轨迹池，固定seed=20261016按链划分48／8／8。聚类只使用48条训练链，即24576对；验证与测试链不参与距离计算。Kuramoto A逐项核对并完全复用了已批准自然MLP试算的训练数组摘要；LV和Kuramoto B锁定同一划分规则。后续三例自然MLP拟合另见独立报告，未混入本次735棵树。',
        '- 状态表示：Kuramoto用sin/cos圆周嵌入，LV沿用u=π log(丰度)；节点内沿训练时间中心化并归一化为单位范数。一步变化表示用wrapped相位差或相邻log丰度差，也逐节点归一化。未在节点间打乱时间，未将独立链首尾拼接成新转移。',
        '- 非线性距离：相位锁定值或绝对Pearson；RBF中心核归一化Frobenius相似度（CKA，1024个共同训练行，逐节点中位距离带宽）；KSG-1互信息（2048个共同训练行、k=5，圆周用torus最近邻，LV用秩copula）；另计算两方向跨时互信息的平均。MI只作为聚类相似度，不称因果或有效信息。KSG负估计原样保留，距离为最大off-diagonal MI减原MI，无截断。',
        '- 凝聚法包括single、complete、average、weighted、Ward、centroid、median；后三者只用于确为Euclidean的轨迹表示，未把一般MI距离代入Ward。Genie使用官方genieclust预计算距离接口，g=.1/.3/.5全部保留；HDBSCAN使用官方完整mutual-reachability单连接树、m=2/3/5，未把丢弃噪声后的平面标签补造成树。DIANA为按splinter规则的独立实现；递归Ncut为谱向量排序后的全部prefix sweep，均递归到单节点。二分k-means使用sklearn官方实现，固定20次内部初始化和随机种子。',
        '- 模块参考与旧实验一致，来自预声明权重阈值连通层级；忽略最小三元模块内部严格细化，所有方法同规则。另在JSON记录严格clade F1、precision/recall、根恢复、上层召回及TP/FP/FN。模块满分不等于恢复了所有重叠物理超边，也不验证最小三元组内部任意二叉结构。网络B对应native C，未加载历史深级联B。',
        '- 四子集各取训练链顺序中的12条完整链，属于同一生成模型／同一轨迹池的敏感性检查。三个案例各一个生成种子，无独立系统重复或显著性检验；不对735棵树作独立样本统计。', '',
        '## 所有预登记配置', '',
        '每格为完整训练集模块F1，方括号为四个12链子集的最小–最大值，不是置信区间。', '',
        '| 配置 | Kuramoto A | HO-LV A | Kuramoto B |', '|---|---:|---:|---:|']
    for config in data['protocol']['methods']:
        cells = []
        for c in records:
            m = value(c, config['id'])
            if not m:
                cells.append('未运行')
                continue
            low, high = ranges(c, config['id'])
            cells.append(f'{m["module_f1"]:.3f} [{low:.3f}, {high:.3f}]')
        lines.append('| ' + config['label'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 解释与下一步方向', '',
        '这些自然轨迹已将部分高阶耦合投影成可由成对非线性依赖辨识的模块关系；这不是说成对统计识别了不可约三体机制。跨时刻与同步相关、有限噪声下的间接作用，都可能使相似度树恰好匹配模块参考。本轮未分离这些因素。', '',
        'HO-LV的MI平均连接在完整训练集F1=1，四个12链子集均降为.5；Kuramoto两例的同一方法在四个子集仍均为1。因此可进一步研究采样覆盖与小样本稳定性，但须在同一自然训练数据上完成可信MLP-SPT后再比较，不能用旧SPT分数推断这一优势。', '',
        '若论文要论证普通依赖聚类的辨识边界，更有力的方向是预先构造／登记“不同高阶机制但低阶统计接近”的对照，并保留聚类成功与失败；不能只挑当前低分的距离或算法。当前结果适合说明相似性结构与干预机制层次是不同对象，尚不能证明MLP-SPT的恢复精度更高。', '',
        '## 方法来源、未运行项目与实现核查', '',
        '常用方法与现代稳健方法已执行，但不声称覆盖所有最前沿层次聚类。另核查NeurIPS 2025的SHiP官方实现：当前解释器／平台无可用binary wheel，源码构建在预设120秒上限停止并终止安装进程组。本轮不长时间编译、不将它记成F1=0。不能称已完成2025／2026最前沿方法的全面比较。', '']
    lines += [f'- [{name}]({url})' for name, url in data['protocol']['sources'].items()]
    lines += ['- [CKA / normalized kernel alignment](https://proceedings.mlr.press/v97/kornblith19a.html)']
    lines += ['', '仅在临时目录安装官方Genie/HDBSCAN及依赖，未升级项目NumPy。Genie用精确预计算距离避开其Euclidean加速路径的NumPy2接口要求。首次运行发现sklearn释放叶节点indices及相关矩阵约1e-15非对称舍入，已读取官方labels_恢复叶身份，并在明确对称误差检查后对距离取对称均值；未改变表示、超参数、评分或真值。原失败与修复原因保存在implementation_repair.json。', '',
        '5项独立软件检查通过：圆周共同旋转不变性及Euclidean等距降维、torus KSG非线性依赖与独立控制、DIANA／Ncut分离控制、官方二分k-means叶身份、模块F1与严格F1区别。图均已实际视检，图例／色条不覆盖数据；这些软件验证不替代方法泛化证据。', '',
        '本次经Zotero重新核对P6UJCVG8题名及全部附件，重新读取唯一主文DXGC7JEA（19页）Methods式（10）–（12）和SI MWIWKSVG（28页）S5.1式（S49）–（S50）。附件无明确稿件日期／版本，保留歧义。本轮不消费Syn，无新增Syn非负估计或容差放宽。', '',
        '复现：先在可用Python环境安装genieclust、deadwood、quitefastmst和hdbscan；本机使用临时路径`PYTHONPATH=/tmp/eisyn_hc_deps`，再以`.venv/bin/python scripts/spt_hierarchical_clustering_screen.py`执行。图表直接读summary.json：`.venv/bin/python scripts/plot_spt_hierarchical_clustering_screen.py`。全部原轨迹、旧实验结果及旧预算均未修改。']
    if (REPORT.parent / 'spt_hierarchical_clustering_literature_audit.md').exists():
        lines[2:2] = ['2026-10-16更新：当前主比较已精简为[三个文献定义代表](spt_hierarchical_clustering_literature_audit.md)，见[新图](assets/kuramoto_hierarchy/spt_hierarchical_clustering_reduced.png)。本页49组仅作为探索归档；KSG组合为适配，不是Song论文复现。所有原始分数保留。', '']
    REPORT.write_text('\n'.join(lines) + '\n')


def plot_literature_screened(data, sha):
    manifest_path = OUT / 'literature_screening.json'
    screening = json.loads(manifest_path.read_text())
    if screening['source_summary_sha256'] != sha or screening['selection_uses_f1']:
        raise ValueError('Literature selection must match unchanged source and exclude F1-based selection')
    if screening.get('current_paper_summary'):
        from scripts.spt_hierarchical_clustering_reduced import plot
        current = ROOT / screening['current_paper_summary']
        if hashlib.sha256(current.read_bytes()).hexdigest() != screening['current_paper_summary_sha256']:
            raise ValueError('Reduced paper-defined result changed since screening')
        return plot(json.loads(current.read_text()))
    names = {'single': 'Single', 'complete': 'Complete', 'average': 'Average', 'ward': 'Ward',
             'diana': 'DIANA', 'bisect': 'Bisecting k-means', 'ncut': 'Recursive Ncut (adapted)',
             'genie_0.3': 'Genie g=0.3', 'hdb_3': 'HDBSCAN hierarchy m=3'}
    fig = plt.figure(figsize=(11.6, 6.5), dpi=250, facecolor='white')
    cmap = LinearSegmentedColormap.from_list('recovery', COLORS)
    groups = [screening['main_ids'], screening['representation_control_ids']]
    rectangles = [[.244, .22, .225, .62], [.720, .282, .225, .558]]
    headings = ['Literature-supported representatives', 'Same algorithms on increments']
    for ids, rectangle, heading in zip(groups, rectangles, headings):
        ax = fig.add_axes(rectangle)
        vals = np.array([[value(c, key)['module_f1'] for c in data['records']] for key in ids])
        plotted = ax.imshow(vals, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        for i, key in enumerate(ids):
            for j, c in enumerate(data['records']):
                marker = '*' if stable(c, key) < 4 else ''
                ax.text(j, i, f'{vals[i,j]:.2f}{marker}', ha='center', va='center', fontsize=9,
                        color='white' if vals[i,j] > .7 else '#26333D')
        labels = [names[key.split('_', 1)[1]] if key != 'mi_average' else 'KSG MI + average (adapted)' for key in ids]
        ax.set_yticks(range(len(ids)), labels, fontsize=8.8)
        ax.set_xticks(range(3), ['Kuramoto\nA', 'HO-LV\nA', 'Kuramoto\nB'], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=6)
        ax.set_title(heading, fontsize=10, weight='bold', pad=39)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.965, .32, .009, .38])
    bar = fig.colorbar(plotted, cax=cax, ticks=[0, .5, 1])
    bar.set_label('Module F1', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.text(.55, .132, '* Scoring modules vary across training-chain subsets; no significance test.', ha='center', fontsize=8.8, color='#53616A')
    fig.text(.55, .084, 'Selection follows literature support and family coverage. All original 49 configurations remain available.', ha='center', fontsize=8.8, color='#53616A')
    fig.text(.55, .037, 'Domain/metric adaptations are documented; these are not literal paper replications. MLP-SPT is pending.', ha='center', fontsize=8.8, color='#53616A')
    dest = ASSETS / 'spt_hierarchical_clustering_literature_screened.png'
    fig.savefig(dest, dpi=250, bbox_inches='tight', pad_inches=.10, metadata={
        'Description': f'Literature-screened view; no new experiments or F1 selection; source SHA256 {sha}.',
        'LiteratureSelectionSHA256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    plt.close(fig)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--literature-screened', action='store_true', help='Only render the literature-audited view; preserve original 49-result outputs')
    args = parser.parse_args()
    source = OUT / 'summary.json'
    data = json.loads(source.read_text())
    if data['status'] != 'complete':
        raise ValueError('Completed screen required')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    ASSETS.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'svg.fonttype': 'none'}):
        if args.literature_screened:
            print(plot_literature_screened(data, sha))
            return
        figures = [plot_scores(data, sha), plot_trees(data, sha)]
    report(data, figures)
    for f in figures:
        print(f)
    print(REPORT)


if __name__ == '__main__':
    main()
