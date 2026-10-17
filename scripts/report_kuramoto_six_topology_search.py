"""Render all search outcomes, model amendments, and actual candidate trees."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.kuramoto_six_topology_core import configuration
from scripts.run_mixed_order_kuramoto_kout_main import NETWORK_POSITIONS
from scripts.synergy_hierarchy_tree_plot import _blend_with_white, EDGE_COLOR, LEAF_FACE_COLOR, LEAF_EDGE_COLOR
from scripts.compare_kuramoto_short_trajectory_clustering import TITLES,group_string
from scripts.run_kuramoto_six_topology_search import HIERARCHICAL

BASE=ROOT/'results/kuramoto_six_topology_search'
LOG=ROOT/'docs/log/kuramoto_six_topology_search'
REPORT=ROOT/'docs/reports/kuramoto_six_topology_search.md'
ASSET=ROOT/'docs/reports/assets/kuramoto_hierarchy/kuramoto_six_topology_search.png'
ARMS=[('', 'Node Fourier, N=8192'),('generic_fourier','All-edge Fourier, N=8192'),('generic_fourier_x4','All-edge Fourier, N=32768')]
GREEN,RED,GRAY,BLUE,ORANGE='#267A70','#B6534B','#AEB7C0','#4477A8','#B86722'


def load(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def candidate_result(folder):
    final=load(folder/'spt.json')
    if final:return final
    levels=load(folder/'spt_levels.json',[])
    if not levels:return None
    return dict(levels[-1],numerically_resolved=False,incomplete=True)


def root_ari(row,cfg):
    supports=[set(s) for s in cfg['supports']]
    maximal=[s for s in supports if not any(s<t for t in supports)]
    labels=np.empty(6,dtype=int);predicted=np.empty(6,dtype=int)
    for i,group in enumerate(maximal):labels[np.array(sorted(group))-1]=i
    for i,group in enumerate(row['groups']):predicted[np.array(group)-1]=i
    return float(adjusted_rand_score(labels,predicted))


def best_methods(baseline,cfg):
    selected={}
    for name in baseline['views']['increment']['methods']:
        rows=[(v,baseline['views'][v]['methods'][name]) for v in ('increment','transition')]
        if name in HIERARCHICAL:
            view,row=max(rows,key=lambda pair:pair[1]['recovery']['module_f1'])
        else:view,row=max(rows,key=lambda pair:root_ari(pair[1],cfg))
        selected[name]=(view,row)
    return selected


def draw_truth(ax,cfg):
    # Reuse the exact six-node coordinates of the original Kuramoto panel.
    positions={i:NETWORK_POSITIONS[f'theta{i}'] for i in range(1,7)}
    for support,weight in sorted(zip(cfg['supports'],cfg['weights']),key=lambda a:-len(a[0])):
        p=np.array([positions[i] for i in support]);size=len(support)
        if size==4:
            # Membership polygon includes node 3 in its interior, as a hyperedge
            # may contain a node that is not a vertex of its geometric hull.
            hull=ConvexHull(p)
            ax.add_patch(Polygon(p[hull.vertices],closed=True,facecolor=to_rgba(ORANGE,.13),
                                 edgecolor=ORANGE,lw=1.8,ls='--',zorder=0))
        elif size==3:
            ax.add_patch(Polygon(p,closed=True,facecolor=to_rgba(BLUE,.22),edgecolor=BLUE,lw=2,zorder=1))
        else:ax.plot(p[:,0],p[:,1],color=ORANGE,lw=3.3,solid_capstyle='round',zorder=2)
    for i,(x,y) in positions.items():
        left=i<=4
        ax.scatter([x],[y],s=390,facecolors='#DCEAF3' if left else '#F4DDC5',
                   edgecolors='#355F7A' if left else '#A95E24',lw=1.5,zorder=4)
        ax.text(x,y,str(i),ha='center',va='center',fontsize=12,color='#24313C',zorder=5)
    ax.text(-.68,.91,'4-body {1,2,3,4}',ha='center',fontsize=9,color=ORANGE)
    ax.text(-.99,-.90,'3-body {1,2,3}',ha='center',fontsize=9,color='#355F7A')
    ax.text(1.12,-.90,'2-body {5,6}',ha='center',fontsize=9,color=ORANGE)
    ax.set(xlim=(-1.55,1.55),ylim=(-1.10,1.12),aspect='equal');ax.axis('off')
    ax.set_anchor('N')
    ax.text(.5,-.075,r'$\kappa_{123}=\kappa_{56}=0.567;\quad\kappa_{1234}=1.134$',
            transform=ax.transAxes,ha='center',fontsize=9)


def draw_tree(ax,tree,cfg,syn=False):
    truth=[set(s) for s in cfg['supports']]
    primitives=[s for s in truth if not any(t<s for t in truth)]
    positions={};counter=[0];nodes=[]
    def locate(node,depth=0):
        nodes.append(node)
        if not node['children']:x=counter[0];counter[0]+=1
        else:x=np.mean([locate(c,depth+1) for c in node['children']])
        positions[tuple(node['sources'])]=(x,-depth)
        return x
    locate(tree)
    total=sum(n.get('syn_nats',0.) for n in nodes) if syn else None
    maximum=max((n.get('syn_nats',0.) for n in nodes),default=1.) if syn else 1.
    if syn and (total<=0 or maximum<=0):raise ValueError('Positive total Syn required for percentage display')
    def draw(node):
        x,y=positions[tuple(node['sources'])];s=set(node['sources']);internal=bool(node['children'])
        for child in node['children']:
            cx,cy=positions[tuple(child['sources'])]
            ax.plot([x,cx],[y,cy],color=EDGE_COLOR,lw=1.25,zorder=1)
            draw(child)
        label=','.join(map(str,node['sources']))
        edge,face,lw=LEAF_EDGE_COLOR,LEAF_FACE_COLOR,1.0
        if internal:
            label='{'+label+'}'
            if syn:
                strength=abs(node['syn_nats'])/maximum
                edge=GREEN;face=_blend_with_white(GREEN,.10+.52*strength);lw=1.1+1.1*strength
                label+=f"\nSyn ≈{100*node['syn_nats']/total:.1f}%"
            else:
                ignored=any(s<p for p in primitives)
                edge=GREEN if s in truth else GRAY if len(s)==6 or ignored else RED
                face=_blend_with_white(edge,.10);lw=1.4 if s in truth else 1.
        ax.text(x,y,label,ha='center',va='center',fontsize=9.0 if internal else 11,
                linespacing=1.12,color='#24313C',zorder=3,
                bbox=dict(boxstyle='round,pad=.30',facecolor=face,edgecolor=edge,linewidth=lw))
    draw(tree)
    depth=max(-y for x,y in positions.values())
    ax.set(xlim=(-.6,5.6),ylim=(-depth-.40,.40));ax.axis('off')
    if syn:
        ax.text(.5,1.03,rf'$\Xi\approx {total:.3f}$ nats; estimated shares',transform=ax.transAxes,
                ha='center',fontsize=9,color='#24313C')


def draw_flat_comparison(ax,methods,cfg):
    for y,name in ((3.2,'kmeans'),(.9,'spectral')):
        view,row=methods[name]
        ax.text(2.5,y+1.05,f"{TITLES[name]}: root ARI {root_ari(row,cfg):.3f}",ha='center',fontsize=10)
        ax.text(2.5,y+.60,'increment' if view=='increment' else 'initial + increment',ha='center',fontsize=8.5,color='#65717A')
        for x,group in zip((1.,4.),row['groups']):
            edge=GREEN if group in ([1,2,3,4],[5,6]) else RED
            ax.text(x,y,'{'+','.join(map(str,group))+'}',ha='center',va='center',fontsize=11,
                    bbox=dict(boxstyle='round,pad=.4',facecolor=LEAF_FACE_COLOR,edgecolor=edge,lw=1.3))
    ax.text(2.5,-.1,'Native two-group partitions',ha='center',fontsize=9,color='#65717A')
    ax.set(xlim=(-.6,5.6),ylim=(-.5,5.0));ax.axis('off')


def collect():
    initial=load(BASE/'screen_summary.json',[])
    arms=[]
    for folder,label in ARMS:
        directory=BASE/folder
        summaries=load(directory/'screen_summary.json',[])
        arms.append(dict(folder=folder,label=label,screens=summaries,
                         discoveries=load(directory/'discovery_results.json',[]),
                         validations=load(directory/'validation_results.json',[])))
    return initial,arms


def plot(initial,arms):
    latest=next((a for a in reversed(arms) if candidate_result(BASE/a['folder']/'B_2_seed0')),None)
    latest=latest or next((a for a in reversed(arms) if any(s['id']=='B_2' for s in a['screens'])),arms[0])
    folder=BASE/latest['folder']/'B_2_seed0';spt=candidate_result(folder)
    baseline=load(folder/'baselines_raw.json') or load(BASE/'B_2_seed0/baselines_raw.json')
    cfg=configuration('B',2);methods=best_methods(baseline,cfg)
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10}):
        fig,axes=plt.subplots(2,4,figsize=(18,10.2),layout='constrained')
        draw_truth(axes[0,0],cfg);axes[0,0].set_title('a  Nested Kuramoto topology',loc='left',fontsize=12,pad=32)
        if spt and spt.get('tree'):
            draw_tree(axes[0,1],spt['tree'],cfg,True)
            axes[0,1].text(.5,-.075,f"Module F1 {spt['recovery']['module_f1']:.3f}",transform=axes[0,1].transAxes,ha='center',fontsize=10)
        else:axes[0,1].axis('off')
        axes[0,1].set_title('b  SPT: exploratory candidate',loc='left',fontsize=12,pad=32)
        names=('ward','correlation_hac','dcor_hac','mi_hac','mic')
        cells=((0,2),(0,3),(1,0),(1,1),(1,2))
        for letter,name,cell in zip('cdefg',names,cells):
            ax=axes[cell];view,row=methods[name];draw_tree(ax,row['tree'],cfg)
            ax.set_title(f"{letter}  {TITLES[name]}",loc='left',fontsize=12,pad=32)
            ax.text(.5,1.025,'increment' if view=='increment' else 'initial + increment',transform=ax.transAxes,ha='center',fontsize=9,color='#65717A')
            ax.text(.5,-.075,f"Module F1 {row['recovery']['module_f1']:.3f}",transform=ax.transAxes,ha='center',fontsize=10)
        draw_flat_comparison(axes[1,3],methods,cfg)
        axes[1,3].set_title('h  K-means and spectral clustering',loc='left',fontsize=12,pad=32)
        fig.legend(handles=[Line2D([],[],color=GREEN,lw=2,label='Baseline outline: planted support'),
                            Line2D([],[],color=RED,lw=2,label='Baseline outline: unsupported merge'),
                            Line2D([],[],color=GRAY,lw=2,label='Baseline outline: unscored refinement / root')],
                   loc='outside lower center',ncols=3,frameon=False,fontsize=10,
                   title='Single data seed; Syn shares are estimates and integration precision is not certified.')
        ASSET.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(ASSET,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
    return latest


def stability_explanation(folder):
    audit=load(folder/'candidate_stability_audit.json')
    if not audit:return []
    lines=['## 复核：“待确认”具体指什么','',
           '2026-10-17按用户要求复核，使用现有逐样本EI缓存；没有重训MLP、增加物理数据、运行新积分或改动原验收条件。重新通过Zotero核对父项P6UJCVG8及唯一正文DXGC7JEA（19/19页）、补充MWIWKSVG（28/28页），读取Methods(5)–(12)、SI S5和S7。附件仍没有明确稿件版本或日期，元数据编辑不消除歧义。','',
           '**可以确认的事实**是：两档平均EI得到的树都含123、1234、56；两档各两次独立scramble也都恢复这三个模块，模块F1均为1。完整二叉树的123内部配对会改变，因此不能说全部二叉细节相同。这个结果适合做探索性候选展示。','',
           '**尚不能确认的部分**包括：积分是否已接近当前冻结MLP通道的精确SPT、Syn强度及其百分比是否足够准确、换物理数据种子后优势是否仍存在。数字指标没有通过原预设门槛，不能靠改一个状态标签消除这些限制。','',
           '重要性积分对每个外层样本、每个源子集，用512个内层补全样本估计边缘密度。其归一化积分权重记为wᵢ，ESS=1/Σᵢwᵢ²。这里的权重不是神经网络权重、不是拓扑边权，也不是训练样本权重。最差一次积分的ESS约4.46，最大wᵢ约0.466，表示该局部密度估计中一个补全样本占约46.6%的加权总和；不表示整个实验只有4.46个有效物理样本。少量样本占主导会使边缘密度和其对数较敏感，提示方差或尚未排除的积分偏差。','',
           '最大跨档EI变化为0.05445 nats、两scramble最大EI SEM为0.01796 nats，均高于0.01 nats门槛。SEM只衡量两次数值随机化的差异，不覆盖共同偏差。Syn是三个EI之差；SPT则比较不同子组EI之和。共同误差有时会抵消，所以绝对EI未收敛不自动意味着模块一定分错；但小的候选差距也可能被误差改变，必须直接检查切分。','',
           '本次发现一个实质敏感点：最高档scramble1880的1234节点，123｜4仅领先第二名0.000229 nats。对该次积分的2048个Sobol点分16块、每次删去128点后，只有6/16次仍恢复全部模块；另一次scramble为16/16。两次平均后，123｜4领先第二名0.01188 nats；对平均结果做下列检查，恢复保持不变。','',
           '| 缓存扰动检查 | 全部模块保留 | 全部通过非负审计 | 123｜4最小领先差 / nats |',
           '|---|---:|---:|---:|']
    for row in audit['summary']:
        if row['scope']=='individual':label=f"单次scramble {row['scramble']}，各删1/16点"
        elif row['scope']=='combined_single_scramble_deleted':label='两次平均，每次仅从其中一次删1/16点'
        else:label='两次平均，同时各删对应1/16点'
        lines.append(f"| {label} | {row['full_module_recovery_count']}/{row['count']} | {row['valid_count']}/{row['count']} | {row['minimum_123_retention_margin_nats']:.6f} |")
    lines += ['', '表中负领先差表示另一切分排名在前，不是负Syn。这些删块结果有重叠，Sobol点也不是独立抽样；它们是数值敏感性检查，不是成功概率、独立重复或置信区间，更不能排除两次积分共有的边缘积分偏差。没有据此把原`numerically_resolved=False`或`qualified=0`改为通过。','',
              'Syn强度同样应保留估计口径：123节点由中档约0.08296变到高档0.06743 nats，变化约19%；1234节点由0.08517变到0.09354 nats。故本图适合展示模块层级的候选差异，不能用小数点后的Syn值、精确百分比或纯三/四阶信息原子支持强定量结论。','',
              '**建议展示口径**：“在同一短轨迹数据集及固定学习通道下，SPT的平均估计树恢复嵌套模块123⊂1234和56，模块F1=1；既定层级基线最优F1=0.667。该结果为单数据种子的探索性候选，模块恢复在平均估计的缓存扰动检查中保持稳定，Syn强度及数值收敛仍待确认。”','',
              '复核入口：`scripts/audit_kuramoto_candidate_stability.py`；缓存中保留逐次切分、最优/次优差距、全部删块结果、原生0.01 nats容差的非负审计和输入缓存SHA256。','']
    return lines


def report():
    initial,arms=collect();latest=plot(initial,arms);budget=load(BASE/'budget.json');exit_record=load(BASE/'exit.json',{})
    folder=BASE/latest['folder']/'B_2_seed0';candidate=candidate_result(folder)
    if candidate and candidate.get('meets_search_target'):
        outcome='开发种子找到达到预设差距的候选；是否可复现仍以独立新种子及数值检查为准。'
    elif candidate and candidate.get('tree'):
        outcome=f"当前缓存平均树恢复全部植入模块（模块F1={candidate['recovery']['module_f1']:.3f}），可作为单数据种子的探索性候选展示。原数值准入仍未通过，不能据此宣称Syn强度已收敛、真实通道SPT已获确认或跨数据种子的稳定优势。"
    elif candidate:
        outcome='本轮尚未找到合格案例：完整六维EI/SPT未通过有效性审计，不报告恢复优势。'
    else:outcome='本轮尚无可报告的完整SPT结果，不能确认恢复优势。'
    lines=['# 六节点 Kuramoto：嵌套拓扑搜索','',
           outcome,'',
           '2026-10-17。用户批准60分钟共享上限；全部计算使用持久化绝对截止，无子代理。目标是探索性寻找同数据下SPT较强、既定基线较弱的案例。', '',
           '![嵌套拓扑与统一风格的SPT和聚类树](assets/kuramoto_hierarchy/kuramoto_six_topology_search.png)','',
           '图沿用原Kuramoto例子的六节点坐标、蓝/橙节点、灰色斜连线及圆角节点框。SPT由两scramble平均EI表构树；节点值为局部Syn占全树Ξ的估计百分比，填色随该估计值变化。数字含“≈”，不代表定量收敛。基线节点框标记真实支持（绿）、错误跨模块合并（红）及未评分细分/共同根（灰）；层级基线取预定两视图中模块F1较高者，平面聚类取根切ARI较高者。树的深度是二叉拓扑排版，不是动力学时间或聚类距离。四体填色多边形包含内部节点3，表示{1,2,3,4}，没有增加成对边。', '',
           *stability_explanation(folder),
           '## 首轮全部候选','',
           '原方案12条登记中A族的三种父/子比例等价，合并为10个唯一动力学配置。所有初态独立Uniform(-pi,pi)^6，预测间隔0.2、dt=0.01、噪声0.08；按独立起始轨迹划分训练/验证/测试6144/1024/1024。每配置所有方法共用同一数组摘要，SPT的MLP从该数据拟合。','',
           '| 配置 | 最强层级基线F1 | MLP独立均值NRMSE | 拟合门槛 |','|---|---:|---:|---|']
    for r in initial:
        if 'diagnostics' not in r:lines.append(f"| {r['id']} | — | — | {r.get('status')} |");continue
        d=r['diagnostics'];lines.append(f"| {r['id']} | {r['best_baseline_f1']:.3f} | {d['independent_mean_nrmse']:.3f} | {'通过' if d['fit_eligible'] else '未通过'} |")
    lines += ['', '“最强基线”取Ward、绝对相关HAC、距离相关HAC、成对MI-HAC及组联合MIC，在增量和初态＋增量两种视图上的最大模块F1。K-means和谱聚类仅有原生两团，不强行补成层级。首轮所有MLP均未过预设0.15均值NRMSE门槛，因此未用其偶然树分数宣称成功。','',
              '## 显式记录的拟合修正','',
              '在计算任何SPT恢复分数之前，增加一个统一表示臂：枚举全部可能的二、三、四节点对称Kuramoto Fourier项，所有候选共用，不传入真实超边列表；网络仍为两层128单元SiLU。它含更强的动力学家族先验，不能把整条路线的改善全部归因于SPT。原36维失败记录未覆盖。','',
              '统一表示在8192条样本上仍未过拟合门槛后，按用户允许共同采样设计的范围追加一次四倍扩样，保持旧训练/验证/测试归属，所有方法读取相同新增训练数据。未按SPT的F1选择表示或样本量；两次修正均先登记，之后才计算SPT。只继续A拟合对照及唯一原始最强基线低于1的B_2。','',
              '| 表示与数据臂 | 配置 | 独立均值NRMSE | 联合95%覆盖 | 拟合门槛 |','|---|---|---:|---:|---|']
    for arm in arms[1:]:
        for r in arm['screens']:
            d=r['diagnostics'];lines.append(f"| {arm['label']} | {r['id']} | {d['independent_mean_nrmse']:.3f} | {d['joint_95_coverage']:.3f} | {'通过' if d['fit_eligible'] else '未通过'} |")
    lines += ['', '均值参考来自未用于拟合的128个独立初态、每点128次带噪转移；有限重复的SEM保留，不扣除或裁剪参考误差。均值NRMSE与Gaussian覆盖仅是工程准入，不证明完整转移密度准确。残差Gaussian保留完整6×6协方差，仍假设状态无关；这不是精确SDE条件分布。']
    state_audit=load(folder/'residual_state_audit.json')
    if state_audit:
        coverage=[b['joint95coverage'] for b in state_audit['phase_bins']]
        lines += ['', f"事后留出残差审计（未用于选模型、改变门槛或构树）：残差及平方残差与36个节点周期特征的最大绝对相关分别{state_audit['max_residual_feature_correlation']:.3f}和{state_audit['max_squared_residual_feature_correlation']:.3f}；每节点初态四等宽相位箱的联合95%覆盖范围为{min(coverage):.3f}–{max(coverage):.3f}。这些描述量不能排除更高阶、非Gaussian或状态相关误差。重算入口为`residual_state_diagnostics`，数据摘要与主实验相同。"]
    lines += ['',
              '## 嵌套候选的逐方法比较','',
              'B_2的真实支持为123、1234、56，耦合系数依次0.56695、1.13389、0.56695。这里的三体/四体按超边节点数定义。以下层级F1评价完整模块恢复；根切ARI仅评价外层1234｜56，不能相互替代。','',
              '| 方法 | 增量模块F1 | 初态＋增量模块F1 | 增量根切ARI | 初态＋增量根切ARI |',
              '|---|---:|---:|---:|---:|']
    baseline=load(folder/'baselines_raw.json')
    cfg=configuration('B',2)
    if baseline:
        for name in baseline['views']['increment']['methods']:
            rows=[baseline['views'][v]['methods'][name] for v in ('increment','transition')]
            f1=[f"{r['recovery']['module_f1']:.3f}" if r.get('recovery') else '—' for r in rows]
            ari=[f"{root_ari(r,cfg):.3f}" for r in rows]
            label=TITLES[name].replace('|',r'\|')
            lines.append(f"| {label} | {f1[0]} | {f1[1]} | {ari[0]} | {ari[1]} |")
    lines += ['', '在当前共享扩样数据上，谱聚类可恢复外层两团；MI-HAC和组联合MIC也可恢复1234及56，但遗漏内层123。因此这个候选比较的是嵌套子模块恢复，不能表述为这些算法完全无法分团。SPT的切分Syn也不等于纯三阶/四阶PID原子；支持恢复和作用阶数强度分解是两个问题。','',
              '## 完整六维SPT结果','']
    for arm in arms:
        for r in arm['discoveries']:
            s=r['result'];score=s.get('recovery',{}).get('module_f1')
            lines += [f"- {arm['label']} / {r['id']}: 状态={s['status']}；模块F1={score}；严格clade F1={s.get('recovery',{}).get('strict_f1')}；数值稳定={s.get('numerically_resolved')}；最强原数据基线={s['best_baseline_f1']:.3f}；差距={s.get('gap')}。"]
            if s.get('audit'):lines += [f"  非负审计：{json.dumps(s['audit'],ensure_ascii=False)}。"]
            lines += [f"  最大EI SEM={s.get('maximum_ei_sem_nats')} nats；跨档最大变化={s.get('grid_change_nats')} nats；最小ESS={s.get('minimum_ess')}；最大权重={s.get('maximum_weight')}。"]
            model_base=load(BASE/arm['folder']/f"{r['id']}_seed0/baselines_model.json")
            if model_base:lines += [f"  同冻结MLP读出上的最强基线F1={model_base['best_hierarchical_f1']:.3f}；该对照用以区分模型去噪与构树差异。"]
    if not any(a['discoveries'] for a in arms):
        lines += ['模型已通过拟合准入，但完整积分协议尚未完成，不能确认SPT恢复优势。' if candidate else '尚无可报告的完整EI/SPT结果，不能确认恢复优势。']
    levels=load(folder/'spt_levels.json',[])
    if levels:
        lines += ['', '| 外层N / 内层P（各两scramble） | 诊断模块F1 | 最小Syn / nats | 显著违规数 | 最大EI SEM / nats | 跨档EI变化 / nats | 最小ESS | 最大权重 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for s in levels:
            f1=s.get('recovery',{}).get('module_f1');audit=s['audit'];delta=s.get('grid_change_nats')
            lines.append(f"| {s['outer']} / {s['particles']} | {f1 if f1 is not None else '—'} | {audit['minimum_syn_nats']:.5f} | {audit['violation_count']} | {s['maximum_ei_sem_nats']:.5f} | {f'{delta:.5f}' if delta is not None else '—'} | {s['minimum_ess']:.2f} | {s['maximum_weight']:.3f} |")
        lines += ['', '所有档位完整保留；末档作为结果，未按最高F1挑档。数值准入同时要求两scramble最大EI SEM与跨档变化均小于0.01 nats、最差ESS至少10、最大归一化权重至多0.25、独立scramble模块树一致且非负审计通过。两scramble间的SEM是有限随机化精度诊断，不是跨数据种子的置信区间。']
    model_base=load(folder/'baselines_model.json')
    if model_base:
        lines += ['',f"冻结MLP额外生成的4096条共同读出上，最强层级基线F1={model_base['best_hierarchical_f1']:.3f}。所有聚类在此对照中获得相同生成数组；它与24576条真实训练数据的主比较分开报告。该对照含数据量差异，只能辅助识别模型读出效应，不能单独证明全部增益来自构树。"]
    if exit_record and candidate and not candidate.get('meets_search_target'):
        lines += ['', '本轮在既定候选及数值档位范围内结束，保留有希望的诊断案例；由于开发种子未通过全部准入条件，没有启动新数据种子确认。不放宽门槛来追求成功。后续若继续，应先改善完整联合边缘积分并验证精度，再冻结同一拓扑、表示、数据量及算法，使用全新种子确认。']
    lines += ['', '## 新数据种子验证','', '| seed | SPT模块F1 | 最强基线F1 | 数值稳定 | 状态 |','|---:|---:|---:|---|---|']
    validation_count=0
    for arm in arms:
        for r in arm['validations']:
            s=r['result'];score=s.get('recovery',{}).get('module_f1');validation_count+=1
            b=load(BASE/arm['folder']/f"B_2_seed{r['seed']}/baselines_raw.json",{})
            lines.append(f"| {r['seed']} | {score if score is not None else '—'} | {b.get('best_hierarchical_f1','—')} | {s.get('numerically_resolved',False)} | {s['status']} |")
    if not validation_count:lines += ['| — | — | — | — | 未进入独立验证 |','','未完成独立新数据种子验证，不能宣称跨数据种子稳定。']
    lines += ['', '## 方法与可复现性','',
              '- 四个支持族固定为A:123/456；B:123⊂1234及56；C:12⊂123⊂1234及56；D:12、34⊂1234及56。父/子强度比0.5、1、2，全系统平均耦合RMS固定0.75，频率固定。',
              '- 原生高阶块内部没有唯一二叉真值，主模块F1忽略其内部细分，错误跨块合并仍计入；严格F1和各支持召回完整保留。',
              '- 完整未来相位作为固定六维目标；所有源子集共享同一因子化干预及同一已学习联合。63项EI、301项二分、单叶终止，无源分量拆分或真值限制切分。',
              '- 本实验使用展开相位变量的未来状态；圆周取模未来、展开未来及相位增量是不同目标，不能假定EI相同。',
              '- Syn容差固定0.01 nats，容差内负估计计数记录，显著违规阻断构树。完整Gaussian解析基准及均匀先验卷积基准已经核验；有限重要性积分仍需独立scramble和跨档检查，不能以通过非负审计代替收敛。',
              '- 当前Zotero父项P6UJCVG8，主文DXGC7JEA、补充MWIWKSVG，重新列举并读取Methods(5)–(12)、SI S5和S7。当前唯一附件组无明确稿件版本日期。相比旧图，本轮改用完整展开未来相位、全联合条件Gaussian TM、通用全超边表示；不是旧图同协议复现。',
              '- 来源入口：`scripts/run_kuramoto_six_topology_search.py`、`scripts/run_kuramoto_six_topology_generic_fit.py`、`scripts/kuramoto_six_topology_core.py`。基线复用既有七方法实现。内部结果位于`results/kuramoto_six_topology_search/`；所有旧臂和失败记录保留，无CSV。',
              '- 11项相关软件/数值测试通过，包含两种MLP表示的解析Jacobian、共同扩样的原划分保留、解析通道EI积分、显著负Syn阻断和模块评分；软件测试不能替代上述非线性通道收敛检查。',
              '- [聚类算法官方文档](https://scikit-learn.org/stable/modules/clustering.html)；[MIC原文](https://arxiv.org/abs/q-bio/0311037)。', '',
              f"本轮计算累计 {(exit_record.get('finished_epoch',time.time())-budget['started_epoch'])/60:.2f} 分钟；执行退出记录：{json.dumps(exit_record,ensure_ascii=False)}。",'']
    REPORT.write_text('\n'.join(lines))
    (LOG/'tuning_report.md').write_text('\n'.join(lines))
    (LOG/'leaderboard.md').write_text('\n'.join(lines[lines.index('| 配置 | 最强层级基线F1 | MLP独立均值NRMSE | 拟合门槛 |'):lines.index('## 显式记录的拟合修正')]))
    return REPORT


if __name__=='__main__':print(report())
