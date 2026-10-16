"""Plot the planted network, inferred full SPT and fixed-cut module F1."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import networkx as nx
from scripts.henon_fixed_lambda_module_recovery import FOLDER,FIG,MODULES,LAMBDA
from scripts.henon_high_order_comparison import save

COLORS=('#167A82','#BC5B27')


def read(p):return json.loads(Path(p).read_text())


def plot_network(ax,case):
    groups=MODULES[case];pos={};nodecolors=[]
    for j,group in enumerate(groups):
        center=-1.3 if j==0 else 1.3
        angles=np.pi/2+np.arange(len(group))*2*np.pi/len(group)
        for node,angle in zip(group,angles):pos[node]=(center+.65*np.cos(angle),.65*np.sin(angle))
        ax.add_patch(Ellipse((center,0),2.0,2.05,fc=COLORS[j],ec='none',alpha=.12))
        ax.add_patch(Ellipse((center,0),2.0,2.05,fc='none',ec=COLORS[j],lw=1.5))
        ax.text(center,-1.23,'{'+','.join(map(str,group))+'}',ha='center',va='top',color=COLORS[j],fontsize=10)
    graph=nx.Graph();graph.add_nodes_from(range(1,7))
    nx.draw_networkx_nodes(graph,pos,ax=ax,node_size=370,node_color='white',edgecolors='#425268',linewidths=1.1)
    nx.draw_networkx_labels(graph,pos,ax=ax,font_size=10)
    ax.set_xlim(-2.5,2.5);ax.set_ylim(-1.6,1.6);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(case+'   Planted '+('3 + 3' if case=='A' else '4 + 2')+' modules',loc='left',fontsize=11)


def plot_tree(ax,result,case):
    if result.get('status')!='complete':
        ax.text(.5,.5,'Numerically unresolved',transform=ax.transAxes,ha='center');ax.axis('off');return
    tree=result['spt']['tree'];positions={};nextleaf=[0]
    def place(node):
        if 'children' not in node:
            p=(nextleaf[0],0);nextleaf[0]+=1
        else:
            ps=[place(c) for c in node['children']]
            p=((ps[0][0]+ps[1][0])/2,max(v[1] for v in ps)+1)
        positions[node['mask']]=p;return p
    place(tree)
    def draw(node):
        x,y=positions[node['mask']]
        if 'children' not in node:
            color=COLORS[0] if node['members'][0] in MODULES[case][0] else COLORS[1]
            ax.scatter([x],[y],s=270,facecolor='white',edgecolor=color,lw=1.6,zorder=3)
            ax.text(x,y,str(node['members'][0]),ha='center',va='center',fontsize=10,zorder=4);return
        for child in node['children']:
            cx,cy=positions[child['mask']]
            ax.plot([cx,cx,x],[cy,y,y],color='#A2ACB7',lw=1,zorder=1);draw(child)
        match=next((j for j,g in enumerate(MODULES[case]) if set(g)==set(node['members'])),None)
        color='#708090' if match is None else COLORS[match]
        label='{'+','.join(map(str,node['members']))+'}'+chr(10)+f"{node['syn_nats']:.3f}"
        ax.text(x,y,label,ha='center',va='center',fontsize=8,zorder=5,
                bbox=dict(boxstyle='round,pad=.22',fc='white',ec=color,lw=1.4 if match is not None else .7))
    draw(tree)
    ax.set_xlim(-.5,5.5);ax.set_ylim(-.6,positions[63][1]+.6);ax.axis('off')
    ax.set_title(r'SPT, $\Xi=$'+f"{result['integrated_ei_nats']:.3f} nats",loc='left',fontsize=11)


def main():
    FIG.mkdir(parents=True,exist_ok=True);rows=[]
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.1),layout='constrained',gridspec_kw={'width_ratios':[1.05,1.55,.8]})
    for j,case in enumerate(MODULES):
        folder=FOLDER/f'{case}_lambda{LAMBDA:g}'
        core=read(folder/'metrics.json');ct=read(folder/'conditional_tm.json');score=read(folder/'module_f1.json')
        pcmci=read(folder/'pcmci.json') if (folder/'pcmci.json').exists() else {}
        plot_network(axes[j,0],case);plot_tree(axes[j,1],ct,case)
        ax=axes[j,2]
        if score.get('status')=='complete':
            values=[score[k] for k in ('precision','recall','f1')]
            bars=ax.bar(range(3),values,width=.62,color=['#94B7C6','#5B98AF','#167A82'])
            for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.3f}',ha='center',va='bottom',fontsize=10)
        else:ax.text(.5,.5,'Not scored',ha='center',transform=ax.transAxes)
        ax.set_xticks(range(3),['Precision','Recall','F1'],fontsize=9)
        ax.set_ylim(0,1.16);ax.set_yticks([0,.5,1]);ax.set_ylabel('Same-module node-pair score')
        valid=[v for v in score.get('numerical_replicates',[]) if v['status']=='valid']
        hits=sum(v['exact_root_partition'] for v in valid)
        ax.set_title('SPT module recovery'+chr(10)+f'Numerical roots: {hits}/{len(valid)} exact',loc='left',fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#DCE2E7',alpha=.5,lw=.6);ax.set_axisbelow(True)
        rows.append(dict(case=case,core=core,conditional=ct,score=score,pcmci=pcmci))
    fig.suptitle(r'Fixed $\lambda=0.05$; natural-data MLP; blind SPT root cut',fontsize=12)
    fig.savefig(FIG/'module_recovery.png',dpi=220,bbox_inches='tight')
    fig.savefig(FIG/'module_recovery.pdf',bbox_inches='tight');plt.close(fig)
    save(FOLDER/'summary.json',dict(lambda_value=LAMBDA,
        rows=[dict(d['score'],case=d['case']) for d in rows]))
    report(rows)
    print(json.dumps({d['case']:{k:d['score'].get(k) for k in ['f1','precision','recall','exact_module_set_f1','predicted_modules']} for d in rows},indent=2))


def report(rows):
    audit=read(FOLDER/'completion_audit.json') if (FOLDER/'completion_audit.json').exists() else None
    lines=['# 固定 λ=0.05：SPT 植入模块恢复与 F1','',
        '本轮将问题固定为：在接近零且非零的参数下，SPT 能否恢复预先植入的模块。λ 在计算前固定为 **0.05**，此后不扫描、不按恢复分数选择 λ、MLP、网络权重或切分。三源与四源案例均使用六节点系统。','',
        '![固定参数下的植入模块、完整 SPT 与 F1](../../fig/henon_fixed_lambda_module_recovery/module_recovery.png)','',
        '**图1。** 每行从左到右为真实模块、依据共同干预 EI 构建的完整 SPT，以及根二分的同模块节点对 Precision／Recall／F1。柱图取均值 EI 对应的一棵树；标题另列各次数值积分中精确恢复根模块的次数。叶子边框颜色表示真实模块，仅用于构树完成后的可视化。树的内部节点标注源集合和 raw Syn（nats）。三项得分的分母定义不同，不能把得分相同理解为额外独立证据。','',
        '## 固定动力学与植入结构','',
        '沿用[上轮有界 Hénon-style 自然动力学](henon_high_order_comparison.md)的全部系数、采样数量、MLP 架构和密度模型，只固定 λ 并将 B 改为可由树表示的非重叠模块：','',
        '| 网络 | 植入模块 | 多源乘积 |','|---|---|---|',
        '| A | {1,2,3}、{4,5,6} | 两个三源乘积模块 |',
        '| B | {1,2,3,4}、{5,6} | 四源乘积模块与二源乘积模块 |','',
        r'固定后 $\gamma=0.3+1.7(0.05)=0.385$、$\kappa=0.5-0.4(0.05)=0.48$。该点的交互系数接近原路径最大值 0.5；实际协同强度仍以测得的 $\Xi$ 和各节点 Syn 报告。四源乘积在均匀支持下的均方幅度较小，本轮不按跨网络总量大小评价阶数优劣。','',
        '原 B 的 {1,2,3,4} 与 {3,4,5,6} 相互重叠，单棵二叉树不能同时以两个完整分支表示它们，因此不适合作为“两个模块都应完整恢复”的主评测真值。原 B 和旧结果完整保留。本轮 B 的拓扑在拟合前固定，没有根据恢复结果修改。','',
        '每个条件生成 64 条自然链，burn-in 256 步，保留 256 步，共 16,384 对。按独立链固定分为训练／验证／测试 12,288／2,048／2,048 对。采样 seed=20261016，只有一个物理数据种子。SPT 与 SHAP 共用同一自然训练数据拟合并冻结的 6–64–64–6 SiLU MLP。WMS、MMI、SURD、Neural Granger 与 PCMCI 继续引用同一自然池和统一分区。MLP 选择只使用自然验证损失。','',
        r'SPT 的源为六维 $\mathbf{x}_t$，未来目标为同一完整六维 $\operatorname{atanh}(\mathbf{x}_{t+1})$，lag=1。独立均匀支持固定为 $(-1,1)^6$，残差 Gaussian 条件 TM 和所有边缘积分均复用该冻结 MLP。遍历全部 63 个非空子集，每个节点精确枚举二分，递归到单叶。真实模块从未作为切分约束输入。','',
        '## F1 的预先固定口径','',
        '真值和恢复结果均为覆盖六个节点的两个互斥模块。预测模块直接取 SPT 根的两个子节点，固定 K=2；不在完整树中搜索对真值最有利的切层。对全部 15 个无序节点对，同属一个模块记为阳性，跨模块记为阴性。','',
        '$$',r'\mathrm{Precision}=\frac{TP}{TP+FP},\qquad\mathrm{Recall}=\frac{TP}{TP+FN},\qquad F_1=\frac{2TP}{2TP+FP+FN}.','$$','',
        '这里的 F1 衡量模块成员关系，和模块编号无关。同时报告精确模块集合 F1：只有节点集合完全相同才算一个命中的模块。模块内部递归产生的二节点、三节点分支不是额外预测模块，不计为假阳性；因此不会把完整树的正常细分误罚成模块恢复错误。','',
        '主 F1 对四次数值积分的 EI 先取均值，再构建一棵 SPT 后评分；它不是四棵树 F1 的平均值。数值积分的根切分稳定性另列，不当作多个独立训练数据集。','',
        '| 网络 | 植入模块 | SPT 根切分 | TP / FP / FN / TN | Precision | Recall | 节点对 F1 | 精确模块集合 F1 |','|---|---|---|---|---:|---:|---:|---:|']
    for d in rows:
        r=d['score']
        if r.get('status')=='complete':
            lines.append(f"| {d['case']} | {r['true_modules']} | {r['predicted_modules']} | {r['tp']} / {r['fp']} / {r['fn']} / {r['tn']} | {r['precision']:.4f} | {r['recall']:.4f} | **{r['f1']:.4f}** | {r['exact_module_set_f1']:.4f} |")
        else:lines.append(f"| {d['case']} | {MODULES[d['case']]} | 数值检查未通过 | — | — | — | N/A | N/A |")
    lines+=['','## 协同、拟合和数值有效性','',
        r'| 网络 | 总 $\Xi$ / nats | 数值 SEM / nats | 自然平均响应 RMSE | 均匀干预平均响应 RMSE | 平均响应误差 / 高阶输入 RMS |','|---|---:|---:|---:|---:|---:|']
    for d in rows:
        ct=d['conditional'];diag=d['core']['diagnostics']
        lines.append(f"| {d['case']} | {ct['integrated_ei_nats']:.5f} | {ct['xi_mc_sem_nats']:.5f} | {diag['natural_test']['mean_function_rmse']:.5f} | {diag['uniform_diagnostic']['mean_function_rmse']:.5f} | {diag['uniform_diagnostic']['model_error_to_joint_signal_rms']:.3f} |")
    lines+=['','本轮模块 F1 只回答当前植入结构、固定参数和一个物理数据集上的恢复情况。它不认证每个节点 Syn 的物理数值准确，也不等于纯高阶 PID 原子识别。尤其当干预域模型误差与真实高阶信号相当时，结构吻合仍需与这一误差同时解释。真实两模块完全分离，真实根残差应为零；学得通道的非零根残差不能直接解释为真实跨模块耦合。','',
        '初始采用四个数值 scramble，各 outer=1024、inner=256。初始单次根切分不完全一致，因此保持网络、λ、自然数据、MLP 和切分目标不变，将外层点加密到 4096、内层保持 256；加密判断仅依据数值根切分一致性，与植入标签无关，初始结果完整保留。数值 scramble 不是独立训练种子。对每个独立 scramble，只有通过原生非负性检查后才给出其诊断 F1，失败项为 N/A，不能记为 0。','',
        '| 网络 | 最小 raw 值 / nats | 非负容差 / nats | 容差带负值数 | 显著违例数 | 数值重复的诊断 F1 |','|---|---:|---:|---:|---:|---|']
    for d in rows:
        r=d['score'];ct=d['conditional'];syn_audit=ct.get('spt',{}).get('audit')
        if syn_audit:
            rep=[f"{v['f1']:.3f}" if v['f1'] is not None else 'N/A（非负性未通过）' for v in r.get('numerical_replicates',[])]
            lines.append(f"| {d['case']} | {syn_audit['minimum_nats']:.6f} | 0.01 | {syn_audit['tolerance_band_negative_count']} | {syn_audit['significant_violation_count']} | {', '.join(rep)} |")
    lines+=['','| 网络 | 初始 outer=1024 的主 F1 | 当前 outer 点数 | 当前主 F1 | 当前数值根切分精确恢复次数 | 内层 128→256 最大 EI 变化 / nats |',
        '|---|---:|---:|---:|---:|---:|']
    for d in rows:
        r=d['score'];ct=d['conditional'];folder=FOLDER/f"{d['case']}_lambda{LAMBDA:g}"
        coarse=read(folder/'module_f1_n1024.json')
        valid=[v for v in r.get('numerical_replicates',[]) if v['status']=='valid']
        hits=sum(v['exact_root_partition'] for v in valid)
        lines.append(f"| {d['case']} | {coarse['f1']:.4f} | {ct.get('outer',ct['replicates'][0]['diagnostic']['outer'])} | {r['f1']:.4f} | {hits} / {len(valid)} 个有效重复 | {ct['inner_prefix_mean_maximum_ei_change_nats']:.6f} |")
    lines+=['','主表与主图始终使用当前加密结果，没有在初始与加密结果之间按 F1 择优。数值根切分精确恢复次数是积分敏感性检查，不是跨物理种子的成功率。','',
        'raw Syn 没有裁剪或投影。低于 −0.01 nats 的显著违例会使对应结果失败；容差内负原值仍完整保留。','']
    if audit:
        stability='；'.join(f"{d['case']} 为 {audit['cases'][d['case']]['exact_numerical_root_recoveries']}/{audit['cases'][d['case']]['valid_numerical_replicates']}" for d in rows)
        gaps='；'.join(f"{d['case']} 为 {audit['cases'][d['case']]['root_objective_gap_to_second_cut_nats']:.5f} nats" for d in rows)
        lines += [f'加密后的单次数值根切分精确恢复次数：{stability}。均值 EI 下，最优根切分与次优切分的目标值差：{gaps}。B 的切分优势很小，仍有数值敏感性，因此不能将均值树 F1=1 表述为稳定、普遍的四源恢复。','']
        lines+=['### 分模块的外推诊断','',
            '以下在模型和 SPT 均冻结后，用原先固定的均匀诊断点逐模块核对平均响应。B 的二源模块较强，因此全网络误差／信号比可能掩盖四源模块的相对误差；这张表用于解释结构 F1 的适用范围。','',
            '| 网络 | 模块 | 平均响应 RMSE | 真实乘积输入 RMS | 误差／信号 RMS |','|---|---|---:|---:|---:|']
        for d in rows:
            for fit in audit['cases'][d['case']]['module_fit']:
                lines.append(f"| {d['case']} | {fit['members']} | {fit['mean_function_rmse']:.5f} | {fit['joint_signal_rms']:.5f} | {fit['error_to_signal_rms']:.3f} |")
        lines+=['','精确模块恢复与平均响应拟合精度分别报告；F1=1 不能抵消四源通道的拟合误差，也不能据此认证四源节点 Syn 的精确数值。','']
    lines+=[
        '## 固定参数下的其他对比读数','',
        '以下保留原请求的七法比较口径。SHAP 为同一冻结 MLP 的三／四阶广义 interaction 幅度汇总，其他算法均使用相同自然池。观测基线的高维特征估计偏差沿用上轮已识别的限制，这些原生数值不等于模块 F1。','',
        '| 网络 | SHAP 三／四阶 | WMS / nats | 分组 MMI / nats | SURD / nats | Neural Granger | PCMCI |','|---|---:|---:|---:|---:|---:|---:|']
    for d in rows:
        c=d['core'];n=c['natural_information'];pc=d['pcmci']
        vals=[c['shap']['high_order_interaction'],n['wms_nats'],n['mmi_grouped_nats'],n['surd_total_synergy_nats'],c['neural_granger']['cross_mean'],pc.get('cross_mean')]
        lines.append('| '+d['case']+' | '+' | '.join('未完成' if v is None else f'{v:.5f}' for v in vals)+' |')
    lines+=['','## 方法依据与复现','',
        '本轮重新通过 Zotero 核验父项 P6UJCVG8 的题名 *Emergent hierarchical organization of causal interactions in complex systems*，重新读取唯一主文 DXGC7JEA、SI MWIWKSVG 全文相关位置：正文 Methods（5）–（12），SI S1.2、S5.1／Algorithm S1、S6.4。附件仍无明确稿件日期／版本；2026-10-04 的元数据时间不能确立稿件版本。固定目标、共同干预、精确二分、原始 Syn 和单叶停止据此核对。自然训练 MLP、条件残差 TM 和本轮模块 F1 是明确的实验适配。','',
        '脚本：`scripts/henon_fixed_lambda_module_recovery.py`；绘图／报告：`scripts/plot_henon_fixed_lambda_module_recovery.py`。结果、轨迹 NPZ、冻结模型、所有 EI 表和 F1 计数位于 `results/henon_fixed_lambda_module_recovery/`。旧 λ 扫描缓存、旧重叠 B 网络和此前 45 分钟预算均未覆盖或重置。', '']
    if audit:
        lines += ['## 完成核验','',
            f"两网络全部七法计算及积分加密在 {audit['computation_elapsed_seconds']/60:.1f} 分钟内完成。训练／验证／测试链与数组摘要、SHAP 背景和评价行、SPT／SHAP 冻结模型摘要及评分树摘要均核验通过。每棵树含五个内部节点，闭合误差低于 $10^{{-9}}$ nats。13 项针对性测试通过，其中新增 5 项覆盖模块评分，另 8 项核对共用计算。单个物理数据种子的结果不能建立总体恢复率。",'']
    if all(d['score'].get('status')=='complete' for d in rows):
        values='；'.join(f"{d['case']} 的节点对 F1={d['score']['f1']:.4f}、精确模块集合 F1={d['score']['exact_module_set_f1']:.4f}" for d in rows)
        caveat=f'数值积分单次精确恢复：{stability}；B 仍有切分敏感性。' if audit else ''
        lines[4:4]=[f'**当前均值 EI 所构建 SPT 的恢复结果：** {values}。{caveat}该分数对应一个物理数据种子；分模块拟合误差另列。','']
    (ROOT/'docs/reports/henon_fixed_lambda_module_recovery.md').write_text('\n'.join(lines))


if __name__=='__main__':
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':.8}):main()
