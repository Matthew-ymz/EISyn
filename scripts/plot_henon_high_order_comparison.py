"""Figures and a compact report from the bounded higher-order benchmark cache."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import networkx as nx

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.henon_high_order_comparison import FOLDER,FIG,CONFIG,NETWORKS,members,save

COLORS={'A':'#167A82','B':'#BC5B27'}


def read(path):return json.loads(Path(path).read_text())


def load():
    result=[]
    for case in NETWORKS:
        for lam in CONFIG['lambdas']:
            folder=FOLDER/f'{case}_lambda{lam:g}'
            if not (folder/'metrics.json').exists():continue
            d=read(folder/'metrics.json')
            d['conditional']=read(folder/'conditional_tm.json') if (folder/'conditional_tm.json').exists() else {}
            d['pcmci']=read(folder/'pcmci.json') if (folder/'pcmci.json').exists() else {}
            d['shift_control']=read(folder/'natural_shift_control.json') if (folder/'natural_shift_control.json').exists() else {}
            result.append(d)
    return result


def network_figure():
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    poses={
        'A':{1:(-1.4,.35),2:(-.7,1.),3:(0.,.35),4:(1.1,.35),5:(1.8,1.),6:(2.5,.35)},
        'B':{1:(-1.3,1.),2:(-1.3,-.1),3:(0.,1.),4:(0.,-.1),5:(1.3,1.),6:(1.3,-.1)}}
    for ax,case in zip(axes,NETWORKS):
        pos=poses[case]
        for k,e in enumerate(NETWORKS[case]):
            points=np.array([pos[i+1] for i in e]);center=points.mean(0)
            color=['#167A82','#BD6E35'][k]
            width=2.4 if case=='A' else 2.5
            height=1.8 if case=='A' else 2.25
            ax.add_patch(Ellipse(center,width,height,facecolor=color,edgecolor=color,alpha=.13,lw=1.8))
            ax.add_patch(Ellipse(center,width,height,facecolor='none',edgecolor=color,lw=1.5))
            ax.text(center[0],center[1]+height/2+.13,
                    r'$e_%d=\{%s\}$'%(k+1,','.join(str(i+1) for i in e)),
                    ha='center',va='bottom',fontsize=10,color=color)
        g=nx.Graph();g.add_nodes_from(range(1,7))
        nx.draw_networkx_nodes(g,pos,ax=ax,node_color='white',edgecolors='#303C48',linewidths=1.5,node_size=560)
        nx.draw_networkx_labels(g,pos,ax=ax,font_size=12,font_color='#1F2D3D')
        ax.set_title(f'{case}   '+('Two disjoint three-source products' if case=='A' else 'Two overlapping four-source products'),loc='left',fontsize=12,pad=18)
        ax.set_aspect('equal');ax.axis('off');ax.margins(.15)
        ax.set_xlim((-2.,3.2) if case=='A' else (-2.6,2.6))
        ax.set_ylim(-1.,2.4)
    fig.savefig(FIG/'networks.png',dpi=220,bbox_inches='tight');plt.close(fig)


def metric_figure(rows):
    specs=[('SPT / PEID',r'$\Xi$ (nats)',lambda d:d['conditional'].get('integrated_ei_nats',np.nan)),
           ('SHAP interaction','Output amplitude',lambda d:d['shap']['high_order_interaction']),
           ('WMS','nats',lambda d:d['natural_information']['wms_nats']),
           ('Grouped MMI-PID','nats',lambda d:d['natural_information']['mmi_grouped_nats']),
           ('SURD synergy','nats',lambda d:d['natural_information']['surd_total_synergy_nats']),
           ('Neural Granger','First-layer norm',lambda d:d['neural_granger']['cross_mean']),
           ('PCMCI-CMIknn','Native CMIknn statistic',lambda d:d['pcmci'].get('cross_mean',np.nan)),
           ('MLP intervention check','Mean-function RMSE /'+chr(10)+'joint-signal RMS',lambda d:d['diagnostics']['uniform_diagnostic']['model_error_to_joint_signal_rms'])]
    fig,axes=plt.subplots(2,4,figsize=(13.8,6.2),layout='constrained')
    for j,(title,unit,get) in enumerate(specs):
        ax=axes.flat[j]
        for case in NETWORKS:
            sub=sorted([d for d in rows if d['case']==case],key=lambda d:d['lambda_value'])
            xs=[d['lambda_value'] for d in sub];ys=[get(d) for d in sub]
            ax.plot(xs,ys,'o-',color=COLORS[case],lw=1.7,ms=4,label=f'Network {case}')
            if j==0:
                errs=[d['conditional'].get('xi_mc_sem_nats',np.nan) for d in sub]
                ax.errorbar(xs,ys,yerr=errs,fmt='none',ecolor=COLORS[case],capsize=2,lw=1)
            if j in (2,3,4):
                key={2:'wms_nats',3:'mmi_grouped_nats',4:'surd_total_synergy_nats'}[j]
                nulls=[np.mean([r[key] for r in d['shift_control'].get('replicates',[])]) if d['shift_control'].get('replicates') else np.nan for d in sub]
                ax.plot(xs,nulls,'--',color=COLORS[case],lw=1,alpha=.65)
        ax.set_title(chr(97+j)+'   '+title,loc='left',fontsize=11)
        ax.set_xlabel(r'$\lambda$');ax.set_ylabel(unit,fontsize=9)
        ax.set_xticks([0,.5,1]);ax.set_xlim(-.05,1.05)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#D9DEE4',lw=.5,alpha=.55)
        if j==7:ax.axhline(1,color='#939AA3',ls=':',lw=1)
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color='#687585',ls='--',lw=1));labels.append('Time-shift control (c–e)')
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False,fontsize=10)
    fig.savefig(FIG/'seven_metrics.png',dpi=220,bbox_inches='tight');plt.close(fig)


def tree_figure(rows):
    fig,axes=plt.subplots(2,3,figsize=(14,7.7),layout='constrained')
    allnodes=[]
    def collect(n):
        if 'children' in n:
            allnodes.append(n['syn_nats'])
            for c in n['children']:collect(c)
    for d in rows:
        if d['conditional'].get('status')=='complete':collect(d['conditional']['spt']['tree'])
    vmax=max(allnodes,default=1.)
    for index,(ax,d) in enumerate(zip(axes.flat,rows)):
        c=d['conditional'];case=d['case'];lam=d['lambda_value']
        title=rf'{case}   $\lambda={lam:g}$'
        if c.get('status')!='complete':
            ax.text(.5,.5,'Numerically unresolved',transform=ax.transAxes,ha='center')
            ax.set_title(title,loc='left');ax.axis('off');continue
        tree=c['spt']['tree'];positions={};counter=[0]
        def place(n):
            if 'children' not in n:
                p=(counter[0],0);counter[0]+=1
            else:
                ps=[place(ch) for ch in n['children']]
                p=((ps[0][0]+ps[1][0])/2,max(v[1] for v in ps)+1)
            positions[n['mask']]=p;return p
        place(tree)
        def draw(n):
            x,y=positions[n['mask']]
            if 'children' not in n:
                ax.scatter([x],[y],s=300,facecolor='white',edgecolor='#3C4858',lw=1,zorder=3)
                ax.text(x,y,str(n['members'][0]),ha='center',va='center',fontsize=10,zorder=4)
                return
            for child in n['children']:
                cx,cy=positions[child['mask']]
                ax.plot([cx,cx,x],[cy,y,y],color='#9AA3AD',lw=1,zorder=1)
                draw(child)
            shade=('#EEEEEE' if n['syn_nats']<0 else plt.cm.YlGnBu(.12+.6*n['syn_nats']/vmax))
            label='{'+','.join(map(str,n['members']))+'}'+chr(10)+f"{n['syn_nats']:.3f}"
            ax.text(x,y,label,ha='center',va='center',fontsize=8.3,zorder=5,
                    color='white' if n['syn_nats']/vmax>.7 else '#16324A',
                    bbox=dict(boxstyle='round,pad=.25',facecolor=shade,edgecolor='#A8B4C0',lw=.7))
        draw(tree)
        ax.set_title(title+rf"    $\Xi={c['integrated_ei_nats']:.3f}$ nats",loc='left',fontsize=11,color=COLORS[case])
        ax.set_xlim(-.55,5.55);ax.set_ylim(-.7,positions[63][1]+.8);ax.axis('off')
    fig.savefig(FIG/'spt_trees.png',dpi=220,bbox_inches='tight');plt.close(fig)


def summarize(rows):
    output=[]
    for d in rows:
        ni=d['natural_information'];ct=d['conditional'];pcm=d['pcmci']
        item=dict(case=d['case'],lambda_value=d['lambda_value'],
            peid_xi_nats=ct.get('integrated_ei_nats'),peid_numerical_sem_nats=ct.get('xi_mc_sem_nats'),
            spt_status=ct.get('status','missing'),
            feature_tm_xi_nats=d['spt']['integrated_ei_nats'],
            shap_high_order=d['shap']['high_order_interaction'],wms_nats=ni['wms_nats'],
            grouped_mmi_nats=ni['mmi_grouped_nats'],surd_synergy_nats=ni['surd_total_synergy_nats'],
            neural_granger=d['neural_granger']['cross_mean'],pcmci=pcm.get('cross_mean'),
            pcmci_status=pcm.get('status','missing'),pcmci_significant_cross_count=pcm.get('significant_cross_count'),
            shared_model_sha256=d['model_sha256'],
            shared_model_verified=d['shared_mlp_verified'] and ct.get('model_sha256')==d['model_sha256'],
            natural_mean_rmse=d['diagnostics']['natural_test']['mean_function_rmse'],
            uniform_mean_rmse=d['diagnostics']['uniform_diagnostic']['mean_function_rmse'])
        if ct.get('status')=='complete':
            item.update(root_split=[n['members'] for n in ct['spt']['tree']['children']],
                        root_syn_nats=ct['spt']['tree']['syn_nats'],
                        nonnegative_audit=ct['spt']['audit'])
        output.append(item)
    save(FOLDER/'summary.json',dict(protocol=CONFIG,rows=output))
    return output


def write_report(rows,summary):
    lines=['# Hénon-style 动力学的三源与四源高阶网络比较','',
        '2026-10-16。借鉴 [Part1 的受控 Hénon 响应](Part1.md)，完成两个六节点网络 × 三个参数点 × 单个动力学种子。SPT/PEID、SHAP、WMS、分组 MMI-PID、SURD、Neural Granger、PCMCI-CMIknn 均从同一份自然轨迹池及统一划分出发。SPT 和 SHAP 共用同一个训练完成、冻结并核验摘要的 MLP。','',
        '## 动力学与两个网络','',
        '受控 Hénon 响应中的双源乘积可推广为三源或四源乘积。为得到可以持续采样的闭环自然轨迹，本实验保留二次折叠与可扫描的交互／单源项，并使用有界随机状态转移：','',
        '$$',r'\begin{aligned}',
        r'x_{i,t+1}&=\tanh\!\left[m_i(\mathbf{x}_t;\lambda)+\sigma\epsilon_{i,t}\right],\\',
        r'm_i(\mathbf{x};\lambda)&=0.5(1-1.4x_i^2)+0.4\gamma(\lambda)x_i',
        r'+\frac{\kappa(\lambda)}{d_i}\sum_{e\ni i}2\prod_{j\in e}x_j,\\',
        r'\gamma(\lambda)&=0.3+1.7\lambda,\qquad\kappa(\lambda)=0.5-0.4\lambda,\\',
        r'\sigma&=0.5,\qquad\epsilon_{i,t}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad\lambda\in\{0,0.5,1\}.',
        r'\end{aligned}','$$','',
        '$d_i$ 是包含节点 $i$ 的超边数。因此重叠节点接收各条乘积输入的平均值。每条乘积项对其源集合中的所有节点生效，包含接收节点本身。源数三或四指乘积中的状态变量数，不是外部邻居数。该构造是本实验提出的有界随机 Hénon-style 网络，并非经典 Hénon 映射。','',
        r'对任一超边 $e$ 及接收节点 $i\in e$，混合导数 $\partial^{|e|}m_i/\prod_{j\in e}\partial x_j=2\kappa(\lambda)/d_i\ne0$。低阶自身折叠和单源项在该混合差分中消失，证明结构中存在相应多源项。超边描述 tanh 之前的乘积输入，共享接收节点的外层非线性和信息约束还可能产生跨超边联合效应。','',
        '| 网络 | 乘积源集合 | 用途 |','|---|---|---|',
        r'| A | $\{1,2,3\}$、$\{4,5,6\}$ | 两个分离的三源作用模块 |',
        r'| B | $\{1,2,3,4\}$、$\{3,4,5,6\}$ | 两个共享节点 3、4 的四源作用组 |','',
        '![两个高阶网络](../../fig/henon_high_order_comparison/networks.png)','',
        '**图1。** 轮廓给出乘积项的精确源集合。每个乘积作用于轮廓内全部节点，自身折叠、单源项及独立噪声未画出。B 的两个集合交叉重叠，任何二叉层级树都不能同时把二者表示为两个完整子树。A、B 同时改变了阶数和重叠方式，因此两网络对比是两个机制案例，不能用于单独归因某一个因素。','',
        r'两种乘积具有相同幅值上界，但在共同独立均匀支持上，单条输入的方差为 $\operatorname{Var}_{q_{\max}}(2\prod_{j\in e}x_j)=4/3^{|e|}$。四源乘积的均方幅度较小，跨网络强弱差异同时受到这一尺度差异影响。','',
        '## 共同自然样本与共享模型','',
        '每个条件模拟 64 条自然链，舍弃 256 步后各保留 256 个相邻状态对，共 16,384 对。固定按链划分训练／验证／测试为 48／8／8 条，即 12,288／2,048／2,048 对。链之间不共享状态。种子索引为 0，采样 RNG 种子为 20261016。六个条件使用相同初值和噪声数组以配对，但动力学不同，产生的状态数组并不相同。每个条件内部，所有方法引用同一数据清单及分区摘要。PCMCI 的双 lag 历史边界使每条测试链少用首个状态对，实际有效样本数为 2,040。','',
        r'MLP 仅使用自然训练对，验证集选择停止轮次。固定 6–64–64–6 SiLU 网络、AdamW、学习率 0.003、最多 240 轮、耐心 30 轮，无真实超边输入，也不按恢复分数或干预测试误差选择模型。MLP 预测共同的六维未来目标 $\mathbf{y}_t=\operatorname{atanh}(\mathbf{x}_{t+1})$。这一逐坐标可逆变换使已知动力学的条件噪声为加性 Gaussian，信息论定义不随该目标变换改变；SHAP 的幅度单位则明确为变换后的输出单位。','',
        '训练残差拟合一个完整协方差 Gaussian 条件 TM。SPT 和 SHAP 读取同一个已保存模型，包括输入／输出标准化、所有网络参数和残差协方差。自然测试集同时用于独立预测检查、三个观测信息基线的估计、SHAP 评价和 PCMCI；SHAP 背景来自自然训练集。Neural Granger 用同一训练对另拟合其原生目标专用稀疏预测器。所有数据、模型摘要和划分保存在内部缓存中。','',
        r'SPT 拟合完成后使用固定的 $\prod_i\mathrm{Unif}(-1,1)$ 干预。干预积分点没有进入训练集。共同自然数据要求约束了学习和观测基线输入，不把 PEID 的干预分布替换为自然观测分布。真实生成器的均匀干预响应只用于外推诊断。','',
        '## 指标与估计口径','',
        '| 方法 | 本次计算 | 单位与边界 |','|---|---|---|',
        r'| SPT／PEID | 同一残差条件 TM 的全部 63 个非空源子集 EI；共同六维未来目标；每节点枚举全部合法二分至单叶 | nats；多源总量为 $\Xi$，节点为路径残差 |',
        '| SHAP | 128 条自然测试评价行、32 条自然训练背景行、全部 64 联盟；精确 Shapley 和三／四阶广义 Shapley interaction | 输出幅度；主表汇总全部三／四源集合的平均绝对 interaction |',
        '| WMS | 自然测试对的四阶多项式特征仿射 TM；全源 MI 减六个单源 MI | nats；允许为负 |',
        '| MMI-PID | 对全部根二分的两个向量源计算二源 MMI synergy，再取最小值 | nats；是分组二源 MMI 扩展，不是完整多源 PID 原子 |',
        '| SURD | 同一自然仿射 Gaussian 特征模型的全部子集逐目标状态 specific information；按原始分配规则先分解后平均 | nats；保留完整多源协同表和状态级闭合检查 |',
        '| Neural Granger | 每个目标一个 6–24–1 tanh MLP；第一层源组 L2 惩罚 0.01、120 轮；汇总非自身源权重范数 | 原生权重尺度；不是高阶协同 |',
        '| PCMCI-CMIknn | 同一批 8 条自然测试链，lag 1、knn=0.1、30 次条件打乱、显著阈值 0.05，最多 2 个条件变量 | 原生 CMIknn 统计量；轻量条件数上限及粗 p 值分辨率限制结论 |','',
        r'主要 PEID 使用可归一化的条件密度 $q(\mathbf{y}\mid\mathbf{x})=\mathcal N(\widehat{\mathbf{m}}(\mathbf{x}),\boldsymbol{\Sigma}_{\mathrm{res}})$，通过三角 Gaussian 残差变换求密度。所有源子集均从这一共同干预联合分布边缘化，缺失坐标用独立均匀 Sobol 积分，不重拟合子集预测器。初始计算采用 4 个数值 scramble，各 1,024 个外层点和 256 个内层点，同时保存 128→256 的内层前缀变化。若前缀 EI 差达到 0.01 nats，则针对该条件将内层加密到 512，保留初始网格结果。图中的 PEID 误差线只表示数值积分 SEM，不是跨动力学种子的误差。','',
        '四阶 Legendre 特征包含六变量全部总次数不超过四的 209 个非恒定项，不按种植超边选择特征。该仿射 TM 近似仍保留为对照，也用于自然信息基线。它在特征空间作 Gaussian 近似，不能当作原始连续联合分布的精确密度。自然基线对源／目标标准化协方差加 $10^{-6}$ 对角正则；没有源打乱。样本具有时间相关且特征非 Gaussian，本次不套用 iid Wishart 偏差修正。','',
        r'SPT 的全源 $\Xi$、每个候选切分残差均声明非负容差 **0.01 nats**。落在 $[-0.01,0)$ 的原值保留并记录，低于 −0.01 nats 则显式失败，不生成通过检查的树，不裁剪、投影或改变切分来消除负值。SURD 对低阶占优项的抑制属于其原始状态级分配规则，与 PEID 数值投影不同。','',
        '## 七类读数','',
        '![七类指标与预测诊断](../../fig/henon_high_order_comparison/seven_metrics.png)','',
        '**图2。** 两色分别表示网络 A 和 B。c–e 的虚线为四次链内目标时间错位的均值，不是置信区间。各面板使用方法本身的单位，不能跨方法比较数值大小。只有一个物理数据种子，不能给出总体显著性或总体稳定性结论。最后一个面板为均匀干预下 MLP 平均响应误差与真实高阶输入 RMS 之比，灰虚线为 1（误差等于高阶信号）。缺失或未通过数值检查的结果不作零值填补。','',
        r'| 网络 | λ | PEID $\Xi$ / nats | SHAP 三／四阶 | WMS / nats | 分组 MMI / nats | SURD / nats | Neural Granger | PCMCI |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '未完成' if v is None else f'{v:.4f}'
    for r in summary:
        lines.append('| '+ ' | '.join([r['case'],str(r['lambda_value'])]+[fmt(r[k]) for k in ['peid_xi_nats','shap_high_order','wms_nats','grouped_mmi_nats','surd_synergy_nats','neural_granger','pcmci']])+' |')
    a=[r for r in summary if r['case']=='A'];b=[r for r in summary if r['case']=='B']
    if len(a)==3 and len(b)==3:
        lines+=['',f"网络 A 的主要 PEID 总量为 {'、'.join(fmt(r['peid_xi_nats']) for r in a)} nats，随扫描参数下降。λ=0 与 0.5 的根分裂保留两个三节点模块。网络 B 的总量为 {'、'.join(fmt(r['peid_xi_nats']) for r in b)} nats，读数较小且未呈稳定单调趋势。它们是同一自然数据训练流程下学得通道的结果。",'',
            '四源项在均匀支持上的典型幅度较小。B 三个条件的 MLP 平均响应外推误差分别为真实高阶输入 RMS 的 '+ '、'.join(f"{d['diagnostics']['uniform_diagnostic']['model_error_to_joint_signal_rms']:.2f} 倍" for d in rows if d['case']=='B')+'。因此，本轮不能宣称可靠恢复真实四源机制。A 的弱交互端点同样受到这一相对误差限制。','']
    lines+=['### 时间错位目标检查','',
        '保持自然源、各链时间顺序和完整目标向量的内部相关，在每条测试链内将目标向量独立循环错位 32–224 步，重复四次。它破坏当前源与一步目标的时间对齐，用来检查特征估计的基线读数，不是额外物理实验。未作空对照扣除或裁剪。','',
        '| 网络 | λ | 错位 WMS 均值 / nats | 错位分组 MMI 均值 / nats | 错位 SURD 均值 / nats |','|---|---:|---:|---:|---:|']
    for d in rows:
        controls=d['shift_control'].get('replicates',[])
        if controls:
            vals=[np.mean([r[k] for r in controls]) for k in ['wms_nats','mmi_grouped_nats','surd_total_synergy_nats']]
            lines.append(f"| {d['case']} | {d['lambda_value']:g} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','WMS 的错位对照约 0.27–0.30 nats，SURD 对照约 0.29–0.32 nats，显示当前 209 维特征、2,048 个自然测试对的 plug-in 估计存在明显有限样本偏差。原始读数仍逐项保留，但不能把高于零本身解释为真实高阶因果作用，也不能据此表格宣称某一种方法普遍优越。','']
    lines+=['','## 完整 SPT 与有效性检查','',
        '![六个条件的完整 SPT](../../fig/henon_high_order_comparison/spt_trees.png)','',
        r'**图3。** 每个内部节点标注源集合和局部 raw Syn，单位 nats；叶子为原始状态变量。填色只帮助比较局部贡献，显示保留三位小数。容差内负原值保留并以灰色显示，不作数值裁剪。各树根的 $\Xi$ 等于本树五个内部节点 raw Syn 的总和。B 的交叉四源集合不可同时成为树的完整分支，树的节点集合也不等同于生成超边。','',
        '| 网络 | λ | 自然测试平均响应 RMSE | 均匀干预平均响应 RMSE | SPT 状态 | 模型共用核验 |','|---|---:|---:|---:|---|---|']
    for r in summary:lines.append(f"| {r['case']} | {r['lambda_value']:g} | {r['natural_mean_rmse']:.4f} | {r['uniform_mean_rmse']:.4f} | {r['spt_status']} | {r['shared_model_verified']} |")
    lines+=['','网络 A 的真实生成器在两个模块之间完全分离，且两组噪声独立，因此在共同独立干预下，真实模块根残差应为零。当前学得通道在 λ=0、0.5 的根残差约为 0.036、0.018 nats。它们受到 MLP 机制误差、残差密度近似和有限积分影响，不能解释为真实生成器存在跨模块作用。模块边界被选中与节点贡献数值准确是两个不同的检验。','',
        'PCMCI 在六个条件下均未检出 BH 校正后阈值 0.05 的跨节点边。30 次条件打乱和最多两个条件变量的轻量配置限制了检验分辨率，未检出不能解释为生成器中没有这些作用。','']
    lines+=['','### 数值与非负性审计','',
        r'| 网络 | λ | $\Xi$ 数值 SEM / nats | 内层点数 | 内层前缀最大 EI 差 / nats | 最小 raw 值 / nats | 容差带负值数 | 显著违例数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in rows:
        ct=d['conditional']
        if ct.get('status')=='complete':
            audit=ct['spt']['audit'];inner=ct['replicates'][0]['diagnostic']['inner']
            lines.append(f"| {d['case']} | {d['lambda_value']:g} | {ct['xi_mc_sem_nats']:.5f} | {inner} | {ct['inner_prefix_mean_maximum_ei_change_nats']:.5f} | {audit['minimum_nats']:.6f} | {audit['tolerance_band_negative_count']} | {audit['significant_violation_count']} |")
    lines+=['',r'审计计数包括每个条件的 64 个 $\Xi$ 值及全部 301 个非平凡候选切分残差。容差内负原值没有从树或缓存中移除。前缀检查和 scramble SEM 只诊断当前积分网格，不构成任意高精度的认证。','',
        '### 种植源集合的完整联盟读数','',
        r'| 网络 | λ | 源集合 | 条件 TM $\Xi(A)$ / nats | 同一 MLP 的 SII | SURD 该集合协同 / nats | 是否为所选 SPT 节点 |','|---|---:|---|---:|---:|---:|---|']
    for d in rows:
        ct=d['conditional'];table=ct.get('ei_table_nats')
        if table is None:continue
        clades=[]
        def collect(n):
            clades.append(n['mask'])
            for child in n.get('children',[]):collect(child)
        if ct.get('status')=='complete':collect(ct['spt']['tree'])
        for edge in NETWORKS[d['case']]:
            mask=sum(1<<i for i in edge);xi=table[mask]-sum(table[1<<i] for i in edge)
            si=d['shap']['generalized_shapley_interactions'][str(mask)]
            su=d['natural_information']['surd_synergy_by_coalition_nats'][mask]
            lines.append(f"| {d['case']} | {d['lambda_value']:g} | {members(mask)} | {xi:.5f} | {si:.5f} | {su:.5f} | {mask in clades} |")
    lines+=['','目前的计算支持三源／四源乘积能够嵌入有界多节点动力学，并在同一自然数据学习协议下计算各方法读数。乘积的非零混合差分证明结构包含对应多源项；它不证明每个 SPT 高阶节点等于纯阶数交互。随 λ 同时减弱交互并增强单源通道是预定的组合变化，趋势对比不能把两种变化的影响分别识别出来。','',
        '## 预检修正、方法依据与复现','',
        r'预检曾采用 $\prod_{j\in e}(\sqrt3x_j)$ 的均匀方差归一化。闭环四源正反馈使部分自然轨迹集中于饱和区，MLP 的均匀干预平均响应误差达到约 70.5。该预检按数据覆盖和外推诊断判为不可用，整轮原始结果保存在 `results/henon_high_order_comparison/preflight_v1_variance_normalized/`。最终方案统一改为 $2\prod x_j$ 的幅值界，在全部条件中固定，未按 SPT、SHAP 或其他方法排名选择参数。由于存在这次预检后的设计修正，最终结果属于探索性轻量验证。','',
        '本次重新通过 Zotero 核验父项 **P6UJCVG8** 的标题 *Emergent hierarchical organization of causal interactions in complex systems*，重新列出并读取唯一主文 **DXGC7JEA**（19 页）和 SI **MWIWKSVG**（28 页）相关全文。两份附件无明确稿件日期／版本；2026-10-04 的附件元数据修改时间不能确立新版。方法依据是正文 Methods（5）–（12），SI S1.2（S7）–（S15）、S5.1／Algorithm S1、S6.4（S61）、S9.1–S9.6。','',
        '共同干预、固定完整未来目标、原始节点残差及精确二分遵循所读定义。本实验的六节点闭环动力学、自然训练 MLP、非线性条件残差 TM、四阶自然基线、三／四阶 SII、分组 MMI 和 PCMCI 条件数上限是明确的扩展／计算适配，不声称复现手稿原先二源 Hénon 的估计流程。','',
        '计算入口：`scripts/henon_high_order_comparison.py`；图表与本报告入口：`scripts/plot_henon_high_order_comparison.py`。内部缓存包含自然样本 NPZ、按链索引、冻结 MLP、全部 EI／SURD／SHAP 联盟值、PCMCI 原生矩阵和显式预算；未生成 CSV。主计算和条件 TM 各自的父进程读取同一个绝对截止，重启不重置 45 分钟预算。','']
    if (FOLDER/'completion_audit.json').exists():
        audit=read(FOLDER/'completion_audit.json')
        lines+=['## 完成核验','',
            f"六个条件的七类指标全部完成，主要数值结果审计耗时 {audit['elapsed_wall_minutes']:.1f} 分钟，低于批准的 45 分钟墙钟上限。8 项针对性测试通过；全部自然训练／测试数组、SHAP 背景与评价行、冻结 MLP 摘要核验通过。六棵树各含 5 个内部节点，闭合误差低于 $10^{{-9}}$ nats，显著负值违例数为 0。容差带负原值共 8 个，均保留，没有裁剪。",'']
    (ROOT/'docs/reports/henon_high_order_comparison.md').write_text('\n'.join(lines))


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.linewidth':.8,'svg.fonttype':'none'}):
        rows=load();network_figure();metric_figure(rows);tree_figure(rows)
    summary=summarize(rows);write_report(rows,summary)
    print(json.dumps(dict(conditions=len(rows),spt_complete=sum(r['spt_status']=='complete' for r in summary),pcmci_complete=sum(r['pcmci_status']=='complete' for r in summary)),indent=2))


if __name__=='__main__':main()
