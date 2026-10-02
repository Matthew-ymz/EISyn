#!/usr/bin/env python3
"""Three evidence-led pilot figures, cost estimates and concise result report."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.dmf_response_benchmark import LABELS, METHODS
BASE=ROOT/'results/dmf_schaefer100/response_benchmark_pilot'
FIG=ROOT/'fig/dmf_response_benchmark'
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#3C3C3C',
        '#7570B3','#8C6D31','#66A61E','#999999','#B3B3B3']
SHORT=['ROI-block Xi','Pairwise Phi-R','Weakest split Phi-R','Phi-R-LU scalar','Dynamic WMS',
       'Target-augmented O','Whole MI','Return SC','Internal SC','Rest FC','Initial response (tie)','Random expectation']


def panel(ax, letter):
    ax.text(-.10,1.05,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
    ax.spines[['top','right']].set_visible(False)


def save(fig,name):
    for ext in ('png','svg','pdf'):
        fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def group_times(records,method,k,layer):
    return [r for r in records if r['method']==method and r['k']==k and r['layer']==layer and r['state']=='complete']


def stats(records,field):
    v=np.array([r[field] for r in records])
    return np.quantile(v,[.25,.5,.75])


def mechanisms(contract,data,summary):
    fig,axes=plt.subplots(2,4,figsize=(15.2,6.7),layout='constrained',gridspec_kw={'width_ratios':[1.15,1.15,1.,1.05]})
    t=np.arange(301)
    for si,site in enumerate(contract['sites']):
        with np.load(BASE/f'response_site{site}.npz') as a:
            rms=a['rms']; rates=a['rates']; loss=a['loss']; early=a['early']; late=a['late']
        ax=axes[si,0]; panel(ax,chr(97+si*4))
        candidates=rms[1:].mean(1)
        ax.plot(t,rms[0].mean(0),color='#222222',lw=1.8,label='Intact pulse − sham')
        ax.fill_between(t,rms[0].min(0),rms[0].max(0),color='#222222',alpha=.12)
        ax.plot(t,candidates.mean(0),color=COLORS[0],lw=1.5,label='Weakened pulse − sham')
        ax.fill_between(t,candidates.min(0),candidates.max(0),color=COLORS[0],alpha=.15)
        ax.axvline(10,color='.6',lw=.8,ls='--')
        ax.axvspan(50,300,color='.8',alpha=.13,zorder=-1)
        ax.set(xlabel='Time from pulse onset (ms)',ylabel='Whole-brain evoked RMS (Hz)',xlim=(0,300))
        ax.text(.97,.95,f'Site {site+1} · G = 1.3',ha='right',va='top',transform=ax.transAxes,fontsize=9)
        if si==0:
            ax.legend(loc='lower left',bbox_to_anchor=(0,1.06),frameon=False,fontsize=8)
        ax=axes[si,1]; panel(ax,chr(98+si*4))
        ax.plot(t[50:],rms[0].mean(0)[50:]*1000,color='#222222',lw=1.8)
        ax.plot(t[50:],candidates.mean(0)[50:]*1000,color=COLORS[0],lw=1.5)
        ax.fill_between(t[50:],candidates.min(0)[50:]*1000,candidates.max(0)[50:]*1000,color=COLORS[0],alpha=.15)
        ax.set(xlabel='Late-window time (ms)',ylabel='Evoked RMS (10⁻³ Hz)',xlim=(50,300))
        ax=axes[si,2]; panel(ax,chr(99+si*4))
        v=loss.mean(1)*1e6; sem=loss.std(1,ddof=1)/np.sqrt(2)*1e6
        ax.errorbar(range(1,9),v,yerr=sem,fmt='o',color='#3C3C3C',capsize=2,ms=4)
        for ri in range(2):
            ax.scatter(np.arange(1,9)+(-.12 if ri==0 else .12),loss[:,ri]*1e6,s=12,alpha=.45,color=COLORS[0])
        ax.axhline(0,lw=.7,color='.7')
        ax.set(xlabel='Frozen candidate index',ylabel='Late response loss (10⁻⁶ Hz·s)',xticks=range(1,9))
        ax=axes[si,3]; panel(ax,chr(100+si*4))
        sham_shift=np.sqrt(np.mean((rates[1:,1]-rates[0,1])**2,axis=(1,3)))
        ax.plot(t,sham_shift.mean(0),color=COLORS[4],label='Sham background shift',lw=1.4)
        ax.fill_between(t,sham_shift.min(0),sham_shift.max(0),color=COLORS[4],alpha=.15)
        ax.set(xlabel='Time from pulse onset (ms)',ylabel='Sham drift RMS (Hz)',xlim=(0,300))
        ax.axvline(10,color='.6',lw=.8,ls='--')
    fig.suptitle('Response pilot · 2 independent noise/initial-state trials per candidate',fontsize=11)
    save(fig,'response_mechanism')


def comparison(data,summary):
    fig,axes=plt.subplots(1,3,figsize=(13.4,5.8),layout='constrained',gridspec_kw={'width_ratios':[1.,1.,.95]})
    nreg=data['nreg']; selected=data['selected_loss']*1e6; rho=data['rank_spearman']
    methods=np.arange(len(LABELS))
    for ai,(values,xlabel) in enumerate(((selected,'Selected late loss (10⁻⁶ Hz·s) ↑'),(nreg,'Normalized regret ↓'),(rho,'Score–loss Spearman ρ'))):
        ax=axes[ai]; panel(ax,chr(97+ai))
        for mi in methods:
            for si in range(2):
                symbol='o' if si==0 else 's'
                for seed in range(2):
                    value=values[seed,si,mi]
                    if np.isfinite(value):
                        ax.scatter(value,mi+(si-.5)*.20+(seed-.5)*.075,marker=symbol,
                                   color=COLORS[mi],s=25,alpha=.85,edgecolors='white',linewidth=.4)
            if np.isfinite(values[:,:,mi]).any():
                ax.plot(np.nanmean(values[:,:,mi]),mi,marker='|',color='#111111',ms=12,mew=1.6)
        ax.set(xlabel=xlabel,yticks=methods,ylim=(len(methods)-.5,-.5))
        if ai==0:
            ax.set_yticklabels(SHORT)
        else:
            ax.set_yticklabels([])
        ax.grid(axis='x',color='.9',lw=.6)
        if ai==1:
            ax.set_xlim(-.03,1.04)
        if ai==2:
            ax.axvline(0,lw=.7,color='.7'); ax.set_xlim(-1.05,1.05)
    axes[2].scatter([],[],marker='o',color='.3',label=f'Site {int(data["sites"][0])+1}')
    axes[2].scatter([],[],marker='s',color='.3',label=f'Site {int(data["sites"][1])+1}')
    axes[2].plot([],[],marker='|',color='.2',ls='none',ms=10,label='Equal-weight mean')
    axes[2].legend(loc='lower left',bbox_to_anchor=(0,1.03),ncols=3,frameon=False,fontsize=8)
    fig.suptitle('Exploratory functional comparison · 2 scoring seeds; shared independent response labels',fontsize=11)
    save(fig,'functional_comparison')


def efficiency(records,anytime):
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.3),layout='constrained',gridspec_kw={'width_ratios':[1.15,1.]})
    kvalues=(2,3,4,6,8)
    names=('xi_fast','pair_phi_r','weak_phi_r','phi_r_lu','wms','o_increment','whole_mi','phi_full')
    labels=('Xi conditional-TC','Pairwise Phi-R','Weakest split Phi-R','Phi-R-LU scalar','Dynamic WMS','Target-augmented O','Whole MI','Full Phi-ID reference')
    colors=COLORS[:7]+['#B2182B']
    ax=axes[0,0]; panel(ax,'a')
    for method,label,color in zip(names,labels,colors):
        ks=[]; med=[]; low=[]; high=[]
        for k in kvalues:
            group=group_times(records,method,k,'core')
            if group:
                q=stats(group,'wall_seconds')*1000; ks.append(k); low.append(q[0]); med.append(q[1]); high.append(q[2])
        if ks:
            ax.plot(ks,med,'o-',color=color,lw=1.2,ms=3,label=label)
            ax.fill_between(ks,low,high,color=color,alpha=.12)
    ax.set(xlabel='ROI blocks, k',ylabel='Core score wall time (ms)',yscale='log',xticks=kvalues)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.04),ncols=3,frameon=False,fontsize=8)
    ax=axes[0,1]; panel(ax,'b')
    # Timing tasks independently reload identical samples. Peak RSS includes runtime/data.
    for ii,(method,label,color) in enumerate(zip(names[:-1],labels[:-1],colors[:-1])):
        for offset,layer,marker in ((-.12,'pool','o'),(.12,'raw','s')):
            group=group_times(records,method,8,layer)
            if group:
                q=stats(group,'wall_seconds')*1000
                ax.errorbar(ii+offset,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt=marker,color=color,capsize=2,ms=5)
    ax.set(yscale='log',ylabel='16-candidate wall time (ms)',xticks=range(7))
    ax.set_xticklabels(['Xi','Pair','Split','LU','WMS','O','MI'],rotation=0)
    ax.plot([],[],marker='o',ls='none',color='.3',label='Fitted backend → rank')
    ax.plot([],[],marker='s',ls='none',color='.3',label='Raw samples → fit → rank')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.04),frameon=False,fontsize=8)
    ax=axes[1,0]; panel(ax,'c')
    for method,label,color in zip(names[:-1],labels[:-1],colors[:-1]):
        curves=[r['curve'] for r in anytime if r['method']==method and r['curve'][-1]['nreg'] is not None]
        if not curves:
            continue
        start=max(c[0]['seconds'] for c in curves); end=max(c[-1]['seconds'] for c in curves)
        t=np.geomspace(start,end,80)
        traces=[]
        for curve in curves:
            ct=np.array([c['seconds'] for c in curve]); cv=np.array([c['nreg'] for c in curve])
            indices=np.searchsorted(ct,t,side='right')-1
            traces.append(cv[np.maximum(indices,0)])
        traces=np.array(traces)
        ax.plot(t*1000,traces.mean(0),color=color,lw=1.5,label=label)
    ax.axhline(.5,color='.7',lw=.7,ls='--')
    ax.set(xscale='log',xlabel='Actual cumulative scoring time (ms)',ylabel='Mean normalized regret ↓',ylim=(-.04,1.04))
    ax=axes[1,1]; panel(ax,'d')
    for ii,(method,label,color) in enumerate(zip(names[:-1],labels[:-1],colors[:-1])):
        group=group_times(records,method,8,'raw')
        if group:
            q=stats(group,'peak_rss_bytes')/1024**2
            ax.errorbar(ii,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='o',color=color,ms=5,capsize=2)
    ax.set(ylabel='Fresh-process peak RSS (MiB)',xticks=range(7),ylim=(0,None))
    ax.set_xticklabels(['Xi','Pair','Split','LU','WMS','O','MI'])
    fig.suptitle('Efficiency pilot · one thread; median/IQR of 5 fresh-process timing tasks',fontsize=11)
    save(fig,'efficiency_comparison')


def report(contract,data,summary,records):
    mean_nreg=np.nanmean(data['nreg'],axis=(0,1)); loss=np.mean(data['selected_loss'],axis=(0,1))
    nonlinear=json.loads((BASE/'nonlinear_tm_check.json').read_text())
    timing=[]
    for method in (*METHODS,'xi_fast','phi_full'):
        k=4 if method=='phi_full' else 8
        group=group_times(records,method,k,'core')
        if group:
            q=stats(group,'wall_seconds')*1000
            timing.append(f'| {method} | {k} | {q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}] | {np.median([r["unique_mi"] for r in group]):.0f} |')
    # Reuse intact pulse/sham per site/G/evaluation seed; 99,840 formal trajectories.
    formal_trajectories=3*4*8*8*2*(64+1)
    pilot_trajectories=2*2*2*(8+1)
    response_seconds=sum(g['elapsed_seconds'] for g in summary['response_gates'])
    estimate_response=response_seconds/pilot_trajectories*formal_trajectories
    estimate_score=np.mean([r['simulation_seconds']+r['fit_seconds'] for r in summary['scoring_cost']])*3*8
    largest_pool=max((np.median([r['wall_seconds'] for r in group_times(records,m,8,'pool')]) for m in METHODS),default=0)
    estimate_ranking=sum(np.median([r['wall_seconds'] for r in group_times(records,m,8,'pool')]) for m in METHODS)*3*8*4*64/16
    cost=dict(formal_response_trajectory_count=formal_trajectories,response_seconds=float(estimate_response),
              scoring_acquisition_and_fit_seconds=float(estimate_score),all_method_ranking_seconds=float(estimate_ranking),
              total_seconds=float(estimate_response+estimate_score+estimate_ranking),
              scope='Linear extrapolation from G=1.3 pilot; excludes implementation repairs, repeat power assessment, nonlinear backend, robustness and machine contention. Not a guaranteed ETA.')
    (BASE/'formal_cost_estimate.json').write_text(json.dumps(cost,indent=2))
    table='\n'.join(f'| {label} | {mean_nreg[i]:.3f} | {loss[i]*1e6:.3f} |' for i,label in enumerate(SHORT))
    response_table='\n'.join(f'| {g["site"]+1} | {g["loss_range_hz_s"]*1e6:.3f} | {g["max_loss_sem_hz_s"]*1e6:.3f} | {g["sham_drift_rms_hz"]:.6f} |' for g in summary['response_gates'])
    text=f'''# DMF 刺激响应比较：A–C 预实验结果

状态：**A–C 已执行，正式 D 未启动。当前没有发现 Ξ 的选择效果优势。**

G=1.3；刺激 ROI {contract['sites'][0]+1}（左半球低 SC 强度层）、{contract['sites'][1]+1}（右半球高强度层）；每池 8 个 k=8 候选，均匀/SC 抽样各半；评分 seeds 901–902，每条件 2 个独立响应初态/噪声重复。ROI 编号在本报告中为 1-based，缓存为 0-based。这不是正式跨 G、跨种子的确认结论，也不能支持非劣性声明。

## 1. 响应检查

![response](../../fig/dmf_response_benchmark/response_mechanism.png)

10 ms 脉冲电流为 i0 的 1%（{contract['pulse_amplitude']}）；10 ms 后启动返回输入削弱；每候选移除相同 SC 权重，并补偿初态对应的 tonic input。JFIC 固定，不重新校准；全程无裁剪。晚期为 50–300 ms，读出是全脑发放率 RMS 面积。误差棒是 2 次响应重复的 SEM，点为原始重复；时程阴影为完整模型重复范围或候选均值的范围，均非置信区间。

| 刺激 ROI | 候选损失范围（10⁻⁶ Hz·s） | 最大标签 SEM（10⁻⁶ Hz·s） | sham 背景变化 RMS（Hz） |
|---|---:|---:|---:|
{response_table}

两个池均通过探索性“范围 > 2 × 最大标签 SEM”门槛。该门槛不是正式统计检验；仅两次重复，尚未量化 noisy oracle 的选择偏差。所有完整/削弱、pulse/sham 条件均保存，无候选剔除。sham 漂移仍非零，tonic 补偿不能保持所有随机背景轨迹不变。

预设的 2 秒确定性基准未通过稳定门槛（0.1254 Hz 漂移），在任何排名/响应标签生成前延长至 10 秒，末端漂移为 {contract['baseline_drift_hz']:.3g} Hz。评分支持从 0.005/0.01/0.02 中选择无初始及积分状态越界的最大半宽 {contract['halfwidth']}。这项调整只依据状态有效性；没有按排名调窗口、幅度或候选。

## 2. 功能对比

![comparison](../../fig/dmf_response_benchmark/functional_comparison.png)

| 选择方法 | 平均 NReg（越小越好） | 选中组合晚期损失（10⁻⁶ Hz·s） |
|---|---:|---:|
{table}

Ξ 与二阶 Phi-R 在两池、两评分种子均选中损失最小的组合，NReg=1；whole MI 均选中池内最大损失组合，NReg=0。结构/普通信息基线比多个协同指标更好，预实验不支持“协同指标更准确识别该返回输入响应依赖”的假设。此处效果针对指定等预算、tonic 补偿操作，不是 Syn 真值或认知水平。

归一化遗憾先在每个位置/评分种子内计算，再等权汇总。两个评分种子共享同一独立响应标签集，不把 4 个点当作 4 个独立受试者。随机基线为池内均匀选择的精确期望；初始响应基线在固定刺激位置内完全并列，固定选择候选 1，其偶然好结果不具有排序信息。前 10% 向上取整为 1 个候选，因此该预实验里与 top-1 重复。

## 3. 计算成本

![efficiency](../../fig/dmf_response_benchmark/efficiency_comparison.png)

核心图从已拟合共同后端开始，Gaussian/affine-TM 的 Xi conditional-TC 简式与直接 MI 差值已核对，最大误差 {summary['xi_direct_fast_max_error_bits']:.3g} bit。主效率曲线使用该同值简式；直接 MI 路径也计时。Phi-R 主高阶对手采用精确 LU 标量算法；完整 ΦID 仅作为 k≤4 的分解参考。共享 MI 与 logdet 缓存对所有方法开放，每次任务重建 MI 缓存，结构热启动可复用。冷结构建立成本另存在 timing.json 中。

| 算法 | k | 核心 wall time 中位数 [Q1,Q3]（ms） | 唯一 MI 查询 |
|---|---:|---:|---:|
{chr(10).join(timing)}

每个任务独立进程，统一预热、单线程、随机计时顺序、5 次重复；记录 CPU 时间和 peak RSS。RSS 包含共同运行时、样本与后端，因此不能把几 MiB 波动解释为算法内存优势。超时限制为进程 60 秒，2 GiB 作为实测峰值判据；本轮并非硬内存限额压力测试。原始样本流程计时不包含模拟生成；数据获取成本另列。

选择质量—时间图使用真实逐候选计时及同一访问次序；每池 5 个预设随机顺序，取末个已访问候选后的当前最好分数。允许共享查询缓存，仅计评分/选择时间，不含模拟。访问未开始的时段不画，截止后保留最终选择。时间曲线显示计算更快并不能弥补该预实验里的错误排序。

## 4. 数值和估计器门槛

主后端为全脑 affine triangular TM：用完整独立盒状源拟合线性转移，使用解析独立先验的 Gaussian 矩与残差协方差，再取候选源/目标边际；不把背景 ROI 固定为零。正则化在全局先验及残差各加一次 1e-6（标准化单位），后续查询不增加 ridge，不 floor 特征值，不截 MI。它近似 Gaussian 依赖，不能声称识别一般非线性高阶机制；线性转移与残差在同一评分样本中拟合，有限样本与模型近似偏差尚未完成确认。

主 Ξ 最小值 {summary['syn_raw_min_bits']:.6g} bit；容差 1e-8 bit；容差内负值计数 {summary['syn_tolerance_zero_count']}。所有原始值保存，无静默非负投影。各阶段均无状态越界、非有限值或异常发放率（异常门槛 500 Hz）。响应损失、观察性差值和 ΦID 原子不套用 Syn 非负规则。

低维检查使用包含刺激位置的冻结 2-ROI 子集，同一评分数据，1536 样本训练 / 512 留出评估，比较 affine 与二次 triangular TM。该独立拟合密度的估计路径未通过非负性审计，可能受密度不一致和有限样本误差影响。非负审计结果为 **{nonlinear['nonnegative_audit']['passed']}**，最小值 {nonlinear['nonnegative_audit']['minimum_bits']:.6g} bit，阈值 -1e-8 bit，违规 {nonlinear['nonnegative_audit']['affected_count']} 个。该路径显式报告失败，未作为 PEID Syn 进入选择比较。它不能作为真实 Syn 为负或存在真实排名翻转的证据；需要共同、稳定的非线性联合密度估计后才能判断 Gaussian 结论是否可迁移。

## 5. 正式阶段预算与下一步

复用完整 pulse/sham 后，原建议 D 配置的实际响应轨迹为 {formal_trajectories:,} 条（未复用上界 196,608）。按本机 G=1.3 预实验线性外推：响应生成约 {estimate_response/60:.1f} 分钟；评分样本与 affine 拟合约 {estimate_score/60:.1f} 分钟；全部指标候选评分约 {estimate_ranking/60:.1f} 分钟；合计约 {(estimate_response+estimate_score+estimate_ranking)/60:.1f} 分钟。其他 G、正式重复数、数值修复、非线性后端和稳健性不在此估计内，不能当作可靠完成时间。

**不建议直接启动 D：先修复低维非线性 TM 的共同密度一致性，确认 estimator 不决定排序；再增加响应重复验证标签稳定性与工作点漂移，最后冻结正式预算。** 按用户本轮批准范围，A–C 到此完成，D 仍需确认。

## 复现与证据

- 执行：`/opt/anaconda3/envs/py311/bin/python scripts/run_dmf_response_benchmark.py`；仅 A–C。
- 重绘已有缓存：`/opt/anaconda3/envs/py311/bin/python scripts/plot_dmf_response_benchmark.py`。
- 正确性验证：`python -m pytest tests/test_dmf_response_benchmark.py -q`，16 项通过。
- 配置、输入哈希、候选、原始分数、响应轨迹、种子、数值诊断和计时：`results/dmf_schaefer100/response_benchmark_pilot/`，NPZ/JSON。
- 本地 Zotero 已搜索 PEID 并阅读 MYATYWAJ（26 页全文）、P7L7F9FT（12 页全文）、26Q48H8Y（正文 12 页、SI 10 页）。定义与引用沿用[实验计划](brain_dmf_response_benchmark_plan.md)；LU 扩展及 tonic 控制仍为本实验操作性选择。
'''
    (ROOT/'docs/reports/brain_dmf_response_benchmark_pilot.md').write_text(text)
    return cost


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    contract=json.loads((BASE/'contract.json').read_text()); summary=json.loads((BASE/'summary.json').read_text())
    records=json.loads((BASE/'timing.json').read_text()); anytime=json.loads((BASE/'anytime.json').read_text())
    with np.load(BASE/'comparison.npz') as a:
        data={k:a[k].copy() for k in a.files}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,
                         'svg.fonttype':'none','pdf.fonttype':42,'lines.linewidth':1.3}):
        mechanisms(contract,data,summary); comparison(data,summary); efficiency(records,anytime)
    print(json.dumps(report(contract,data,summary,records),indent=2))


if __name__=='__main__':
    main()
