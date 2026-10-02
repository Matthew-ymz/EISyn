#!/usr/bin/env python3
"""Curve agreement on frozen DMF subjects; no new simulations or peak alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_subject_consistency import load
from scripts.dmf_subject_consistency import transition_interval
from scripts.run_dmf_subject_consistency import BASE, atomic_json

LABELS = {'xi': 'Interventional $\\Xi$', 'whole_ei': 'Whole EI',
          'phi_r': 'BOLD-like pairwise $\\Phi^R$', 'wms': 'Observational source WMS'}
ZH = {'xi': '干预 Ξ', 'whole_ei': 'whole EI', 'phi_r': 'BOLD-like ΦR', 'wms': '自然态 source-WMS'}
COLORS = {'xi': '#B64D64', 'whole_ei': '#BA8F34', 'phi_r': '#287C9C', 'wms': '#737B85'}
SUBJECT_MARKERS = ['o','s','^','D','v','P','X','>']


def standard_curve(curve):
    a = np.asarray(curve, float)
    sd = a.std(ddof=0)
    if not np.isfinite(a).all() or sd <= 1e-12 * max(1., abs(a).max()):
        raise ValueError('Curve is constant or nonfinite; shape correlation is undefined')
    return (a-a.mean())/sd


def agreement(curves):
    """Held person/seed never occurs in training; normalize each training curve.

    Three rotations are averaged within person. Per-person summaries, rather
    than 28 pairs or 24 seeds, define the descriptive mean/SD.
    """
    v = np.asarray(curves, float)
    normalized = np.array([[standard_curve(c) for c in row] for row in v])
    person = []
    for i in range(len(v)):
        rows = []
        for si in range(v.shape[1]):
            train = np.delete(np.delete(normalized, i, axis=0), si, axis=1).mean((0,1))
            raw_train = np.delete(np.delete(v, i, axis=0), si, axis=1).mean((0,1))
            standard_curve(train)
            rows.append([pearsonr(train, normalized[i,si]).statistic,
                         spearmanr(train, normalized[i,si]).statistic,
                         pearsonr(raw_train, v[i,si]).statistic])
        person.append(np.mean(rows, axis=0))
    values = np.array(person)
    return dict(person_count=len(v), independent_seed_rotations=v.shape[1],
        pearson_mean=float(values[:,0].mean()), pearson_sd=float(values[:,0].std(ddof=1)),
        spearman_mean=float(values[:,1].mean()), spearman_sd=float(values[:,1].std(ddof=1)),
        pearson_by_person=values[:,0].tolist(), spearman_by_person=values[:,1].tolist(),
        raw_amplitude_consensus_pearson_mean=float(values[:,2].mean()),
        raw_amplitude_consensus_pearson_sd=float(values[:,2].std(ddof=1)))


def repeatability(curves):
    """Three within-person seed pairs, averaged before describing eight people."""
    person=[]
    for row in curves:
        person.append(np.mean([[pearsonr(row[a],row[b]).statistic,
                                spearmanr(row[a],row[b]).statistic]
                               for a,b in combinations(range(len(row)),2)],axis=0))
    values=np.asarray(person)
    return dict(pearson_mean=float(values[:,0].mean()),pearson_sd=float(values[:,0].std(ddof=1)),
                spearman_mean=float(values[:,1].mean()),spearman_sd=float(values[:,1].std(ddof=1)),
                pearson_by_person=values[:,0].tolist(),spearman_by_person=values[:,1].tolist())


def analyze(d, baseline_path):
    ids=d['ids'][:8]; g=d['g']
    curves={'xi':d['metrics']['xi_nats'][:8], 'whole_ei':d['metrics']['whole_ei_nats'][:8]}
    reference={'xi':d['metrics']['xi_nats'][8], 'whole_ei':d['metrics']['whole_ei_nats'][8]}
    baseline_contract=None;baseline_audit=None
    if baseline_path.exists():
        with np.load(baseline_path) as a:
            if not np.array_equal(a['G'],g) or not np.array_equal(a['seeds'],d['contract']['seeds']):
                raise ValueError('Baseline G/seed mismatch')
            if a['subject_ids'].tolist()!=d['ids'] or a['completed'].shape!=(9,3,7) or not a['completed'].all():
                raise ValueError('Baseline subjects mismatch or incomplete')
            for key in ('phi_r','wms'):
                if key+'_nats' not in a.files or a[key+'_nats'].shape!=(9,3,7) or not np.isfinite(a[key+'_nats']).all():
                    raise ValueError(f'Native baseline {key} missing, nonfinite or wrong shape')
                curves[key]=a[key+'_nats'][:8].copy()
                reference[key]=a[key+'_nats'][8].copy()
            baseline_contract=json.loads(str(a['contract_json']))
            baseline_audit=json.loads(str(a['audit_json'])) if 'audit_json' in a.files else None
            if baseline_contract.get('pilot_contract') != d['contract']:
                raise ValueError('Native baseline model/input contract does not match the pilot')
    transitions=[transition_interval(g,r.mean(0)) for r in d['dyn'][:8]]
    selected=[i for i,t in enumerate(transitions) if t['located']]
    upper=min((g[-1]-transitions[i]['midpoint'])/transitions[i]['midpoint'] for i in selected)
    # Shared support and point count derive from independent diagnostics only.
    grid=np.linspace(-1.,upper,len(g))
    summaries={}; aligned={}
    for name,values in curves.items():
        if values.shape!=(8,3,7) or not np.isfinite(values).all():
            raise ValueError(f'Invalid complete metric {name}: {values.shape}')
        aligned[name]=np.array([[np.interp(grid,(g-transitions[i]['midpoint'])/transitions[i]['midpoint'],
                                             values[i,si]) for si in range(3)] for i in selected])
        mean=values.mean(1); max_i=mean.argmax(1); min_i=mean.argmin(1)
        landmark=min_i if name=='wms' else max_i
        rows=[]
        for i,subject in enumerate(ids):
            k=int(landmark[i]);t=transitions[i]
            rows.append(dict(subject=subject,maximum_G=float(g[max_i[i]]),minimum_G=float(g[min_i[i]]),
                landmark_G=float(g[k]),landmark_interior=0<k<len(g)-1,
                seed_maximum_G=g[values[i].argmax(1)].tolist(),
                seed_minimum_G=g[values[i].argmin(1)].tolist(),
                maximum_value_nats=float(mean[i,max_i[i]]),minimum_value_nats=float(mean[i,min_i[i]]),
                transition=t,landmark_offset_from_transition_midpoint=(float(g[k]-t['midpoint']) if t['located'] else None),
                landmark_in_transition_interval=(bool(t['interval'][0]<=g[k]<=t['interval'][1]) if t['located'] else None)))
        summaries[name]=dict(raw_G_all8=agreement(values),raw_G_located7=agreement(values[selected]),
            aligned_state7=agreement(aligned[name]),within_person_seed_repeatability=repeatability(values),extrema=rows,
            landmark='minimum for WMS; maximum for the other metrics',
            interior_landmark_count=sum(r['landmark_interior'] for r in rows),
            landmark_G_sd_all8=float(g[landmark].std(ddof=1)),
            landmark_G_sd_located7=float(g[landmark[selected]].std(ddof=1)),
            landmark_in_transition_interval_count=sum(rows[i]['landmark_in_transition_interval'] for i in selected),
            modal_landmark_count=int(np.unique(g[landmark],return_counts=True)[1].max()))
    summary=dict(metrics=summaries,subject_ids=ids,G=g.tolist(),seeds=d['contract']['seeds'],
        located_subject_ids=[ids[i] for i in selected],aligned_state_grid=grid.tolist(),
        alignment='Independent mean-rate maximum-slope interval midpoint; linear interpolation on common support; no extrapolation',
        normalization='Each seed/person curve centered and divided by its own across-grid SD before averaging training consensus; no sign flip',
        evaluation='Leave one person and one seed out; other persons use the other two seeds; average three rotations within person',
        inference='Purposively selected development subjects; descriptive subject mean/SD; no population inference',
        missing_native_baselines=[key for key in ('phi_r','wms') if key not in curves],
        baseline_contract=baseline_contract,
        baseline_audit=baseline_audit,
        group_mean_reference={key:value.tolist() for key,value in reference.items()},
        group_mean_transition=transition_interval(g,d['dyn'][8].mean(0)),
        manuscript_recheck=dict(parent='P6UJCVG8',attachment='DXGC7JEA',
            title='Emergent hierarchical organization of causal interactions in complex systems',
            version_date=None,sections='Brain/Fig.2; Methods Eqs.4-8',missing='supplementary appendices',
            discrepancy='Main prose describes both observational comparisons as BOLD-like; repository source-WMS uses full E/I natural inputs and full future target, PhiR uses BOLD-like pairs.'),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return summary,curves,aligned,selected


def draw(d,s,curves,aligned,selected,path):
    names=list(curves);n=len(names);subject_colors=plt.cm.viridis(np.linspace(.08,.9,8))
    fig,axes=plt.subplots(3,n,figsize=(4*n,9.2),layout='constrained',squeeze=False)
    for col,name in enumerate(names):
        values=curves[name]
        for i in range(8):
            mean=values[i].mean(0)
            style=dict(color=subject_colors[i],lw=1.25,marker=SUBJECT_MARKERS[i],ms=3)
            axes[0,col].plot(d['g'],mean,label=d['ids'][i],**style)
            axes[1,col].plot(d['g'],standard_curve(mean),**style)
        for row,i in enumerate(selected):
            mean=aligned[name][row].mean(0)
            axes[2,col].plot(s['aligned_state_grid'],standard_curve(mean),
                            color=subject_colors[i],lw=1.25,marker=SUBJECT_MARKERS[i],ms=3)
        ref=np.mean(s['group_mean_reference'][name],axis=0)
        axes[0,col].plot(d['g'],ref,color='#222222',ls='--',lw=1.3,label='Mean SC (93)')
        axes[1,col].plot(d['g'],standard_curve(ref),color='#222222',ls='--',lw=1.3)
        t=s['group_mean_transition']
        if t['located']:
            z=(d['g']-t['midpoint'])/t['midpoint']
            grid=np.asarray(s['aligned_state_grid'])
            if grid.min()<z.min()-1e-12 or grid.max()>z.max()+1e-12:
                raise ValueError('Mean-SC reference cannot cover the common state grid')
            axes[2,col].plot(grid,standard_curve(np.interp(grid,z,ref)),color='#222222',ls='--',lw=1.3)
        axes[0,col].set_title(LABELS[name],fontsize=10)
        axes[0,col].set(xlabel='$G$',ylabel='Information (nats)')
        axes[1,col].set(xlabel='$G$',ylabel='Standardized curve (z score)')
        axes[2,col].set(xlabel='$(G-G_c)/G_c$',ylabel='Standardized curve (z score)')
    for letter,ax in zip('abcdefghijkl',axes.ravel()):
        ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    fig.savefig(path,dpi=240,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
    categories=['Raw $G$\n(all 8)','Raw $G$\n(located 7)','State aligned\n(located 7)']
    for ni,name in enumerate(names):
        for ax,metric in zip(axes,['pearson','spearman']):
            rows=[s['metrics'][name][key] for key in ['raw_G_all8','raw_G_located7','aligned_state7']]
            x=np.arange(3)+(ni-(n-1)/2)*.10
            ax.errorbar(x,[r[metric+'_mean'] for r in rows],yerr=[r[metric+'_sd'] for r in rows],
                        fmt='o-',ms=4,lw=1.1,capsize=3,color=COLORS[name],label=LABELS[name])
            ax.set(xticks=np.arange(3),xticklabels=categories,ylabel=f'Independent-seed LOSO\n{metric.capitalize()} correlation')
            ax.axhline(0,color='#cccccc',lw=.6)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,fontsize=8,frameon=False)
    for label,ax in zip('ab',axes):ax.text(-.13,1.04,label,transform=ax.transAxes,fontweight='bold')
    fig.savefig(path.with_name('curve_agreement.png'),dpi=240,bbox_inches='tight');plt.close(fig)
    draw_extrema(d,s,path.with_name('curve_extrema.png'))


def draw_extrema(d,s,path):
    names=list(s['metrics'])
    fig,axes=plt.subplots(1,len(names),figsize=(3.8*len(names),4.8),layout='constrained',squeeze=False)
    for col,name in enumerate(names):
        ax=axes[0,col]
        for i,row in enumerate(s['metrics'][name]['extrema']):
            t=row['transition']
            if t['located']:
                ax.plot(t['interval'],[i,i],color='#b8bdc3',lw=5,solid_capstyle='butt',zorder=1)
            positions=row['seed_minimum_G'] if name=='wms' else row['seed_maximum_G']
            for seed,offset,mark,g in zip(s['seeds'],[-.18,-.06,.06],['o','s','^'],positions):
                ax.scatter(g,i+offset,marker=mark,s=20,color=COLORS[name],zorder=2)
            ax.scatter(row['landmark_G'],i+.20,marker='D',s=22,facecolor='white',edgecolor='#222222',lw=.8,zorder=3)
        ax.set(xlim=(-.12,3.12),xticks=d['g'],xlabel='Observed $G$',yticks=np.arange(8),
               yticklabels=d['ids'][:8] if col==0 else [],title=LABELS[name]+(' (min)' if name=='wms' else ' (max)'))
        ax.invert_yaxis()
        ax.text(-.13,1.04,'abcd'[col],transform=ax.transAxes,fontweight='bold')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker=m,color='none',markerfacecolor='#65717d',markeredgecolor='#65717d',
                    markersize=4,label=f'Seed {seed}') for seed,m in zip(s['seeds'],['o','s','^'])]
    handles += [Line2D([],[],marker='D',color='none',markerfacecolor='white',markeredgecolor='#222222',
                       markersize=4,label='Extremum of seed-mean curve'),
                Line2D([],[],color='#b8bdc3',lw=5,label='Independent rate transition interval')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    fig.savefig(path,dpi=240,bbox_inches='tight');plt.close(fig)


def report(s,path):
    rows=[];ranks=[];peaks=[];both=[]
    for name,v in s['metrics'].items():
        cells=[f"{v[k]['pearson_mean']:.3f} ± {v[k]['pearson_sd']:.3f}" for k in ['raw_G_all8','raw_G_located7','aligned_state7']]
        rows.append('| '+ZH[name]+' | '+' | '.join(cells)+' |')
        ranks.append(f"| {ZH[name]} | {v['raw_G_all8']['spearman_mean']:.3f} | {v['raw_G_located7']['spearman_mean']:.3f} | {v['aligned_state7']['spearman_mean']:.3f} | {v['within_person_seed_repeatability']['pearson_mean']:.6f} ± {v['within_person_seed_repeatability']['pearson_sd']:.6f} |")
        peaks.append(f"| {ZH[name]} | {'最小值' if name=='wms' else '最大值'} | "+'、'.join(f"{r['landmark_G']:g}" for r in v['extrema'])+f" | {v['landmark_G_sd_all8']:.3f} | {v['interior_landmark_count']}/8 | {v['landmark_in_transition_interval_count']}/7 |")
        both.append(f"| {ZH[name]} | "+'、'.join(f"{r['maximum_G']:g}" for r in v['extrema'])+' | '+'、'.join(f"{r['minimum_G']:g}" for r in v['extrema'])+' |')
    ranking=[]
    for key,label in [('raw_G_all8','原 G、8人'),('aligned_state7','状态对齐、7人')]:
        order=sorted(s['metrics'],key=lambda n:s['metrics'][n][key]['pearson_mean'],reverse=True)
        ranking.append(label+'：'+' > '.join(f"{ZH[n]}（{s['metrics'][n][key]['pearson_mean']:.3f}）" for n in order)+'。')
    audit=s.get('baseline_audit')
    interpretation=''
    peak_interpretation=''
    if not s['missing_native_baselines']:
        shape=[s['metrics'][k]['aligned_state7']['pearson_mean'] for k in ['xi','phi_r','wms']]
        interpretation=(f"Ξ、ΦR、WMS在独立状态对齐后均达到较高的曲线相似度（Pearson {min(shape):.3f}–{max(shape):.3f}）。"
            "这一维度没有显示Ξ明显占优；三者的均值次序也随是否对齐改变。whole EI的形状最一致，但主要表现为随G下降，"
            "不能据此将它视作最好的内部转折标记。8人开发样本及原生协议差别限定了这项描述性排名。")
        peak_interpretation=(f"ΦR的峰与WMS的谷在8人均值曲线上位于相同格点，中间6人均为G=1，"
            f"原G位置SD均为{s['metrics']['phi_r']['landmark_G_sd_all8']:.3f}，小于Ξ的{s['metrics']['xi']['landmark_G_sd_all8']:.3f}。"
            f"相对于独立转折区间，Ξ有{s['metrics']['xi']['landmark_in_transition_interval_count']}/7命中，"
            f"ΦR与WMS各{s['metrics']['phi_r']['landmark_in_transition_interval_count']}/7。"
            "因此，极值更集中与更靠近各人的动力学转折给出了不同排序。这里只比较粗格点，不据此证明真实相变点或机制优势。")
    audit_text='原生基线尚未完整，不给出其一致性排名。'
    if audit:
        floor_conditions=sum(any(row['phi_audit'][key] for key in
                            ['source_eigenvalue_floor_count','target_eigenvalue_floor_count','joint_eigenvalue_floor_count'])
                             for row in audit['conditions'])
        eigenvalue_queries=audit['condition_count']*4950*8
        audit_text=(f"189个条件均给出有限的 ΦR/WMS；ΦR 原始逐对最小值为 {audit['phi_r_minimum_raw_bits']:.6g} bits，"
            f"落入非负数值零容差的对数为 {audit['phi_r_numerical_zero_count']}。自然轨迹边界命中共 {audit['natural_boundary_hit_count']} 次，"
            f"未裁剪未来的越界次数为 {audit['future_outside_state_count']}；稳定判据在 {audit['stabilization_detected_condition_count']}/189 条件检测到，"
            "未检测到时沿用0.3s burn-in，而非宣称达到稳态。\n\n"
            f"ΦR 协方差特征值下限被触发 {audit['phi_r_covariance_floor_count']}/{eigenvalue_queries} 次，涉及 {floor_conditions}/189 条件；单变量相关分母下限被触发 {audit['phi_r_singleton_correlation_floor_count']} 次；"
            f"WMS 协方差特征值下限被触发 {audit['wms_covariance_floor_count']} 次。计数是所有条件、所有估计子查询中"
            "低于下限的特征值/分母总数，不是异常被试数。ΦR 的高自相关和短 BOLD-like 记录使这一数值稳定处理影响解释，"
            "本轮保留原生估计器，未据结果调下限。\n\n"
            f"WMS 的最大源/残差条件数分别为 {audit['source_condition_number_max']:.3g}、{audit['noise_condition_number_max']:.3g}；"
            f"2048次有放回采样中，最少只有 {audit['minimum_sampled_unique_timepoints']} 个不同时间点。"
            "这不是2048段独立自然轨迹；高维 Gaussian 拟合和短记录均是本次原生比较的限制。")
    native_text=('原生 ΦR＋WMS 已完成同一8人×7 G×3 seed，并补算平均SC参照，共189条件。平均SC只作虚线参照，不进入被试一致性训练或统计。'
                 if not s['missing_native_baselines'] else '当前仅完成已有 Ξ/EI 缓存分析；原生 ΦR/WMS 未完整，未纳入结果。')
    text='''# DMF 跨个体曲线：Ξ、ΦR 与 WMS

2026-10-02。@NATIVE@ 本比较检验曲线随耦合 G 变化时的跨个体一致性，同时区分随机重复、峰/谷位置与转折标记。8个体依据原生SC谱半径等距秩选取，属于开发样本；均值±SD是描述性结果，不作人口推断。

## 1. 整条曲线是否一致

@RANKING@

@INTERPRETATION@

每次留出1人的1个seed，训练共识使用其他人的另2个seed，3次轮换先在个体内平均。训练曲线各自按跨G均值/SD标准化后等权平均；不取绝对值、不翻转、不按自身峰对齐。Pearson评价形状，Spearman评价排序。不同指标原始量级不直接比较。

| 指标 | 原始G，全部8人 | 原始G，可定位7人 | 独立状态对齐，可定位7人 |
|---|---:|---:|---:|
@AGREEMENT@

**表1。** 独立seed留一人形状Pearson的个体均值±SD。状态参照来自独立发放率诊断：最大变化区间的中点Gc。sub-10377处于扫描边界，不能定位；7人是同一组，故原G的7人结果提供对齐前参照。共同状态支持为−1至0.579，取7个等距点，对原7个G观测线性插值，不外推。截短区间与插值会改变比较问题，相关提高不能全部归因于状态差异消除。

| 指标 | 原G8人 Spearman | 原G7人 Spearman | 对齐7人 Spearman | 同人不同seed Pearson，8人均值±SD |
|---|---:|---:|---:|---:|
@RANKS@

**表2。** Spearman沿用表1的留一人计算。同人重复性则先平均每人的3对seed相关，再汇总8人；这与跨个体一致性是不同端点。原始幅度共识的敏感性结果及每人分数保留在JSON中，未将seed、28个被试对或4950个ROI对当作独立被试。

![各指标原始量级、原G形状与独立状态对齐形状](assets/dmf_subject_consistency/metric_curves.png)

**图1。** 每种指标一列，依次为3seed均值的原始nats、原G下标准化形状、共同状态区间的标准化形状。颜色固定对应个体，虚线为93人平均SC的独立模拟参照；它不是8条曲线的均值。图中每人先平均seed再标准化，表1每个seed先独立标准化后评估，不能以图替代表中计算。第三行仅含7个可定位个体及可定位的平均SC参照，连线/插值没有增加独立观测。

![跨个体独立seed相关比较](assets/dmf_subject_consistency/curve_agreement.png)

**图2。** Pearson与Spearman使用相同评估和人群。点为个体均值，误差棒为个体SD，不是置信区间；均值±SD可能超出相关的合法范围，误差棒不截断。原G7人与对齐7人使用相同个体，横轴支持和采样点不同。

## 2. 峰/谷位置与独立转折

个体顺序固定为sub-10377、sub-10249、sub-10321、sub-10274、sub-10631、sub-10565、sub-10325、sub-10228，SC谱半径递增。沿用原生DMF比较的极值方向：Ξ、ΦR、whole EI主报最大值，WMS主报最小值；同时给出所有指标的最大值和最小值。位置在原观测格点上查找，未用插值找峰。格点极值不能认定为连续G的真实峰/谷；边界极值可能是单调趋势或扫描不足。

| 指标 | 主参照 | 8人均值曲线极值G，依上述顺序 | 极值G的SD | 扫描内部 | 落在独立转折区间 |
|---|---|---|---:|---:|---:|
@PEAKS@

**表3。** 极值G的SD使用全部8人；转折区间命中只计7个可定位人，含区间端点。共同G的极值集中与相对于自身动力学转折的位置一致是不同问题，JSON保留每人到独立转折中点的偏移和3个seed的峰/谷位置。

![各指标原格点极值与独立发放率转折区间](assets/dmf_subject_consistency/curve_extrema.png)

**图3。** 小标记显示各seed的极值G，空心菱形是3seed均值曲线的极值；灰条为独立发放率最大变化区间。无法定位的sub-10377不画灰条，仍保留其观测极值。WMS列采用谷的位置。

| 指标 | 8人最大值G | 8人最小值G |
|---|---|---|
@BOTH@

@PEAK_INTERPRETATION@

whole EI主要单调下降，其高一致性和边界最大值不能说明它更适合标记内部转折。Ξ的两个SC尺度极端在共同G下峰位错开，因此还需对照同一7人的独立状态对齐结果。对原生ΦR/WMS，一致性分数只说明各自观测和估计协议下的曲线表现，不能把它们与Ξ的差别完全归因于指标公式，也不能据8人的均值排名宣布全面优势。

## 3. 原生协议与数值审计

沿用冻结的个体SC、平均SC参照及93人平均图在G=1校准的固定JFIC，不归一化个体SC，也不为各指标重新调参。G为0、0.5、1、1.3、1.6、2.2、3；名义seed为3、4、5。新观测基线每个名义seed使用62000+seed的自然轨迹、63000+seed的时间采样和64000+seed的未来噪声，跨人/跨G成对，三条流互异且与Ξ及独立诊断分离。G=0同seed输出在个体间相同；表1采用不同seed评价，不作G=0背景扣除。

- **ΦR：** 自然DMF记录1.5s，burn-in至少0.3s；稳定判据窗口0.05s、发放率漂移阈值0.15Hz、连续2窗口。将尾部E发放率转换为Balloon–Windkessel BOLD-like信号，Gaussian-MMI ΦR对全部4950个ROI对取均值，延迟1步即1ms。原生Gaussian估计保留10⁻¹⁰协方差特征值/相关分母下限及MI非负投影；逐对ΦR在[−10⁻¹⁰,0) bits视作数值零并记录，低于−10⁻¹⁰ bits显式失败。短BOLD-like记录、1ms延迟和数值下限限制其生理解释。
- **WMS：** 从自然轨迹尾部有放回抽2048个完整200维E/I源状态，以无状态裁剪的同模型预测300ms后的完整200维未来。保留相关的经验源先验，用仓库原生Gaussian线性拟合估计整体MI减200个标量源对完整未来的MI之和；继承10⁻⁶ ridge及10⁻¹²特征值下限。WMS有符号，负值不属于PEID Syn非负违反，未取绝对值或作非负投影。
- **Ξ及whole EI：** 复用原预实验的独立均匀干预[0.3,0.7]、2048样本、300ms未来、统一affine-TM密度、矩匹配对角Gaussian源先验及10⁻⁸ nats Syn容差。其源分布、观测变量、目标维数、时距及正则化与ΦR/WMS并不全部相同。本轮是原生协议下的曲线比较，不是只改变指标公式的受控消融。若在Ξ的同一因子化共同密度上直接求source whole-minus-sum，按定义等于Ξ，不构成独立WMS基线。

观测自然轨迹继承[0,1]硬裁剪，所有记录步骤及最终状态均审计；未来预测不裁剪，状态越界或异常发放率显式失败。原生输出bits统一乘ln2后作nats曲线，保留原始逐对值、极值、数值零计数和估计器下限审计。

@AUDIT@

本轮重新核对Zotero P6UJCVG8（*Emergent hierarchical organization of causal interactions in complex systems*），正文附件DXGC7JEA，19页，读取Brain/Fig.2、Methods式（4）–（8）。附件没有明确稿件版本/日期，元数据编辑不能确定新版本；补充附录不可用。正文把两项观测对照并列描述为BOLD-like，而仓库原生WMS实际使用自然态完整E/I状态，只有ΦR使用BOLD-like。本轮保留已获批准的仓库原生实现，记录这一正文/代码差别，未声称完全核验最新稿件一致性。

## 4. 复用

[分析汇总](../../results/dmf_schaefer100/subject_consistency_pilot/curve_comparison_summary.json)、[原生冻结协议](../../results/dmf_schaefer100/subject_consistency_pilot/curve_native_contract.json)、[原生数值审计](../../results/dmf_schaefer100/subject_consistency_pilot/curve_native_audit.json)、[原开发预实验](brain_dmf_subject_consistency_pilot.md)。

[原生计算与断点复用](../../scripts/run_dmf_subject_curve_baselines.py)、[分析与绘图](../../scripts/analyze_dmf_subject_curves.py)。昂贵模拟保存为189个轻量NPZ条件与完整curve_baselines.npz，分析只读匹配协议的完整缓存，无需重算，不创建CSV。

```bash
.venv/bin/python scripts/analyze_dmf_subject_curves.py
```
'''
    for key,value in {'@NATIVE@':native_text,'@RANKING@':'\n\n'.join(ranking),'@AGREEMENT@':'\n'.join(rows),
                      '@RANKS@':'\n'.join(ranks),'@PEAKS@':'\n'.join(peaks),'@BOTH@':'\n'.join(both),
                      '@AUDIT@':audit_text,'@INTERPRETATION@':interpretation,'@PEAK_INTERPRETATION@':peak_interpretation}.items():
        text=text.replace(key,value)
    path.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE)
    args=p.parse_args()
    with threadpool_limits(limits=1):
        d=load(args.base,organization=False)
        s,curves,aligned,selected=analyze(d,args.base/'curve_baselines.npz')
        atomic_json(args.base/'curve_comparison_summary.json',s)
        out=ROOT/'docs/reports/assets/dmf_subject_consistency'
        with plt.rc_context({'font.family':'sans-serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
            draw(d,s,curves,aligned,selected,out/'metric_curves.png')
        report_path=ROOT/'docs/reports/brain_dmf_subject_curve_comparison.md'
        report(s,report_path)
        if not s['missing_native_baselines']:
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            atomic_json(args.base/'curve_native_completed.json',dict(status='complete',analysis_complete=True,
                condition_count=189,baseline_sha256=digest(args.base/'curve_baselines.npz'),
                summary_sha256=digest(args.base/'curve_comparison_summary.json'),
                report_sha256=digest(report_path),analysis_sha256=s['implementation_sha256'],
                figures_sha256={name:digest(out/name) for name in
                               ['metric_curves.png','curve_agreement.png','curve_extrema.png']}))
        print(json.dumps({k:{axis:v[axis] for axis in ['raw_G_all8','raw_G_located7','aligned_state7']} for k,v in s['metrics'].items()},indent=2))


if __name__=='__main__':main()
