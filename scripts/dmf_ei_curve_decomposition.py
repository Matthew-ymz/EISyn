"""Read the two terms of Xi from the same cached affine-TM joint density."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.dmf_joint_readout import audit_nonnegative
from scripts.report_sections import write_report_section

COMPONENT_COLORS = {'whole': '#287C9C', 'partial': '#BA8F34', 'xi': '#B64D64'}


def decompose(base, d):
    """Audit S directly from scalar marginal MIs, rather than infer S=E-Xi."""
    whole=d['metrics']['whole_ei_nats'].copy();xi=d['metrics']['xi_nats'].copy()
    partial=np.empty_like(whole);minimum_scalar=np.inf;maximum_closure=0.
    tol=float(d['contract']['syn_tolerance_nats'])
    zero_count=0
    for pi,subject in enumerate(d['ids']):
        for si,seed in enumerate(d['contract']['seeds']):
            for gi,g in enumerate(d['g']):
                with np.load(base/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz') as a:
                    if json.loads(str(a['contract_json']))!=d['contract']:
                        raise ValueError('Decomposition cache contract mismatch')
                    cov=a['covariance'];conditional=a['conditional']
                dim=len(conditional);prior=cov[:dim,:dim]
                if not np.allclose(prior,np.diag(np.diag(prior)),atol=1e-12,rtol=0):
                    raise ValueError('The common source prior must remain factorized')
                if (np.diag(prior)<=0).any() or (np.diag(conditional)<=0).any():
                    raise ArithmeticError('Invalid scalar marginal covariance')
                scalar=.5*(np.log(np.diag(prior))-np.log(np.diag(conditional)))
                audit=audit_nonnegative(scalar,tolerance=tol)
                minimum_scalar=min(minimum_scalar,audit['minimum_nats'])
                zero_count+=audit['tolerance_negative_count']
                partial[pi,si,gi]=scalar.sum()
                error=abs(whole[pi,si,gi]-partial[pi,si,gi]-xi[pi,si,gi])
                maximum_closure=max(maximum_closure,float(error))
                if error>tol:
                    raise ArithmeticError(f'EI decomposition does not close: {subject}, G={g}, seed={seed}, error={error} nats')
    if not np.isfinite(whole).all() or not np.isfinite(partial).all():
        raise ArithmeticError('Nonfinite EI components')
    xi_audit=audit_nonnegative(xi,tolerance=tol)
    h=np.diff(d['g']);rate_whole=-np.diff(whole,axis=-1)/h
    rate_partial=-np.diff(partial,axis=-1)/h
    xi_slope=np.diff(xi,axis=-1)/h
    rate_error=float(np.max(abs(rate_partial-rate_whole-xi_slope)))
    if rate_error>tol/h.min():
        raise ArithmeticError(f'EI rate accounting failed: {rate_error} nats/G')
    rows=[]
    for pi,subject in enumerate(d['ids']):
        w=whole[pi].mean(0);p=partial[pi].mean(0);v=xi[pi].mean(0)
        dw=np.diff(w);dp=np.diff(p);peak=int(v.argmax())
        def increases(delta):
            return [dict(interval=d['g'][[i,i+1]].tolist(),increase_nats=float(delta[i]))
                    for i in np.flatnonzero(delta>tol)]
        rows.append(dict(subject=subject,whole_mean_nats=w.tolist(),partial_sum_mean_nats=p.tolist(),
            xi_mean_nats=v.tolist(),whole_seed_sd_nats=whole[pi].std(0,ddof=1).tolist(),
            partial_sum_seed_sd_nats=partial[pi].std(0,ddof=1).tolist(),
            xi_seed_sd_nats=xi[pi].std(0,ddof=1).tolist(),
            whole_nonincreasing_on_grid=bool((dw<=tol).all()),
            partial_nonincreasing_on_grid=bool((dp<=tol).all()),
            whole_increases=increases(dw),partial_increases=increases(dp),
            whole_seed_increase_counts=(np.diff(whole[pi],axis=-1)>tol).sum(axis=0).tolist(),
            partial_seed_increase_counts=(np.diff(partial[pi],axis=-1)>tol).sum(axis=0).tolist(),
            whole_decrease_rate_nats_per_g=rate_whole[pi].mean(0).tolist(),
            partial_decrease_rate_nats_per_g=rate_partial[pi].mean(0).tolist(),
            xi_slope_nats_per_g=xi_slope[pi].mean(0).tolist(),
            xi_peak_G=float(d['g'][peak]),xi_peak_interior=bool(0<peak<len(d['g'])-1)))
    summary=dict(definition='Xi = E-S; E=EI(full 200 scalar sources -> full future); S=sum of 200 scalar-source EIs to the SAME full future',
        estimator='Marginal MIs of the existing one affine-TM joint density; no refit or new simulations',
        source_partition='Finest 200-scalar E/I partition, not 100 ROI blocks',
        horizon_seconds=.3,units='nats',G=d['g'].tolist(),seeds=d['contract']['seeds'],
        subject_ids=d['ids'],curve_center='Mean of three seeds; spread where drawn is across-seed sample SD',
        monotonicity_tolerance_nats=tol,syn_tolerance_nats=tol,xi_audit=xi_audit,
        scalar_numerical_tolerance_negative_count=zero_count,minimum_scalar_ei_nats=float(minimum_scalar),
        maximum_decomposition_closure_error_nats=maximum_closure,maximum_rate_closure_error_nats_per_g=rate_error,
        rate_definition='R_E=-Delta E/Delta G, R_S=-Delta S/Delta G; Delta Xi/Delta G=R_S-R_E; interval averages, not continuous derivatives',
        whole_nonincreasing_subject_count=sum(r['whole_nonincreasing_on_grid'] for r in rows[:8]),
        partial_nonincreasing_subject_count=sum(r['partial_nonincreasing_on_grid'] for r in rows[:8]),
        manuscript_recheck=dict(date='2026-10-03',parent_key='P6UJCVG8',attachment_key='DXGC7JEA',pages=19,
            version_date=None,sections='Methods Eqs.5,7,8 (pp.15-16); Brain/Fig.2',supplementary_appendices='unavailable'),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),subjects=rows)
    return summary,dict(whole=whole,partial=partial,xi=xi,
                        rate_whole=rate_whole,rate_partial=rate_partial,xi_slope=xi_slope)


def draw_components(d,values,out,markers):
    colors=plt.cm.viridis(np.linspace(.08,.9,8))
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout='constrained')
    for ax,key,title in zip(axes,['whole','partial','xi'],
                           ['Whole EI ($E$)','Sum of partial EI ($S$)','Integrated EI ($\\Xi=E-S$)']):
        for i in range(8):
            ax.plot(d['g'],values[key][i].mean(0),color=colors[i],marker=markers[i],lw=1.2,ms=3,label=d['ids'][i])
        ax.plot(d['g'],values[key][8].mean(0),color='#222222',ls='--',lw=1.4,label='Mean SC (93)')
        ax.set(xlabel='$G$',ylabel='Information (nats)',title=title,ylim=(0,21) if key=='xi' else (0,160))
    for label,ax in zip('abc',axes):ax.text(-.13,1.04,label,transform=ax.transAxes,fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    fig.savefig(out/'ei_decomposition_curves.png',dpi=240,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for key,label in [('whole','$E$: whole EI'),('partial','$S$: sum of partial EI')]:
        raw=values[key][8];mean=raw.mean(0);sd=raw.std(0,ddof=1)
        axes[0].plot(d['g'],mean,color=COMPONENT_COLORS[key],marker='o',ms=3,lw=1.3,label=label)
        axes[0].fill_between(d['g'],mean-sd,mean+sd,color=COMPONENT_COLORS[key],alpha=.15,lw=0)
    axes[0].fill_between(d['g'],values['partial'][8].mean(0),values['whole'][8].mean(0),
                         color=COMPONENT_COLORS['xi'],alpha=.13,lw=0)
    mean=values['xi'][8].mean(0);sd=values['xi'][8].std(0,ddof=1)
    axes[1].plot(d['g'],mean,color=COMPONENT_COLORS['xi'],marker='o',ms=3,lw=1.3,label='$\\Xi=E-S$')
    axes[1].fill_between(d['g'],mean-sd,mean+sd,color=COMPONENT_COLORS['xi'],alpha=.18,lw=0)
    for key,color in [('rate_whole','whole'),('rate_partial','partial'),('xi_slope','xi')]:
        axes[2].stairs(values[key][8].mean(0),d['g'],baseline=None,color=COMPONENT_COLORS[color],lw=1.4,
                       linestyle='--' if key=='xi_slope' else '-')
    axes[2].axhline(0,color='#adb4ba',lw=.7)
    axes[0].set(xlabel='$G$',ylabel='Information (nats)',title='Mean SC: the two terms',ylim=(0,160))
    axes[1].set(xlabel='$G$',ylabel='Information (nats)',title='Mean SC: the difference',ylim=(0,21))
    axes[2].set(xlabel='$G$ interval',ylabel='Interval change rate (nats per $G$)',
                title='$R_E$, $R_S$ and $\\Delta\\Xi/\\Delta G$')
    handles,labels=axes[0].get_legend_handles_labels();h,l=axes[1].get_legend_handles_labels()
    fig.legend(handles+h,labels+l,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    for label,ax in zip('abc',axes):ax.text(-.13,1.04,label,transform=ax.transAxes,fontweight='bold')
    fig.savefig(out/'ei_decomposition_rates.png',dpi=240,bbox_inches='tight');plt.close(fig)


def write_report(s,path):
    ref=s['subjects'][8];tables=[]
    for i,(lo,hi) in enumerate(zip(s['G'][:-1],s['G'][1:])):
        tables.append(f"| {lo:g}–{hi:g} | {ref['whole_decrease_rate_nats_per_g'][i]:.3f} | {ref['partial_decrease_rate_nats_per_g'][i]:.3f} | {ref['xi_slope_nats_per_g'][i]:+.3f} |")
    exceptions=[]
    for row in s['subjects'][:8]:
        for key,label in [('whole_increases','整体EI'),('partial_increases','部分EI之和')]:
            for event in row[key]:
                lo,hi=event['interval'];exceptions.append(f"{row['subject']} 的{label}在G={lo:g}→{hi:g}回升{event['increase_nats']:.6f} nats")
    text=r'''# DMF 整合有效信息峰值的两项分解

2026-10-03。整体EI从跨指标一致性比较中移出，在同一冻结8人×7 G×3 seed及平均SC参照中单独分析。只读取原有affine-TM缓存，无新增模拟。

## 定义和口径

令 $E(G)=EI_{300}(V\to V)$ 为整体EI，$S(G)=\sum_{i\in V}EI_{300}(\{i\}\to V)$ 为各标量源的部分EI之和。这里 $V$ 包含100脑区的200个E/I标量变量，所有部分EI都使用相同完整200维未来、同一干预密度和300ms时距。当前整合有效信息满足

$$
\Xi(G)=E(G)-S(G),\qquad E(G)=S(G)+\Xi(G).
$$

本图分解的是相减得到的差值。若按100个ROI的E/I二元块求和，得到的是另一个“跨ROI残差”，不能替代当前Ξ的200标量源分解。$S$直接从缓存密度的200个单变量MI求和，并独立核对 $E-S=\Xi$；未仅以 $E-\Xi$反推后宣称核验。

![整体EI、部分EI之和及差值Ξ](assets/dmf_subject_consistency/ei_decomposition_curves.png)

**图1。** 三列依次为整体EI、部分EI之和及差值Ξ，均为原始nats，未分别标准化后相减。每人曲线为3seed均值，颜色/标记对应同一人；虚线为93人平均SC独立模拟参照。前两列共用0–160 nats范围，差值列使用0–21 nats范围。平均SC参照不等于8条个体曲线的平均。

## 两项是否单调下降

平均SC参照的整体EI和部分EI之和都在全部7个观测G点下降。8个体中，整体EI有 @WHOLE_COUNT@/8 在观测格点不增，部分EI之和有 @PARTIAL_COUNT@/8 不增；中间6人两项均下降。端点例外为：@EXCEPTIONS@。这些回升保留原值，不能把“总体下降”写成所有被试严格单调，也不能以Syn数值零容差抹去它们。各区间发生回升的seed数量与跨seed SD保留在分析汇总。

## 为什么两个下降量的差会有峰

定义每个观测区间的平均下降速度 $R_E=-\Delta E/\Delta G$、$R_S=-\Delta S/\Delta G$，则

$$
\frac{\Delta\Xi}{\Delta G}=R_S-R_E.
$$

当部分EI之和降得更快，差值Ξ增大；当整体EI降得更快，差值Ξ减小。峰对应两项相对下降速度交换的附近，并不要求整体EI增加。这是对曲线的数学分解，尚不单独解释动力学机制或证明纯高阶相互作用增强。

![平均SC上的两项、差值与区间下降速度](assets/dmf_subject_consistency/ei_decomposition_rates.png)

**图2。** 平均SC参照，3seed均值。左图蓝/金曲线分别为 $E$/$S$，浅红色两线之间的间隙为Ξ；蓝/金细带为跨seed SD。中图单独显示Ξ及跨seed SD。右图保持同一颜色：蓝/金阶梯为 $R_E$/$R_S$，红色虚线为 $\Delta\Xi/\Delta G=R_S-R_E$。阶梯是各真实G区间的平均变化率，不是连续导数，也未插值定位斜率交点。

| G区间 | 整体EI下降速度 $R_E$ | 部分EI之和下降速度 $R_S$ | Ξ变化速度 $R_S-R_E$ |
|---|---:|---:|---:|
@RATES@

**表1。** 单位nats/G。平均SC在G=1→1.3时，部分EI之和下降更快，Ξ从15.961升到18.013 nats；在G=1.3→1.6时，整体EI下降更快，Ξ降到16.793 nats，形成观测格点上的G=1.3峰。8人中7人的Ξ有扫描内部极值；最低SC尺度个体仍在扫描末端G=3达到最大值，当前范围未观察到其下降支，不能强行认定所有人都发生速度交换。

## 估计与核验边界

保持原干预支持[0.3,0.7]、2048样本、300步未来、无状态裁剪、固定JFIC与共同affine-TM近似：线性转移、Gaussian残差、矩匹配对角源先验及原ridge。所有分量来自同一拟合密度，未重拟合或对分量另作正则化。Ξ/Syn非负容差为10⁻⁸ nats；本次189条件的分解最大闭合误差为 @CLOSURE@ nats，Ξ容差负值计数 @ZERO_COUNT@，显著负值计数 @VIOLATIONS@。原G=0拟合背景保留，未扣除。

本次重新定位Zotero主稿 *Emergent hierarchical organization of causal interactions in complex systems*，父条目P6UJCVG8、当前唯一正文附件DXGC7JEA（19页），读取Methods式（5）、（7）、（8）（第15–16页）及Brain/Fig.2。正文没有明确稿件版本/日期；Zotero版本号/编辑时间不是稿件版本证据，补充附录不可用。上述分解依照正文式（7），代码使用既有affine-TM Gaussian近似，未把它称为已验证的非线性高阶PID原子。

[跨指标曲线报告](brain.md#dmf-subject-curves)现仅比较Ξ、ΦR、WMS。[完整分析汇总](../../results/dmf_schaefer100/subject_consistency_pilot/curve_comparison_summary.json)的ei_decomposition字段保留逐人曲线、seed SD、单调性例外及区间速度。
'''
    replacements={'@WHOLE_COUNT@':str(s['whole_nonincreasing_subject_count']),
        '@PARTIAL_COUNT@':str(s['partial_nonincreasing_subject_count']),'@EXCEPTIONS@':'；'.join(exceptions),
        '@RATES@':'\n'.join(tables),'@CLOSURE@':f"{s['maximum_decomposition_closure_error_nats']:.3g}",
        '@ZERO_COUNT@':str(s['xi_audit']['tolerance_negative_count']),
        '@VIOLATIONS@':str(s['xi_audit']['violation_count'])}
    for key,value in replacements.items():text=text.replace(key,value)
    write_report_section(path, "dmf-ei-components", text)
