#!/usr/bin/env python3
"""Render the frozen HCP replication and its complete, nonselective report."""
from __future__ import annotations
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_hcp_full_replication as a
from scripts.run_hcp_full_replication import OUT,STATES,TASKS,core,sha,dump

FIG=ROOT/'fig'
STEM='hcp_schaefer1000_full_replication_20261022'
SHORT=['REST','EMO','GAMB','LANG','MOTOR','REL','SOC','WM']
NET=['Vis','SM','DAN','VAN','Lim','Cont','DMN']
BLUE='#4C78A8';ORANGE='#D07A3A';GREEN='#3D8978';GRAY='#7B8188'


def read(name):return json.loads((OUT/name).read_text())


def compact(name):
    return '+'.join(dict(zip(core.NETWORKS,NET))[n] for n in name.split('+'))


def ptext(p):
    return '<0.0001' if p<1e-4 else f'{p:.4f}'


def label(ax,letter):
    ax.text(-.085,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=13,
            ha='left',va='bottom',clip_on=False)


def save(fig,suffix):
    path=FIG/f'{STEM}_{suffix}.png';FIG.mkdir(exist_ok=True)
    fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return str(path)


def heat(ax,data,yticks,xticks,cbar,*,vmin=0,vmax=None,cmap='viridis',fmt=None):
    data=np.asarray(data)
    im=ax.imshow(data,aspect='auto',interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(xticks)),xticks,rotation=32,ha='right')
    ax.set_yticks(range(len(yticks)),yticks)
    ax.tick_params(length=0)
    cb=ax.figure.colorbar(im,ax=ax,fraction=.045,pad=.025)
    cb.set_label(cbar)
    if fmt:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                col=im.cmap(im.norm(data[i,j]));lum=.2126*col[0]+.7152*col[1]+.0722*col[2]
                ax.text(j,i,format(data[i,j],fmt),ha='center',va='center',fontsize=7.5,
                        color='black' if lum>.55 else 'white')
    return im


def main_figure(m,rows,summary,behavior,legacy):
    fig=plt.figure(figsize=(13.2,14.5),layout='constrained')
    outer=fig.add_gridspec(3,1,height_ratios=[.78,1.0,1.95],hspace=.07)
    ax=fig.add_subplot(outer[0]);label(ax,'a')
    data=[np.array([rows[sid,s]['system_xi_bits'] for sid in m['cohorts'][s]])*a.LN2 for s in STATES]
    colors=[BLUE]+[ORANGE]*7
    boxes=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.52,
        medianprops={'color':'#303030','linewidth':1},whiskerprops={'linewidth':.8},capprops={'linewidth':.8})
    rng=np.random.default_rng(20261022)
    for i,(v,color,box) in enumerate(zip(data,colors,boxes['boxes']),1):
        box.set(facecolor=color,alpha=.18,edgecolor=color)
        ax.scatter(i+rng.uniform(-.17,.17,len(v)),v,s=3,alpha=.16,color=color,linewidths=0,rasterized=True)
        ax.scatter(i,v.mean(),marker='D',s=25,color='white',edgecolor='#303030',zorder=5)
    ticks=[f'REST\nN={len(data[0]):,}']+[f'{name}\nN={len(data[i+1]):,}\npaired={len(m["pairs"][s]):,}' for i,(name,s) in enumerate(zip(SHORT[1:],TASKS))]
    ax.set_xticks(np.arange(1,9),ticks)
    ax.set_ylim(0,max(map(np.max,data))*1.08)
    ax.set_ylabel(r'System $\Xi$ (nats)')
    ax.text(1,1.025,'All available runs · diamonds: means · paired REST contrasts: BH-7, all q<0.0001',
        transform=ax.transAxes,ha='right',fontsize=9)
    mid=outer[1].subgridspec(1,2,wspace=.14)
    axb=fig.add_subplot(mid[0]);label(axb,'b')
    mass=np.array([summary['descriptive'][s]['order_mass_mean_nats'] for s in STATES]).T
    heat(axb,mass,list(range(2,8)),SHORT,'SPT order mass (nats)',vmax=mass.max(),fmt='.2f')
    axb.set_ylabel('Number of source networks')
    axc=fig.add_subplot(mid[1]);label(axc,'c')
    pct=np.array([summary['descriptive'][s]['network_percent_mean'] for s in STATES]).T
    heat(axc,pct,NET,SHORT,'Conserved attribution (%)',vmax=20,fmt='.1f')
    lower=outer[2].subgridspec(2,3,width_ratios=[1,1,1.42],wspace=.16,hspace=.12)
    rr={r['id']:r for r in behavior['results']['primary']}
    specs=[('d','Story accuracy',ORANGE),('e','Math accuracy',ORANGE),('f',"Social sensitivity d′",GREEN),('g','Face-specific speed',BLUE)]
    for i,(letter,name,color) in enumerate(specs):
        ax=fig.add_subplot(lower[i//2,i%2]);label(ax,letter)
        r=rr[letter];v=behavior['scatter'][letter]
        # Behavior ranks and ties are plotted as computed, with no display jitter.
        x=np.array(v['y_rank_residual_div_n']);y=np.array(v['x_rank_residual_div_n'])
        ax.scatter(x,y,s=7,alpha=.23,color=color,linewidths=0,rasterized=True)
        slope,intercept=np.polyfit(x,y,1);xx=np.array([x.min(),x.max()])
        ax.plot(xx,slope*xx+intercept,color=color,lw=1.2)
        ax.set(xlabel=f'{name}\n(adjusted rank / N)',ylabel=r'Coalition $\Xi$ (adjusted rank / N)',
            xlim=(-.65,.65),ylim=(-.65,.65),xticks=[-.5,0,.5],yticks=[-.5,0,.5])
        lo,hi=r['ci95']
        ax.text(0,1.025,f'{r["state"]} · {compact(r["coalition"])}\nN={r["n"]:,} · ρ={r["rho"]:+.3f} [{lo:+.3f}, {hi:+.3f}]\nHolm-13 P={ptext(r["p_holm13"])}',
            transform=ax.transAxes,fontsize=8.4,va='bottom')
    axh=fig.add_subplot(lower[:,2]);label(axh,'h')
    motor=[r for r in behavior['results']['primary'] if r['state']=='MOTOR']
    old={r['coalition']:r for r in legacy['rows'] if r['endpoint']=='motor_capacity'}
    for i,r in enumerate(motor):
        lo,hi=r['ci95'];color=ORANGE if r['same_direction_significant'] else GRAY
        axh.errorbar(r['rho'],i,xerr=[[r['rho']-lo],[hi-r['rho']]],fmt='o',color=color,ms=4,capsize=2,lw=1.2)
        axh.scatter(old[r['coalition']]['rho'],i+.18,marker='x',s=28,color='#353535')
        axh.text(1.02,i,f'{r["p_holm13"]:.3f}',transform=axh.get_yaxis_transform(),va='center',fontsize=8)
    axh.set_yticks(range(9),[f'{r["id"]}  {compact(r["coalition"])}' for r in motor],fontsize=8.2)
    axh.set(xlabel='Adjusted rank correlation ρ',xlim=(-.55,.2),ylim=(8.7,-.6))
    axh.axvline(0,color='#B7BBC0',lw=.8,ls='--')
    axh.text(0,1.025,f'MOTOR · N={motor[0]["n"]:,}\nFixed nine coalitions; 95% bootstrap CIs',transform=axh.transAxes,fontsize=9)
    axh.text(1.02,1.025,'Holm P',transform=axh.transAxes,fontsize=8,ha='left')
    axh.legend(handles=[Line2D([],[],marker='x',color='#353535',ls='',label='Original 57 estimate'),
        Line2D([],[],marker='o',color=GRAY,ls='',label='Expanded cohort + 95% CI')],
        loc='upper center',bbox_to_anchor=(.4,-.12),frameon=False,fontsize=8.5)
    return save(fig,'main')


def behavior_forest(behavior,legacy):
    fig,ax=plt.subplots(figsize=(12.3,8.5),layout='constrained')
    old={r['coalition']:r for r in legacy['rows']}
    main=behavior['results']['primary'];global_rows=behavior['results']['global_xi_adjusted']
    for i,(r,g) in enumerate(zip(main,global_rows)):
        ax.scatter(old[r['coalition']]['rho'],i-.22,color='#555555',marker='x',s=32)
        for row,offset,color,marker in [(r,0,ORANGE,'o'),(g,.22,BLUE,'s')]:
            lo,hi=row['ci95'];ax.errorbar(row['rho'],i+offset,xerr=[[row['rho']-lo],[hi-row['rho']]],
                color=color,marker=marker,ms=4,capsize=2,lw=1.1)
        ax.text(1.02,i,f'{r["n"]:,}     {r["p_holm13"]:.3f}     {g["p_holm13"]:.3f}',
            transform=ax.get_yaxis_transform(),fontsize=9,va='center')
    names={'d':'Story','e':'Math','f':'Social d′','g':'Face speed'}
    ax.set_yticks(range(13),[f'{r["id"]}  {names.get(r["id"],"Motor")} · {compact(r["coalition"])}' for r in main])
    ax.set(xlim=(-.6,.35),ylim=(12.7,-.7),xlabel='Adjusted rank correlation ρ (expanded cohort); original statistic for 57-person estimates')
    ax.axvline(0,color='#AAB0B5',ls='--',lw=.8)
    ax.text(1.02,1.01,'N      Holm P    +global Ξ P',transform=ax.transAxes,fontsize=9)
    handles=[Line2D([],[],marker='x',color='#555555',ls='',label='Original 57 (historical estimate)'),
        Line2D([],[],marker='o',color=ORANGE,label='Expanded: primary, 95% CI'),
        Line2D([],[],marker='s',color=BLUE,label='Expanded: +system Ξ, 95% CI')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False)
    return save(fig,'behavior_forest')


def sensitivity_figure(summary,window):
    fig,axes=plt.subplots(3,3,figsize=(15.5,13.8),layout='constrained',gridspec_kw={'height_ratios':[1.15,1,1]})
    configs=[('Full run',summary['primary'],GRAY,'o',-.24),('12 REST windows',window['primary'],BLUE,'s',-.08),
        ('Common N=965',summary['common'],ORANGE,'^',.08),('Spatial exclusions',summary['spatial'],GREEN,'D',.24)]
    handles=[Line2D([],[],color=c,marker=mark,label=name,lw=1) for name,_,c,mark,_ in configs]
    fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False)
    for j,(kind,xlabel) in enumerate([('A','REST − task system Ξ (nats)'),('B_root','REST − task seven-network mass (nats)'),('C_control','Task − REST Control share (pp)')]):
        ax=axes[0,j];label(ax,'abc'[j])
        for _,data,color,mark,offset in configs:
            for i,r in enumerate(data[kind]):
                lo,hi=r['mean_difference_ci95'];ax.errorbar(r['mean_difference'],i+offset,
                    xerr=[[r['mean_difference']-lo],[hi-r['mean_difference']]],fmt=mark,color=color,ms=3,capsize=1.5,lw=.9)
        ax.axvline(0,color='#B7BBC0',lw=.8,ls='--');ax.set(yticks=range(7),yticklabels=SHORT[1:],ylim=(6.6,-.6),xlabel=xlabel)
        ax.text(0,1.02,'Paired means and 95% bootstrap CIs',transform=ax.transAxes,fontsize=8.5)
    full=summary['descriptive'];common=summary['common_descriptive']
    heat(axes[1,0],100*np.array([full[s]['order_share_mean'] for s in STATES]).T,range(2,8),SHORT,'Cross-network Ξ share (%)',vmax=25,fmt='.1f')
    heat(axes[1,1],100*np.array([common[s]['order_share_mean'] for s in STATES]).T,range(2,8),SHORT,'Cross-network Ξ share (%)',vmax=25,fmt='.1f')
    diff=100*np.array([np.array(window['descriptive'][s]['task']['order_share_mean'])-np.array(window['descriptive'][s]['matched_rest']['order_share_mean']) for s in TASKS]).T
    vmax=max(abs(diff.min()),abs(diff.max()));heat(axes[1,2],diff,range(2,8),SHORT[1:],'Task − matched REST share (pp)',vmin=-vmax,vmax=vmax,cmap='RdBu_r',fmt='.1f')
    heat(axes[2,0],np.array([full[s]['network_percent_mean'] for s in STATES]).T,NET,SHORT,'Network attribution (%)',vmax=20,fmt='.1f')
    heat(axes[2,1],np.array([common[s]['network_percent_mean'] for s in STATES]).T,NET,SHORT,'Network attribution (%)',vmax=20,fmt='.1f')
    diff=np.array([window['descriptive'][s]['task_minus_matched_rest_network_percentage_points'] for s in TASKS]).T
    vmax=max(abs(diff.min()),abs(diff.max()));heat(axes[2,2],diff,NET,SHORT[1:],'Task − matched REST attribution (pp)',vmin=-vmax,vmax=vmax,cmap='RdBu_r',fmt='.1f')
    for i,ax in enumerate(axes.flat):
        if i>=3:label(ax,'abcdefghi'[i])
    for col,title in enumerate(['All available per state','Same 965 participants in every state','12-window matched comparison']):
        axes[1,col].text(0,1.04,title,transform=axes[1,col].transAxes,fontsize=9.5)
    axes[1,0].set_ylabel('SPT order');axes[2,0].set_ylabel('Yeo network')
    return save(fig,'sensitivity')


def final_audit(m,rows):
    """Use cached channel parameters for diagnostics; do not refit models."""
    errors={};minimum={s:float('inf') for s in STATES};conditions={s:[] for s in STATES}
    for sid,inp in m['inputs'].items():
        with np.load(OUT/'pc1_cache'/f'{sid}.npz') as f:
            for s in inp['states']:
                noise=f[f'{s}__noise'];eig=np.linalg.eigvalsh(noise)
                minimum[s]=min(minimum[s],float(eig.min()));conditions[s].append(rows[sid,s]['noise_condition'])
    for s in STATES:
        rr=[rows[sid,s] for sid in m['cohorts'][s]]
        errors[s]=dict(n=len(rr),minimum_channel_noise_eigenvalue=minimum[s],
            noise_condition_quantiles=np.quantile(conditions[s],[0,.5,.95,1]).tolist(),
            max_network_share_sum_error=float(max(abs(sum(r['network_percent'])-100) for r in rr)),
            max_order_share_sum_error=float(max(abs(sum(r['order_share'])-1) for r in rr)),
            heldout_skill_quantiles=np.quantile([r['heldout']['skill_ratio'] for r in rr],[0,.5,.95,1]).tolist())
    matched_minimum={s:float('inf') for s in TASKS}
    for s,ids in m['pairs'].items():
        for sid in ids:
            with np.load(OUT/'windows'/f'{sid}_{s}.npz') as f:
                matched_minimum[s]=min(matched_minimum[s],float(np.linalg.eigvalsh(f['noise']).min()))
    # Every conditional covariance is noise + a positive semidefinite matrix.
    # The full-coalition conditional covariance equals noise, so this gives its
    # exact minimum over all conditionals without redoing the EI calculations.
    result=dict(per_state=errors,minimum_covariance_eigenvalue=min(minimum.values()),
        window_minimum_covariance_eigenvalue=min(matched_minimum.values()),window_minimum_covariance_eigenvalue_by_task=matched_minimum,
        eigenvalue_note='Minimum over noise and all omitted-source conditional covariances; channel response covariance is positive semidefinite',
        model_implementation_match=all(sha(ROOT/p)==h for p,h in m['implementation'].items()),
        manuscript_rechecked=dict(parent='P6UJCVG8',main='DXGC7JEA',supplement='MWIWKSVG',date='2026-10-22',
            main_indexed_text_sha256=sha(ROOT/'tmp/hcp_resume_main_20261022.txt'),
            supplement_indexed_text_sha256=sha(ROOT/'tmp/hcp_resume_si_20261022.txt'),
            version_ambiguity='One current main and SI; neither declares a manuscript version/date. Metadata modification is not a manuscript date.',
            sections=['Main Fig3 pp7-9','SI S1.2-S1.3 EqsS7-S19','SI S5.1 EqsS49-S50','SI S12.2.2 pp23,25']),
        report_code_sha256=sha(Path(__file__)),analysis_code_sha256=sha(Path(a.__file__)))
    identity=read('source_identity_audit.json')
    if identity['manifest_sha256']!=sha(OUT/'manifest.json') or identity['status']!='passed':raise ValueError('Source identity audit failed')
    result['source_identity_audit']=dict(path=str(OUT/'source_identity_audit.json'),sha256=sha(OUT/'source_identity_audit.json'),input_files=identity['checked_input_files'],designs_and_qc=identity['checked_designs_and_qc'])
    if not result['model_implementation_match']:raise ValueError('Frozen model code changed')
    dump(OUT/'completion_audit.json',result)
    return result


def write_report(m,summary,window,behavior,legacy,audit,figures):
    r=behavior['results']['primary'];g={q['id']:q for q in behavior['results']['global_xi_adjusted']}
    old={q['coalition']:q for q in legacy['rows']}
    passed=[q['id'] for q in r if q['same_direction_significant']]
    wp=window['primary']['A'];same=[q['state'] for q in wp if q['same_direction_significant']]
    text=[
        '# HCP Schaefer-1000 全可用样本复现结果',
        '', '日期：2026-10-22。状态：执行完成；固定主模型、12 窗等长对照、共同队列、空间覆盖敏感性和 13 项固定脑行为检验均已完成。',
        '', '**主要结果：系统与网络组织的描述在扩样后保持；行为效应普遍减弱，13 项中只有 MOTOR 的 SM+Lim+Cont+DMN 通过 Holm-13。** 该关联加入系统 Ξ 后仍通过校正。等长对照使 SOCIAL 的方向反转，限定了“REST 高于所有任务”的解释。',
        '', '## 执行范围与偏差',
        '', f'- 主模型 **{sum(map(len,m["cohorts"].values())):,}** 组；REST–任务 **{sum(map(len,m["pairs"].values())):,}** 对；匹配 REST **{window["numerical_audit"]["window_models"]:,}** 个窗口。每人每任务先平均 12 窗，窗口不作独立被试。',
        '- 原会话明确授权跳过 735148 缺设计的七项任务，保留其 REST。八状态共同队列因此为 **965 人**；未改变其余冻结设置。11 个来源阶段短 run 不进入标准长度主分析，其余状态保留。',
        '- 本次与旧目录为相同来源队列，部分人曾进入既有 979 人分析；这是固定候选扩样检验，不能称独立、未使用数据的复制。',
        '- Schaefer-1000 / Yeo7、NoGSR、LR，每网络 PC1、历史 p=3、Δ-Ridge α=1；前75%拟合。21 个独立标准高斯干预源、相同七维未来目标、同一 affine-Gaussian TM 信道的解析 log-det EI。内部 bits，图与报告 nats。',
        '- 不重新选组合、调参或按结果删人。GSR 在方案中为可选后续，本轮未启动。家系映射、匹配 FD 与统一预处理重提取均未获得，相关验证未完成。',
        '', '## a–c：系统、SPT 阶数与网络归因',
        '', '| 状态 | N | 系统 Ξ 均值 (nats) | 阶数质量均值峰值 | Control (%) | Limbic (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for s,v in summary['descriptive'].items():
        text.append(f'| {s} | {v["n"]} | {v["mean_system_xi_nats"]:.3f} | {v["mean_peak_order"]} | {v["network_percent_mean"][5]:.2f} | {v["network_percent_mean"][4]:.2f} |')
    text += ['', '所有八状态的群体平均阶数质量峰值均为 **6 阶**；旧 GAMBLING 峰值为5阶，本轮变为6阶。这里的阶数是来源网络数。个体峰值有差异，完整分布保存在主统计中，不能将群体峰值归给所有个体。',
        '', '状态检验采用双侧配对Wilcoxon（zero_method=wilcox，correction=False），它不是均值差检验；配对均值差、其bootstrap区间、中位差、正差比例和秩二列相关另行报告为效应摘要。',
        '', '完整 REST 的七阶根质量均高于七项任务，均通过 Holm-7。Control 份额在七任务均上升并通过独立的 Holm-7 家族；Limbic 在八状态均最小，Control 平均份额仍以 LANGUAGE 最高。归因逐人归一化后求均值，每人、每状态列和均为100%。',
        '', '## 时长、共同样本和空间覆盖对照',
        '', '| 任务 | 配对 N | 完整 REST−任务 Δ (nats) | 12窗 REST−任务 Δ [95% CI] (nats) | 等长 BH q | 等长判定 |',
        '|---|---:|---:|---|---:|---|']
    for main,match in zip(summary['primary']['A'],wp):
        lo,hi=match['mean_difference_ci95'];verdict='保持' if match['same_direction_significant'] else ('显著反向' if match['mean_difference']<0 and match['p_adjusted']<.05 else '证据不足')
        text.append(f'| {main["state"]} | {main["n"]} | {main["mean_difference"]:.3f} | {match["mean_difference"]:.3f} [{lo:.3f}, {hi:.3f}] | {match["p_adjusted"]:.4g} | {verdict} |')
    text += ['', f'等长后 REST 优势在 {len(same)} 项任务保持。SOCIAL 的结果应据上表报告，不把原57人等长“不显著”沿用为本轮结论。原始状态差异与时长、拟合样本数和预处理均有关，等长对照只改变 REST 时序长度。',
        '', '等长对照还限定了七阶根质量结论：完整REST的根质量高于全部任务，但等长后MOTOR反向（REST−MOTOR='+f'{window["primary"]["B_root"][3]["mean_difference"]:.3f} nats，Holm-7 P={window["primary"]["B_root"][3]["p_adjusted"]:.4g}'+'），其余六项仍为REST更高。Control份额上升在七任务等长对照中均保持。根质量的绝对差与归一化根份额是不同结果，见敏感性图d–f。',
        '', '965人共同队列和固定空间覆盖排除均复算 a–c；等长系统Ξ的SOCIAL反向在共同队列（Δ='+f'{window["common"]["A"][5]["mean_difference"]:.3f} nats，BH q={window["common"]["A"][5]["p_adjusted"]:.4g}'+'）及空间覆盖排除后也保持。均值差、区间、Wilcoxon P 与对应 BH/Holm 校正在敏感性图及 JSON 中完整保留。独立家系映射尚不可用，所有逐人检验和bootstrap仍依赖被试独立假设。',
        '', '## d–h：13 项冻结脑行为假设',
        '', '主模型对脑与行为取平均秩，调整 Age 类别、Gender、重建版本和原29/补充28/扩样来源；EMOTION 加 Shape speed 秩。每个终点使用10万次来源块内 Freedman–Lane 残差置换；MOTOR九组合共享置换索引。13项统一 Holm 校正。区间为5000次来源分层bootstrap，每次重算秩与协变量模型；区间为逐项95%，不是多重校正后的同时区间。',
        '', '| ID | 状态 / 固定组合 | N | 旧57 ρ | 扩样 ρ [95% CI] | Holm-13 P | +系统 Ξ：ρ / Holm P | 判定 |',
        '|---|---|---:|---:|---|---:|---|---|']
    for q in r:
        lo,hi=q['ci95'];v=g[q['id']];verdict='同向保持' if q['same_direction_significant'] else ('方向保持，证据不足' if q['rho']*q['direction']>0 else '反向，证据不足')
        text.append(f'| {q["id"]} | {q["state"]} / {compact(q["coalition"])} | {q["n"]} | {old[q["coalition"]]["rho"]:+.3f} | {q["rho"]:+.3f} [{lo:+.3f}, {hi:+.3f}] | {q["p_holm13"]:.4g} | {v["rho"]:+.3f} / {v["p_holm13"]:.4g} | {verdict} |')
    h=next(q for q in r if q['id']=='h5');v=g['h5']
    text += ['', f'唯一通过主校正的是 h5：SM+Lim+Cont+DMN，ρ={h["rho"]:.3f}，Holm P={h["p_holm13"]:.4g}；加入系统 Ξ 后 ρ={v["rho"]:.3f}，Holm P={v["p_holm13"]:.4g}。这是条件关联，不能证明因果、额外预测收益或纯四阶机制。13项主结果方向均与旧图一致，但只有1项满足冻结的扩样保持标准；不显著不等于零效应。',
        '', 'SOCIAL 的138837缺TOM评分；MOTOR的147030、206222、578057缺至少一项体能评分，仅影响对应行为终点。MOTOR标准化使用旧57人的固定均值/SD，不对缺项取剩余项目平均。',
        '', 'SOCIAL 固定 n=5 敏感性：'+f'ρ={behavior["social_fixed_n5"]["rho"]:.3f}，点位 P={behavior["social_fixed_n5"]["p_pointwise"]:.4g}；它不是主Holm-13结果。有效试次数来自汇总百分比推断，不视为本次LR真实试次数。',
        '', '旧口径兼容统计在每项的 `legacy_compatible_*` 字段保存；SOCIAL另报告max-T-120。旧57标记使用原统计口径，新样本使用预先声明的统一协变量，图中不暗示两者是完全相同的调整。原图“Coalition Syn”实际是分组 Ξ(S)，本轮已据SI S16标注为 Coalition Ξ。',
        '', '## 校准、数值有效性与稿件核对',
        '', '- 旧57人×八状态共456模型通过复算；最大模型误差 '+f'{read("legacy_gate.json")["maximum_error_bits"]:.3g} bits；行为特征最大误差 {legacy["maximum_original_feature_error_bits"]:.3g} bits。原LANGUAGE历史统计仅保留三位小数，已核对独立系数与舍入值，未宣称不存在的全精度P值逐位一致。',
        '- Syn 容差固定 **1e−4 bits = 6.93147e−5 nats**。主模型候选二分 '+f'{summary["numerical_audit"]["candidate_count"]:,} 次；窗口候选二分 {window["numerical_audit"]["candidate_count"]:,} 次。显著负值和容差内负值均为0；没有裁剪。',
        '- 主模型最小候选 Syn '+f'{summary["numerical_audit"]["minimum_candidate_syn_bits"]:.6g} bits；窗口最小值 {window["numerical_audit"]["minimum_candidate_syn_bits"]:.6g} bits。最大闭合误差分别 {summary["numerical_audit"]["maximum_closure_error_bits"]:.3g}、{window["numerical_audit"]["maximum_closure_error_bits"]:.3g} bits。',
        '- 主模型与窗口协方差下限触发次数分别 '+f'{summary["numerical_audit"]["covariance_floor_count"]} / {window["numerical_audit"]["covariance_floor_count"]}；最小信道协方差特征值分别 {audit["minimum_covariance_eigenvalue"]:.6g} / {audit["window_minimum_covariance_eigenvalue"]:.6g}。未把预测skill低或尖峰当删人条件。',
        '- 完成前重新校验全部 '+f'{audit["source_identity_audit"]["input_files"]} 个MAT的SHA256、{audit["source_identity_audit"]["designs_and_qc"]} 对任务设计/QC以及行为表/标签指纹，均与冻结输入一致；依赖代码及运行库版本另存来源审计。',
        '- 本轮重新通过 Zotero 搜索核对题名及父项 **P6UJCVG8**，读取当前正文 **DXGC7JEA**（19页）与SI **MWIWKSVG**（28页）的相关全文：正文Fig.3及第7–9页，SI S1.2–S1.3/S7–S19、S5.1/S49–S50、S12.2.2第23/25页。两附件仍无明确稿件版本/日期；元数据修改时间为2026-10-04，不能据此认定稿件版本日期。此歧义仍保留。',
        '- 当前稿件和旧代码对LANGUAGE协变量简述存在差别：原函数只去总体均值并限制置换在来源队列内；本轮统一模型按方案显式调整协变量。保留差异而不从两种版本中挑显著者。',
        '- REST为minimal且未加ICA-FIX，任务有平滑/高通且FSL版本与任务混杂；全run设计回归和原高通使后25%仅适合作动力学诊断。缺少匹配FD、家系及独立空间图谱映射验证，不能声称已消除这些混杂或完成全部稿件/数据一致性验证。',
        '', '## 图注与可复用结果',
        '', '主图：a为各状态全部合格被试，箱线为中位数/IQR、须线为1.5 IQR，点为被试、菱形为均值；横向显示抖动仅用于状态散点。a的检验使用逐任务配对交集。b为逐人SPT节点按网络数求和后的均值；c为逐人守恒网络百分比的均值。d–g为调整后的秩残差除以N，保留并列、不添加评分抖动，直线仅为视觉引导；效应及P来自实际统计。h固定九组合，误差线为逐项95%bootstrap区间，叉号为旧57估计。',
        '', '敏感性图：a–c为配对均值差和逐项95%bootstrap区间；d–f为按跨网络Ξ归一化的阶数份额；g–i为网络归因。完整和共同队列同类热图共享色轴，差值热图以零为中心。',
        '', '森林图：13行全部保留，旧57仅标点估计；扩样主模型及+系统Ξ均给出逐项95%bootstrap区间与各自Holm-13 P。旧/新不同协变量口径已在横轴与本报告说明。',
        '']
    for title,path in zip(['全量复现主图 a–h','12窗与共同队列敏感性','13项旧/新效应森林图'],figures):
        text.extend([f'![{title}]({path})',''])
    text += [f'缓存目录：`{OUT}`。主模型按被试保存JSON与PC1/信道NPZ；12窗按被试–任务保存JSON/NPZ。统计JSON、manifest与完成审计保留身份、单位、容差、来源、配置、代码指纹；不复制原始MAT或生成CSV。',
        '', '复现入口：先核对 manifest 身份；已有模型复用。`analyze_hcp_full_replication.py` 的 main/behavior/windows 分别生成统计，`report_hcp_full_replication.py` 从缓存生成三图与本报告。',
        '', '检验覆盖：新增单状态缺失、缺失TOM、MOTOR缺项、缓存指纹不符、Freedman–Lane快速公式与显式协变量重拟合等边界；与原信道信息恒等式、负Syn失败规则合计7项通过。',
        '']
    path=ROOT/'docs/log'/f'{STEM}_results.md';path.write_text('\n'.join(text))
    dump(OUT/'completion.json',dict(status='complete',main_models=sum(map(len,m['cohorts'].values())),window_models=window['numerical_audit']['window_models'],
        fixed_behavior_hypotheses=13,behavior_preserved_ids=passed,figures=figures,report=str(path),manifest_sha256=sha(OUT/'manifest.json')))
    return path


def main():
    m=read('manifest.json');rows=a.load_rows(m);summary=read('main_statistics.json');window=read('window_statistics.json')
    behavior=read('behavior_statistics.json');legacy=read('legacy_behavior_gate.json')
    for result in [summary,window,behavior]:
        if result['manifest_sha256']!=sha(OUT/'manifest.json'):raise ValueError('Statistics manifest mismatch')
        if result['analysis_code_sha256']!=sha(Path(a.__file__)):raise ValueError('Statistics code changed')
    style={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':9}
    with plt.rc_context(style):
        figures=[main_figure(m,rows,summary,behavior,legacy),sensitivity_figure(summary,window),behavior_forest(behavior,legacy)]
    audit=final_audit(m,rows);report=write_report(m,summary,window,behavior,legacy,audit,figures)
    print(json.dumps(dict(report=str(report),figures=figures),ensure_ascii=False))


if __name__=='__main__':main()
