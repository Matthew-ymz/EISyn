#!/usr/bin/env python3
"""Run the user-approved all93 BOLD appendix sensitivity, not the main figure."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dmf_bold_appendix_metrics as kernel
from scripts.dmf_dense_statistics import subject_landmarks
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.run_dmf_subject_consistency import atomic_json,digest
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

DENSE=ROOT/'results/dmf_schaefer100/subject_curves_93_dense'
OUTPUT=ROOT/'results/dmf_schaefer100/bold_all_metrics'
FIGURE=ROOT/'docs/reports/assets/dmf_subject_dense/bold_all_metrics_appendix'
W={}


def prepare(output=OUTPUT):
    dense=json.loads((DENSE/'contract.json').read_text())
    old=json.loads((DENSE/'fixed_window_view.json').read_text())
    for f,sha in old['source_sha256'].items():
        if digest(DENSE/f)!=sha: raise ValueError(f'Frozen source changed: {f}')
    c=dict(version='all93-BOLD-fitted-model-appendix-v1',
        subject_ids=dense['subject_ids'],G=dense['G'],seeds=dense['seeds'],
        authorization='2026-10-11 user approved all93 complete G scan (~1.5–2h), BOLD-fitted intervention approximation Xi and natural PhiR/WMS',
        native_parameters=dict(dense['native_protocol']['native_parameters'],t_total=20.,burn_in=5.),
        stabilization=dense['native_protocol']['stabilization'],
        hemodynamic_parameters=dense['native_protocol']['hemodynamic_parameters'],
        readout='Balloon-Windkessel of full 20-s natural E-rate trace; discard fixed first5s; 100ROI BOLD remaining15s; dt1ms; no downsampling',
        readout_initialization='Inherited zero hemodynamic signal and unit flow/volume/deoxy; rates centered over full20s by original transform function',
        natural_simulation_seed='62000 + nominal seed, same stream across subjects and G',
        horizon_steps=300,horizon_seconds=.3,source_dimension=100,target_dimension=100,
        model='First-order delta-Ridge alpha1; first75% of lagged pairs fit; purge300-source-step gap before heldout prediction diagnostics',
        standardization='Training-source mean/std only, shared source/target units; no normalization of output information curves',
        Xi='Scalar-fine I_q(full source;full future)-sum_i I_q(source_i;full future)',
        Xi_prior='Independent uniform[-sqrt(3),sqrt(3)] standardized inputs (unit variance), affine-TM Gaussian-moment approximation; diagonal natural marginal variances + ridge',
        natural_prior='Empirical correlated covariance of the same BOLD training inputs + ridge',
        shared_channel='Same fitted transition and training-residual noise; only source prior differs between Xi and natural baselines',
        estimator='Repository affine triangular Gaussian TM; covariance/noise ridge1e-6; no extra query floors or MI/Syn projection',
        phi_r='Whole-system Gaussian-MMI PhiR; fixed left50|right50 ROI partition, no MIP or pair averaging',
        partition_provenance='Supplement Fig4/p24: input order left hemisphere followed by right; ROI identities remain inferred in repository',
        WMS='Signed observational SOURCE WMS I_p(full;fullfuture)-sum_i I_p(single source_i;fullfuture); not temporal self-target WMS or Syn',
        units='nats',nonnegative_tolerance_nats=kernel.TOL_NATS,
        landmarks='Global Xi/PhiR maximum; global signed WMS minimum; first tie, flat/boundary extrema miss',
        window='Original frozen independent neural-rate maximum-slope interval +0.2G each side; width0.5G, posthoc',
        model_diagnostics='Heldout RMSE and persistence RMSE; worse-than-persistence fraction retained, never retune on hit rate',
        inference='Descriptive appendix sensitivity; no new hypothesis tests; not a BOLD-only causal ablation or physical neural intervention Xi',
        limitations=['BOLD alone is an incomplete readout of neural/hemodynamic state; first-order observable model need not be Markov or causally identified',
            'Fitted-model independent BOLD interventions can leave the natural data manifold',
            '20s traces and fixed5s warmup do not establish stationary fMRI or independent samples',
            'Trace length, readout dimension, fitted channel, prior support, regularization and PhiR partition also change from the neural main experiment',
            'Pilot heldout prediction underperforms persistence; keep this limitation with the full result'],
        dense_contract_sha256=digest(DENSE/'contract.json'),
        dense_inputs_sha256=digest(DENSE/'inputs.npz'),
        dense_windows_sha256=digest(DENSE/'window_summary.json'),
        kernel_sha256=digest(Path(kernel.__file__)),dmf_implementation_sha256=digest(ROOT/'exp/brain/dmf_fig6.py'),
        manuscript=dict(parent='P6UJCVG8',body='DXGC7JEA',supplement='MWIWKSVG',checked_date='2026-10-11',
            explicit_draft_date_or_version=None,ambiguity='Current attachments lack draft dates/revisions; metadata is not version evidence',
            locations='Main pp6-7 and15-16 Eqs5-8; Supplement pp23-25 S12.2.1/2',
            method_relation='Follow factorized common-intervention definition and affine-TM preference. Borrow fixed delta-Ridge alpha1 from HCP, but use100ROI first-order state, not its7network/third-order model. Appendix approximation, not manuscript original DMF protocol.'))
    output.mkdir(parents=True,exist_ok=True);(output/'conditions').mkdir(exist_ok=True)
    path=output/'contract.json'
    if path.exists() and json.loads(path.read_text())!=c: raise ValueError('BOLD appendix contract changed')
    if not path.exists(): atomic_json(path,c)
    return c


def initialize(output):
    global W
    output=Path(output);c=json.loads((output/'contract.json').read_text())
    if digest(Path(kernel.__file__))!=c['kernel_sha256']: raise ValueError('BOLD scientific kernel changed')
    limiter=threadpool_limits(limits=1);dmf=load_dmf_module()
    with np.load(DENSE/'inputs.npz') as a:
        if a['subject_ids'].tolist()!=c['subject_ids']: raise ValueError('Input subject order changed')
        matrices,jf=a['connectivity'].copy(),a['j_fic'].copy()
    W=dict(output=output,c=c,sha=digest(output/'contract.json'),limiter=limiter,dmf=dmf,matrices=matrices,jf=jf,
        p=dmf.DMFParameters(**c['native_parameters']),stabilization=dmf.StabilizationParameters(**c['stabilization']))


def job(indices):
    pi,gi,si=indices;w=W;c=w['c'];subject,g,seed=c['subject_ids'][pi],c['G'][gi],c['seeds'][si]
    path=w['output']/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['contract_sha256'])!=w['sha']: raise ValueError('Wrong BOLD condition cache')
        return dict(condition=path.name,reused=True)
    started=time.perf_counter()
    sim=w['dmf'].simulate_dmf(w['matrices'][pi],g,w['jf'],parameters=w['p'],
        stabilization_parameters=w['stabilization'],seed=62000+seed,record_rate_trace=True,record_state_trace=True)
    rates=np.asarray(sim['region_rate_trace_hz'])
    se,si_trace=np.asarray(sim['state_se_trace']),np.asarray(sim['state_si_trace'])
    if (not all(np.isfinite(a).all() for a in (rates,se,si_trace))
        or min(se.min(),si_trace.min())<0 or max(se.max(),si_trace.max())>1
        or rates.min()<0 or rates.max()>500): raise ArithmeticError(f'Invalid natural trajectory: {path.name}')
    bold=w['dmf'].transform_rates_to_bold(rates,dt=.001)[5000:]
    if bold.shape!=(15000,100): raise ValueError(f'Unexpected fixed BOLD length: {bold.shape}')
    r=kernel.bold_metrics(bold)
    natural,intervention=r.pop('natural_covariance'),r.pop('intervention_covariance')
    r.update(subject=subject,G=g,seed=seed,elapsed_seconds=time.perf_counter()-started,
        BOLD_sha256=hashlib.sha256(bold.tobytes()).hexdigest(),BOLD_timepoints=15000,
        mean_E_rate_hz=float(rates[5000:].mean()),
        natural_state_boundary_hit_count=int(((se==0)|(se==1)).sum()+((si_trace==0)|(si_trace==1)).sum()))
    atomic_savez(path,metrics_json=json.dumps(r,allow_nan=False),natural_covariance=natural,
        intervention_covariance=intervention,contract_sha256=w['sha'])
    return dict(condition=path.name,reused=False,xi_nats=r['xi_nats'],phi_r_nats=r['phi_r_nats'],
        wms_nats=r['wms_nats'],elapsed_seconds=round(r['elapsed_seconds'],2))


def aggregate(output=OUTPUT):
    c=json.loads((output/'contract.json').read_text());shape=(len(c['subject_ids']),len(c['seeds']),len(c['G']))
    values={k:np.empty(shape) for k in ('xi_nats','phi_r_nats','wms_nats')};records=[]
    for pi,subject in enumerate(c['subject_ids']):
        for si,seed in enumerate(c['seeds']):
            for gi,g in enumerate(c['G']):
                with np.load(output/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz') as a:
                    if str(a['contract_sha256'])!=digest(output/'contract.json'): raise ValueError('BOLD condition mismatch')
                    r=json.loads(str(a['metrics_json']));records.append(r)
                for k,v in values.items(): v[pi,si,gi]=r[k]
    with np.load(DENSE/'analysis_curves.npz') as a: rates=a['rate'].copy()
    scenario=next(s for s in json.loads((DENSE/'window_summary.json').read_text())['scenarios'] if s['k_each_side']==2)
    rows=[]
    for pi,old in enumerate(scenario['subject_rows']):
        row=json.loads(json.dumps(old))
        lm=subject_landmarks(c['G'],rates[pi].mean(0),{k:values[k+'_nats'][pi].mean(0) for k in ('xi','phi_r','wms')})
        if lm['transition']!=old['independent_transition']: raise ValueError('Frozen independent window changed')
        lo,hi=old['window']['interval']
        for m in lm['methods'].values():
            point=m['extremum_G'];m['hit']=bool(m['interior'] and point is not None and lo-1e-12<=point<=hi+1e-12)
        row['methods']=lm['methods'];rows.append(row)
    dense=json.loads((DENSE/'contract.json').read_text());cohorts={}
    for key,ids in [('all93',c['subject_ids'][:-1]),('primary85',dense['holdout_subject_ids']),('development8',dense['development_subject_ids'])]:
        selected=[r for r in rows if r['subject'] in ids]
        cohorts[key]=dict(subject_count=len(selected),methods={k:dict(hit_count=sum(r['methods'][k]['hit'] for r in selected),
            denominator=len(selected),hit_rate=sum(r['methods'][k]['hit'] for r in selected)/len(selected)) for k in ('xi','phi_r','wms')})
    atomic_savez(output/'curves.npz',subject_ids=c['subject_ids'],G=c['G'],seeds=c['seeds'],rate=rates,**values,contract_sha256=digest(output/'contract.json'))
    individual=[r for r in records if r['subject']!='group_mean_93']
    summary=dict(contract_sha256=digest(output/'contract.json'),curves_sha256=digest(output/'curves.npz'),
        condition_count=len(records),cohorts=cohorts,subject_rows=rows,
        audit=dict(tolerance_nats=kernel.TOL_NATS,minimum_nats=min(r['audit']['minimum_nats'] for r in records),
            tolerance_negative_count=sum(r['audit']['tolerance_negative_count'] for r in records),significant_violation_count=0,
            natural_boundary_hit_count=sum(r['natural_state_boundary_hit_count'] for r in records)),
        prediction_diagnostics=dict(condition_count=len(individual),
            worse_than_persistence_count=sum(r['mean_holdout_RMSE_standardized']>r['mean_persistence_RMSE_standardized'] for r in individual),
            mean_holdout_RMSE_standardized=float(np.mean([r['mean_holdout_RMSE_standardized'] for r in individual])),
            mean_persistence_RMSE_standardized=float(np.mean([r['mean_persistence_RMSE_standardized'] for r in individual]))),
        inference=c['inference'],limitations=c['limitations'])
    atomic_json(output/'summary.json',summary);return summary


def plot(output=OUTPUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from scripts.dmf_curve_shape import METHOD_COLORS
    c=json.loads((output/'contract.json').read_text());s=json.loads((output/'summary.json').read_text())
    if s['curves_sha256']!=digest(output/'curves.npz'): raise ValueError('BOLD appendix curves changed')
    with np.load(output/'curves.npz') as a:
        g=a['G'].copy();v={k:a[k+'_nats'].mean(1) for k in ('xi','phi_r','wms')}
    with np.load(DENSE/'inputs.npz') as a: rho=a['spectral_radius'][:-1].copy()
    style={'font.family':'sans-serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=(11.8,7.2),layout='constrained')
        norm=Normalize(float(rho.min()),float(rho.max()));colors=plt.cm.viridis(norm(rho))
        titles=[r'BOLD-fitted intervention approximation $\Xi$',r'Natural BOLD $\Phi^R$ (fixed LH | RH)',r'Natural BOLD signed source WMS']
        for ax,k,title,letter in zip(axes.ravel()[:3],('xi','phi_r','wms'),titles,'abc'):
            for i in range(93): ax.plot(g,v[k][i],color=colors[i],alpha=.5,lw=.7)
            ax.plot(g,v[k][-1],color='#202020',ls='--',lw=1.7)
            ax.set(xlim=(0,4),xlabel='$G$',ylabel='Information (nats)',title=title)
            if k!='wms': ax.set_ylim(bottom=0)
            ax.text(-.1,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
        ax=axes.ravel()[3]
        for y,k in zip([2,1,0],('xi','phi_r','wms')):
            m=s['cohorts']['all93']['methods'][k];percent=m['hit_rate']*100
            ax.barh(y,percent,color=METHOD_COLORS[k],height=.43)
            ax.text(percent+1.5,y,f"{percent:.1f}%  {m['hit_count']}/93",va='center',fontsize=9)
        ax.set(xlim=(0,118),xticks=[0,25,50,75,100],yticks=[2,1,0],
            yticklabels=[r'Approx. $\Xi$',r'$\Phi^R$ (LH | RH)','Source WMS'],
            xlabel='Hits in own neural transition window (%)',title='All93; frozen 0.5 G window; exploratory')
        ax.text(-.1,1.04,'d',transform=ax.transAxes,fontweight='bold',fontsize=12)
        fig.legend(handles=[Line2D([],[],color=plt.cm.viridis(.5),lw=1,label='93 individual SCs; 3-seed means'),
            Line2D([],[],color='#202020',ls='--',lw=1.7,label='Mean SC (separate simulation)')],loc='outside upper center',ncol=2,frameon=False)
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axes.ravel()[:3].tolist(),
            orientation='horizontal',shrink=.65,aspect=40,pad=.04)
        cb.set_label('Native SC spectral radius (same individual colors)')
        FIGURE.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('png','pdf'): fig.savefig(FIGURE.with_suffix('.'+ext),dpi=260,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    return s


def write_appendix(summary):
    report=ROOT/'docs/reports/brain.md';text=report.read_text()
    def result(k,cohort='all93'):
        m=summary['cohorts'][cohort]['methods'][k]
        return f"{m['hit_count']}/{m['denominator']}（{m['hit_rate']*100:.1f}%）"
    d=summary['prediction_diagnostics'];a=summary['audit']
    edge_counts={k:{g:sum(r['methods'][k]['extremum_G']==g for r in summary['subject_rows'])
        for g in (0.,4.)} for k in ('xi','phi_r','wms')}
    lo,hi='<!-- bold-all-metrics:start -->','<!-- bold-all-metrics:end -->'
    block=f'''{lo}
<a id="dmf-bold-all-metrics"></a>

## 附录 V：三指标均使用 BOLD 的模型敏感性对照

全部93人使用同一100ROI BOLD-like读出和300 ms未来。**BOLD拟合模型上的干预近似Ξ为{result('xi')}，自然系统级ΦR为{result('phi_r')}，自然source WMS为{result('wms')}。** 这组结果仅作附录备用，不替换主图的完整E/I神经状态比较。

![全BOLD读出下的三指标曲线与个体神经转折邻域命中率](assets/dmf_subject_dense/bold_all_metrics_appendix.png)

**图V1。** a为BOLD拟合模型上的干预近似Ξ，b为固定左／右半球50|50ROI分区的自然系统级ΦR，c为自然signed source WMS，d为原冻结0.5 G神经发放率转折窗口的命中率。细线为每人seed 3/4/5均值，颜色编码原生SC谱半径；黑色虚线为平均SC独立模拟，不进入93人统计。信息单位nats，G=0–4、步长0.1，无平滑、幅度归一化或峰位对齐。Ξ/ΦR取峰，signed WMS取谷，平坦或边界极值未命中，窗口未重选；图例在数据外。原85人子队列分别为Ξ{result('xi','primary85')}、ΦR{result('phi_r','primary85')}、WMS{result('wms','primary85')}，仅描述、不新增假设检验。

**曲线形状与固定判据。** Ξ的全局峰全部位于扫描边界：G=0有{edge_counts['xi'][0.]}人，G=4有{edge_counts['xi'][4.]}人；ΦR的边界峰分别为{edge_counts['phi_r'][0.]}人和{edge_counts['phi_r'][4.]}人。因此，两者的0/93首先反映本拟合模型下的曲线形状与原峰值判据不匹配，不能直接解释为与转折毫无关系。尤其Ξ常在转折附近呈谷形；本对照没有事后改用谷值来增加Ξ命中率。WMS有{edge_counts['wms'][4.]}人的全局谷位于G=4而判为未命中，其余83人的内部谷均命中各自冻结窗口。本结果仅描述该模型和判据下的表现，不能据此建立三指标的一般优劣排序。

各条件生成20秒自然DMF轨迹，经原Balloon-Windkessel函数转为BOLD；固定舍弃前5秒，剩余15秒均按1 ms记录。所有指标均从同一BOLD训练样本拟合的一阶Δ-Ridge模型（α=1）读取，前75%拟合、训练目标与留出源之间留300 ms间隔，剩余样本仅作预测诊断。训练源的均值／标准差用于共同标准化。两自然基线保留相关经验源先验；Ξ采用相同拟合通道与残差噪声，但使用独立、单位方差的标准化输入先验（均匀[-√3,√3]的Gaussian-moment近似），EI由仓库affine triangular TM计算。ΦR不做MIP搜索，也不是两两ΦR均值；这里使用的是source WMS，单源项均读取完整未来。

**解释边界。** BOLD不是完整神经／血流动力学状态；这个一阶拟合通道不自动成为已识别的因果模型。干预是BOLD拟合模型的反事实，可能离开自然数据流形，不能把a称为原DMF因果Ξ的直接BOLD转换。{d['worse_than_persistence_count']}/{d['condition_count']}个个体条件（{100*d['worse_than_persistence_count']/d['condition_count']:.1f}%）的留出RMSE高于保持当前状态的预测，平均标准化RMSE为{d['mean_holdout_RMSE_standardized']:.4g}，持久预测为{d['mean_persistence_RMSE_standardized']:.4g}；必须把这些预测限制和指标曲线一起解释。高频重叠时间点不是独立观测，20秒轨迹／5秒预热不等于稳定的真实fMRI。相比神经状态主实验，本对照还改变了维度、轨迹长度、模型、输入支持、ΦR分区和正则化，不能单独估计血流动力学转换的因果效果。

Ξ/ΦR原生非负容差10⁻⁸ nats，容差内负值查询计数{a['tolerance_negative_count']}，显著违反0；没有裁剪、投影或新增边缘查询下限。signed WMS不施加Syn非负限制。自然模拟继承原[0,1]状态截断，边界命中数{a['natural_boundary_hit_count']}另行记录。

2026-10-11重新查询Zotero父条目P6UJCVG8，核验正文DXGC7JEA和补充MWIWKSVG；相关定义为正文第15–16页式（5）–（8），DMF为补充第23页，左／右ROI顺序为第24页，HCP拟合模型／affine-TM方法为第25页。当前附件无明确稿件日期／修订号，元数据不能确立版本。本附录借用HCP的固定Δ-Ridge α=1，但采用100ROI一阶BOLD状态，未复刻其7网络三阶模型；与原DMF协议的差别在上文完整记录。
{hi}'''
    if lo in text:
        start,end=text.index(lo),text.index(hi)+len(hi);text=text[:start]+block+text[end:]
    else: text=text.rstrip()+'\n\n'+block+'\n'
    report.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--limit-conditions',type=int)
    p.add_argument('--render-only',action='store_true');args=p.parse_args()
    if not args.render_only:
        c=prepare(args.output);tasks=[(pi,gi,si) for pi in range(len(c['subject_ids'])) for gi in range(len(c['G'])) for si in range(len(c['seeds']))]
        if args.limit_conditions: tasks=tasks[:args.limit_conditions]
        print(json.dumps(dict(status='started',conditions=len(tasks),workers=args.workers,appendix_only=True)),flush=True)
        started=time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(args.output),)) as pool:
            for f in as_completed([pool.submit(job,t) for t in tasks]): print(json.dumps(f.result()),flush=True)
        if args.limit_conditions:
            print('Bounded smoke complete; no appendix cohort result.');return
        s=aggregate(args.output);print(json.dumps(dict(status='metrics_complete',elapsed_seconds=time.perf_counter()-started,
            cohorts=s['cohorts'],audit=s['audit'],prediction_diagnostics=s['prediction_diagnostics'])),flush=True)
    s=plot(args.output);write_appendix(s)
    atomic_json(args.output/'completed.json',dict(status='rendered_visual_review_pending',condition_count=s['condition_count'],
        contract_sha256=digest(args.output/'contract.json'),figures={ext:digest(FIGURE.with_suffix('.'+ext)) for ext in ('png','pdf')}))


if __name__=='__main__': main()
