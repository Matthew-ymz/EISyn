#!/usr/bin/env python3
"""Reconstruct the original natural WMS samples for full-E/I PhiR, resumably.

Xi and WMS arrays are read-only. Natural source hashes and recomputed native WMS
must match every old condition before a new PhiR result can enter either figure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dmf_natural_state_comparison as kernel
from scripts.dmf_dense_statistics import subject_landmarks
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.run_dmf_subject_consistency import atomic_json,digest

DENSE=kernel.DENSE
OUTPUT=ROOT/'results/dmf_schaefer100/natural_state_phi_r'
W={}


def prepare(output=OUTPUT):
    dense=json.loads((DENSE/'contract.json').read_text())
    native=dense['native_protocol']
    for path,sha in native['implementation_sha256'].items():
        if digest(ROOT/path)!=sha:
            raise ValueError(f'Original natural implementation changed: {path}')
    c=dict(version='natural-full-EI-fixed-bipartition-phiR-v2-square-root',
        subject_ids=dense['subject_ids'],G=dense['G'],seeds=dense['seeds'],
        source='Original natural correlated full 200-scalar E/I gating states',
        target='Original WMS independently evolved 300-ms full 200-scalar E/I futures',
        scheme='original_sampling',sample_count=2048,source_hash_bridge_required=True,
        original_Xi_and_WMS='Reused unchanged from dense analysis_curves.npz',
        natural_protocol=native,units='nats',phi_r_partition='Fixed E|I (100|100); not pair average or MIP',
        phi_r_formula='I(X;Y)-I(X_E;Y_E)-I(X_I;Y_I)+min_ab I(X_a;Y_b)',
        estimator='Native empirical correlated source + Gaussian least-squares transition, ridge 1e-6; PhiR marginal queries of one regularized joint covariance',
        numerical_limit='Preserve native WMS values/query-specific ridge; record native whole MI vs coherent joint MI difference. Same samples does not mean identical marginal query regularization.',
        nonnegative_tolerance_nats=kernel.TOL_NATS,phi_r_no_floor_or_projection=True,
        WMS='Signed observational source WMS; negative values are permitted, not PEID Syn',
        landmarks='Original Xi/PhiR maxima, signed WMS minima, first ties and boundary misses',
        window='Frozen 0.5 G post-hoc independent-rate window, no reselection',
        dense_contract_sha256=digest(DENSE/'contract.json'),
        dense_inputs_sha256=digest(DENSE/'inputs.npz'),
        dense_curves_sha256=digest(DENSE/'analysis_curves.npz'),
        dense_windows_sha256=digest(DENSE/'window_summary.json'),
        kernel_sha256=digest(Path(kernel.__file__)),
        manuscript=dict(parent='P6UJCVG8',body='DXGC7JEA',supplement='MWIWKSVG',
            checked_date='2026-10-11',explicit_draft_version_or_date=None,
            locations='Brain pp6-7; Methods Eqs5-8 pp15-16; Supplement S12.2.1/EqsS82-S87 p23',
            ambiguity='No explicit draft revision/date in current attachments',
            discrepancies=['New neural E/I PhiR and fixed E|I partition replace original BOLD pairwise comparator',
                'Source WMS remains natural full E/I, despite manuscript BOLD-like description',
                'Individual 93/3seed/G0..4 versus manuscript meanSC/8seed/G0..3',
                'Gaussian moment approximations and inherited per-query WMS regularization remain explicit']),
        authorization='2026-10-11 user: WMS用原来的数据就可以，phi R用跟它一样的数据; after original-reconstruction cost estimate of ~3h')
    output.mkdir(parents=True,exist_ok=True)
    (output/'conditions').mkdir(exist_ok=True)
    path=output/'contract.json'
    equivalence=output/'numerical_diagnostics'/'legacy_square_root_equivalence.json'
    if equivalence.exists():
        migration=json.loads(equivalence.read_text())
        if migration['new_kernel_sha256']!=c['kernel_sha256']:
            raise ValueError('Numerical equivalence audit belongs to a different kernel')
        if migration['maximum_MI_difference_nats']>kernel.TOL_NATS:
            raise ValueError('Legacy covariance-query numerical equivalence exceeds 1e-8 nats')
        c['numerical_implementation']=dict(method='Affine Gaussian square-root factors and QR log determinants',
            unchanged='Native source covariance, least-squares coefficient, residual and both original ridge contributions',
            no_joint_covariance_repair=True,legacy_equivalence_tolerance_nats=kernel.TOL_NATS,
            legacy_contract_sha256=migration['old_contract_sha256'],
            legacy_cache_equivalence_sha256=digest(equivalence),
            legacy_condition_count=migration['condition_count'],
            legacy_maximum_MI_difference_nats=migration['maximum_MI_difference_nats'])
    else:
        c['numerical_implementation']=dict(method='Affine Gaussian square-root factors and QR log determinants',
            unchanged='Native source covariance, least-squares coefficient, residual and both original ridge contributions',
            no_joint_covariance_repair=True)
    if path.exists() and json.loads(path.read_text())!=c:
        previous=json.loads(path.read_text())
        expected_previous={k:v for k,v in c.items() if k!='numerical_implementation'}
        expected_previous['version']='natural-full-EI-fixed-bipartition-phiR-v1'
        old_kernel=output/'numerical_diagnostics'/'kernel_before_square_root.py'
        if not old_kernel.exists(): raise ValueError('Missing original numerical kernel')
        expected_previous['kernel_sha256']=digest(old_kernel)
        if (not equivalence.exists() or previous!=expected_previous
                or digest(path)!=migration['old_contract_sha256']):
            raise ValueError('Natural PhiR contract changed outside the audited numerical revision')
        atomic_json(path,c)
    if not path.exists(): atomic_json(path,c)
    return c


def legacy_index(output,c):
    numerical=c['numerical_implementation']
    if 'legacy_contract_sha256' not in numerical: return {}
    path=output/'numerical_diagnostics'/'legacy_square_root_equivalence.json'
    if digest(path)!=numerical['legacy_cache_equivalence_sha256']:
        raise ValueError('Legacy numerical equivalence audit changed')
    return {r['condition']:r for r in json.loads(path.read_text())['conditions']}


def validate_cache(a,path,c,sha,legacy):
    if str(a['contract_sha256'])!=sha:
        old=c['numerical_implementation'].get('legacy_contract_sha256')
        if (str(a['contract_sha256'])!=old or path.name not in legacy
                or digest(path)!=legacy[path.name]['sha256']):
            raise ValueError(f'Wrong cache: {path.name}')
    r=json.loads(str(a['metrics_json']))
    kernel.audit_nonnegative([r['whole_mi_nats'],r['phi_r_nats'],*a['partition_mi_nats'].ravel()],kernel.TOL_NATS)
    if float(a['phi_r_nats'])!=r['phi_r_nats'] or str(a['source_sha256'])!=r['source_sha256']:
        raise ValueError(f'Inconsistent cache fields: {path.name}')
    with np.load(DENSE/'conditions'/path.name) as original:
        source=str(original['native_source_sha256'])
        native=json.loads(str(original['native_metrics_json']))
    if source!=r['source_sha256']:
        raise ValueError(f'Original source hash mismatch: {path.name}')
    bridge=abs(r['source_wms_nats']-native['wms_bits']*np.log(2))
    if not np.isfinite(bridge) or bridge>kernel.TOL_NATS:
        raise ArithmeticError(f'Original WMS bridge mismatch: {path.name}: {bridge} nats')
    return r


def initialize(output):
    global W
    output=Path(output)
    c=json.loads((output/'contract.json').read_text())
    if digest(Path(kernel.__file__))!=c['kernel_sha256']:
        raise ValueError('Natural kernel changed')
    W=dict(output=output,c=c,sha=digest(output/'contract.json'),legacy=legacy_index(output,c),
           limiter=threadpool_limits(limits=1))


def job(indices):
    pi,gi,si=indices;w=W;c=w['c']
    subject,g,seed=c['subject_ids'][pi],c['G'][gi],c['seeds'][si]
    name=f'{subject}_G{g:.2f}_seed{seed}.npz'
    path=w['output']/'conditions'/name
    if path.exists():
        with np.load(path) as a:
            validate_cache(a,path,c,w['sha'],w['legacy'])
        return dict(condition=name,reused=True)
    r=kernel.condition(subject,g,seed,scheme='original_sampling')
    covariance=r.pop('covariance');partition_mi=r.pop('partition_mi_nats')
    source=r.pop('source_covariance');coefficient=r.pop('transition_coefficient')
    noise=r.pop('effective_noise_covariance')
    atomic_savez(path,covariance=covariance,partition_mi_nats=partition_mi,
        source_covariance=source,transition_coefficient=coefficient,effective_noise_covariance=noise,
        metrics_json=json.dumps(r,allow_nan=False),phi_r_nats=r['phi_r_nats'],
        source_sha256=r['source_sha256'],contract_sha256=w['sha'])
    return dict(condition=name,reused=False,phi_r_nats=r['phi_r_nats'],
        WMS_bridge_error_nats=r['native_wms_bridge_error_nats'],elapsed_seconds=round(r['elapsed_seconds'],2))


def aggregate(output=OUTPUT):
    c=json.loads((output/'contract.json').read_text());sha=digest(output/'contract.json')
    legacy=legacy_index(output,c)
    shape=(len(c['subject_ids']),len(c['seeds']),len(c['G']))
    values=np.empty(shape);records=[]
    for pi,subject in enumerate(c['subject_ids']):
        for si,seed in enumerate(c['seeds']):
            for gi,g in enumerate(c['G']):
                path=output/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz'
                with np.load(path) as a:
                    record=validate_cache(a,path,c,sha,legacy)
                    values[pi,si,gi]=float(a['phi_r_nats'])
                    records.append(record)
    windows=json.loads((DENSE/'window_summary.json').read_text())
    scenario=next(s for s in windows['scenarios'] if s['k_each_side']==2)
    with np.load(DENSE/'analysis_curves.npz') as a:
        rates=a['rate'].copy()
        if a['subject_ids'].tolist()!=c['subject_ids']: raise ValueError('Subject order changed')
    rows=[]
    for pi,old in enumerate(scenario['subject_rows']):
        row=json.loads(json.dumps(old))
        lm=subject_landmarks(c['G'],rates[pi].mean(0),{'phi_r':values[pi].mean(0)})
        if lm['transition']!=row['independent_transition']: raise ValueError('Independent transition changed')
        m=lm['methods']['phi_r'];point=m['extremum_G'];lo,hi=row['window']['interval']
        m['hit']=bool(m['interior'] and point is not None and lo-1e-12<=point<=hi+1e-12)
        m['distance_to_window_G']=None if point is None else max(lo-point,0.,point-hi)
        row['methods']['phi_r']=m;rows.append(row)
    dense=json.loads((DENSE/'contract.json').read_text());cohorts={}
    for key,ids in [('all93',c['subject_ids'][:-1]),('primary85',dense['holdout_subject_ids']),
                     ('development8',dense['development_subject_ids'])]:
        selected=[r for r in rows if r['subject'] in ids]
        cohorts[key]=dict(subject_count=len(selected),methods={k:dict(
            hit_count=sum(r['methods'][k]['hit'] for r in selected),denominator=len(selected),
            hit_rate=sum(r['methods'][k]['hit'] for r in selected)/len(selected),
            flat_count=sum(r['methods'][k]['flat'] for r in selected),
            boundary_count=sum(not r['methods'][k]['interior'] and not r['methods'][k]['flat'] for r in selected))
            for k in ('xi','phi_r','wms')})
    audit=dict(minimum_phi_r_nats=float(values.min()),tolerance_nats=kernel.TOL_NATS,
        tolerance_negative_query_count=sum(r['nonnegative_audit']['tolerance_negative_count'] for r in records),
        significant_violation_count=0,source_hash_match_count=len(records),
        maximum_native_WMS_bridge_error_nats=max(r['native_wms_bridge_error_nats'] for r in records),
        maximum_native_whole_vs_joint_MI_error_nats=max(r['native_whole_vs_joint_mi_error_nats'] for r in records),
        mean_native_whole_vs_joint_MI_error_nats=float(np.mean([r['native_whole_vs_joint_mi_error_nats'] for r in records])),
        minimum_unique_natural_timepoints=min(r['unique_source_count'] for r in records))
    audit['minimum_native_whole_MI_nats']=min(r['native_whole_mi_nats'] for r in records)
    audit['negative_native_whole_MI_condition_count']=sum(r['native_whole_mi_nats']<0 for r in records)
    audit['inherited_native_WMS_eigenvalue_floor_count']=sum(r['native_wms_audit']['eigenvalue_floor_count'] for r in records)
    audit['square_root_condition_count']=sum(r.get('numerical_method')=='affine-square-root-QR' for r in records)
    audit['equivalent_legacy_condition_count']=len(records)-audit['square_root_condition_count']
    audit['legacy_maximum_MI_difference_nats']=c['numerical_implementation'].get('legacy_maximum_MI_difference_nats',0.)
    audit['original_Xi_nonnegative_audit']=windows['audit']['original_syn_audit']
    atomic_savez(output/'curves.npz',phi_r_nats=values,subject_ids=c['subject_ids'],G=c['G'],seeds=c['seeds'],contract_sha256=sha)
    summary=dict(contract_sha256=sha,curves_sha256=digest(output/'curves.npz'),condition_count=len(records),
        cohorts=cohorts,subject_rows=rows,audit=audit,inference='Descriptive fixed-window proportions; no new hypothesis tests',
        original_Xi_and_WMS_unchanged=True,reconstructed_natural_condition_count=len(records))
    atomic_json(output/'summary.json',summary)
    return summary


def read_inputs(output=OUTPUT):
    from scripts.plot_dmf_subject_fixed_window import read_inputs as read_original
    config=json.loads((DENSE/'fixed_window_view.json').read_text())
    old,g,values,rho,_,_=read_original(config)
    c=json.loads((output/'contract.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    if summary['curves_sha256']!=digest(output/'curves.npz'): raise ValueError('Natural PhiR curves changed')
    if c['dense_curves_sha256']!=digest(DENSE/'analysis_curves.npz'): raise ValueError('Original Xi/WMS changed')
    with np.load(output/'curves.npz') as a:
        if str(a['contract_sha256'])!=digest(output/'contract.json'): raise ValueError('Natural PhiR contract differs')
        values['phi_r']=a['phi_r_nats'].mean(1)
    return c,g,values,rho,summary,summary['audit']


def render(output=OUTPUT):
    from scripts import plot_dmf_subject_fixed_window as view
    c,g,values,rho,summary,audit=read_inputs(output)
    view.TITLE['xi']=r'Integrated EI $\Xi$ (intervention)'
    view.TITLE['phi_r']=r'Natural full-system $\Phi^R$ (fixed 100E | 100I)'
    view.TITLE['wms']='Natural signed source WMS'
    view.draw(g,values,rho,summary)
    from scripts import compose_dmf_brain_main_figure as composite
    # The user selected the existing 19,900-pair mean for the main figure.
    # This fixed-partition run now supplies only its supplementary overview.
    # Re-rendering it must not overwrite the selected main/report protocol.
    selected = composite.read_main_comparison()[4]
    if selected.get('phi_r_protocol') in ('interventional-pairwise-mean', 'natural-pairwise-mean'):
        print('Natural fixed-partition overview rendered as supplementary; '
              'user-selected pairwise main figure and report retained.')
        return audit
    composite.draw(450)
    write_report(summary)
    return audit


def write_report(summary):
    report=ROOT/'docs/reports/brain.md'
    text=report.read_text()
    methods=summary['cohorts']['all93']['methods']
    primary=summary['cohorts']['primary85']['methods']
    def value(name,rows=methods):
        m=rows[name]
        return f"{m['hit_count']}/{m['denominator']}（{m['hit_rate']*100:.1f}%）"
    lo,hi='<!-- dense-fixed-view:start -->','<!-- dense-fixed-view:end -->'
    body=fr'''{lo}
<a id="dmf-subject-fixed-window"></a>

### 1.1 同一神经状态、保留指标分布定义的三指标比较

**三指标均读取完整200维E/I神经状态，未来目标为300 ms后的同一状态；Ξ使用干预分布，ΦR与WMS使用自然分布。** 全93人中，Ξ为 **{value('xi')}**，自然态系统级ΦR为 **{value('phi_r')}**，原自然source WMS为 **{value('wms')}**。Ξ及WMS原数据完全保留；ΦR使用原WMS的自然源样本和未来响应，逐条件核对源样本SHA256及WMS复现。

每人使用自己的原生100ROI结构连接，G=0–4、步长0.1，seed 3/4/5；每條曲线为同人三seed均值。Ξ沿用2048个独立均匀干预状态U(0.3,0.7)及affine-TM估计；ΦR/WMS的源从原自然轨迹尾部按原seed有放回抽取2048个完整E/I状态，再按原未来噪声seed独立演化300步。**仅统一神经状态表示和预测时距，不把观察性基线改成干预分布，也不改用轨迹上的300 ms间隔配对。** 原WMS保留其相关自然源先验、Gaussian最小二乘估计和带符号结果。

系统级ΦR用完整E群体及完整I群体作为固定两块：

$$\Phi^R_{{E\mid I}}=I(\boldsymbol{{x}};\boldsymbol{{y}})-I(\boldsymbol{{x}}_E;\boldsymbol{{y}}_E)-I(\boldsymbol{{x}}_I;\boldsymbol{{y}}_I)+\min_{{a,b\in\{{E,I\}}}}I(\boldsymbol{{x}}_a;\boldsymbol{{y}}_b).$$

ΦR所有边缘互信息来自同一保留相关自然源的正则化Gaussian联合模型，沿用原WMS的源协方差、最小二乘转移和噪声模型构造。用平方根因子与QR分解计算同一Gaussian模型的边缘／条件行列式，避免大型联合矩阵的尺度差异破坏数值正定性；不改变ridge、最小二乘系数或相关先验，不加特征值下限，不裁剪或投影ΦR。固定E/I分区由模型两类状态决定，不根据命中选择；没有MIP搜索，也不是原BOLD两两ΦR均值。本结果不能代替全分区ΦID或分区敏感性检验。

窗口沿用每人独立100ROI平均E发放率最大正斜率小区间[L,U]，左右各扩0.2 G；当前93人的窗口均宽0.5 G，无端点截断。Ξ/ΦR取全局峰，signed source WMS保留原全局谷；首并列点、含端点、平坦或边界极值未命中的规则不变。原85人主要队列中，Ξ为{value('xi',primary)}，ΦR为{value('phi_r',primary)}，WMS为{value('wms',primary)}。这里只报告描述性比例，不新增假设检验或沿用旧ΦR的p值；窗口为同队列事后固定，不能视为独立确认性结果。

![同一E/I神经状态下，Ξ干预分布与ΦR/WMS自然分布的93人比较](assets/dmf_subject_dense/fixed_window_overview.png)

**识别总览图。** a为独立序参量，b为干预Ξ，c为固定E/I分区的自然系统级ΦR，d为原自然signed source WMS，e为全部93人的窗口命中率。每条细线为同人三seed均值，颜色按原生SC谱半径编码；黑色虚线及圆点为平均SC的独立模拟，不计入93人统计。信息单位为nats，原生纵轴显示各指标的实际符号和量级，无平滑、幅度归一化或峰位对齐，图例位于数据外。

非负容差为10⁻⁸ nats，仅用于Ξ及ΦR，ΦR容差内负值查询计数为{summary['audit']['tolerance_negative_query_count']}，显著违反为0；signed WMS允许负值，不能把它当作PEID Syn。{summary['condition_count']:,}个条件的自然源哈希均匹配，原WMS重现最大误差为{summary['audit']['maximum_native_WMS_bridge_error_nats']:.3g} nats。原WMS保留条件协方差子查询的额外ridge、目标协方差伪逆及历史特征值下限，whole MI与严格共同联合模型的MI差异为平均{summary['audit']['mean_native_whole_vs_joint_MI_error_nats']:.3g}、最大{summary['audit']['maximum_native_whole_vs_joint_MI_error_nats']:.3g} nats；原whole MI近似中有{summary['audit']['negative_native_whole_MI_condition_count']}个负值条件（最小{summary['audit']['minimum_native_whole_MI_nats']:.3g} nats），这是保留原WMS计算所带来的数值／近似局限，不是负互信息或负Syn的理论证据。历史WMS特征值下限触发总计{summary['audit']['inherited_native_WMS_eigenvalue_floor_count']}次，ΦR没有新增下限。这里报告**样本相同**，不宣称两者查询正则化或数值近似相同。自然样本有重复且来自短轨迹，最少仅{summary['audit']['minimum_unique_natural_timepoints']}个不同时间索引；2048不是2048个独立自然观测。

原Ξ审计沿用{summary['audit']['original_Xi_nonnegative_audit']['count']:,}个条件，最小{summary['audit']['original_Xi_nonnegative_audit']['minimum_nats']:.6g} nats，容差内负值{summary['audit']['original_Xi_nonnegative_audit']['tolerance_negative_count']}个、显著违反{summary['audit']['original_Xi_nonnegative_audit']['violation_count']}个。数值修复保留{summary['audit']['equivalent_legacy_condition_count']:,}个旧ΦR缓存，平方根等价查询最大差异{summary['audit']['legacy_maximum_MI_difference_nats']:.3g} nats，低于10⁻⁸ nats；其余{summary['audit']['square_root_condition_count']:,}条件采用平方根计算，所有条件的ΦR定义与正则化相同。

2026-10-11重新核对Zotero父条目P6UJCVG8（*Emergent hierarchical organization of causal interactions in complex systems*），读取正文DXGC7JEA的Brain/Fig.2（第6–7页）与Methods式（5）–（8）（第15–16页），补充MWIWKSVG的S12.2.1/式S82–S87（第23页）。两附件无明确稿件日期／修订号；元数据修改不能确定版本。此图替换原观察性BOLD pairwise ΦR，原WMS仍是自然神经E/I状态，与稿件BOLD-like WMS措辞的差异继续记录。

{hi}'''
    a,b=text.index(lo),text.index(hi)+len(hi)
    text=text[:a]+body+text[b:]
    paragraphs=text.split('\n\n')
    for i,s in enumerate(paragraphs):
        if s.startswith('**分布口径更正'):
            paragraphs[i]=(f'**统一神经状态，保留各指标的分布定义。** 在93个体SC驱动的DMF模型中，'
                f'完整200维E/I状态和300 ms未来下，干预Ξ为 **{value("xi")}**，自然系统级ΦR为 **{value("phi_r")}**，'
                f'原自然source WMS为 **{value("wms")}**。ΦR使用与原WMS完全相同的自然源样本与未来响应；'
                'Ξ/WMS原值保留。固定E/I分区与事后0.5 G窗口的适用边界见第1.1节。')
        elif s.startswith('**候选图注｜当前恢复的原比较'):
            paragraphs[i]=(f'**候选图注｜DMF模型的转折定位与信息组织。** **a–d，** 独立平均E发放率、干预Ξ、'
                '自然系统级ΦR（固定E/I分区）及原自然signed source WMS。三指标均读取完整200维E/I门控状态及300 ms未来，'
                'ΦR/WMS自然样本相同，Ξ保留独立均匀干预。细线为同人seed 3/4/5均值，原生SC谱半径编码颜色，'
                '黑色虚线及圆点为平均SC独立模拟；G=0–4、步长0.1，无平滑、归一化或峰位对齐。'
                f'**e，** Ξ为{value("xi")}，ΦR为{value("phi_r")}，WMS为{value("wms")}。'
                'Ξ/ΦR取峰、WMS取谷；0.5 G窗口由同队列事后固定，系统级ΦR不包含MIP搜索。')
        elif s.startswith('用户已明确：统一神经E/I状态'):
            paragraphs[i]=('用户明确保留Ξ干预分布和两基线自然分布。此前干预两两ΦR的35/93、三项都干预时的系统级ΦR24/93与WMS65/93'
                '均不属于本图的自然基线，已撤回其主比较用途。原BOLD-like pairwise ΦR为28/93；当前自然E/I系统级ΦR为'
                f'{value("phi_r")}。同时变化了信号类型、时距和ΦR分区／聚合定义，不能单独归因于BOLD转换。'
                '旧协议及统计保留于[附录P](#dmf-subject-dense)。')
    text='\n\n'.join(paragraphs)
    lines=text.splitlines()
    for i,s in enumerate(lines):
        if s.startswith('| **当前展示原协议，等待自然E/I基线**'):
            lines[i]=(f'| **统一神经状态后的转折邻域比较** | Ξ{value("xi")}、自然ΦR{value("phi_r")}、原自然WMS{value("wms")} | '
                      'Ξ保留干预；两基线共用自然样本与300 ms未来；ΦR固定E/I分区 |')
        elif s.startswith('2026-10-11口径更正：已撤回'):
            lines[i]='2026-10-11更新：上部a–e使用统一神经状态、Ξ干预与两基线自然分布的结果；下部层级树及皮层图沿用原缓存。图版190 × 211 mm，PNG450 dpi，PDF保留矢量曲线／文字。'
        elif s.startswith('## 1. 序参量转折识别：原结果与自然'):
            lines[i]='## 1. 序参量转折识别：统一神经状态、保留分布定义'
    report.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--limit-conditions',type=int)
    p.add_argument('--render-only',action='store_true')
    a=p.parse_args()
    if a.render_only:
        render(a.output);return
    c=prepare(a.output)
    tasks=[(pi,gi,si) for pi in range(len(c['subject_ids']))
           for gi in range(len(c['G'])) for si in range(len(c['seeds']))]
    if a.limit_conditions: tasks=tasks[:a.limit_conditions]
    print(json.dumps(dict(status='started',conditions=len(tasks),workers=a.workers,
                          kernel_sha256=c['kernel_sha256'],scheme=c['scheme'])),flush=True)
    started=time.perf_counter()
    # Bound submissions so an exception cannot leave the complete cohort queued
    # behind the executor context manager's shutdown(wait=True).
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(str(a.output),)) as pool:
        iterator=iter(tasks);pending={}
        def submit_one():
            t=next(iterator,None)
            if t is not None: pending[pool.submit(job,t)]=t
        for _ in range(2*a.workers): submit_one()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                indices=pending.pop(f)
                try:
                    result=f.result()
                except BaseException as error:
                    failure=dict(status='failed',indices=indices,error_type=type(error).__name__,error=str(error))
                    atomic_json(a.output/'failure.json',failure)
                    print(json.dumps(failure),flush=True)
                    for queued in pending: queued.cancel()
                    raise
                print(json.dumps(result),flush=True)
                submit_one()
    if a.limit_conditions:
        print('Bounded smoke complete; no cohort figure produced.');return
    s=aggregate(a.output)
    print(json.dumps(dict(status='metrics_complete',elapsed_seconds=time.perf_counter()-started,
                          cohorts=s['cohorts'],audit=s['audit'])),flush=True)
    render(a.output)
    atomic_json(a.output/'completed.json',dict(status='rendered_visual_review_pending',
        condition_count=s['condition_count'],contract_sha256=digest(a.output/'contract.json')))


if __name__=='__main__': main()
