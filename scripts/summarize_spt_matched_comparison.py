"""Read-only scientific result collection for the genuinely matched three cases."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.spt_matched_common import FOLDER,read
from scripts.spt_kuramoto_finite_tm import save_json
from scripts.summarize_spt_kuramoto_ac_selected import collect,signature,module_signature
from scripts.summarize_spt_glv_a_comparison import provisional_surd_result
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS

METHODS=('spt','surd','block_phi_r','wms_natural','shap_profiles')
LABELS=('SPT','SURD + max-weight tree','Block PhiR','WMS natural tree','SHAP output-profile tree')
REPORT=ROOT/'docs/reports/spt_matched_three_case_comparison.md'
FIGURE=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_matched_three_case_comparison.png'
NETWORK_FIGURE=FIGURE.with_name('spt_matched_networks.png')
COMPOSITE_FIGURE=FIGURE.with_name('spt_matched_cases_composite.png')


def display_case(name):
    return 'Kuramoto B' if name=='Kuramoto C' else name


def optional(path):
    return read(path) if path.exists() else {}


def displayed_metrics(record):
    """Validated/provisional output, or explicitly authorized diagnostic output."""
    return record.get('metrics') or (record.get('diagnostic_display') or {}).get('metrics')


def attach_diagnostic(raw, candidate, source_path, *, basis, precision):
    result=copy.deepcopy(raw)
    if result.get('metrics'):return result
    metric=candidate.get('diagnostic_metrics') or candidate.get('metrics')
    if not metric:raise ValueError('No complete stored diagnostic hierarchy available')
    result['diagnostic_display']=dict(metrics=copy.deepcopy(metric),
        integration_replicates=copy.deepcopy(candidate.get('integration_replicates') or []),
        source_path=str(source_path.relative_to(ROOT)),basis=basis,precision=precision,
        strict_precision_passed=False,ranking_eligible=False,
        authorization='User requested display despite failed precision checks, 2026-10-16',
        selection_rule='highest completed registered grid; complete stored mean; no selection by F1',
        original_status=candidate.get('status'),original_metrics_field_unchanged=True)
    return result


def annotate(raw,case,path):
    record=copy.deepcopy(raw)
    record.setdefault('status','not_started')
    record['source_path']=str(path.relative_to(ROOT))
    metric=displayed_metrics(record)
    if metric:
        if 'clade_f1' not in metric or not 0<=metric['clade_f1']<=1:
            raise ValueError('Finite valid module F1 required')
        truth={frozenset(e) for e in SUPPORTS[case] if len(e)<9}
        predicted={frozenset(c) for c in metric['clades']}
        metric['strict_clade_f1']=2*len(truth&predicted)/(len(truth)+len(predicted))
        repeats=(record.get('diagnostic_display') or {}).get('integration_replicates') or record.get('integration_replicates') or record.get('numerical_replicate_metrics') or []
        record.update(numerical_replicate_count=len(repeats),
            raw_tree_agreement_count=sum(signature(m)==signature(metric) for m in repeats),
            scored_clade_agreement_count=sum(module_signature(m,case)==module_signature(metric,case) for m in repeats),
            numerical_f1_range=[min(m['clade_f1'] for m in repeats),max(m['clade_f1'] for m in repeats)] if repeats else None)
        native_ok=record['status'] in ('screen_passed','complete')
        existing_ok=record.get('numerically_stable',True)
        record['stable']=not record.get('diagnostic_display') and native_ok and existing_ok and len(repeats)==4 and record['scored_clade_agreement_count']==4
        record['marker']='‡' if record.get('diagnostic_display') else '†' if record.get('provisional') else '' if record['stable'] else '*'
    else:
        record['stable']=False
    return record


def records():
    original=ROOT/'results/spt_local_kuramoto_tm/ac_comparison'
    cases=[]
    for case in 'AC':
        intervention=collect(case)
        natural=optional(original/case/'natural_baselines.json')
        methods={m:annotate(intervention['methods'][m],case,original/case) for m in METHODS[:3]}
        for method in METHODS[3:]:
            raw=natural.get('methods',{}).get(method,{'status':natural.get('status','not_computed')})
            source=original/case/'natural_baselines.json'
            refined_path=FOLDER/f'kuramoto_wms_{case}.json'
            if method=='wms_natural' and refined_path.exists():
                raw=read(refined_path);source=refined_path
            if method=='wms_natural' and not raw.get('metrics'):
                original_method=natural.get('methods',{}).get(method,{})
                candidate=raw if raw.get('diagnostic_metrics') else original_method
                rows=natural.get('wms_replicates',[])
                if candidate is original_method and (len(rows)!=4 or any(set(r['information_nats'])!=set(map(str,range(1,512))) for r in rows)):
                    raise ValueError('WMS diagnostic display requires all four complete 511-subset tables')
                raw=attach_diagnostic(raw,candidate,original/case/'natural_baselines.json' if candidate is original_method else source,
                    basis='stored canonical WMS tree; four complete 511-subset tables; original N1024/P64',
                    precision={k:candidate.get(k) for k in ['failed_checks','maximum_information_sem_nats','maximum_inner_change_nats','minimum_ess','maximum_weight']})
            methods[method]=annotate(raw,case,source)
        cases.append(dict(case=f'Kuramoto {case}',network=f'simplified {case}',methods=methods))
    spt=optional(FOLDER/'spt.json');block=optional(FOLDER/'block.json');surd=optional(FOLDER/'surd.json');natural=optional(FOLDER/'natural.json')
    if block and not block.get('metrics'):
        completed=[r for r in block.get('levels',[]) if r.get('metrics') and len(r.get('integration_replicates') or [])==4]
        if not completed:raise ValueError('No completed Block grid exists for diagnostic display')
        candidate=max(completed,key=lambda r:(r['outer'],r['particles']))
        queries=[json.loads(line) for line in (FOLDER/f"queries_n{candidate['outer']}.jsonl").read_text().splitlines()]
        block=attach_diagnostic(block,candidate,FOLDER/'block.json',
            basis=f"highest completed Block grid N{candidate['outer']}/P{candidate['particles']}; high N512/P512 remains incomplete",
            precision=dict(outer=candidate['outer'],particles=candidate['particles'],
                completed_queries=candidate['queried_information_count'],failed_queries=len(candidate['failed_queries']),
                maximum_information_sem_nats=max(q['sem_nats'] for q in queries),
                minimum_raw_mi_nats=min(v for q in queries for v in q['values_nats']),
                original_mi_negative_tolerance_nats=.01,
                significant_raw_mi_violation_count=sum(v<-.01 for q in queries for v in q['values_nats']),
                failed_issue_types=sorted({issue for q in queries for issue in q['issues']}),
                phi_audit={k:v for k,v in candidate['audit'].items() if k!='selected_gaps'}))
    methods={m:annotate(raw,'A',FOLDER/f'{stage}.json') for m,stage,raw in
             [('spt','spt',spt),('block_phi_r','block',block),('surd','surd',surd)]}
    if surd and surd.get('status')!='screen_passed':
        provisional=provisional_surd_result({**surd,'grid':{'outer':128,'posterior':128,'inner':256}})
        if provisional is not None:methods['surd']=annotate(provisional,'A',FOLDER/'surd.json')
    for method in METHODS[3:]:
        raw=natural.get('methods',{}).get(method,{'status':natural.get('status','not_started'),'stage':natural.get('stage')})
        methods[method]=annotate(raw,'A',FOLDER/'natural.json')
    return [cases[0],dict(case='HO-LV A',network='simplified A, identical weighted supports',methods=methods),cases[1]]


def cell(record):
    metric=displayed_metrics(record)
    if metric:
        return f"{metric['clade_f1']:.4f}{record.get('marker','')}"
    status=record.get('status','not_started')
    return '待完成' if status in ('not_started','running','not_computed') else f'N/A：{status}'


def render(data,finalized=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    values=np.array([[displayed_metrics(case['methods'][m])['clade_f1'] if displayed_metrics(case['methods'][m]) else np.nan for case in data] for m in METHODS])
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9}):
        fig=plt.figure(figsize=(8.1,5.1),layout='constrained')
        grid=fig.add_gridspec(2,1,height_ratios=(5,1.15))
        ax=fig.add_subplot(grid[0]);note=fig.add_subplot(grid[1]);note.axis('off')
        cmap=LinearSegmentedColormap.from_list('f1',['#faf5ef','#b9d9cc','#167663']);cmap.set_bad('#e9ecef')
        ax.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(3),['Kuramoto\nNetwork A','HO-LV\nNetwork A','Kuramoto\nNetwork B']);ax.xaxis.tick_top()
        ax.set_yticks(range(5),['SPT','SURD +\nmax-weight tree',r'Block $\Phi_R$','WMS tree\n(natural sampling)','SHAP output-\nprofile tree']);ax.tick_params(length=0,pad=9)
        for i,method in enumerate(METHODS):
            for j,case in enumerate(data):
                r=case['methods'][method];v=values[i,j]
                label=f'{v:.2f}{r.get("marker","")}' if np.isfinite(v) else 'pending' if cell(r)=='待完成' else 'N/A'
                ax.text(j,i,label,ha='center',va='center',color='white' if v>.65 else '#293740',weight='bold' if v>.65 else 'normal')
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xlabel('Module-reference F1 (higher is better)',labelpad=8)
        all_max=all(c['methods']['spt'].get('metrics',{}).get('clade_f1')==1 for c in data)
        headline='SPT attains F1 = 1.00 in all three cases; ties are retained.' if all_max else 'Three matched cases; ties and incomplete results are retained.'
        note.text(0,.98,headline,va='top',fontsize=9,fontweight='bold')
        note.text(0,.70,'A: identical weighted hyperedges across dynamics. A vs B: different networks, same Kuramoto dynamics.',va='top',fontsize=7.3)
        note.text(0,.44,'* Sensitivity unresolved. † Incomplete repeats. ‡ Precision not passed; diagnostic display.',va='top',fontsize=7.3)
        foot='‡ LV Block uses completed N256/P256; high grid incomplete. One fitted dataset per case.' if finalized else 'Pending: computation ongoing. N/A: unavailable. One fitted dataset per case.'
        note.text(0,.20,foot,va='top',fontsize=7.3)
        note.text(1,-.04,('Finalized ' if finalized else 'Snapshot ')+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M CST'),ha='right',va='top',fontsize=6.6,color='#66717b')
        fig.savefig(FIGURE,dpi=320,bbox_inches='tight',facecolor='white');plt.close(fig)


def summarize(plot=False):
    data=records();budget=read(FOLDER/'budget.json');workflow=optional(FOLDER/'workflow.json');final=optional(FOLDER/'finalization.json')
    for case in data:case['display_case']=display_case(case['case'])
    all_max=all(c['methods']['spt'].get('metrics',{}).get('clade_f1')==1 and c['methods']['spt'].get('stable') for c in data)
    overall='SPT reaches maximum module F1 in all three cases; ties retained; incomplete baselines excluded from ranking' if all_max else 'Cross-case comparison incomplete'
    output=dict(updated_epoch=time.time(),budget=budget,workflow=workflow,finalization=final,records=data,overall_assessment=overall,
                presentation_policy=dict(precision_gate_waived_for_display=True,strict_thresholds_unchanged=True,
                    diagnostic_marker='‡',diagnostic_ranking_eligible=False,complete_stored_trees_only=True,
                    native_results_unchanged=True,new_estimation_performed=False,no_clipping=True),
                inference='one independently fitted dataset per case; numerical/background repeats do not establish statistical significance')
    save_json(FOLDER/'summary.json',output)
    lines=['# 同网络与同动力学：三案例五法比较','',
        '2026-10-16。三例固定为 Kuramoto 网络 A、高阶 LV 的同权同超边网络 A、Kuramoto 网络 B。按用户要求仅用A/B展示两个网络；图中B对应本轮原始实验键C，冻结缓存及评分参考的键不变。旧高阶 LV 原 A 只作为历史结果保留。',
        '', '拟合口径：本表是旧条件模型结果，尚不满足用户最新要求的MLP管线。Kuramoto使用Fourier候选字典／Lasso拟合的流模型，高阶LV使用指数特征字典／Lasso，均不是MLP。现有共同自然数据MLP仅完成接口试算，未生成新的完整SPT／Block比较；见[共同MLP约束](../log/spt_natural_mlp_training_contract.md)。',
        '', '| 方法 | Kuramoto 网络 A | 高阶 LV 同网络 A | Kuramoto 网络 B |','|---|---:|---:|---:|']
    for method,label in zip(METHODS,LABELS):lines.append('| '+label+' | '+' | '.join(cell(case['methods'][method]) for case in data)+' |')
    if all_max:
        lines.extend(['','SPT在三种配置上模块F1均为1，最高档四次积分重复的评分模块结构均一致。Kuramoto两例与普通Block并列，新LV与SHAP并列。按用户最新要求，图表另行呈现精度未通过的诊断输出：新LV Block粗档候选也是1‡，但这项及两个WMS的0‡不作为数值可靠排名。综合评价保留并列，不据诊断零分推断方法优势。'])
    lines.extend(['','* 表示数值或树结构敏感性未完全解决。† 仅表示完整511子集表构成的暂定结果，重复精度未过。‡ 表示用户授权放宽展示门槛后呈现的完整诊断输出，原精度检查仍失败，不参与可靠排名。没有完整树时仍不补造。',
        '', '模块 F1 沿用预声明的耦合阈值连通分量参考，忽略 primitive 三元模块内部的任意二叉细化；不是全部物理超边恢复率。网络A/B的桥边可与模块重叠。严格 clade F1 和根恢复另列于各方法详情。',
        '', '本轮只更换高阶 LV 网络以补齐对照，保持模型/干预/未来 target/时距、评分及原 .01 nats 容差。四积分重复和 SHAP 的背景子样本共用每例一个拟合数据集，不能据此报告跨系统总体优越性的显著性。'])
    lines.extend(['','## 宽松展示与精度问题','',
        '按用户2026-10-16最新要求，原三项N/A改为显示已经保存的完整诊断树和F1，统一标‡。本次豁免进入图表的精度门槛，没有将原始失败状态改写为通过，也没有另设一个事后数值阈值。原SEM<.01 nats、ESS≥10、最大归一化权重<.25、内层变化<.01 nats及非负容差判据均保留供审计。没有新增模拟或积分，没有放宽先前计算的绝对截止，没有裁剪负估计。',
        '', '| 原N/A项目 | 现展示输出 | 计算来源与精度限制 |','|---|---:|---|'])
    for case in data:
        for method,label in zip(METHODS,LABELS):
            r=case['methods'][method];diagnostic=r.get('diagnostic_display')
            if not diagnostic:continue
            p=diagnostic['precision']
            if 'completed_queries' in p:
                note=f"使用最高已完整结束的N{p['outer']}/P{p['particles']}粗档，{p['completed_queries']}条查询中{p['failed_queries']}条未过原门槛；最大MI SEM={p['maximum_information_sem_nats']:.6g} nats，最小raw MI={p['minimum_raw_mi_nats']:.6g} nats，低于−.01的原始MI估计{p['significant_raw_mi_violation_count']}项。Phi候选本身的显著非负违规数为{p['phi_audit']['significant_violation_count']}，但组成MI未获认证。N512/P512高档未完成，其部分表不构树。"
            else:
                weights_pass=p['minimum_ess']>=10 and p['maximum_weight']<.25
                note=f"复用原N1024/P64的四份完整511子集表及其均值诊断树。最大MI SEM={p['maximum_information_sem_nats']:.6g} nats，高于.01；最小ESS={p['minimum_ess']:.6g}，最大归一化权重={p['maximum_weight']:.6g}（权重检查{'通过' if weights_pass else '失败'}）；代表查询内层64→256最大变化={p['maximum_inner_change_nats']:.6g} nats。原追加精度复核仍未启动。"
            lines.append(f"| {display_case(case['case'])} / {label} | {cell(r)} | {note} |")
    lines.extend(['', '这些分数是已产生输出的展示，不是精度收敛或统计可靠性的认证。选用规则只依据已完成的登记精度和完整均值结果，不按F1挑选重复或精度档；‡项保留原始metrics为空，诊断结果单独存于summary的diagnostic_display字段。'])
    if final:
        coverage=final['query_coverage'];surd_coverage=final['surd_cache_coverage_before_deadline']
        lines.extend(['','## 本轮终态与未完成项','',
            '全部新阶段现已终止，唯一预算仍为2026-10-16 07:57:47—15:57:47。本轮补齐了三案例对应关系及SPT，但完整五法精度比较没有全部完成，未启动替代计算。',
            '', '- 新LV SPT：两档均511子集×4次积分重复，完整切分审计与原数值门槛通过。',
            f"- 新LV SURD：最高登记精度128/128/256下，仅截止前完成的全部完整表scramble `{final['surd_complete_scrambles_admitted']}`进入暂定均值，2/4完整；另两重复分别有{surd_coverage[2]['valid_cached_subsets_before_deadline']}/511和{surd_coverage[3]['valid_cached_subsets_before_deadline']}/511个可用缓存子集（含pilot），均不能补零构树。",
            f"- 新LV普通Block：N256/P256完成{coverage['queries_n256.jsonl']['unique_queries']}条查询，{coverage['queries_n256.jsonl']['numerically_invalid_queries']}条未通过单查询数值门槛；N512/P512只完成{coverage['queries_n512_p512.jsonl']['unique_queries']}条查询，其中{coverage['queries_n512_p512.jsonl']['numerically_invalid_queries']}条未过门槛。高档尚无完整枚举树，现按授权展示完整粗档候选F1=1‡；严格精度认证状态不变。",
            '- 新LV自然SHAP/WMS：共用自然轨迹和冻结模型，均完成。WMS的数值检查通过，但恢复F1=0且树敏感；SHAP模块F1=1。',
            '- Kuramoto网络A/B的新增WMS数值复核未在截止前启动；现复用旧完整诊断树，显示两例F1=0‡，原数值未分辨状态保留。',
            '', '截止保护异常：SURD守护进程未按绝对截止停止其工作进程；16:06:41核对完整命令后立即强制停止，比截止晚约8分54秒。留存日志不足以确认根因。35份截止后生成的specific缓存及其越时中间计算均不进入本轮汇总。原始检查点与异常记录保留，两份被采用完整表的全部输入缓存均早于截止；分数未使用越时数据。Block和调度器在原截止结束，预算文件未重置。',
            '', '后续若继续，应先修复绝对墙钟截止保护，再另行授权复用缓存补齐Block、SURD和Kuramoto WMS，随后用独立物理数据种子检验三案例结论。当前并列结果已符合用户的综合评价口径，不需要为了独占第一重新设计网络或按F1选参。'])
    if COMPOSITE_FIGURE.exists():
        lines.extend(['','![网络A/B、三个案例的五法层级树及模块F1](assets/kuramoto_hierarchy/spt_matched_cases_composite.png)',
            '', '上方展示两种实际超边网络：绿色节点、红色二体包络、蓝色三体包络，边框粗细与耦合权重成比例，面积和重叠不表示Syn或耦合强度。网络A由Kuramoto和高阶LV共用，B为Kuramoto的结构对照。下方三行分别为Kuramoto–A、HO-LV–A和Kuramoto–B，五列为SPT、SURD、普通Block、自然WMS及SHAP；树使用原生均值结果，不按恢复分数挑选积分重复。绿色树节点为恢复的参考模块，红色为错误clade，灰色为根、叶或不计分的primitive内部细化。SURD暂定树仅取全部两份完整表的均值；‡树是精度未通过的完整诊断输出，其中LV Block只用完整粗档，不使用不完整高档。右侧为同一批输出的模块F1。',
            '', '[独立网络图](assets/kuramoto_hierarchy/spt_matched_networks.png) · [独立分数图](assets/kuramoto_hierarchy/spt_matched_three_case_comparison.png) · [组合图SVG](assets/kuramoto_hierarchy/spt_matched_cases_composite.svg)'])
    else:
        if NETWORK_FIGURE.exists():lines.extend(['','![两种网络A/B](assets/kuramoto_hierarchy/spt_matched_networks.png)'])
        if plot or FIGURE.exists():lines.extend(['','![同网络三案例五法](assets/kuramoto_hierarchy/spt_matched_three_case_comparison.png)'])
    for case in data:
        lines.extend(['',f"## {display_case(case['case'])}",''])
        for method,label in zip(METHODS,LABELS):
            r=case['methods'][method];metric=displayed_metrics(r)
            line=f"- {label}：`{r['status']}`，来源 `{r['source_path']}`。"
            if metric:
                line+=f" 模块 F1={metric['clade_f1']:.4f}，严格 clade F1={metric['strict_clade_f1']:.4f}，根={metric['root_split']}。"
                count=r['numerical_replicate_count']
                line+=f" 已有重复的评分结构一致 {r['scored_clade_agreement_count']}/{count}。" if count else ' 未提供完整重复树，稳定性未评价。'
            if r.get('failed_checks'):line+=f" 未通过项：`{r['failed_checks']}`。"
            if r.get('error'):line+=f" 限制：`{r['error']}`。"
            if r.get('source_error'):line+=f" 原阶段限制：`{r['source_error']}`。"
            if r.get('original_source_path'):line+=f" 原始数值结果：`{r['original_source_path']}`，状态 `{r.get('original_status')}`。"
            if r.get('provisional'):line+=f" 完整重复 {r['completed_integration_replicates']}/4，采用全部 `{r['selected_scrambles']}`，不按 F1 选择。"
            if r.get('diagnostic_display'):
                d=r['diagnostic_display'];line+=f" ‡诊断展示来源：`{d['source_path']}`；{d['basis']}。原精度未通过，不参与可靠排名。"
            lines.append(line)
    lines.extend(['','## 输入可比性与结果解释','',
        '同一案例的五法共用拟合训练数据和冻结转移模型，但后续统计输入不完全相同。SPT、SURD和Block使用因子化均匀干预联合；SPT和SURD固定完整未来，Block查询块内及跨块未来变量。WMS和SHAP另用共用自然轨迹池：WMS从拟合自然源联合重采样，SHAP抽取经验评价/背景行，因此二者也不是逐行相同的积分样本。不同方法的积分点数、估计近似和树适配同样存在差异。',
        '', '在相同因子化干预联合下，SPT的节点Syn残差与本实验固定完整未来WMS树的切分目标，都是父块MI减去两个子块MI。若WMS采用与SPT相同的EI表、候选集合及并列规则，二者构树目标一致。因此当前SPT–WMS差异混合了干预/自然源分布的差异，不能全部归功于构树规则。SURD最大权重树和SHAP模式聚类也是声明过的层级化适配，分数评价的是整套适配流程。',
        '', 'SPT成功的合理解释是：稀疏的强内弱外结构与最小跨块残差的切分方向相符，因子化干预展示联合响应，九节点规模允许充分枚举和数值核验。该解释尚未被逐因素消融证实。评分只衡量耦合阈值参考模块，并忽略primitive内部二叉细化；满分不等于所有物理超边或完整二叉树均恢复。当前每例仅一个拟合数据集，且部分基线未完成，结论限定为这些开发案例的模块恢复，不能称完全相同统计输入下的普遍算法优越性。',
        '', '## 实现与原手稿的边界','',
        'Zotero 父项 P6UJCVG8、主文 DXGC7JEA、SI MWIWKSVG 已于2026-10-16重新获取全文并核对相关定义。附件无明确稿件日期/版本，元数据时间不作为新稿证据。主文 Methods（4）–（12）、SI S1.2、S5.1、S7、S9.1/S9.3/S9.4 是本轮依据。',
        '', 'SPT 固定共同因子化干预和完整未来，逐节点枚举精确二分并递归至单节点；这是逐节点最优的贪心层级。条件非线性 TM 是本实验的估计近似，区别于 SI 的 affine 路径。gLV 使用对数丰度、Kuramoto 使用圆周变量，条件密度族按状态域适配。',
        '', 'SURD 使用已声明的干预联合及完整未来进行逐状态分配，再用最大权重层级树适配；不同于 SI S9.3 的自然轨迹设置。SHAP 使用完整512联盟输出模式树，区别于 SI S9.4 的替换差分。自然 WMS 与 SHAP 共用每例真实自然轨迹，WMS 保留源相关性和固定完整未来；源密度采用已公开的近似，不把该近似称为真实自然分布。',
        '', '原生 Syn/Phi 的 .01 nats 容差、显著负值失败、minimum/threshold/count和无裁剪规则保持。WMS 的负余额合法，组成 MI 仍审计非负性及源包含单调性。',
        '', '[对照核查、试算与预算记录](spt_three_case_supervision.md)。'+('所有新阶段已停止，小时监督已删除；未追加计算。' if final.get('monitoring_deleted') else '所有新阶段共用一份绝对截止，按用户要求每小时监督，健康无新变化保持安静。')])
    REPORT.write_text('\n'.join(lines)+'\n')
    if plot:render(data,finalized=bool(final))
    print(json.dumps({'report':str(REPORT),'cases':[{c['case']:{m:cell(c['methods'][m]) for m in METHODS}} for c in data]},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plot',action='store_true');args=parser.parse_args();summarize(args.plot)
