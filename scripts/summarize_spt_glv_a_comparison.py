"""Merge active baselines; never score unresolved estimates as zero."""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_glv_a_comparison import quality
from scripts.compare_spt_surd_three_cases import maximum_weight_tree
DIRECTORY=ROOT/'results/spt_local_glv_tm/a_comparison'
FIGURE=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_glv_a_active_methods.png'
REPORT=ROOT/'docs/reports/spt_glv_kuramoto_parallel.md'
METHODS=[('spt','SPT'),('surd_maximum_weight','SURD + max-weight tree'),('block_phi_r',r'Block $\Phi_R$'),
         ('wms_natural','WMS tree (natural sampling)'),('shap_profiles','SHAP attribution-profile tree')]


def provisional_surd_result(full):
    """Use every completed full-table replicate, never select by recovery F1.

    A prefix of the 511 coalitions cannot support a SURD tree. Atom validity
    remains mandatory even for an explicitly provisional numerical result.
    """
    complete=[r for r in full.get('replicates',[]) if r.get('complete') and r.get('subsets')==511]
    if not complete: return None
    for r in complete:
        audit=r.get('audit',{})
        if (audit.get('significant_specific_violation_count',1) or
            audit.get('significant_atom_violation_count',1) or audit.get('clipping',True) or
            audit.get('maximum_statewise_closure_error_nats',1)>1e-8):
            raise ArithmeticError('Completed SURD replicate has no valid unclipped allocation audit')
    atoms=np.asarray([r['synergy_nats'] for r in complete],dtype=float)
    if atoms.shape!=(len(complete),512) or not np.isfinite(atoms).all():
        raise ValueError('Incomplete or nonfinite SURD atom table')
    if atoms.min()<-.01:
        raise ArithmeticError(f'SURD atom minimum={atoms.min()}; threshold=-0.01; affected_count={int((atoms<-.01).sum())}')
    tree=maximum_weight_tree(atoms.mean(0)); trees=[maximum_weight_tree(v) for v in atoms]
    checks=sorted(set(full.get('failed_checks',[]))|set(s for r in complete for d in r.get('diagnostics',{}).values() for s in d.get('issues',[])))
    return dict(status='provisional_completed_full_tables',provisional=True,
        metrics=quality(tree['root_split'],tree['clades']),
        integration_replicates=[quality(t['root_split'],t['clades']) for t in trees],
        completed_integration_replicates=len(complete),selected_scrambles=[r['scramble'] for r in complete],
        grid=full['grid'],failed_checks=checks,source_status=full.get('status'),
        source_error=full.get('error'),
        selection_rule='highest registered completed grid; average every complete 511-coalition replicate; no F1 selection',
        provisional_reason='full four-scramble precision audit has not passed')


def summarize(directory=DIRECTORY):
    directory=Path(directory)
    payload=json.loads((directory/'comparison_1024.json').read_text())
    side_path=directory/'natural_baselines.json'
    side=json.loads(side_path.read_text()) if side_path.exists() else {}
    payload['methods'].update(side.get('methods',{}))
    surd=payload['methods'].get('surd_maximum_weight',{})
    full_path=directory/'surd_full.json'
    if surd.get('status')!='screen_passed' and full_path.exists():
        fallback=provisional_surd_result(json.loads(full_path.read_text()))
        if fallback is not None: payload['methods']['surd_maximum_weight']=fallback
    for method in ('wms_natural','shap_profiles'):
        if method in payload['methods']: continue
        if side.get('status')=='running':
            payload['methods'][method]={'status':'running' if side.get('stage')==method else 'not_yet_evaluated'}
        elif side.get('status'):
            payload['methods'][method]={'status':side['status'],'error':side.get('error')}
    if side.get('status')=='running' and payload['status']!='running':
        payload['status']='running_natural_baselines'
    budget=json.loads((directory/'budget.json').read_text())
    if payload['status'].startswith('running'):
        payload['elapsed_wall_seconds']=time.time()-budget['started_epoch']
    payload['natural_baseline_protocol']={k:side.get(k) for k in ('version','natural_sampling','source_tm_audit')}
    (directory/'comparison_active_methods.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    records=[]
    for key,label in METHODS:
        result=payload.get('methods',{}).get(key,{})
        metrics=result.get('metrics')
        row=dict(method=key,label=label,status=result.get('status','not_yet_evaluated'),
                 f1=None,strict_f1=None,correct_root=None,upper_recall=None,f1_range=None,
                 provisional=result.get('provisional',False),
                 completed_integration_replicates=result.get('completed_integration_replicates'))
        if metrics:
            repeats=result.get('integration_replicates') or []
            row.update(f1=metrics['clade_f1'],strict_f1=metrics.get('strict_clade_f1'),
                correct_root=metrics['correct_root'],upper_recall=metrics['upper_support_recall'],
                f1_range=[min(r['clade_f1'] for r in repeats),max(r['clade_f1'] for r in repeats)] if repeats else None,
                replicate_roots_agree=result.get('all_replicate_roots_match_mean'))
        row['failed_queries']=len(result.get('failed_queries',{}))
        if row['f1'] is not None:
            row['difference_from_spt']=payload['methods']['spt']['metrics']['clade_f1']-row['f1']
        records.append(row)
    summary=dict(status=payload['status'],stage=payload.get('stage'),
        query_count=payload['query_count'],elapsed_wall_seconds=payload['elapsed_wall_seconds'],
        records=records,statistical_significance='not tested; one fitted dataset, integration repeats only')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.left':False}):
        fig,ax=plt.subplots(figsize=(8.8,4.4),layout='constrained')
        for index,row in enumerate(records):
            y=len(records)-1-index
            if row['f1'] is None:
                active=(row['method']==payload.get('stage') or
                    (row['method']=='surd_maximum_weight' and payload.get('stage')=='surd_specific_information') or
                    (payload.get('stage')=='pairwise' and row['method']=='pair_phi_r_hac') or
                    (payload.get('stage')=='pairwise' and row['method']=='future_mi_hac' and 'pair_phi_r_hac' in payload.get('methods',{})))
                text=('Running' if active or row['status']=='running' else 'Pending') if row['status'] in ('not_yet_evaluated','running') and payload['status'].startswith('running') else 'Not resolved'
                ax.text(.5,y,text,color='#747f8a',ha='center',va='center')
                continue
            interval=row['f1_range']
            if interval:
                ax.plot(interval,[y,y],color='#9da8b1',lw=3,zorder=1)
            ax.scatter(row['f1'],y,s=58,facecolor='white' if row['provisional'] else '#168572',edgecolor='#168572',zorder=3)
            if row['strict_f1'] is not None:
                ax.scatter(row['strict_f1'],y-.17,s=28,marker='s',facecolor='white',edgecolor='#357cb3',zorder=3)
            ax.annotate(f"{row['f1']:.2f}"+('†' if row['provisional'] else ''),(row['f1'],y),xytext=(0,10),textcoords='offset points',
                        ha='center',fontsize=8,color='#176c5c')
        ax.set_yticks(range(len(records)-1,-1,-1),[r['label'] for r in records])
        ax.tick_params(axis='y',length=0,pad=10)
        ax.set_xlim(-.03,1.04); ax.set_ylim(-.4,len(records)-.35)
        ax.set_xticks(np.linspace(0,1,6)); ax.set_xlabel('Hierarchy F1 (higher is better)')
        ax.grid(axis='x',lw=.5,color='#e2e6e8'); ax.set_axisbelow(True)
        legend=[Line2D([],[],marker='o',ls='',color='#168572',label='Support F1*'),
                Line2D([],[],marker='s',ls='',mfc='white',color='#357cb3',label='Strict clade F1'),
                Line2D([],[],color='#9da8b1',lw=3,label='Numerical / sampling range')]
        if any(r['provisional'] for r in records):
            legend.append(Line2D([],[],marker='o',ls='',mfc='white',color='#168572',label='† Provisional SURD'))
        ax.legend(handles=legend,loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
        fig.suptitle('A · finite-data HO-gLV · one fitted transition TM',fontsize=10,x=.54)
        FIGURE.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(FIGURE,dpi=320,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def number(value): return '未评价' if value is None else f'{value:.3f}'
    rows=['| 方法 | 支持F1* | 严格clade F1 | 上层Recall | 正确根 | 数值／子样本重复F1*范围 | 数值状态 |',
          '|---|---:|---:|---:|---|---|---|']
    for row in records:
        name=row['label'].replace('$','')
        root='未评价' if row['correct_root'] is None else ('是' if row['correct_root'] else '否')
        interval='未评价' if row['f1_range'] is None else f"{row['f1_range'][0]:.3f}–{row['f1_range'][1]:.3f}"
        mark='†' if row['provisional'] else ''
        rows.append(f"| {name} | {number(row['f1'])}{mark} | {number(row['strict_f1'])}{mark} | {number(row['upper_recall'])} | {root} | {interval} | {row['status']} |")
    surd=payload['methods'].get('surd_maximum_weight',{})
    surd_note=''
    if surd.get('provisional'):
        surd_note=f"SURD暂定结果†采用网格`{surd['grid']}`的全部{surd['completed_integration_replicates']}/4份完整511组合表，先平均全部可用完整表的协同原子，再构树；未按F1挑选。完整四scramble精度审计尚未通过；来源状态`{surd['source_status']}`，已完成表的检查问题`{surd['failed_checks']}`。暂定分数不作为SPT超过SURD的确证证据。"
    text='\n'.join([
        '### 8.5 当前五项 F1 比较（由缓存合并汇总）','',
        f"状态：`{summary['status']}`；已完成局部信息查询{summary['query_count']}项；本轮含smoke／数值修复／追加自然基线的墙钟累计{summary['elapsed_wall_seconds']/3600:.3f}小时（原上限8小时）。SPT复用第7节结果，SURD和Block读取同一有限数据拟合联合分布；追加WMS自然固定target树与SHAP归因模式树，当前共五项；归一化Block-PhiR已按用户要求取消。WMS仅增加自然源密度近似，转移核仍是同一非线性TM；SHAP解释同一非线性均值。已取消的pair/future-MI＋HAC指标不恢复。",'',
        *rows,'',
        surd_note,'',
        '![A的五项有限数据F1比较](assets/kuramoto_hierarchy/spt_glv_a_active_methods.png)','',
        '信息方法先平均全部采用的完整积分表再构树，通常为四份；若SURD只有部分完整重复或最终精度审计未过，其空心圆点与†仅表示暂定结果，采用的完整重复数见文字说明。SHAP先平均四份归因模式再构树。方点是对全部非根二叉clade打分的严格F1；灰线为已采用积分重复或SHAP的四组自然评价／背景子样本的F1*范围，不是置信区间。最底层123内部必须二叉细分，因此支持F1*=1时严格F1仍可能是12/13。不完整511组合表不构SURD树，显著负specific／atom仍显式失败；未完成项目不填0，也不据此归因为算法失败。','',
        '本轮只有seed0的一份训练数据。四个scramble共用同一拟合TM，仅刻画积分重复稳定性；它们不能用于独立数据显著性检验。表中差距只适用于当前网络和拟合模型，不能据此声称SPT统计显著超过其他方法。SPT也尚未对全511项统一完成4096外层复核，沿用探索性筛查限定。','',
        '可复现入口：`scripts/spt_glv_a_comparison.py`与`scripts/spt_glv_natural_baselines.py`；合并结果为`results/spt_local_glv_tm/a_comparison/comparison_active_methods.json`，当前主进程三项与自然两项分别保留原始输出。汇总入口仅读取缓存，不重新积分。',''])
    start='<!-- A_FIVE_METHOD_RESULTS_START -->'; end='<!-- A_FIVE_METHOD_RESULTS_END -->'
    original=REPORT.read_text()
    section=start+'\n'+text+'\n'+end
    if start in original:
        before,rest=original.split(start,1); _,after=rest.split(end,1); original=before+section+after
    else: original=original.rstrip()+'\n\n'+section+'\n'
    REPORT.write_text(original)
    return summary


if __name__=='__main__':
    result=summarize(Path(sys.argv[1]) if len(sys.argv)>1 else DIRECTORY)
    print(json.dumps(result,ensure_ascii=False))
