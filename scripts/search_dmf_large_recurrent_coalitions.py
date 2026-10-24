#!/usr/bin/env python3
"""Exhaustive descriptive screening of registered exact 11–99 ROI SPT nodes.

Only previously observed native tree sets are searched, not all 2^100 subsets.
Primary frequency: above the original order-specific background threshold in
>=2/3 seeds at >=4/6 individual transition-window G points. Natural-presence
frequency and >=1-G sensitivity are kept distinct. 95–99 ROI nodes are retained
and separately reported because they approach the common root.

This is an exploratory all-cohort search, with no claim of held-out validation,
no new simulations, density fitting, EI/Syn queries, tree search, or p values.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_dmf_sc_topology_comparison import BASE,load_data,sha
from scripts.analyze_dmf_critical_coalitions_all93 import (
    M,METRICS,mask_members,word_mask,read_subject,empty_audit,
)
from scripts.run_dmf_paired_spt_pilot import paired_sources
from scripts.plot_dmf_spt_peel_groups import load_comparison,PAIRED_SUMMARY,YEO_METADATA
from scripts.spt import flatten_nodes

OUT=ROOT/'results/dmf_schaefer100/large_coalition_recurrence'
REPORT=ROOT/'docs/reports/dmf_large_coalition_recurrence.md'


def main():
    started=time.perf_counter()
    _,subjects,words,orders,counts,original,fingerprints,_=load_data()
    c=json.loads((BASE/'contract.json').read_text())
    bg=json.loads((BASE/'background_reference.json').read_text())
    if len(subjects)!=93 or c['repeat_seed_count']!=2 or c['syn_tolerance_nats']!=1e-8:
        raise ValueError('Frozen cohort/repeat/validity rule changed')
    eligible=np.flatnonzero((orders>=11)&(orders<=99))
    if any(bg['thresholds'][str(k)]['threshold_nats'] is None for k in range(11,100)):
        raise ValueError('At least one screened order lacks its background calibration')
    names=['VIS','SM','DA','SA/VA','LIM','FPN','DMN']
    with np.load(YEO_METADATA,allow_pickle=True) as a:membership=a['network_membership']
    ref,_=load_comparison(PAIRED_SUMMARY,4,[1.3])[0]
    displayed={frozenset(n.indices) for n in flatten_nodes(ref)}

    def maxima(ix):
        return {mode:{metric:int(counts[ix,mi,M[metric]].max(initial=0))
                      for metric in ['critical_ge4','critical_ge3','critical_ge1','presence_subjects']}
                for mi,mode in enumerate(['natural_presence','above_background'])}

    by_order=[dict(order=k,registered_sets=int(np.count_nonzero(orders==k)),
                   maxima=maxima(np.flatnonzero(orders==k))) for k in range(11,100)]
    bands=[dict(minimum=lo,maximum=hi,
                registered_sets=int(np.count_nonzero((orders>=lo)&(orders<=hi))),
                maxima=maxima(np.flatnonzero((orders>=lo)&(orders<=hi))))
           for lo,hi in [(11,19),(20,49),(50,79),(80,94),(95,99)]]

    def ranking(mi,metric,maximum=99,limit=15):
        ix=eligible[(orders[eligible]<=maximum)&(counts[eligible,mi,M[metric]]>0)]
        # Deterministic mask-word tie break has no biological/statistical role.
        keys=(words[ix,0],words[ix,1],orders[ix],
              -counts[ix,0,M[metric]].astype(int),-counts[ix,mi,M[metric]].astype(int))
        return ix[np.lexsort(keys)[:limit]].tolist()

    ids={
        'primary_above_background_ge4':ranking(1,'critical_ge4'),
        'natural_presence_ge4':ranking(0,'critical_ge4'),
        'sensitivity_above_background_ge1_11_to_94':ranking(1,'critical_ge1',94,10),
        'sensitivity_natural_presence_ge1_11_to_94':ranking(0,'critical_ge1',94,10),
    }
    targets=sorted(set(i for v in ids.values() for i in v))
    masks={word_mask(words[i]):i for i in targets}
    check={i:dict(natural_ge4=0,strong_ge4=0,natural_ge1=0,strong_ge1=0,
                  window_any_tree=0,qualifying_subjects=[],qualifying_G_counts=[]) for i in targets}
    audit=empty_audit()
    source={seed:hashlib.sha256(np.concatenate(paired_sources(seed,c['sample_count'],100),axis=1)
                               .tobytes()).hexdigest() for seed in c['seeds']}
    for pi,subject in enumerate(subjects):
        win=c['windows'][subject]
        if len(win['grid_indices'])!=6 or win['required_hits']!=4:
            raise ValueError('Frozen window changed')
        local=dict(c,G=[c['G'][i] for i in win['grid_indices']])
        events=read_subject(BASE,local,subject,bg['thresholds'],sha(BASE/'contract.json'),audit,source)
        for mask,i in masks.items():
            values=events.get(mask,np.zeros((2,6),np.uint8))
            n,s=[sum(int(x).bit_count()>=2 for x in row) for row in values]
            q=check[i]
            q['natural_ge4']+=int(n>=4);q['strong_ge4']+=int(s>=4)
            q['natural_ge1']+=int(n>=1);q['strong_ge1']+=int(s>=1)
            q['window_any_tree']+=int(values[0].any())
            if s>=4:q['qualifying_subjects'].append(subject);q['qualifying_G_counts'].append(s)
        if (pi+1)%25==0:print(f'Verified ranking leaders: {pi+1}/93',flush=True)
    for i,q in check.items():
        for field,mi,metric in [('natural_ge4',0,'critical_ge4'),('strong_ge4',1,'critical_ge4'),
                                ('natural_ge1',0,'critical_ge1'),('strong_ge1',1,'critical_ge1')]:
            if q[field]!=int(counts[i,mi,M[metric]]):
                raise ArithmeticError(f'Leader scan differs from registry: {i}, {field}')

    def record(i):
        members=mask_members(word_mask(words[i]))
        return dict(order=int(orders[i]),members_one_based=[j+1 for j in members],
            missing_one_based=[j+1 for j in range(100) if j not in members],
            network_counts={n:int(np.count_nonzero(membership[members]==j)) for j,n in enumerate(names)
                            if np.any(membership[members]==j)},
            displayed_in_current_mean_SC_tree=frozenset(members) in displayed,
            threshold_nats=bg['thresholds'][str(len(members))]['threshold_nats'],
            counts={mode:{name:int(v) for name,v in zip(METRICS,counts[i,mi])}
                    for mi,mode in enumerate(['natural_presence','above_background'])},
            direct_window_verification=check[i])

    result=dict(status='complete',subject_count=93,screened_registered_sets=len(eligible),
        protocol=dict(scope='Every registered native SPT set with 11–99 ROI; root100 excluded; no fuzzy membership',
            primary='Original above-background >=2/3 seeds at >=4/6 individual transition-window points',
            natural_presence='Same repeat/window rule without the background-strength filter',
            sensitivity='>=2/3 seeds at >=1/6 window points, explicitly not the primary rule',
            ranking='Decreasing selected frequency, natural frequency, increasing size and uint128 membership mask',
            units='Individual SC; seeds and G points are within-participant repetitions',
            inference='Exploratory descriptive all-cohort screening, no confirmatory p values/independent test set'),
        maximum_primary_frequency=int(counts[eligible,1,M['critical_ge4']].max()),
        by_size_band=bands,by_order=by_order,
        rankings={k:[record(i) for i in v] for k,v in ids.items()},
        original_full_registry_audit=original['audit'],leader_verification_audit=audit,
        verified_leader_count=len(targets),source_fingerprints=fingerprints,
        background_sha256=sha(BASE/'background_reference.json'),
        new_simulations=0,new_information_queries=0,elapsed_seconds=time.perf_counter()-started)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    primary=result['rankings']['primary_above_background_ge4']
    text=['# 大于 10 ROI 的完整 SPT 集合频率筛查','',
        '2026-10-24。93 个个体 SC；复用冻结结果，不重跑模拟、估计或树搜索。',
        '',f'检索全部 {len(eligible):,} 个已注册 11–99 ROI 完整集合。原始全 G 注册表覆盖 11,439 棵树；候选排名直接通过个体窗口的 1,674 棵树及其密度元数据复核。',
        '', '**结论：没有找到高频且持续复现的大范围完整 ROI 集合。最高超过背景的原规则频率只有 2/93 人。**',
        '', '原规则：同一 G 至少 2/3 个种子重复，且六点个体转变窗口内至少 4/6 个 G 点满足。每人至多计一次。',
        '', '| ROI 数范围 | 已注册完整集合 | 原规则：超过背景最高人数 | 原规则：自然出现最高人数 | 放宽至一 G：超过背景最高人数 |',
        '|---|---:|---:|---:|---:|']
    for r in bands:
        q=r['maxima'];text.append(f"| {r['minimum']}–{r['maximum']} | {r['registered_sets']:,} | {q['above_background']['critical_ge4']}/93 | {q['natural_presence']['critical_ge4']}/93 | {q['above_background']['critical_ge1']}/93 |")
    text+=['','## 原规则中排名最前的集合','',
            '| ROI 数 | 完整成员（编号从 1 开始） | 原规则人数 | 任一个 G 重复人数 | 任一次出现人数（全 G） |',
            '|---|---|---:|---:|---:|']
    for r in primary:
        if r['order']>=95:label='除 {'+', '.join(map(str,r['missing_one_based']))+'} 外的全部 ROI'
        else:label='{'+', '.join(map(str,r['members_one_based']))+'}'
        q=r['counts']['above_background'];text.append(f"| {r['order']} | {label} | {q['critical_ge4']}/93 | {q['critical_ge1']}/93 | {q['presence_subjects']}/93 |")
    text+=['', '95–99 ROI 的集合接近共同根，可能仅区别于少量末端分离；它们没有被删除，但单列以避免用根附近的普遍包含关系代替具体功能模块复现。',
        '', '即便将窗口持续要求从四个 G 点放宽到一个 G 点，11–19 ROI 的最高超过背景频率也只有 11/93；20–49 为 1/93，50–79 为 2/93，80–94 为 1/93。该放宽结果属于敏感性描述，不可替代原标准后称为高频。',
        '', '前两名 11/12 ROI 集合都没有出现在当前平均 SC、G=1.3、seed 4 的展示树中。没有理由将 E/F 换成“高频大模块”。精确大集合低复现不等于缺少大尺度协同信息；不应把这一检索外推为所有可能源子集都不共享信息。',
        '', '## 检索和推断边界','',
        '- 只覆盖已经实际出现于 SPT 的源集合，不枚举所有 2^100 个可能组合；原始大节点分裂是谱候选内的贪心优化。',
        '- 不要求树中父子分裂也完全相同；本次完整节点成员重复已很少，更强的拓扑一致性未另行声称。',
        '- 全 93 人用于筛选，排名与人数仅为样本内描述；没有独立验证或筛选后显著性检验。',
        '- 自然出现、超背景出现、单次出现、跨种子重复、跨 G 持续分开记录。',
        '- 沿用 1e-8 nats 容差；复核的选中/候选/成对/查询负 Syn 计数均为零，闭合误差不超过 1.60e-14 nats。没有裁剪。',
        '- 脑区解剖映射继续受原 SC 行序 inferred 的限制；本次不据稀少大集合添加新的功能故事。',
        '', '## 方法及来源','',
        '按 experimental-design 的探索性筛查与敏感性区分规则记录。Zotero 本次重新核对父条目 P6UJCVG8（Emergent hierarchical organization of causal interactions in complex systems），读取正文 DXGC7JEA 脑模型部分及补充 MWIWKSVG 的 S5.1/S5.2、S12.2.1。附件无明确修订日期；2026-10-04 为附件元数据日期，不作为最新修订日期。旧稿 Yeo 约束树与当前个体无约束 ROI 树不同，具体结果来自冻结注册表。',
        '', '- 注册表：results/dmf_schaefer100/critical_coalitions/all93_registry.npz。',
        '- 完整排名、逐阶最大值、来源指纹、直接复核：results/dmf_schaefer100/large_coalition_recurrence/summary.json。',
        '- 可复核入口：scripts/search_dmf_large_recurrent_coalitions.py。']
    REPORT.write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(screened=len(eligible),maximum_primary=result['maximum_primary_frequency'],
                          verified_leaders=len(targets),elapsed_seconds=result['elapsed_seconds']),indent=2))
    print(REPORT)


if __name__=='__main__':main()
