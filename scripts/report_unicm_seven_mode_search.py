#!/usr/bin/env python3
"""Render and summarize the completed seven-mode parameter search."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.search_unicm_seven_mode_parameters import OUT,LOG,TARGETS,METHODS,sha256
from scripts.run_unicm_information_prior_comparison import COLORS

PALETTE={**COLORS,'shap':'#BE6B32'}
LABELS=dict(xi_shapley=r'$\Xi$-Shapley',ei_shapley='EI-Shapley',shap='SHAP',surd='Gaussian SURD',
            uniform='Uniform ridge',univariate='Univariate',phi_r=r'$\Phi^R$',phi_wms=r'$\Phi^{WMS}$',
            phi_si=r'$\Phi_{SI}$',causal_density='Causal density')
FAMILY_LABELS=dict(standard_global='Shared parameters',standard_targetwise='Per-mode parameters',
    block_global='Shared + block ratio',block_targetwise='Per-mode + block ratio',
    anchor_global='Shared + anchored penalty',anchor_targetwise='Per-mode + anchored penalty')


def letter(ax,value):
    ax.set_title(value,loc='left',fontsize=11,fontweight='bold',pad=10)


def figure(result):
    fig,axes=plt.subplots(2,2,figsize=(13.2,8.3),layout='constrained',
                          gridspec_kw={'width_ratios':(1.1,1)})
    original=result['fixed_original_methods']['xi_shapley']
    families=result['families'];primary=families[result['primary_family']]
    ax=axes[0,0]
    names=['original',*FAMILY_LABELS]
    for i,name in enumerate(names):
        e=original if name=='original' else families[name]['methods']['xi_shapley']
        point=e.get('reduction_vs_frozen_percent',e.get('rmse_reduction_vs_frozen_percent'))
        ci=e.get('reduction_vs_frozen_ci95_percent',e.get('rmse_reduction_vs_frozen_ci95_percent'))
        color='#808B98' if name=='original' else ('#007D79' if name==result['primary_family'] else '#3F6F9F')
        ax.plot(ci,[i,i],color=color,lw=1.2)
        ax.scatter(point,i,color=color,s=30,zorder=3)
        ax.annotate(f'{point:.2f}%',(point,i),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    ax.set(yticks=range(len(names)),yticklabels=['Current fixed baseline',*FAMILY_LABELS.values()],
        xlabel='Ξ test nRMSE reduction vs Frozen (%)',ylim=(-.6,len(names)-.4))
    ax.invert_yaxis();ax.axvline(0,color='.6',ls='--',lw=.7)
    ax.grid(axis='x',color='#E9EDF2',lw=.6);letter(ax,'a')
    ax=axes[0,1]
    order=('xi_shapley','ei_shapley','shap','surd','uniform','univariate')
    for i,key in enumerate(order):
        g=families['standard_global']['methods'][key]
        p=primary['methods'][key]
        ax.plot([g['test_nrmse'],p['test_nrmse']],[i,i],color='#C5CCD3',lw=1)
        ax.scatter(g['test_nrmse'],i,color=PALETTE[key],s=32)
        ax.scatter(p['test_nrmse'],i,facecolor='white',edgecolor=PALETTE[key],s=32)
    ax.set(yticks=range(len(order)),yticklabels=[LABELS[k] for k in order],xlabel='Test nRMSE (lower is better)')
    ax.invert_yaxis();ax.grid(axis='x',color='#E9EDF2',lw=.6)
    handles=[plt.Line2D([],[],ls='',marker='o',color='#58687D',label='Shared-parameter diagnostic'),
             plt.Line2D([],[],ls='',marker='o',color='#58687D',mfc='white',label='Predeclared per-mode main family')]
    ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=7)
    letter(ax,'b')
    ax=axes[1,0]
    global_=families['standard_global']['methods']['xi_shapley']
    delta=np.asarray(primary['methods']['xi_shapley']['target_nrmse'])-global_['target_nrmse']
    for i,value in enumerate(delta):
        ax.plot([0,value],[i,i],color='#58687D',lw=1.2)
        ax.scatter(value,i,color='#BE6B32' if value>0 else '#007D79',s=30)
    ax.set(yticks=range(7),yticklabels=TARGETS,
        xlabel='Per-mode main minus shared-parameter nRMSE\nPositive: more flexible parameters perform worse')
    ax.invert_yaxis();ax.axvline(0,color='.6',ls='--',lw=.7);ax.grid(axis='x',color='#E9EDF2',lw=.6);letter(ax,'c')
    ax=axes[1,1]
    contrasts=primary['xi_simultaneous_comparisons']
    for i,(key,e) in enumerate(contrasts.items()):
        low,high=e['bonferroni_family_ci95']
        ax.plot([low,high],[i,i],color=PALETTE[key],lw=1.2)
        ax.scatter(e['point'],i,color=PALETTE[key],s=25)
    ax.set(yticks=range(len(contrasts)),yticklabels=[LABELS[k] for k in contrasts],
        xlabel='nRMSE advantage of Ξ main family\nBonferroni-adjusted family 95% intervals; positive favors Ξ')
    ax.invert_yaxis();ax.axvline(0,color='.5',ls='--',lw=.7);ax.grid(axis='x',color='#E9EDF2',lw=.6);letter(ax,'d')
    path=OUT/'parameter_search_comparison.png'
    fig.savefig(path,dpi=350,bbox_inches='tight');plt.close(fig)
    return path


def report(result,figure_path):
    families=result['families'];primary=families[result['primary_family']]
    original=result['fixed_original_methods']['xi_shapley']
    global_=families['standard_global']['methods']
    anchor=primary['methods']
    screen=json.loads((OUT/'screen_summary.json').read_text())
    refine=json.loads((OUT/'refine_summary.json').read_text())
    searched=json.loads((OUT/'anchor_summary.json').read_text())
    x=global_['xi_shapley'];gain=x['gain_over_original_xi']
    lines=['# UniCM七模态预测：释放参数搜索结果','',
        '**结论：本轮没有找到能稳定扩大Ξ相对EI／SHAP预测优势的参数方案。** 原统一参数方法经验证选参后的诊断结果可从10.65%提高到11.89%，但新增改善区间跨零；更灵活的预定主方案在验证集更好，在测试集反而退化到7.23%。正式主图和 `earth.md` 均未修改。','',
        f'![参数搜索对比]({figure_path})','',
        '图中a为Ξ的各预先声明家族及12月块逐项95%区间，灰色是当前固定参数；b全部方法各自选参，实心为共享参数诊断、空心为预先声明的逐模态主方案；c显示主方案相对共享参数方案的逐模态误差变化；d为主方案对九项竞品的Bonferroni家族95%区间。主方案在看测试之前已冻结，没有根据测试结果改成得分最高的家族。','',
        '## 对当前结果的实际改善','',
        '| Ξ方案 | 验证nRMSE ↓ | 测试nRMSE ↓ | 相对Frozen降幅 |',
        '|---|---:|---:|---:|',
        f"| 当前固定α=30000、γ=3、floor=.05 | 0.827326 | {original['test_nrmse']:.6f} | {original['rmse_reduction_vs_frozen_percent']:.2f}% |"]
    for name,label in FAMILY_LABELS.items():
        e=families[name]['methods']['xi_shapley']
        lines.append(f"| {label}{'（预定主方案）' if name==result['primary_family'] else ''} | {e['validation_nrmse']:.6f} | {e['test_nrmse']:.6f} | {e['reduction_vs_frozen_percent']:.2f}% |")
    lines+=['',f"共享参数诊断的降幅增加 **{x['reduction_vs_frozen_percent']-original['rmse_reduction_vs_frozen_percent']:.2f}个百分点**。相对当前固定Ξ的绝对nRMSE改善为 **{gain['point']:.6f}**，95%配对时间块区间 **[{gain['ci95'][0]:.6f}, {gain['ci95'][1]:.6f}]**，跨零，不能确认其改进稳定。该结果是预先保留的诊断家族，不能在看过测试分数后替换主方案。",'',
        '## 同预算、各自选参后的方法比较','',
        '| 方法 | 共享参数诊断nRMSE | 相对Frozen降幅 | 逐模态主方案nRMSE | 相对Frozen降幅 |',
        '|---|---:|---:|---:|---:|']
    for method in (*METHODS,'uniform','univariate'):
        g,p=global_[method],anchor[method]
        lines.append(f"| {method} | {g['test_nrmse']:.6f} | {g['reduction_vs_frozen_percent']:.2f}% | {p['test_nrmse']:.6f} | {p['reduction_vs_frozen_percent']:.2f}% |")
    lines+=['',
        '共享参数诊断中，EI为12.08%，Ξ为11.89%，SHAP为11.80%。Ξ相对SHAP的nRMSE差异区间跨零；相对EI的点估计略差。不能据这些结果声称释放参数后扩大了Ξ独有优势。','',
        '| 主方案Ξ对比较方法的nRMSE改善 | 点估计 | 逐项95%区间 | 九项比较校正后的家族95%区间 |',
        '|---|---:|---:|---:|']
    for method,e in primary['xi_simultaneous_comparisons'].items():
        lo,hi=e['pointwise_ci95'];a,b=e['bonferroni_family_ci95']
        lines.append(f"| {method} | {e['point']:.6f} | [{lo:.6f}, {hi:.6f}] | [{a:.6f}, {b:.6f}] |")
    alt=primary['block24_sensitivity']['frozen']
    lines+=['',f"保留原12月块为主要不确定性约定，同时检查24月块以匹配最长预测窗。24月块下，主方案Ξ相对Frozen的绝对改善区间为 **[{alt['ci95'][0]:.6f}, {alt['ci95'][1]:.6f}]**，也跨零。因测试起报只有96个月，不能把168个target–lead单元当作独立重复。",'',
        '## 搜索了什么','',
        '- 原十一源、12月历史、三个冻结checkpoint、原target-specific Ξ/EI及其他先验定义保持不变；评分和验证目标仅ENSO、NPMM、SPMM、IOB、IOD、SIOD、TNA的7×24单元平均nRMSE。未采用此前joint7共享先验，也未把Φ改为21对。',
        '- 初筛每信息先验468组：α13值（0.1—1e6）、γ9值（0.125—8）、floor四值（0.005、0.05、0.5、5）。',
        '- 每先验再做standard家族1064组、block家族1368组的统一/逐模态局部细化，含两轮256个逐目标候选及256个全局候选；同一Sobol设计/相同相对半径用于所有信息先验。',
        '- 新增块比例r，将11个冻结预测特征与33个历史特征分别正则化；r=1退回原四组重复惩罚。所有方法都能调此参数，Uniform也搜索α×r。',
        '- 局部边界为α∈[1e-3,1e9]、γ∈[0.0625,16]、floor∈[1e-4,50]、r∈[1e-3,1000]。信息先验保持γ>0，Uniform保留显式无信息对照。',
        '- 每信息先验再做1034组κ∈[0,1]的收缩中心搜索。κ=0为原零中心系数正则，κ=1向自身Frozen预测斜率收缩；截距仍为fit目标均值，因此其极限是均值已校正的Frozen斜率，并非原Frozen原值。系数容量仍44/target/lead。',
        '- 逐模态家族每个模态用一组参数、固定应用全部24lead，没有逐起报或逐测试事件选参。主家族的扩展依据只有验证结果，演进过程保留于搜索记录。',
        '- 每种信息先验共尝试3934组配置布局（含重复/嵌套候选，不视为独立试验）；所有8种信息先验同预算。Univariate只需搜索α，用49值完整范围而非无效γ/floor轴。',
        '- 先复现原拟合数值，再检查新批量求解与原feature_penalty一致；收缩中心通过独立残差拟合等价核查。', '',
        '整个搜索不使用测试目标计算任何候选目标函数。加载器会读取包含完整时间序列的文件并构造测试特征，但所有参数和家族的选择只依据原36个验证起报。配置与家族先落盘为 `frozen_configurations.json`，随后一次统一评价原96个测试起报；冻结SHA在评价前后完全一致。','',
        '## 为什么验证更好，测试反而更差','',
        '逐模态主方案的验证nRMSE从当前约0.827降到0.767，测试却从1.120升至1.163。比共享参数诊断退化最明显的是NPMM、IOB和ENSO；这与在有限重叠验证窗口中选择更多自由度后泛化变差的现象一致。它提示验证选择不稳定，不能仅据此确定唯一成因。', '',
        '部分逐模态最优参数达到搜索边界（例如很大α或块比例），也表明局部验证最优不一定是稳健配置。本轮按事先停止规则完成搜索与评估后，没有因不利测试结果继续增添候选或改换主方案。', '',
        '## 参数记录','',
        '共享参数诊断Ξ：','',
        '```json',json.dumps(x['parameters'][0],indent=2),'```','',
        '逐模态主方案Ξ：','',
        '| 目标 | α | γ | floor | forecast/history ratio | κ |','|---|---:|---:|---:|---:|---:|']
    for name,p in zip(TARGETS,anchor['xi_shapley']['parameters']):
        lines.append(f"| {name} | {p['alpha']:.6g} | {p['gamma']:.6g} | {p['floor']:.6g} | {p['forecast_ratio']:.6g} | {p.get('anchor_strength',0):.6g} |")
    lines+=['',
        '## 结论的边界与建议','',
        '保持当前正式10.65%方案，暂不替换。可以将统一参数搜索11.89%记录为探索性改进，但不能称为稳定显著提升；未找到能显著优于EI／SHAP的配置。若继续做新一轮方法改造，应先预留新的时间验证／测试资料或采用带足够预测窗隔离的滚动时间验证，避免在这段已反复查看的测试期上继续追分。本轮没有启动新模型训练，也没有以不公平的竞品固定参数制造差距。','',
        '所有区间均条件于固定先验、checkpoint与已选参数，不包含调参/家族选择误差。测试期此前已被研究查看，本轮结果属于探索性。九项比较校正改善了多重比较解释，但不消除这些边界。','',
        '## 方法依据与有效性','',
        '本轮重新经Zotero查验父条目P6UJCVG8，标题 *Emergent hierarchical organization of causal interactions in complex systems*；当前正文DXGC7JEA、补充MWIWKSVG，无明确稿件版本日期。重新读取补充S12.1.2 pp21–22关于逐目标先验、44维特征、原固定参数、读出及时间划分的方法。原稿固定超参数，本轮放开超参数/收缩中心属于用户授权的探索扩展，不声称稿件原方法已经采用。','',
        '原Syn原生容差1e-8 bit：缓存归因逐项检查，容差内归零与显著违例计数均记录于identity，超阈值失败；本轮没有重新估计Syn，也没有静默截断。干预范围和原affine degree-1 TM读出不变。输入文件、先验及配置冻结SHA均保存；独立子代理核查参数选择、预测分数、同时区间及正式文件未变。','',
        '局部细化末尾曾遇JSON序列化类型错误，九方法计算结果已逐项保存至JSONL；修正后从这些完成记录恢复配置，没有按测试结果重跑或更换参数。此失败及恢复记录保留，未丢失选参结果。','',
        '## 复现与证据','',
        '- 搜索入口：`scripts/search_unicm_seven_mode_parameters.py`，阶段依次为screen、refine、anchor、freeze、evaluate。',
        '- 图文入口：`scripts/report_unicm_seven_mode_search.py`。',
        '- 本说明及状态：`docs/log/unicm_seven_mode_tuning/`。',
        '- 原始选择记录、冻结配置、测试结果及预测：`results/unicm_seven_mode_tuning/`。',
        f"- 冻结配置SHA256：`{result['configuration_freeze_sha256']}`。",'']
    text='\n'.join(lines)
    (LOG/'tuning_report.md').write_text(text)
    (OUT/'report.md').write_text(text)
    (LOG/'leaderboard.md').write_text('# 各家族验证结果与测试结果\n\n'+
        '\n'.join(lines[lines.index('| Ξ方案 | 验证nRMSE ↓ | 测试nRMSE ↓ | 相对Frozen降幅 |'):lines.index('| Ξ方案 | 验证nRMSE ↓ | 测试nRMSE ↓ | 相对Frozen降幅 |')+9])+'\n')
    (LOG/'live_status.md').write_text('# 当前状态\n\n参数搜索与预冻结评价完成。未找到稳定扩大Ξ对EI/SHAP差距的配置。标准global诊断11.89%，原方案10.65%，改进区间跨零；预定逐模态主方案7.23%。正式主图/报告未变。\n\n证据：tuning_report.md、parameter_search_comparison.png、冻结配置与evaluation_summary.json。\n')
    (LOG/'next_steps.md').write_text('# 停止与后续\n\n本轮搜索计划已完成，未按不利测试结果追加候选。正式方案保持10.65%。若未来继续，应先设计新的带24月隔离的滚动时间验证或预留新的测试期；本轮不自动扩大模型或继续对相同测试期资料追分。\n')


def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    result=json.loads((OUT/'evaluation_summary.json').read_text())
    hashes=json.loads((LOG/'official_artifact_hashes.json').read_text())
    for path,expected in hashes.items():
        if sha256(ROOT/path)!=expected: raise ValueError('Official artifact changed: '+path)
    path=figure(result);report(result,path)
    print(path);print(LOG/'tuning_report.md')


if __name__=='__main__': main()
