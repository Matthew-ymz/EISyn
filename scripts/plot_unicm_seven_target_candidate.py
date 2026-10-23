#!/usr/bin/env python3
"""Draw the seven-future-mode candidate without replacing manuscript assets."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / 'scripts'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from scripts.analyze_unicm_11mode_xi_hierarchy_tree import _tree_from_record, render_trees
from scripts.run_unicm_information_prior_comparison import COLORS
from scripts.plot_unicm_mode_regions import MODES, region_center
from scripts.plot_runge_gateway_mediator_map import (
    LAND_URL, COASTLINE_URL, load_geojson, extract_polygons, extract_lines, draw_world,
)
from scripts.recompute_unicm_seven_target_forecast import TARGETS

OUTPUT = ROOT / 'results/unicm_seven_target_candidate'
STRUCTURE = OUTPUT / 'structure'
LEADS = (1, 8, 24)
LABELS = dict(frozen='Frozen', xi_shapley=r'$\Xi$-Shapley', ei_shapley='EI-Shapley',
    shap='SHAP', univariate='Univariate', uniform='Uniform ridge', phi_r=r'$\Phi^R$',
    phi_wms=r'$\Phi^{\mathrm{WMS}}$', phi_si=r'$\Phi_{\mathrm{SI}}$',
    causal_density='Causal density', surd='Gaussian SURD')
PALETTE = {**COLORS, 'shap': '#BE6B32'}
METHOD_ORDER = ('frozen', 'xi_shapley', 'ei_shapley', 'shap', 'univariate', 'uniform',
                'phi_r', 'phi_wms', 'phi_si', 'causal_density', 'surd')
SOURCE_ORDER = ('nino', 'nino12', 'nino3', 'nino4', 'NPMM', 'SPMM', 'IOB', 'IOD', 'SIOD', 'TNA', 'WWV')


def configure():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'savefig.facecolor': 'white'})


def letter(ax, value, x=-.12):
    ax.text(x, 1.04, value, transform=ax.transAxes, weight='bold', fontsize=10, va='bottom')


def maps(canvas):
    panels = json.loads((STRUCTURE / 'joint7/displayed_edges.json').read_text())['panels']
    land = extract_polygons(load_geojson(LAND_URL))
    coast = extract_lines(load_geojson(COASTLINE_URL))
    grid = canvas.add_gridspec(2,3,height_ratios=(1,.16),hspace=0)
    axes = [canvas.add_subplot(grid[0,i], projection='mollweide') for i in range(3)]
    maximum = max(row['mean_bits'] for rows in panels for row in rows)
    for i, (ax, rows, lead) in enumerate(zip(axes, panels, LEADS)):
        draw_world(ax, land, coast)
        ax.set_xticks([])
        ax.set_yticks(np.radians([-60, -30, 0, 30, 60]), ['60°S', '30°S', '0°', '30°N', '60°N'], fontsize=5)
        ax.set_title(f'Lead = {lead} '+('month' if lead==1 else 'months'), fontsize=9, pad=5)
        letter(ax, 'abc'[i], -.04)
    canvas.canvas.draw() if hasattr(canvas, 'canvas') else canvas.figure.canvas.draw()
    for ax, rows in zip(axes, panels):
        positions = []
        for mode in MODES:
            lon, lat = np.array([region_center(r) for r in mode.regions]).mean(axis=0)
            xy = ax.transData.transform(np.radians([lon, lat]))
            xy += np.array({9:(-3, 10), 10:(-6, -12)}.get(mode.index, (0, 0)))*ax.figure.dpi/72
            positions.append(ax.transAxes.inverted().transform(xy))
        for rank, row in enumerate(rows):
            a, b = [positions[row[k]] for k in ('a', 'b')]
            if row['target'] != 'joint7':
                raise ValueError('Main source-pair map must use the joint target.')
            mid = (a+b)/2
            direction = b-a
            perp = np.array([-direction[1], direction[0]])
            perp /= max(np.linalg.norm(perp), 1e-9)
            hub = mid+(-1 if rank%2 else 1)*(.025+.008*(rank//2))*perp
            strength = row['mean_bits']/maximum
            width, alpha = .5+2.3*strength, .22+.56*strength
            for source in (a, b):
                ax.plot([source[0], hub[0]], [source[1], hub[1]], transform=ax.transAxes,
                    color='#8056a5', lw=.75*width, alpha=alpha*.65, zorder=3)
            ax.scatter(*hub, transform=ax.transAxes, s=10+10*strength, c='#8056a5',
                edgecolors='white', linewidths=.3, zorder=5)
        for i, xy in enumerate(positions):
            ax.text(*xy, str(i), transform=ax.transAxes, ha='center', va='center',
                fontsize=6, color='white' if i<7 else '#3976a1', zorder=6,
                bbox=dict(boxstyle='circle,pad=.3', fc='#3976a1' if i<7 else 'white', ec='#3976a1', lw=.7))
    key = canvas.add_subplot(grid[1,:])
    key.axis('off')
    key.text(.5, .80,
        '0 ENSO · 1 NPMM · 2 SPMM · 3 IOB · 4 IOD · 5 SIOD · 6 TNA · 7 Niño 1+2 · 8 Niño 3 · 9 Niño 4 · 10 WWV',
        ha='center', va='center', fontsize=7, transform=key.transAxes)
    key.text(.5, .40, 'All pairs read the same joint future: (ENSO, NPMM, SPMM, IOB, IOD, SIOD, TNA)',
                ha='center', va='center', fontsize=7, color='#172033', transform=key.transAxes)
    key.text(.5, .02, 'Filled nodes: joint-target members; open nodes: additional sources | line width: pair Syn',
                ha='center', va='center', fontsize=7, color='#66717a', transform=key.transAxes)


def prediction_panels(ax_i, ax_j, forecast):
    methods = forecast['methods']
    frozen = methods['frozen']['test_nrmse']
    null = 100*(1-np.asarray(forecast['shuffled_xi_control']['scores'])/frozen)
    observed = methods['xi_shapley']['reduction_vs_frozen_percent']
    ax_i.scatter(null, 1+np.random.default_rng(20260728).uniform(-.16, .16, len(null)),
        color='#B4BFCC', edgecolor='white', lw=.3, s=15, alpha=.7)
    ax_i.scatter([observed], [0], color=PALETTE['xi_shapley'], s=36)
    ax_i.text(observed, .13, f'{observed:.2f}%', ha='center', color=PALETTE['xi_shapley'], fontsize=7)
    ax_i.text(.98, .92, f"Conditional null rank = {forecast['shuffled_xi_control']['fraction_null_at_least_as_good']:.3f}",
        transform=ax_i.transAxes, ha='right', fontsize=7)
    ax_i.set(yticks=[1,0], yticklabels=['Shuffled Ξ priors', 'Ξ-Shapley prior'], ylim=(-.4,1.4),
             xlabel='Seven-target test nRMSE reduction vs Frozen (%)')
    ax_i.set_xlim(min(0., float(null.min())-.5), max(observed, float(null.max()))+1)
    ax_i.grid(axis='x', color='#E9EDF2', lw=.5)
    letter(ax_i, 'i', -.22)
    for i, key in enumerate(METHOD_ORDER):
        value = methods[key]['reduction_vs_frozen_percent']
        ax_j.scatter(value, i, color=PALETTE[key], s=27, zorder=3)
        ax_j.annotate(f'{value:.2f}%', (value, i), xytext=(5,0), textcoords='offset points', va='center', fontsize=7)
    values = [methods[k]['reduction_vs_frozen_percent'] for k in METHOD_ORDER]
    ax_j.set(yticks=range(len(METHOD_ORDER)), yticklabels=[LABELS[k] for k in METHOD_ORDER],
             xlabel='Seven-target test nRMSE reduction vs Frozen (%)')
    ax_j.invert_yaxis()
    ax_j.set_xlim(min(0., min(values))-.7, max(values)+2)
    ax_j.axvline(0, color='#7A8288', ls='--', lw=.6)
    ax_j.grid(axis='x', color='#E9EDF2', lw=.5)
    hp = forecast['shared_hyperparameters']
    ax_j.set_title(rf"Shared $\Xi$ validation settings: $\alpha={hp['alpha']:g}$, $\gamma={hp['gamma']:g}$",
                   loc='right', fontsize=7, color='#66717a')
    letter(ax_j, 'j', -.25)


def composite(forecast):
    joint = STRUCTURE / 'joint7'
    hierarchy = json.loads((joint/'hierarchy_trees.json').read_text())
    shapley = json.loads((joint/'shapley_summary.json').read_text())
    rows = hierarchy['results']
    trees = [_tree_from_record(next(r for r in rows if r['checkpoint']==2 and r['lead']==lead)['tree']) for lead in LEADS]
    fig = plt.figure(figsize=(13.2, 12.2), layout='constrained')
    grid = fig.add_gridspec(4, 2, height_ratios=(2.15, 2.7, 2.15, 2.25), hspace=.09)
    map_canvas = fig.add_subfigure(grid[0,:])
    maps(map_canvas)
    tree_canvas = fig.add_subfigure(grid[1,:])
    render_trees(trees, [2]*3, OUTPUT/'earth_unicm_seven_target_candidate.png', lead=8,
        split_objective='raw_residual', dpi=400, syn_tolerance=1e-8, canvas=tree_canvas,
        show_checkpoint=False, show_tree_metrics=False, show_root_info=False,
        core_highlights=[False]*3, compact_node_labels=True, display_scale=np.log(2), display_unit='nats',
        node_label_fontsize=7.6, terminal_label_fontsize=8.2)
    for ax, label, lead, tree in zip(tree_canvas.axes[:3], 'def', LEADS, trees):
        letter(ax, label, -.06)
        ax.text(.5,1.01,rf'$\ell={lead}$; joint-target $\Xi={tree.phi_value*np.log(2):.3f}$ nats',
                transform=ax.transAxes, ha='center', fontsize=8)
    tree_canvas.text(.995, .995, 'Joint future target: seven modes | checkpoint 2 | n = 16,384',
        ha='right', va='top', fontsize=7, color='#66717a')
    with np.load(joint/'order_mass.npz') as cache:
        order_mass = cache['mass_bits'].mean(axis=0).T*np.log(2)
    ax_g, ax_h = fig.add_subplot(grid[2,0]), fig.add_subplot(grid[2,1])
    image = ax_g.imshow(order_mass, aspect='auto', origin='upper', cmap='viridis', vmin=0)
    ax_g.set(yticks=range(10), yticklabels=range(2,12), ylabel='SPT node order')
    fig.colorbar(image, ax=ax_g, fraction=.035, pad=.03, label='SPT order mass (nats)')
    shares = np.asarray([[r['shapley_percent_mean'][name] for r in shapley['lead_summary']] for name in SOURCE_ORDER])
    image = ax_h.imshow(shares, aspect='auto', cmap='YlGnBu')
    ax_h.scatter(np.arange(24), np.argmax(shares, axis=0), s=7, facecolor='white', edgecolor='#172033', lw=.3)
    ax_h.set(yticks=range(11), yticklabels=['ENSO' if n=='nino' else n for n in SOURCE_ORDER], ylabel='Source mode')
    fig.colorbar(image, ax=ax_h, fraction=.035, pad=.03, label='Mean Shapley share (%)')
    for ax, label in ((ax_g,'g'), (ax_h,'h')):
        ax.set(xticks=[0,3,7,11,15,19,23], xticklabels=[1,4,8,12,16,20,24], xlabel='Prediction lead (months)')
        ax.tick_params(labelsize=7)
        ax.set_title('Seven-mode joint target | three-checkpoint mean', loc='right', fontsize=7)
        letter(ax,label)
    bottom = grid[3,:].subgridspec(1,2,width_ratios=(1,1.35),wspace=.16)
    prediction_panels(fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1]), forecast)
    path = OUTPUT/'earth_unicm_seven_target_candidate.png'
    fig.savefig(path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return path


def diagnostics(forecast):
    with np.load(OUTPUT/'forecast/evaluation_arrays.npz') as cache:
        target, scale = cache['test_target'], cache['target_scale']
        predictions = {k: cache['prediction_'+k] for k in METHOD_ORDER}
    from scripts.run_unicm_synergy_guided_forecast import cell_rmse
    methods = forecast['methods']
    fig, axes = plt.subplots(2,2,figsize=(12.8,7.2),layout='constrained',gridspec_kw={'width_ratios':(1,1.3)})
    ax = axes[0,0]
    gains = np.asarray(methods['xi_shapley']['gain_over_frozen']['target_ci95'])
    point = np.asarray(methods['frozen']['target_nrmse'])-methods['xi_shapley']['target_nrmse']
    for i,(v,low,high) in enumerate(zip(point,*gains)):
        ax.plot([low,high],[i,i],color=PALETTE['xi_shapley'],lw=1.2)
        ax.scatter(v,i,color=PALETTE['xi_shapley'],s=26)
    ax.set(yticks=range(7),yticklabels=TARGETS,xlabel='Ξ nRMSE reduction vs Frozen (95% interval)')
    ax.invert_yaxis();ax.axvline(0,color='.5',ls='--',lw=.7);letter(ax,'a')
    ax = axes[0,1]
    for k in ('frozen','xi_shapley','ei_shapley','shap','uniform'):
        ax.plot(range(1,25),methods[k]['lead_nrmse'],color=PALETTE[k],label=LABELS[k],lw=1.4)
    ax.set(xlabel='Prediction lead (months)',ylabel='Seven-target mean nRMSE')
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8);letter(ax,'b')
    ax = axes[1,0]
    frozen = cell_rmse(predictions['frozen'],target,scale)
    gain = 100*(1-cell_rmse(predictions['xi_shapley'],target,scale)/frozen)
    limit = np.max(np.abs(gain))
    image=ax.imshow(gain,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.set(yticks=range(7),yticklabels=TARGETS,xticks=[0,3,7,11,15,19,23],xticklabels=[1,4,8,12,16,20,24],
           xlabel='Prediction lead (months)')
    fig.colorbar(image,ax=ax,fraction=.035,pad=.03,label='Ξ reduction vs Frozen (%)');letter(ax,'c')
    ax=axes[1,1]
    independent=forecast['independently_tuned_methods']
    keys=('xi_shapley','ei_shapley','shap','uniform','surd','phi_r','phi_wms','phi_si','causal_density')
    for i,k in enumerate(keys):
        ax.scatter(methods[k]['test_nrmse'],i-.12,color=PALETTE[k],s=26)
        ax.scatter(independent[k]['test_nrmse'],i+.12,facecolor='white',edgecolor=PALETTE[k],s=26)
    ax.set(yticks=range(len(keys)),yticklabels=[LABELS[k] for k in keys],xlabel='Test nRMSE (lower is better)')
    ax.invert_yaxis();letter(ax,'d')
    handles=[plt.Line2D([],[],marker='o',ls='',color='#58687D',label='Shared Ξ settings'),
             plt.Line2D([],[],marker='o',ls='',color='#58687D',mfc='white',label='Each prior tuned on validation')]
    ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8)
    path=OUTPUT/'seven_target_forecast_diagnostics.png'
    fig.savefig(path,dpi=350,bbox_inches='tight');plt.close(fig)
    return path


def structure_diagnostics():
    joint = STRUCTURE/'joint7'
    with np.load(joint/'ei_tables.npz') as cache:
        ei,xi = cache['ei_bits'],cache['xi_bits'][...,-1]
    with np.load(joint/'order_mass.npz') as cache:
        share=cache['share']
    with np.load(STRUCTURE/'scalar_pair_syn_all_leads.npz') as cache:
        pair=cache['values_bits']
    leads=np.arange(1,25)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    ax=axes[0,0]
    series=(ei[...,-1],ei[...,1<<np.arange(11)].sum(axis=2),xi)
    for values,label,color in zip(series,('Joint EI','Sum of singleton EI',r'Joint-target $\Xi$'),
                                  ('#3F6F9F','#8A929D','#007D79')):
        mean,sd=values.mean(axis=0)*np.log(2),values.std(axis=0,ddof=1)*np.log(2)
        ax.plot(leads,mean,color=color,label=label,lw=1.4)
        ax.fill_between(leads,mean-sd,mean+sd,color=color,alpha=.12,lw=0)
    ax.set(xlabel='Prediction lead (months)',ylabel='Information (nats)')
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8);letter(ax,'a')
    ax=axes[0,1]
    for checkpoint,values,color in zip((1,2,3),xi,('#9BACCA','#007D79','#BE6B32')):
        ax.plot(leads,values*np.log(2),label=f'Checkpoint {checkpoint}',color=color,lw=1.4)
    ax.set(xlabel='Prediction lead (months)',ylabel='Joint-target Ξ (nats)')
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8);letter(ax,'b')
    ax=axes[1,0]
    for values,label,color in ((share[...,:4].sum(axis=2),'Orders 2–5','#007D79'),
                              (share[...,4:].sum(axis=2),'Orders 6–11','#7B5EA7')):
        mean,sd=100*values.mean(axis=0),100*values.std(axis=0,ddof=1)
        ax.plot(leads,mean,label=label,color=color,lw=1.4)
        ax.fill_between(leads,mean-sd,mean+sd,color=color,alpha=.12,lw=0)
    ax.set(xlabel='Prediction lead (months)',ylabel='Within-tree order share (%)',ylim=(0,100))
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8);letter(ax,'c')
    ax=axes[1,1]
    # Distinct-source/target triples only, matching the map eligibility rule.
    values=np.zeros((7,24))
    from itertools import combinations
    for pi,(a,b) in enumerate(combinations(range(11),2)):
        for target in range(7):
            if target not in (a,b):
                values[target]+=pair[:,:,pi,target].mean(axis=0)*np.log(2)/45
    image=ax.imshow(values,aspect='auto',cmap='Purples',vmin=0)
    ax.set(yticks=range(7),yticklabels=TARGETS,xticks=[0,3,7,11,15,19,23],xticklabels=[1,4,8,12,16,20,24],
           xlabel='Prediction lead (months)',ylabel='Scalar future receiver')
    fig.colorbar(image,ax=ax,fraction=.035,pad=.03,label='Mean pair Syn (nats)');letter(ax,'d')
    output=OUTPUT/'seven_target_structure_diagnostics.png'
    fig.savefig(output,dpi=350,bbox_inches='tight');plt.close(fig)
    return output


def checkpoint_trees():
    rows=json.loads((STRUCTURE/'joint7/hierarchy_trees.json').read_text())['results']
    fig=plt.figure(figsize=(13,14),layout='constrained')
    grid=fig.add_gridspec(3,1)
    for i,lead in enumerate(LEADS):
        canvas=fig.add_subfigure(grid[i])
        trees=[_tree_from_record(next(r for r in rows if r['checkpoint']==cp and r['lead']==lead)['tree'])
               for cp in (1,2,3)]
        render_trees(trees,[1,2,3],OUTPUT/'seven_target_checkpoint_trees.png',lead=lead,
            split_objective='raw_residual',dpi=350,syn_tolerance=1e-8,canvas=canvas,
            show_checkpoint=True,show_tree_metrics=False,show_root_info=False,core_highlights=[False]*3,
            compact_node_labels=True,display_scale=np.log(2),display_unit='nats',
            node_label_fontsize=7.6,terminal_label_fontsize=8.2)
        canvas.text(.005,.99,f'Lead {lead} month'+('' if lead==1 else 's')+' · seven-mode joint future',
                    va='top',fontsize=9,weight='bold')
        for ax,cp,tree in zip(canvas.axes[:3],(1,2,3),trees):
            ax.text(.5,.97,rf'Checkpoint {cp} · $\Xi={tree.phi_value*np.log(2):.3f}$ nats',
                    transform=ax.transAxes,ha='center',va='top',fontsize=8)
    output=OUTPUT/'seven_target_checkpoint_trees.png'
    fig.savefig(output,dpi=350,bbox_inches='tight');plt.close(fig)
    return output


def report(forecast):
    methods=forecast['methods']
    structure=json.loads((STRUCTURE/'joint7/shapley_summary.json').read_text())
    hierarchy=json.loads((STRUCTURE/'joint7/hierarchy_trees.json').read_text())
    pair=json.loads((STRUCTURE/'joint7/displayed_edges.json').read_text())
    selected=forecast['shared_hyperparameters']
    xi=methods['xi_shapley']
    null=np.asarray(forecast['shuffled_xi_control']['scores'])
    matched=int(np.count_nonzero(null<=xi['test_nrmse']))
    lines=['# Earth：每月七维联合目标的候选重算', '',
        '2026-10-23。候选已完成，尚未替换 `docs/reports/earth.md`、正式主图或任何旧缓存。', '',
        '## 结果', '',
        f"主预测采用联合七模态 Ξ 归因，测试平均 nRMSE 为 **{xi['test_nrmse']:.6f}**，相对 Frozen 降低 **{xi['reduction_vs_frozen_percent']:.2f}%**，配对时间块95%区间为 **[{xi['reduction_vs_frozen_ci95_percent'][0]:.2f}%, {xi['reduction_vs_frozen_ci95_percent'][1]:.2f}%]**。",
        '该改善没有转化为优于校准基线的证据：与同参数 Uniform 的差异区间跨零，SHAP 的误差更低。联合信息组织依然可分析，但当前联合先验未显示额外预测优势。', '',
        '| 方法 | 测试 nRMSE ↓ | 相对 Frozen 降幅 |', '|---|---:|---:|']
    for key in METHOD_ORDER:
        e=methods[key]
        lines.append(f"| {key} | {e['test_nrmse']:.6f} | {e['reduction_vs_frozen_percent']:.2f}% |")
    lines += ['', 'Ξ相对各比较方法的绝对 nRMSE 改善（正数有利于Ξ）：', '',
              '| 比较对象 | 改善 | 逐项95%区间 |', '|---|---:|---:|']
    for key in ('frozen','uniform','univariate','ei_shapley','shap'):
        e=xi['gain_over_'+key]
        lines.append(f"| {key} | {e['point']:.6f} | [{e['ci95'][0]:.6f}, {e['ci95'][1]:.6f}] |")
    lines += ['', f"在固定真实联合先验验证最优参数下，**{matched}/{len(null)}** 个标签打乱先验达到或优于真实先验。条件排名为 {(1+matched)/(1+len(null)):.6f}，没有真实标签优于打乱的证据。这是固定选参结果下的诊断排名，未校正真实先验的选参过程，不是严格随机化检验的P值。每个lead只置换一次源标签，再共享给七个预测输出，保持真实先验的共享结构。", '',
        '逐模态点估计如下，不能将总体改善解释为七模态逐项改善；置信区间见预测诊断图。', '',
        '| 目标 | Ξ相对Frozen降幅 |', '|---|---:|']
    for name,value in zip(TARGETS,xi['target_reduction_vs_frozen_percent']):
        lines.append(f'| {name} | {value:.2f}% |')
    lines += ['', '## 结构变化', '',
        '| lead（月） | 联合 Ξ（bits，三checkpoint均值±SD） |', '|---|---:|']
    for row in structure['lead_summary']:
        if row['lead'] in LEADS:
            lines.append(f"| {row['lead']} | {row['interaction_mean_bits']:.6f} ± {row['interaction_std_bits']:.6f} |")
    peak=max(structure['lead_summary'],key=lambda r:r['interaction_mean_bits'])
    order=np.load(STRUCTURE/'joint7/order_mass.npz')
    within=order['share'][...,:4].sum(axis=2).mean(axis=0)
    lines += ['', f"联合Ξ的均值峰位仍为第{peak['lead']}个月，但相对短期的增强幅度小于旧十一目标图的描述，且checkpoint间趋势并不一致。第1、8、24个月二至五阶质量占比分别为 {within[0]:.2%}、{within[7]:.2%}、{within[23]:.2%}。以上SD表示三个冻结checkpoint之间的差异，不是置信区间。", '',
        '主图第8个月的代表树不再出现旧版相同的五模态 ENSO–IOD 嵌套子树；旧核心累计占比及其强度不应移植。目标子空间缩小后Ξ不必单调下降，因为整体EI与单源EI之和同时改变。', '',
        '## 图及图注', '',
        '![七维联合目标完整候选主图](earth_unicm_seven_target_candidate.png)', '',
        '**图1（候选）** a–c，第1、8、24个月的前十个源对，同一个七维未来向量为共同target；每个lead从55个源对中按三checkpoint均值排序。紫色线只连接源对、共同target不赋予虚构的地理位置；线宽共享尺度。实心节点是七个共同target成员，全部11节点仍可作源。d–f，同lead的checkpoint 2完整SPT。g，72棵树分别按阶求和，再跨checkpoint平均的绝对order mass。h，每个checkpoint/lead内归一化后平均的联合Ξ精确Shapley份额，白点标均值首位。i，联合Ξ真实先验与200个按lead共享标签置换的先验，参数固定为真实先验验证最优；横轴仅评七输出，条件排名不作选参校正。j，仅七输出的平均nRMSE相对Frozen降幅；Ξ/EI使用相同联合target的归因，其他先验保留各自定义。所有先验共享Ξ所选参数；正值表示改善。', '',
        '![七目标预测诊断](seven_target_forecast_diagnostics.png)', '',
        '**图2** a，Ξ对Frozen的逐模态绝对nRMSE改善及4000次共享时间块重采样的逐项95%区间。b，七目标平均逐lead nRMSE。c，逐目标/lead降幅点估计。d，固定Ξ参数与每方法独立验证选参的测试误差；空心/实心含义见外置图例。', '',
        '![七目标结构诊断](seven_target_structure_diagnostics.png)', '',
        '**图3** a，整体EI、单源EI之和及Ξ，均值±三个checkpoint的SD。b，各checkpoint联合Ξ。c，先逐树归一化的二至五阶与六至十一阶质量份额，均值±SD。d，补充的七个标量接收端诊断：每个target只平均45对不含目标自身的源对Syn，再跨checkpoint平均。d是标量target的定位图，不能将其跨target相加当作七维联合Syn。', '',
        '![三个checkpoint的代表SPT](seven_target_checkpoint_trees.png)', '',
        '**图4** 三行分别为第1、8、24个月，三列为checkpoint 1、2、3；各行共享Syn色标。节点值为局部Syn（nats），每棵树局部量之和闭合到根Ξ。不同lead色标不同，比较时应读取数值。', '',
        '## 目标定义与控制', '',
        r'每个源是完整12月历史向量 $\mathbf{h}_m$。所有11源组成 $\mathbf{h}$，其132个坐标均独立采样自 $\mathcal{U}[-4,4]$；复用同一批16,384条历史和三个冻结checkpoint的预测。每个lead单独定义', '',
        r'$$\mathbf{y}_{\ell}=(Y_{\mathrm{ENSO},\ell},Y_{\mathrm{NPMM},\ell},Y_{\mathrm{SPMM},\ell},Y_{\mathrm{IOB},\ell},Y_{\mathrm{IOD},\ell},Y_{\mathrm{SIOD},\ell},Y_{\mathrm{TNA},\ell})^\top\in\mathbb{R}^{7}.\tag{1}$$', '',
        r'式(1)的24个目标状态分别计算，不拼成168维target，也不加入Niño 1+2、Niño 3、Niño 4或WWV的未来值。对应的联合game为', '',
        r'$$\Xi_{\ell}(S)=EI(\mathbf{h}_S\to\mathbf{y}_{\ell})-\sum_{m\in S}EI(\mathbf{h}_m\to\mathbf{y}_{\ell}).\tag{2}$$', '',
        '式(2)的全部2048个源联盟用于精确Shapley和完整SPT；二源情况即主图a–c的Syn。SPT遍历当前节点全部无序二分，最小化原始跨组残差，递归至单源；不固定旧核心。', '',
        r'采用原生坐标、已知独立源协方差 $(16/3)\mathbf{I}_{132}$ 的仿射degree-1 TM/Gaussian读出，残差正则1e-6。每lead固定干预月份相位start_month=0，sampling_seed=20260901。这是有限正则读出近似，不能称为非线性模型精确连续MI。此处“TM”是transport map。', '',
        '三个checkpoint的联合Ξ/EI Shapley先在原生bit单位平均，再将每lead同一条11源权重广播给七输出的44特征校准器。此变更也消除了旧版结构读出与校准先验在目标维度/坐标正则上的差别；因此旧新版分数之差不只由少评四个输出造成。', '',
        'Frozen内部仍按既有11通道checkpoint运行；源历史及校准特征维数不变。仅七个校准输出被拟合和保存（96×7×24），额外四通道不进入预测评分或新的信息target。', '',
        f"fit/validation/test为253/36/96个起报，期间保留24月目标隔离，fit-only预处理及nRMSE分母不变。α用8值、γ用6个正值，共48组/信息先验，floor=0.05固定；主比较由七目标平均验证误差选出α={selected['alpha']:g}、γ={selected['gamma']:g}，其他先验固定此参数；公平性诊断为每方法同网格独立选参。Univariate另外用原6值α网格重新验证选择。未根据测试结果追加网格。", '',
        '评分对每个target/lead的96个起报先求RMSE、除以该target的fit期标准差，再在7×24单元等权平均。4000次12月循环时间块bootstrap沿用原共享月份权重，区间逐项、未调整多重性。', '',
        '## 比较边界与诊断保留', '',
        '- Ξ/EI使用七维联合game；SHAP与Gaussian SURD按其标量输出定义只保留七目标，因此target分辨率不同，不能把主比较视为只替换一个信息公式的严格单因子试验。',
        '- Φ类仅保留七目标内部21个双输出pair；额外四源raw归因显式为零，再应用已有非负权重接口。CD仍保留11源，汇总指向七未来输出的非自连接，归一化分母为70。它们均不读取额外四未来模态。',
        '- SHAP仍为既有小预算拟合期直接归因；SURD仍是全联盟Gaussian密度近似。二者复用匹配的七个标量边缘缓存，未重复昂贵冻结推理。',
        '- 只把输出评分改七维、保留旧逐target先验的诊断在 `forecast_scalar_target_diagnostic/`：其Ξ提升11.83%，与主joint7先验不是同一game。该诊断没有用于从两种定义中按测试择优。',
        '- 此测试期已在旧研究中使用。当前是用户指定目标修订后的探索性比较，不能称新增独立确认。区间不包含预训练checkpoint、归因估计和调参的不确定性；12月时间块沿用历史约定。',
        '- 本轮只重算UniCM部分。SLP观测场是独立实验，目标变量不是此七模态，本轮没有更改。', '',
        '## 有效性与版本依据', '',
        '72棵SPT均有效。原生Syn容差1e-8 bit，所有联盟Ξ、候选/选中拆分及pair Syn均显式审计；无显著负值、无容差负值、无截断。order mass最大闭合误差不超过1.78e-15 bit。预测缓存元数据逐项核验，归档SHA256和数值非负/闭合审计均在结构与预测summary中。26项相关测试通过，另有独立数值核查fit-only尺度、bootstrap样本及EI读出。', '',
        '本轮重新经Zotero核验父条目 `P6UJCVG8`，标题 *Emergent hierarchical organization of causal interactions in complex systems*。正文附件 `DXGC7JEA`（19/19页）与补充 `MWIWKSVG`（28/28页）全文均已读取。依据：正文Earth pp.9–11、Methods pp.15–17式(5)–(12)；补充S1.3 pp.4–5式(S16)–(S18)、S5 pp.8–9、S10.3 p.16、S12.1.2 pp.20–22。附件分别登记于2026-10-02及2026-10-04，没有明确稿件版本号或稿件日期，不能依据元数据修改时间认定更新版本。当前可用稿仍写11目标，本候选为用户授权的方法修订，尚未同步稿件。', '',
        '正文图4打乱说明与补充材料对是否重调null存在冲突；本轮明确固定真实先验参数，并报告条件排名，未继承旧P值。', '',
        '## 复现入口', '',
        '- 结构：`scripts/recompute_unicm_seven_target_structure.py`，默认 `joint7`。',
        '- 预测：`scripts/recompute_unicm_seven_target_forecast.py`，默认 `joint7`；原scalar7诊断须显式指定 `--prior-scope scalar7` 并用单独输出目录。',
        '- 图与本说明：`scripts/plot_unicm_seven_target_candidate.py`。',
        '- 原始联盟表、72棵树与pair结果：`structure/joint7/`；标量接收端pair诊断：`structure/scalar_pair_syn_all_leads.npz`。',
        '- 仅七输出的预测/配对重采样：`forecast/evaluation_arrays.npz`；搜索分数、参数与区间：`forecast/summary.json`。', '',
        '实际重算采用既有环境及单线程BLAS，未重训或重新调用UniCM。候选文件保持独立，是否替换正式结果由用户后续决定。', '']
    content='\n'.join(lines)
    for name in ('earth_unicm_seven_target_candidate.png','seven_target_forecast_diagnostics.png',
                 'seven_target_structure_diagnostics.png','seven_target_checkpoint_trees.png'):
        content=content.replace(f']({name})',f']({OUTPUT/name})')
    (OUTPUT/'report.md').write_text(content)


def main():
    configure()
    forecast=json.loads((OUTPUT/'forecast/summary.json').read_text())
    if (forecast['target_mode_names']!=list(TARGETS) or forecast['prediction_shape']!=[96,7,24]
            or forecast['information_prior_scope']!='joint7'):
        raise ValueError('Candidate forecast target scope mismatch.')
    outputs=[diagnostics(forecast),structure_diagnostics(),checkpoint_trees(),composite(forecast)]
    report(forecast)
    print('\n'.join(str(p) for p in outputs))


if __name__=='__main__':
    main()
