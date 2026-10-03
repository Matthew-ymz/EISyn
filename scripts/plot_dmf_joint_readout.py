#!/usr/bin/env python3
"""Render and report the frozen joint-readout smoke test, including negative results."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_sections import write_report_section
BASE=ROOT/'results/dmf_schaefer100/joint_readout_pilot'
FIG=ROOT/'fig/dmf_joint_readout'
REPORT=ROOT/'docs/reports/brain.md'
COLORS=dict(factorized='#0072B2',conditional_pairwise='#E69F00',joint='#009E73')
LABELS=dict(factorized='Factorized',conditional_pairwise='Conditional pairwise',joint='Joint')


def panel(ax,letter):
    ax.text(-.13,1.03,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
    ax.spines[['top','right']].set_visible(False)


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    scores=json.loads((BASE/'scores.json').read_text())
    contract=json.loads((BASE/'contract.json').read_text())
    controls=json.loads((BASE/'analytic_controls.json').read_text())
    FIG.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':10,'axes.labelsize':10,'legend.fontsize':9,
                         'font.family':'DejaVu Sans','axes.linewidth':.7}):
        fig,axes=plt.subplots(2,2,figsize=(12.8,7.4),layout='constrained',
                             gridspec_kw={'width_ratios':[1.15,1.]})
        for row,k in enumerate((2,4)):
            records=summary['test'][str(k)]; ax=axes[row,0]; panel(ax,chr(97+row*2))
            for offset,family,marker in ((-.18,'factorized','o'),(0,'conditional_pairwise','s'),(.18,'joint','^')):
                v=[100*r['models'][family]['mode_accuracy'] for r in records]
                ax.scatter(np.arange(1,len(v)+1)+offset,v,color=COLORS[family],marker=marker,s=36,label=LABELS[family])
            ax.axhline(100*2**(-k),color='.55',ls='--',lw=.9)
            ax.set(xlabel=f'Frozen candidate (k = {k} ROI)',ylabel='Full-pattern test accuracy (%)',
                   xticks=range(1,len(records)+1),ylim=(0,50))
            ax.text(.99,.95,f'Chance = {100*2**(-k):g}%',ha='right',va='top',transform=ax.transAxes,color='.4',fontsize=9)
            if row==0:
                ax.legend(loc='lower left',bbox_to_anchor=(0,1.035),frameon=False,ncol=3,columnspacing=1.,handletextpad=.4)
            ax=axes[row,1]; panel(ax,chr(98+row*2))
            u=np.array([r['cross_roi_u'] for r in scores['candidates'][str(k)]])
            for name,key,color,marker in (('Joint − factorized','Q','#0072B2','o'),
                                         ('Joint − pairwise','joint_minus_pairwise','#E69F00','^')):
                values=np.array([r[key]['difference'] for r in records])*100
                sem=np.array([r[key]['trajectory_sem'] for r in records])*100
                ax.errorbar(u,values,yerr=sem,fmt=marker,color=color,ms=5,capsize=2,lw=.8,label=name)
            ax.axhline(0,color='.5',ls='--',lw=.9)
            ax.set(xlabel='Cross-ROI u(S) (nats)',ylabel='Accuracy difference (percentage points)',ylim=(-6.5,3.5))
            if row==0:
                ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
        fig.savefig(FIG/'dmf_readout_smoke.png',dpi=220,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.6,3.4),layout='constrained')
        for noise,color in ((0.,'#0072B2'),(.1,'#D55E00')):
            subset=[r for r in controls if r['bit_flip_probability']==noise]
            n=[r['n'] for r in subset]
            ax.plot(n,[r['xi_nats'] for r in subset],'-o',color=color,label=f'Full Xi, flip probability {noise:g}',ms=4)
            ax.plot(n,[r['pair_syn_sum_nats'] for r in subset],'--s',color=color,label=f'Pairwise sum, flip probability {noise:g}',ms=4)
        ax.set(xlabel='Number of independent binary sources',ylabel='Information (nats)',xticks=[2,3,4,6,8],ylim=(-.025,.75))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
        fig.savefig(FIG/'noisy_xor_control.png',dpi=220,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    total=summary['totals']; lines=[
        '# DMF：全脑整合量与联合扰动读取预实验',
        '',
        '**新计划的小规模预实验已执行。预算和数值检查通过；当前解码配置没有显示联合读取优势，因此尚不进入正式比较。**',
        '',
        '执行日期：2026-10-02。对应[旧实验合同](brain.md#dmf-retired-contract)中的一次小规模检验：固定 G、k＝2/4 读取烟测。',
        '',
        '## 已执行的合同',
        '',
        'Schaefer100、200 个 E/I 标量；G＝1.3，固定在 G＝1 校准的 JFIC，1 ms 积分，300 ms 全脑未来目标。围绕确定性参考状态的全系统独立均匀盒，半宽 0.02，噪声 sigma＝0.01。标签是候选 ROI 兴奋性初态偏移的真实正/负符号。此次只运行 sham，未增加电流脉冲或连接削弱。',
        '',
        f'每个 k 冻结 {contract["split_sizes"]["train"]} 条训练、{contract["split_sizes"]["validation"]} 条验证及 {contract["split_sizes"]["test"]} 条测试轨迹；每个 k 的八个候选包括四个均匀候选和四个 SC 候选，候选在查看分数前生成。三个数据划分用独立初态和噪声种子，同一划分供所有候选和方法使用。候选重叠不构成独立重复。',
        '',
        '评分复用原 G＝1.3 的 seed901 缓存：已核对 SC/JFIC 文件、模拟与估计实现的哈希，重建完全相同的源样本，并重新拟合共同 affine TM 核对协方差。所有本方法查询均边缘化同一全系统密度，目标固定为全脑未来。原生 Phi-R/WMS 保留候选对应未来，因此与共同目标比较存在支持差别。',
        '',
        '## 全脑预算与多源核对',
        '',
        '| 量 | nats |', '|---|---:|',
        f'| 全脑 EI | {total["whole_ei_nats"]:.6f} |',
        f'| 标量最细分区全脑 Ξ | {total["xi_nats"]:.6f} |',
        f'| ROI 内 E/I 整合之和 | {total["roi_local_xi_nats"]:.6f} |',
        f'| 跨 ROI 残差 | {total["cross_roi_nats"]:.6f} |',
        '',
        f'预算闭合误差 {total["closure_error_nats"]:.3g} nats。非负容差为 1e−8 nats，查询审计容差内负值 {summary["syn_audit"]["tolerance_negative_count"]} 个、显著负值 {summary["syn_audit"]["violation_count"]} 个；各候选精确 SPT 节点和 Shapley 非负审计另存原值与数量。没有裁剪 Ξ、Syn 或任务差值。全部新轨迹状态越界和异常发放率计数为零。',
        '',
        '全脑值来自该独立盒＋共同 affine TM 预实验，不能与新稿 Fig. 2 的 18.093 nats 直接等同：当前缺少新稿补充附录，尚无法完整核对其后端、支持和参考状态。该数值差别是剩余协议核对项，不是对稿件结果的复现。',
        '',
        'n＝2、3、4、6、8 的离散 XOR 和 10% bit-flip 对照均由完整分布精确计算。n≥3 时成对 Syn 总和为零，全体 Ξ 仍为正；Shapley 和 SPT 闭合。whole EI 同样能识别 XOR，不能据此声称胜过所有信息指标。SPT 阶数是所选树的层级规模。',
        '',
        '![独立二元源的无噪声和带噪 XOR 对照](../../fig/dmf_joint_readout/noisy_xor_control.png)',
        '',
        '每个 DMF 候选已计算精确 SPT 与跨 ROI Shapley（目标仍为全脑）；它们只归因该候选的残差，没有运行 100 ROI 全脑 SPT 或全脑 Shapley。',
        '',
        '## 留出任务结果',
        '',
        '解码器使用训练集标准化的 200 个未来状态和偏置。逐源、条件二阶最大熵、完整联合模型分别使用至 1、2、k 阶的正交 Walsh 标签项，各项系数是未来状态的线性函数。三者共享优化器、收敛规则及正则化候选；只依据验证集负对数似然选正则化。k＝2 时二阶与联合模型完全相同。k＝4 时参数量分别为 804、2010、3015，容量是显式方法差异，当前样本量可能使联合模型受损。二阶基线包含全部二阶标签项，但仍受限于线性未来特征，不能代表所有二阶预测模型。',
        '',
        '![各候选的绝对识别率和联合读取差](../../fig/dmf_joint_readout/dmf_readout_smoke.png)',
        '',
        '左列为最终测试的全模式准确率，虚线为机会水平；右列为联合模型减逐源／条件二阶模型的准确率差，误差条是同一测试集内配对轨迹的 1 SEM。一个训练/验证/测试划分不提供跨种子稳定性或确认性置信区间。',
        '',
        '| k | 候选 | ROI（1-based） | 联合准确率 | 逐源准确率 | 二阶准确率 | Q（百分点） | 联合−二阶（百分点） |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for k,records in summary['test'].items():
        for ci,r in enumerate(records):
            m=r['models']
            lines.append(f'| {k} | {ci+1} | '+', '.join(str(i+1) for i in r['members'])+
                f' | {100*m["joint"]["mode_accuracy"]:.2f}% | {100*m["factorized"]["mode_accuracy"]:.2f}% | {100*m["conditional_pairwise"]["mode_accuracy"]:.2f}% | {100*r["Q"]["difference"]:+.2f} | {100*r["joint_minus_pairwise"]["difference"]:+.2f} |')
    lines += ['',
        'k＝4 的全部候选 Q 均为负；只有个别候选联合模型略优于二阶，且差值与轨迹 SEM 同量级。本次没有证明 Ξ 选组或联合读取的实用优势，也不能将这种有限样本、受限解码器的结果解释为理论整合量不存在。',
        '',
        '分数选组和验证集 Q 选组均在最终测试预测前冻结。随机基线为八个候选的精确均值。',
        '',
        '此候选池中，跨 ROI u 与成对 Syn 总和在 k＝2 和 k＝4 均给出相同排序；k＝2 两者及共同目标 O 增量完全同值。k＝4 的 u 与成对和仍有数值差别（最大约 0.000503 nats），但未产生不同选组，不能宣称本次发现了二阶汇总遗漏的 DMF 组合。',
        '',
        '| k | 选择方法 | 候选 | 测试 Q（百分点） |', '|---:|---|---:|---:|']
    for k,methods in summary['selected'].items():
        for method,r in methods.items():
            q=r['Q'] if method=='random' else r['Q']['difference']
            lines.append(f'| {k} | {method} | '+('均匀期望' if method=='random' else str(r['candidate']+1))+f' | {100*q:+.2f} |')
    lines += ['',
        '## 实测成本与继续门槛',
        '',
        f'本次总耗时 {summary["elapsed_seconds"]:.2f} s；新轨迹生成合计 {sum(summary["simulation_seconds"].values()):.2f} s，共同密度拟合后的评分及候选组织核对 {summary["scoring_seconds"]:.2f} s，解码拟合合计 {summary["decoder_fit_seconds"]:.2f} s。单线程、同进程预实验计时；不是隔离、重复的效率基准，不据此外推正式效率优势。',
        '',
        '读取具有部分可检测信号，k＝4 个别候选接近机会水平。下一步先在独立开发数据上诊断学习曲线和解码表达能力，明确容量／样本限制；不得依据本次最终测试改候选、支持、时距或排名后再称为确认性结果。正式阶段需新冻结合同、独立最终测试和重复种子。',
        '',
        '全脑归因、候选搜索、G/k 扫描、逐行预算方向干预和真实脑数据迁移仍是计划后续阶段，本次未执行。当前 formal_go=False。',
        '',
        '## 方法依据与复现',
        '',
        '本轮已重新经 Zotero 查询并核对父条目 `P6UJCVG8` 的标题 *Emergent hierarchical organization of causal interactions in complex systems*，唯一附件 `DXGC7JEA`，19/19 页正文，2026-10-02 入库，无明确稿件版本号／日期且不含 S1–S4/S12 附录。已读取 Methods 式（5）—（12）、Brain 与 Discussion；正文支持共同干预、标量 Ξ、层级分解与闭合，尚不能声称完整后端一致性已验证。旧稿不替代该主稿。',
        '',
        '连续评分使用共同 affine TM 的 Gaussian 近似：先验替换为独立盒的 Gaussian 矩，残差为 Gaussian；真实任务继续使用均匀盒。这一近似保证共同密度查询一致，却不证明一般非线性高阶能力。精确离散对照无需 TM。',
        '',
        '- 执行：`/opt/anaconda3/envs/py311/bin/python scripts/run_dmf_joint_readout.py`。匹配完整合同的结果可直接复用。',
        '- 重绘：`/opt/anaconda3/envs/py311/bin/python scripts/plot_dmf_joint_readout.py`。',
        '- 验证：`/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_dmf_joint_readout.py tests/test_dmf_response_benchmark.py -q`。',
        '- 原生输出：`results/dmf_schaefer100/joint_readout_pilot/` 中 JSON 合同、评分、审计与 NPZ 轨迹／后验；不生成 CSV。',
        '']
    write_report_section(REPORT, 'dmf-joint-pilot', '\n'.join(lines))
    print(REPORT)


if __name__=='__main__':
    main()
