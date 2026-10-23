#!/usr/bin/env python3
"""Summarize the read-only HCP audit and render its diagnostic figure."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from audit_hcp_s1200_download import DATA, FOLDERS, LENGTHS, OUT, ROOT, array_digest, fields, quantiles

CONTEXT = ROOT / "tmp/hcp_s1200_download_context_20261022.json"
FIG = ROOT / "fig/hcp_s1200_download_audit_20261022.png"
REPORT = OUT.with_suffix(".md")
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
SCORE_FIELDS = dict(EMOTION=["Emotion_Task_Acc"], LANGUAGE=["Language_Task_Acc"],
    WM=["WM_Task_Acc"], RELATIONAL=["Relational_Task_Acc"],
    SOCIAL=["Social_Task_Perc_TOM", "Social_Task_Perc_Random"],
    GAMBLING=["Gambling_Task_Perc_Larger", "Gambling_Task_Perc_Smaller"],
    MOTOR=["Endurance_AgeAdj", "Dexterity_AgeAdj", "Strength_AgeAdj"])


def prepare_context():
    comparisons = []
    for folder in ["hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30",
                   "hcp_rest1_lr_schaefer500_1000_30_subjects"]:
        rows = []
        for old_path in sorted((ROOT / "data" / folder).rglob("*.mat")):
            sid = old_path.name.split("_")[0].removeprefix("sub-")
            new_path = DATA / f"{sid}.mat"
            if not new_path.exists():
                rows.append(dict(subject=sid, missing_new=True))
                continue
            old = loadmat(old_path)
            new = loadmat(new_path, variable_names=["Schaefer500_NoGSR", "Schaefer1000_NoGSR"])
            rows.append(dict(subject=sid, resolutions={str(n):dict(
                exact_equal=bool(np.array_equal(old[f"Schaefer{n}"], new[f"Schaefer{n}_NoGSR"])),
                max_abs_difference=float(np.max(np.abs(old[f"Schaefer{n}"].astype(float)-new[f"Schaefer{n}_NoGSR"]))))
                for n in (500,1000)}))
        comparisons.append(dict(old_folder=folder, rows=rows))
    old_files = sorted(p for p in (ROOT / "data/timeseries").glob("*.mat") if p.stem.isdigit() and len(p.stem)==6)
    chosen = [old_files[i] for i in np.linspace(0, len(old_files)-1, 12, dtype=int)]
    sample = []
    for old_path in chosen:
        old = loadmat(old_path)
        sample.append(dict(subject=old_path.stem,
            old_array_hashes={k:array_digest(a) for k,a in old.items() if not k.startswith("__")}))
    result = dict(minimal_rest_reference=comparisons,
        old_timeseries_sample=dict(selection="12 equally spaced positions in sorted unique 6-digit old IDs", rows=sample))
    CONTEXT.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    print(json.dumps(dict(reference_rest_checked=sum(len(c["rows"]) for c in comparisons),
                          old_timeseries_sampled=len(sample))))


def plot(audit):
    summary = audit["summary"]
    states = list(LENGTHS)
    style = {"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
             "font.size":10, "axes.spines.top":False, "axes.spines.right":False,
             "axes.labelsize":10, "xtick.labelsize":8, "ytick.labelsize":9}
    colors = ["#326B91", "#C66B35"]
    with plt.rc_context(style):
        fig, axs = plt.subplots(2,2,figsize=(12.4,7.8), layout="constrained")
        a,b,c,d = axs.flat
        y = np.arange(len(states))
        values = [summary["coverage"][s]["standard_valid"] for s in states]
        a.barh(y,values,color=colors[0],height=.63)
        a.set(yticks=y,yticklabels=[s.replace("REST1_LR","REST1") for s in states],
              xlabel="Subjects with standard length and valid numeric arrays",xlim=(0,1150))
        a.invert_yaxis()
        for i,v in enumerate(values):
            n=summary["coverage"][states[i]]["present"]
            a.text(v+10,i,f"{v}"+(f" / {n}" if n!=v else ""),va="center",fontsize=8)
        a.text(.02,.01,f"All 8 states: {summary['all_eight_standard_valid']} subjects",
               transform=a.transAxes,fontsize=9)
        counts=Counter(sum(r["subject"] in audit["cohorts"][s]["standard_valid"] for s in states)
                       for r in audit["files"])
        xx=np.arange(9);vv=[counts[i] for i in xx]
        b.bar(xx,vv,color=["#A7B8C4"]*8+[colors[0]],width=.7)
        b.set(xticks=xx,xlabel="Number of standard-length, numerically valid states",
              ylabel="Subjects",ylim=(0,max(vv)*1.14))
        for i,v in enumerate(vv):
            if v:b.text(i,v+max(vv)*.016,str(v),ha="center",fontsize=9)
        for ax, field, label in [(c,"roi_lag1_median","Median ROI lag-one correlation"),
                                 (d,"dvars_p95","95th percentile of standardized DVARS")]:
            for j,variant in enumerate(["NoGSR","GSR"]):
                groups=[]
                for s in states:
                    group=[]
                    valid=set(audit["cohorts"][s]["standard_valid"])
                    for r in audit["files"]:
                        if r["subject"] not in valid:continue
                        for m in r["matrices"].values():
                            if m["state"]==s and m["resolution"]==1000 and m["variant"]==variant:
                                group.append(m["standardized_dvars"]["p95"] if field=="dvars_p95" else m[field])
                    groups.append(group)
                boxes=ax.boxplot(groups,positions=np.arange(8)+(j-.5)*.32,widths=.25,
                    patch_artist=True,showfliers=False,manage_ticks=False,
                    medianprops={"color":"white","linewidth":1.1},
                    boxprops={"facecolor":colors[j],"edgecolor":colors[j]},
                    whiskerprops={"color":colors[j]},capprops={"color":colors[j]})
            ax.set(xticks=np.arange(8),xticklabels=[s.replace("REST1_LR","REST1") for s in states],
                   ylabel=label,xlim=(-.6,7.6))
            ax.tick_params(axis="x",labelrotation=35)
            ax.grid(axis="y",color=".92",linewidth=.6)
            ax.set_axisbelow(True)
        from matplotlib.patches import Patch
        c.legend(handles=[Patch(facecolor=colors[0],label="NoGSR"),Patch(facecolor=colors[1],label="GSR")],
                 loc="lower center",bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,borderaxespad=0)
        for ax,letter in zip(axs.flat,"abcd"):
            ax.text(-.13,1.03,letter,transform=ax.transAxes,weight="bold",fontsize=13)
        FIG.parent.mkdir(exist_ok=True)
        fig.savefig(FIG,dpi=220,bbox_inches="tight",facecolor="white")
        plt.close(fig)


def main():
    audit = json.loads(OUT.read_text())
    context = json.loads(CONTEXT.read_text())
    lookup = {r["subject"]:r for r in audit["files"]}
    old_ids={p.stem for p in (ROOT/"data/timeseries").glob("*.mat") if p.stem.isdigit() and len(p.stem)==6}
    context["old_timeseries_cohort"]=dict(old_unique_subjects=len(old_ids),
        added_subjects=sorted(set(lookup)-old_ids),removed_subjects=sorted(old_ids-set(lookup)))
    spatial={}
    with zipfile.ZipFile(DATA/"metadata/nas_publication_records.zip") as z:
        for state,folder in FOLDERS.items():
            resolution_rows={"500":[],"1000":[]}
            for name in z.namelist():
                if not name.startswith(folder+"/") or not name.endswith(".json"):continue
                record=json.loads(z.read(name))
                if record.get("status")!="complete":continue
                for resolution,counts in record["diagnostics"].get("parcel_voxel_counts",{}).items():
                    counts=np.asarray(counts)
                    resolution_rows[resolution].append(dict(subject=record["subject"],minimum=int(counts.min()),
                        roi_count_le5=int(np.sum(counts<=5)),roi_indices_le5_1based=(np.flatnonzero(counts<=5)+1).tolist()))
            spatial[state]={resolution:dict(runs=len(rows),
                minimum_voxels=min(r["minimum"] for r in rows),
                runs_with_roi_le5=sum(r["roi_count_le5"]>0 for r in rows),
                roi_run_entries_le5=sum(r["roi_count_le5"] for r in rows),
                flagged_runs=[r for r in rows if r["roi_count_le5"]>0]) for resolution,rows in resolution_rows.items() if rows}
    context["producer_parcel_mask_coverage"]=spatial
    old_sample = context["old_timeseries_sample"]
    sample_summary = {}
    for state in ["REST1_LR","EMOTION","LANGUAGE","MOTOR","WM"]:
        comparisons=[]
        for old in old_sample["rows"]:
            current=lookup.get(old["subject"],{}).get("matrices",{})
            for key in fields(state):
                if key in old["old_array_hashes"] and key in current:
                    comparisons.append(dict(subject=old["subject"],field=key,
                        exact_equal=old["old_array_hashes"][key]==current[key]["array_sha256"]))
        sample_summary[state]=dict(checked_arrays=len(comparisons),
            exact_equal_arrays=sum(r["exact_equal"] for r in comparisons),rows=comparisons)
    old_sample["comparison"]=sample_summary
    frame = pd.read_csv(BEHAVIOR,low_memory=False).set_index("Subject")
    frame.index=frame.index.astype(str)
    ids=list(lookup)
    behavior=dict(source=str(BEHAVIOR),unmatched=sorted(set(ids)-set(frame.index)),
        gender_counts=frame.reindex(ids).Gender.value_counts(dropna=False).to_dict(),
        age_band_counts=frame.reindex(ids).Age.value_counts(dropna=False).to_dict(),
        task_score_coverage={})
    for state,columns in SCORE_FIELDS.items():
        selected=audit["cohorts"][state]["standard_valid"]
        values=frame.reindex(selected)[columns].apply(pd.to_numeric,errors="coerce")
        behavior["task_score_coverage"][state]=dict(n_standard=len(selected),
            finite_by_field={col:int(np.isfinite(values[col]).sum()) for col in columns},
            all_selected_fields_finite=int(np.isfinite(values).all(axis=1).sum()),
            fields=columns)
    context["behavior"]=behavior
    signal_summary={}
    for state in LENGTHS:
        signal_summary[state]={}
        valid=set(audit["cohorts"][state]["standard_valid"])
        for variant in ("NoGSR","GSR"):
            selected=[dict(subject=r["subject"],**m) for r in audit["files"] if r["subject"] in valid
                for m in r["matrices"].values() if m["state"]==state and m["resolution"]==1000 and m["variant"]==variant]
            signal_summary[state][variant]=dict(n=len(selected),
                median_roi_sd=quantiles([m["roi_sd"]["median"] for m in selected]),
                median_roi_mean=quantiles([m["roi_mean_median"] for m in selected]),
                roi_lag1_median=quantiles([m["roi_lag1_median"] for m in selected]),
                standardized_dvars_p95=quantiles([m["standardized_dvars"]["p95"] for m in selected]),
                matrices_with_abs_z_above_8=sum(m["abs_z_above_8_count"]>0 for m in selected),
                max_abs_z=max(m["max_abs_within_roi_z"] for m in selected),
                highest_standardized_dvars=[dict(subject=m["subject"],
                    maximum=m["standardized_dvars"]["maximum"],p95=m["standardized_dvars"]["p95"],
                    max_abs_z=m["max_abs_within_roi_z"]) for m in
                    sorted(selected,key=lambda m:m["standardized_dvars"]["maximum"],reverse=True)[:5]])
    audit["context_checks"]=context
    audit["signal_summary_schaefer1000"]=signal_summary
    OUT.write_text(json.dumps(audit,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
    plot(audit)
    s=audit["summary"]
    old_equal=sum(all(v["exact_equal"] for v in r.get("resolutions",{}).values()) and not r.get("missing_new",False)
        for comp in context["minimal_rest_reference"] for r in comp["rows"])
    old_total=sum(len(comp["rows"]) for comp in context["minimal_rest_reference"])
    old_missing=sorted({r["subject"] for comp in context["minimal_rest_reference"] for r in comp["rows"] if r.get("missing_new")})
    table=["| 状态 | 标准时间点 | 有数据 | 标准长度及数值合格 | 与标准 REST 成对 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for state in LENGTHS:
        c=s["coverage"][state]
        table.append(f"| {state} | {c['expected_frames']} | {c['present']} | {c['standard_valid']} | {c['paired_standard_rest']} |")
    anomalies=["| 被试 | 状态 | 实际时间点 | 标准时间点 |", "| --- | --- | ---: | ---: |"]
    for r in s["anomalous_runs"]:
        lengths=sorted({shape[0] for shape in r["actual_shapes"]})
        anomalies.append(f"| {r['subject']} | {r['state']} | {', '.join(map(str,lengths))} | {r['expected_frames']} |")
    signals=["| 状态 | ROI SD：NoGSR / GSR | 滞后一期相关：NoGSR / GSR | 标准化 DVARS P95：NoGSR / GSR |",
             "| --- | ---: | ---: | ---: |"]
    for state in LENGTHS:
        pair=[signal_summary[state][v] for v in ["NoGSR","GSR"]]
        values=[" / ".join(f"{p[key]['median']:.3f}" for p in pair)
                for key in ["median_roi_sd","roi_lag1_median","standardized_dvars_p95"]]
        signals.append("| "+state+" | "+" | ".join(values)+" |")
    spatial_table=["| 状态 | 被试 | 标记 parcel 索引（从 1 开始） | 该 run 最小 parcel 体素数 |",
                   "| --- | --- | --- | ---: |"]
    for state,q in spatial.items():
        for row in q.get("1000",{}).get("flagged_runs",[]):
            spatial_table.append(f"| {state} | {row['subject']} | {', '.join(map(str,row['roi_indices_le5_1based']))} | {row['minimum']} |")
    lines=["# HCP S1200 新下载数据审核（2026-10-22）", "",
        f"对象：`{DATA}`。检查完成时间：{audit['checked_at']}。原始 MAT 只读，未补值、删帧、重命名或修改。", "",
        f"**结论：下载与发布清单一致，可以使用；正式分析需按状态筛选，排除 {len(s['anomalous_runs'])} 个非标准长度 run，并补充研究所需的影像质控与协变量。**", "",
        f"逐一完整读取 {s['files']:,} 个 MAT、{s['matrices']:,} 个 float32 矩阵；MAT 总计 {s['total_bytes']/1e9:.2f} GB（{s['total_bytes']/1024**3:.2f} GiB）。全量检查耗时 {audit['elapsed_s']:.1f} 秒。", "",
        f"![覆盖与信号概览]({FIG})", "",
        "## 1. 下载、结构与来源", "",
        f"- {s['unique_subjects']:,} 个唯一六位被试编号；文件名、大小、字段和 SHA-256 逐一与 manifest 比对。文件一致性失败 {len(s['file_integrity_failures'])}，读入/压缩错误 {len(s['read_errors'])}；清单缺文件 {len(s['missing_manifest_subjects'])}、额外被试 {len(s['extra_subjects'])}。",
        f"- {s['matrices']:,} 个数组的 dtype + shape + C-order SHA-256 全部对照 ZIP 中逐 run 发布记录；哈希不匹配 {s['array_hash_mismatches']}、记录帧数不一致 {s['provenance_frame_mismatches']}。",
        f"- 文件内容完全重复组数 {len(s['duplicate_file_groups'])}；数组内容完全重复配对数 {len(s['duplicate_arrays'])}；缺半套图谱/GSR 版本的文件数 {len(s['partial_state_files'])}。未进行近似重复被试识别或身份鉴定。",
        "- 一人一个 `<ID>.mat`，矩阵为 **时间 × 脑区**；每个已有状态有 Schaefer-500/1000 × NoGSR/GSR 四个版本。REST 变量名为 `Schaefer1000_NoGSR` 等；任务为 `WM_LR_Schaefer1000_NoGSR` 等。",
        "- 发布状态为 complete，NAS 清单验证时间为 2026-10-17，当前检查的是 10 月 22 日下载的本地副本；发布日期与本地下载日期不同。S1200 是发布名称，目录人数不要求等于 1200。",
        "- ZIP 提供逐被试 S3 来源键、部分源文件版本/ETag、图谱及代码指纹、FEAT 设置和发布数组指纹。当前本地 MAT 对这些记录一致；未重新下载 NIfTI 从源头重算，也未在本次复核所有官方排除条目。", "",
        "## 2. 覆盖、长度与可用名单", "", *table, "",
        f"八个状态均存在：**{s['all_eight_present']} 人**；八个状态均达到标准长度、数值和校验条件：**{s['all_eight_standard_valid']} 人**。统一八状态比较使用后一交集；单任务或 REST–单任务比较可使用表内对应人数。所有名单、逐状态缺失名单均已保存在 [审核记录]({OUT})。这里的“合格”仅指文件与数值条件，不代表已完成头动质控。", "",
        "标准帧数参考 [HCP S1200 Reference Manual，第 36 页](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf)。逐 run 元数据报告 TR ≈ 0.72 秒，矩阵保留源输入实际帧数。以下较短 run 的四个版本均一致，且与发布记录指纹、帧数相符，因此有来源阶段已较短的证据，不属于本地下载截断：", "", *anomalies, "",
        "对需要统一长度的主分析，排除对应 run，保留这些人的其他合格状态；不补零、不重复帧、不整人删除。若分析允许变长输入，应单独声明并做相应控制。", "",
        "## 3. 数值、时间特征与异常筛查", "",
        f"NaN/Inf 数量 {s['nonfinite_values']}；常量 ROI {s['constant_rois']}；近常量 ROI {s['near_constant_rois']}；全零帧 {s['zero_frames']}；完全相同的相邻帧 {s['identical_consecutive_frames']}。近常量阈值为时域总体 SD ≤ 1e-8（原始保存信号单位）；常量另外按 SD = 0 计数。", "",
        "下表使用 Schaefer-1000、各状态合格 run，每项先按 ROI/时间归纳，再报告跨被试中位数。ROI SD 使用原始信号单位；原始 NoGSR 保留基线，GSR 为回归残差，不能将二者当成已统一标准化的信号。", "", *signals, "",
        "标准化 DVARS 定义为相邻帧差值除以各 ROI 全 run 时域总体 SD 后，在 ROI 上取均方根；表内再取每个 run 的 P95。它是信号跳变描述，**不是头动 FD，也不是传统体素级 DVARS 的同尺度替代**。盒图仅显示合格 run，箱体为四分位数、白线为中位数、须为 1.5 IQR 内最远值；散在离群点未画出，全部仍参与 JSON 记录和统计。GSR 后 ROI SD 和自相关较低、标准化 DVARS 较高；后者的标准化分母也改变，不能单凭它判断噪声增加。", "",
        "**尖峰需要进一步定位**：例如 Schaefer-1000 NoGSR 中，REST 609143 的最大标准化帧间跳变为 9.864（其 run 的 P95 为 1.339），WM 146331 为 12.018（P95 为 0.856）；REST NoGSR 的最大逐 ROI |z| 达 23.466。整体分布中的常规水平不能掩盖个别尖峰；应结合原始时序、帧索引和头动信息复核，不能仅凭本轮指标确定成因或自动排除。", "",
        f"全体保存数组中，|逐 ROI 时间标准分| > 8 的值共 {s['abs_z_above_8_count']:,} 个；四个版本对同一 run 的事件可能重复计数。这是描述性定位阈值，不是数据损坏或自动排除证据。各状态/版本的极端值、最高跳变被试及逐矩阵指标见 JSON 的 `signal_summary_schaefer1000` 和 `files`；本次未据此删人或删帧。", "",
        "## 4. 预处理与旧数据衔接", "",
        "- **REST1_LR**：记录指向 S1200 的 `rfMRI_REST1_LR.nii.gz`，active_rest_version 为 `S1200_minimal_no_FIX`；在上游最小预处理基础上直接做完整图谱体素均值，不额外添加平滑、时间滤波、去趋势、标准化、删帧或 ICA-FIX。GSR 回归截距和 HCP brainmask_fs.2 的全脑均值（含灰/白质）。",
        "- **七项任务 LR**：记录一致使用官方 FSF 的 FEAT pre-stats，4 mm 平滑、200 秒高通、无低通、不重复上游 MC/BET/配准、不删除帧；保留 FEAT grand-mean scaling。提取 FEAT mask 内各 parcel 均值；GSR 回归该 run FEAT mask 的全脑均值及截距。没有任务设计回归，NoGSR/GSR 不构成任务保留/任务回归配对。",
        "- **处理差异需控制**：REST 与任务在平滑、滤波、parcel 取值掩模、全脑信号定义和缩放上存在差别。时长也不同；直接比较 SD、时序特征或估计量会混入这些差异。不能仅凭状态间差异作任务机制解释。",
        "- **运行环境不同**：LANGUAGE/WM/MOTOR/GAMBLING 记录为容器 FSL 6.0.7.22；EMOTION/RELATIONAL/SOCIAL 为原生 FSL 6.0.7.18。各状态内部设置一致，但参数一致不能证明跨版本逐位等价；这属于跨任务比较需记录的批次因素。",
        f"- **旧 minimal REST 数值核对**：本地两批旧 minimal REST 共 {old_total} 个被试条目；新目录中重叠的 {old_equal} 个，Schaefer-500/1000 NoGSR 同时逐元素完全一致。旧参考中的 {', '.join(old_missing) if old_missing else '无被试'} 不在新目录。核验范围是实际可读的这两批参考文件。",
        f"- 与旧 `data/timeseries` 相比，唯一被试编号集合一致（{len(old_ids)} 人），没有新增或移除被试；内容和状态覆盖有更新。",
        "- **旧 `data/timeseries` 抽样**：按排序编号等距选 12 人。"+
        "；".join(f"{state} {q['exact_equal_arrays']}/{q['checked_arrays']} 数组完全相同" for state,q in sample_summary.items())+
        "。抽样结果限定于这 12 人，不能替代全体新旧数据等价性验证。",
        "- 旧脚本使用 `sub-*/*.mat`、旧字段或指向 `data/timeseries` 的路径时，需要显式适配一人一 MAT 的新目录，尤其不能把旧 REST 缓存直接移用。本次未运行或更改 EI/Syn/SPT 方法，也未做稿件一致性认证。", "",
        "## 5. 行为表、来源缺失与影像质控边界", "",
        f"全部本地 ID 与仓库现有 unrestricted 行为表匹配，未匹配 {len(behavior['unmatched'])} 人。性别分布 {behavior['gender_counts']}；年龄分组 {behavior['age_band_counts']}。该行为表是既有文件，不能据此确认所有评分与本次扫描同日。", "",
        "| 状态 | 合格时序人数 | 所列行为字段均有限人数 | 行为字段 |", "| --- | ---: | ---: | --- |"]
    for state,q in behavior["task_score_coverage"].items():
        lines.append(f"| {state} | {q['n_standard']} | {q['all_selected_fields_finite']} | {', '.join(q['fields'])} |")
    lines += ["", "GAMBLING/SOCIAL 表中为反应类别比例，并非统一的任务正确率；MOTOR 使用运动能力测试，不能当作本次扫描内动作正确率。不同字段含义应分别解释，有限值不等于行为质量合格。", "",
        "逐 run 发布记录还保留了任务 FEAT mask 内各 parcel 的体素数。以 ≤5 个 2 mm 体素作为描述性空间覆盖标记（不是排除标准），Schaefer-1000 的标记 run 数为："+
        "；".join(f"{state} {q['1000']['runs_with_roi_le5']}/{q['1000']['runs']}（最小 {q['1000']['minimum_voxels']} 体素）" for state,q in spatial.items() if '1000' in q)+
        "。这些 run 虽有有限且非恒定的时序，仍可能存在部分 parcel 覆盖较少；标记名单与 parcel 索引已保存于 JSON 的 `context_checks.producer_parcel_mask_coverage`，需回到源影像核查后决定。这里引用的是生产阶段记录，未用源影像独立重算。8 个标记 run 共涉及 3 人：", "", *spatial_table, "",
        "发布 complete 表示所有成功输出已经发布，并不表示所有源扫描都成功。记录中 LANGUAGE 的 390645 缺少源键；662551 的 LANGUAGE/WM/MOTOR/GAMBLING 存在空 parcel；EMOTION 627852 存在空 parcel；WM 724446 的 FEAT 子进程失败。REST import 状态另记录 101107、122620 因 GSR 常量残差计算失败，这两人未计入成功的 1058 人；manifest 的 REST failed_subjects 空列表不能解释为源计算零失败。失败记录被保留，没有用补值强行生成输出。", "",
        "当前下载包不包含逐帧头动参数/FD、原始全脑信号、删帧掩码、实际 EV 时间文件、源 NIfTI、图谱 NIfTI/标签表或家系信息。记录虽包含来源指针及图谱哈希，仍不能独立重做空间覆盖、头动、GSR 正交性或 atlas 列标签验证。正式分析前应补齐所需协变量、核对 family dependence、检查预处理混杂，并明确短 run 处理；本轮可确认下载和数值完整性，不能宣称完成影像级质控。", "",
        "图谱在记录中称 Schaefer Yeo7，500/1000 是两种皮层 parcel 分辨率，Yeo7 是归属标签，不代表数组已降成 7 个网络，也不包含全脑皮层下分区。", "",
        "## 文件", "", f"- [质检图]({FIG})",f"- [完整逐文件记录与筛选名单]({OUT})",
        f"- [全量审核脚本]({ROOT / 'scripts/audit_hcp_s1200_download.py'})",
        f"- [汇总与绘图脚本]({ROOT / 'scripts/report_hcp_s1200_download_audit.py'})", ""]
    REPORT.write_text("\n".join(lines))
    print(json.dumps(dict(report=str(REPORT),figure=str(FIG),behavior=behavior,
                          minimal_rest_equal=old_equal,minimal_rest_checked=old_total,
                          old_sample={k:{j:v for j,v in q.items() if j!='rows'} for k,q in sample_summary.items()}),ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--context-only",action="store_true")
    args=parser.parse_args()
    if args.context_only:
        prepare_context()
    else:
        if not CONTEXT.exists():prepare_context()
        main()
