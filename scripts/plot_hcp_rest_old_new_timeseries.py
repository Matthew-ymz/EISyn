#!/usr/bin/env python3
"""Plot native paired REST parcel traces without any signal transformation."""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hcp_timeseries_abc_979 import LABELS, atomic_json, sha

OUT = ROOT / "results/hcp_timeseries_abc_replication_979/rest_diagnostic"
SUBJECTS = ("100206", "106319", "911849")
PARCELS = (1, 813, 318)  # 1-based Schaefer indices; same columns in old and new inputs.
DISPLAY = ("Visual · LH Vis 1", "Limbic · RH OFC 2", "Control · LH Par 1")
COLORS = ("#4877A0", "#CF7C43")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.loadtxt(LABELS, dtype=str)[:, 1]
    expected = ("7Networks_LH_Vis_1", "7Networks_RH_Limbic_OFC_2", "7Networks_LH_Cont_Par_1")
    if tuple(labels[np.array(PARCELS)-1]) != expected:
        raise ValueError("Parcel labels no longer match the frozen plotting selection")
    pairs, records = [], []
    for subject in SUBJECTS:
        op = ROOT / ("data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain/"
            f"sub-{subject}/sub-{subject}_hcp_s1200_rfMRI_REST1_LR_schaefer500-1000_yeo7.mat")
        npth = ROOT / f"data/timeseries/{subject}.mat"
        old = np.asarray(loadmat(op, variable_names=["Schaefer1000"])["Schaefer1000"], float)
        new = np.asarray(loadmat(npth, variable_names=["Schaefer1000_NoGSR"])["Schaefer1000_NoGSR"], float)
        if old.shape != (1200, 1000) or new.shape != old.shape or not np.isfinite([old, new]).all():
            raise ValueError(f"Unexpected input for {subject}")
        ix = np.array(PARCELS)-1
        a, b = old[:, ix], new[:, ix]
        pairs.append((a, b))
        diagnostic = []
        for j, parcel in enumerate(PARCELS):
            diagnostic.append(dict(parcel_1based=parcel, label=expected[j],
                old_mean=float(a[:, j].mean()), new_mean=float(b[:, j].mean()),
                old_sd=float(a[:, j].std(ddof=1)), new_sd=float(b[:, j].std(ddof=1)),
                old_new_correlation=float(np.corrcoef(a[:, j],b[:, j])[0,1]),
                old_lag1=float(np.corrcoef(a[:-1,j],a[1:,j])[0,1]),
                new_lag1=float(np.corrcoef(b[:-1,j],b[1:,j])[0,1])))
        records.append(dict(subject=subject, old_file=str(op.resolve()), new_file=str(npth),
            old_sha256=sha(op), new_sha256=sha(npth), parcels=diagnostic))
    # Each old/new pair shares its native full-range limits, also reused in the
    # zoom. Different subjects retain their actual baseline and amplitude.
    limits = []
    for pair in pairs:
        row=[]
        for j in range(3):
            low=min(float(pair[k][:,j].min()) for k in (0,1))
            high=max(float(pair[k][:,j].max()) for k in (0,1))
            margin=.06*(high-low)
            row.append((low-margin,high+margin))
        limits.append(row)
    style = {"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
        "font.size":9, "axes.labelsize":9, "axes.titlesize":10,
        "axes.spines.top":False, "axes.spines.right":False}
    for stop, name in ((1200,"full"),(240,"zoom")):
        with plt.rc_context(style):
            fig, axes = plt.subplots(3,3,figsize=(12.0,6.8), sharex=True,layout="constrained")
            for i, (subject, pair) in enumerate(zip(SUBJECTS,pairs)):
                for j, axis in enumerate(axes[i]):
                    for k, label in enumerate(("Old REST","New REST (NoGSR)")):
                        axis.plot(np.arange(1,stop+1),pair[k][:stop,j],color=COLORS[k],
                            lw=.7 if stop==1200 else .9,alpha=.85,label=label)
                    axis.set(xlim=(1,stop),ylim=limits[i][j])
                    if i==0:axis.set_title(DISPLAY[j],pad=7)
                    if j==0:axis.set_ylabel(f"Subject {subject}\nMAT signal value (a.u.)")
                    if i==2:axis.set_xlabel("Frame number")
                    axis.tick_params(labelsize=8)
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=2,frameon=False)
            fig.savefig(OUT/f"rest_old_new_timeseries_{name}.png",dpi=260,bbox_inches="tight",facecolor="white")
            plt.close(fig)
    atomic_json(OUT/"timeseries_plot_provenance.json",dict(
        subjects=SUBJECTS, selection="First, middle and last positions of the sorted overlapping 57-person cohort; fixed before trace inspection",
        parcels=PARCELS, parcel_selection="First visual and control atlas parcels; RH Limbic OFC 2 is a parcel implicated in the preceding PC1 diagnostic, used as a targeted illustration, not a representative population estimate",
        old_key="Schaefer1000",new_key="Schaefer1000_NoGSR",n_frames=1200,
        transformations="None. Plot MAT entries exactly, with no centering, filtering, smoothing, detrending, z-scoring, interpolation, time realignment or PCA",
        windows="All 1200 frames and fixed first 240 frames; each old/new subject-parcel pair shares a native-value axis, with identical limits in the full and zoom exports",
        time_units="Frame number, not seconds; new run identifier and TR not documented", records=records))
    report=["# 新旧REST的原始脑区时间序列对照", "",
        "固定展示重叠57人名单排序的首位、中位、末位：100206、106319、911849。列为同一Schaefer脑区：视觉网络LH Vis 1（第1列）、边缘网络RH OFC 2（第813列）、控制网络LH Par 1（第318列）。OFC 2是此前PCA诊断涉及的脑区，属于针对性示例，不用于估计总体效应。", "",
        "蓝色为旧REST，橙色为新NoGSR REST。直接绘制MAT中的数值，没有去均值、标准化、平滑、滤波或时间对齐。每个被试/脑区面板的新旧数据共用原值纵轴；其全长和放大图使用相同纵轴范围。不同被试面板的纵轴范围可不同，应读取实际刻度比较振幅。横轴为采样帧；新数据TR和REST run未记录，不能把坐标直接解释为秒或确认同一次扫描。", "",
        f"![1200帧全长]({OUT/'rest_old_new_timeseries_full.png'})", "",
        f"![前240帧放大]({OUT/'rest_old_new_timeseries_zoom.png'})", "",
        "可以观察到旧序列较大的慢变化和新序列中减弱的慢成分，OFC示例的新旧快速变化仍有对应。这与输入经过不同时间/混杂处理的解释相容；仅靠曲线不能识别具体处理步骤，也不能判断哪一版更接近神经信号。", "",
        "原文件校验值、parcel标签、原均值/标准差、时间相关和lag1诊断记录在同目录timeseries_plot_provenance.json。"]
    (OUT/"timeseries_comparison.md").write_text("\n".join(report)+"\n")
    print("Two paired REST trace figures saved")


if __name__=="__main__":main()
