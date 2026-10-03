#!/usr/bin/env python3
"""Visualize the actual maximum-Syn node, only when its order is <=10.

Cache-only: no dynamics, estimator fits, or partition searches. A blank means
the maximum node fails the user's display rule, not absence of all synergy.
The displayed set is the full winning node, not the Xi-following C10 branch.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_spt_order_distribution import (
    INPUT, SEEDS, STYLE, TOLERANCE, bin_edges, load_analysis, selected_nodes, setup_G_axis,
)
from scripts.plot_dmf_spt_max_syn import max_by_order, NATS_PER_BIT
from scripts.brain_surface_plot import _draw_surface, _face_colors
from scripts.plot_dmf_schaefer100_summary import load_schaefer100_surface_map

OUTPUT = ROOT / "fig/dmf_schaefer100/dmf_spt_peak_core_evolution"
SUMMARY = ROOT / "results/dmf_schaefer100/spt_order_distribution/peak_core_evolution_summary.json"
SURFACE = ROOT / "results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz"
CUTOFF, DEMO_SEED = 10, 4
PEAK_COLOR = "#C74328"
COLORS = dict(Vis="#6A3D9A", SomMot="#1F78B4", DorsAttn="#33A02C",
              SalVentAttn="#E31A1C", Limbic="#B15928", Cont="#FF7F00", Default="#66A61E")
NETWORK_NAMES = dict(Vis="Visual", SomMot="Somatomotor", DorsAttn="Dorsal attention",
                     SalVentAttn="Salience / ventral attention", Limbic="Limbic",
                     Cont="Frontoparietal control", Default="Default mode")


def load_winners(data, input_dir):
    """Match the existing max-by-order rule and retain actual node identity."""
    shape = (len(SEEDS), len(data["G"]))
    orders, peaks = np.zeros(shape, dtype=int), np.empty(shape)
    selected = np.zeros((*shape, 100), dtype=bool)
    nodes = [[None for _ in data["G"]] for _ in SEEDS]
    node_ties = np.zeros(shape, dtype=int)
    with np.load(input_dir / "membership.npz") as cache:
        labels = cache["labels"].astype(str)
        np.testing.assert_array_equal(cache["G"], data["G"])
        np.testing.assert_array_equal(cache["seeds"], SEEDS)
    if labels.shape != (100,) or len(set(labels)) != 100:
        raise ValueError("Expected 100 unique, ordered Schaefer100 labels")
    paths = sorted((input_dir / "shards").glob("seeds*/trees/seed*.json"))
    seen = set()
    for path in paths:
        record = json.loads(path.read_text())
        si = int(np.flatnonzero(SEEDS == record["seed"])[0])
        gi = int(np.flatnonzero(np.isclose(data["G"], record["G"], atol=1e-12, rtol=0))[0])
        if (si, gi) in seen:
            raise ValueError("Duplicate cached seed/G condition")
        seen.add((si, gi))
        candidates = selected_nodes(record["tree"])
        raw = np.asarray([n["syn_bits_raw"] for n in candidates], dtype=float)
        bad = raw < -TOLERANCE
        if bad.any():
            raise ValueError(f"Syn violation: minimum={raw.min():.12g}, threshold={-TOLERANCE:.12g}, "
                             f"affected_count={int(bad.sum())}")
        values = raw.copy()
        values[values < 0] = 0.0  # Audited tolerance-scale negatives; load_analysis records count.
        peak = float(values.max())
        tied = [n for n, v in zip(candidates, values, strict=True) if v == peak]
        winner = min(tied, key=lambda n: (len(n["indices"]), tuple(sorted(n["indices"]))))
        members = sorted(winner["indices"])
        orders[si, gi], peaks[si, gi], node_ties[si, gi] = len(members), peak, len(tied)
        if len(members) <= CUTOFF:
            selected[si, gi, members] = True
        nodes[si][gi] = dict(members=members, child_members=[sorted(c["indices"]) for c in winner["children"]])
    if len(seen) != shape[0] * shape[1]:
        raise ValueError("Incomplete winner grid")
    existing = max_by_order(data)
    np.testing.assert_array_equal(orders, existing["orders"])
    np.testing.assert_allclose(peaks, existing["peak"], rtol=0, atol=0)
    eligible = orders <= CUTOFF
    np.testing.assert_array_equal(selected.sum(axis=2), np.where(eligible, orders, 0))
    return dict(orders=orders, peaks=peaks, selected=selected, labels=labels, nodes=nodes,
                node_ties=node_ties, eligible=eligible, frequency=selected.sum(axis=0),
                active=eligible.sum(axis=0))


def peak_column(data, winners):
    """Peak of the equal-seed mean of each tree's all-order maximum Syn."""
    mean = winners["peaks"].mean(axis=0)
    index = int(np.argmax(mean))
    edges = bin_edges(data["G"])
    return dict(G=float(data["G"][index]), index=index,
                definition="Peak of the eight-seed mean of per-tree maximum Syn over all orders, before the <=10 display filter",
                mean_maximum_syn_nats=float(mean[index] * NATS_PER_BIT),
                exact_peak_tie_G=data["G"][mean == mean[index]].tolist(),
                column_edges_G=edges[index:index + 2].tolist())


def mark_peak_column(ax, gs, peak):
    """Outline the actual sampled-G cell without recoloring its data."""
    left, right = peak["column_edges_G"]
    ax.add_patch(Rectangle((left, 0), right - left, 1, transform=ax.get_xaxis_transform(),
                           facecolor="none", edgecolor=PEAK_COLOR, linewidth=1.4,
                           zorder=10, clip_on=False))
    ticks = np.sort(np.unique(np.r_[np.arange(0, 3.01, .5), peak["G"]]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" if t == peak["G"] else f"{t:.1f}" for t in ticks])
    for tick, value in zip(ax.get_xticklabels(), ticks, strict=True):
        if value == peak["G"]:
            tick.set_color(PEAK_COLOR)
            tick.set_fontweight("bold")


def plot_summary(data, winners, output):
    """Actual G spacing, all seeds, and all ROIs that ever enter the shown node."""
    roi = np.flatnonzero(winners["selected"].any(axis=(0, 1)))
    gs = data["G"]
    peak = peak_column(data, winners)
    with mpl.rc_context({**STYLE, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(11.6, 12.4), layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 8.5))
        ax = fig.add_subplot(grid[0])
        palette = ListedColormap(mpl.colormaps["YlGnBu"](np.linspace(.3, .95, 9)))
        palette.set_bad("white")
        shown = np.where(winners["eligible"], winners["orders"], np.nan)
        image = ax.pcolormesh(bin_edges(gs), np.arange(len(SEEDS) + 1) - .5, shown,
                             cmap=palette, norm=BoundaryNorm(np.arange(1.5, 11.5), 9), rasterized=True)
        ax.set(ylim=(len(SEEDS) - .5, -.5), yticks=np.arange(len(SEEDS)),
               yticklabels=[f"Seed {s}" for s in SEEDS])
        setup_G_axis(ax, gs)
        ax.set_title("a  Order of the maximum-Syn node, shown only at 2–10 ROI", loc="left", fontweight="bold", pad=25)
        ax.text(0, 1.04, "White = maximum order >10; every column uses the same eight seeds",
                transform=ax.transAxes, fontsize=8, color="#555555")
        ax.text(1, 1.04, f"Mean maximum Syn peak: G = {peak['G']:.2f}",
                transform=ax.transAxes, ha="right", fontsize=8, color=PEAK_COLOR, fontweight="bold")
        mark_peak_column(ax, gs, peak)
        cb = fig.colorbar(image, ax=ax, fraction=.023, pad=.02, ticks=[2, 4, 6, 8, 10])
        cb.set_label("Shown node order (ROI)")

        ax = fig.add_subplot(grid[1])
        palette = ListedColormap(["white", *mpl.colormaps["Blues"](np.linspace(.3, .95, 8))])
        image = ax.pcolormesh(bin_edges(gs), np.arange(len(roi) + 1) - .5,
                             winners["frequency"][:, roi].T,
                             cmap=palette, norm=BoundaryNorm(np.arange(-.5, 9.5), 9), rasterized=True)
        short = [winners["labels"][i].replace("7Networks_LH_", "LH: ").replace("7Networks_RH_", "RH: ") for i in roi]
        ax.set(ylim=(len(roi) - .5, -.5), yticks=np.arange(len(roi)), yticklabels=short)
        ax.tick_params(axis="y", length=0, labelsize=8)
        for tick, i in zip(ax.get_yticklabels(), roi, strict=True):
            tick.set_color(COLORS[winners["labels"][i].split("_")[2]])
        split = int(np.sum(roi < 50))
        ax.axhline(split - .5, lw=.75, color="#707070")
        setup_G_axis(ax, gs)
        ax.set_title("b  Members of the shown maximum-Syn node", loc="left", fontweight="bold", pad=26)
        ax.text(0, 1.012, f"{len(roi)} ROIs ever shown; each seed contributes its full winning node or no members",
                transform=ax.transAxes, fontsize=8, color="#555555")
        mark_peak_column(ax, gs, peak)
        cb = fig.colorbar(image, ax=ax, fraction=.023, pad=.02, ticks=[0, 2, 4, 6, 8])
        cb.set_label("Seeds including this ROI (of 8)")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=280, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def render_core_images(winners):
    """Render each distinct actual node once, then reuse it across G and seeds."""
    labels = winners["labels"]
    left, right, lv, rv = load_schaefer100_surface_map(SURFACE, labels, np.arange(100, dtype=float))
    colors = np.array([mpl.colors.to_rgb(COLORS[s.split("_")[2]]) for s in labels])
    unique = sorted({tuple(n["members"]) for row in winners["nodes"] for n in row if len(n["members"]) <= CUTOFF})
    images = {}
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(8, 3.6), dpi=125, facecolor="white")
        specs = [(left, lv, 180, "LH lateral"), (right, rv, 0, "RH lateral"),
                 (left, lv, 0, "LH medial"), (right, rv, 180, "RH medial")]
        cache = []
        for k, (mesh, vertex_roi, azim, title) in enumerate(specs):
            x, y = (k % 2) * .49, (1 - k // 2) * .47
            ax = fig.add_axes((x + .005, y + .015, .49, .48), projection="3d")
            _draw_surface(ax, mesh, np.full(len(vertex_roi), np.nan), cmap=mpl.colormaps["viridis"],
                          norm=Normalize(0, 1), elev=0, azim=azim, background_color="#D7D7D7", zoom=1.6)
            ax.text2D(.02, .94, title, transform=ax.transAxes, fontsize=8, color="#666666")
            raw = vertex_roi[mesh.faces]
            valid = np.isfinite(raw).all(axis=1) & np.all(raw == raw[:, :1], axis=1)
            ids = np.full(len(raw), -1, dtype=int)
            ids[valid] = raw[valid, 0].astype(int)
            background = _face_colors(mesh, np.full(len(vertex_roi), np.nan),
                                      cmap=mpl.colormaps["viridis"], norm=Normalize(0, 1), background_color="#D7D7D7")
            shade = background[:, 0] / mpl.colors.to_rgb("#D7D7D7")[0]
            cache.append((ax.collections[0], ids, background, shade))
        for number, members in enumerate(unique, 1):
            for collection, ids, background, shade in cache:
                rgba = background.copy()
                mask = np.isin(ids, members)
                rgba[mask, :3] = np.clip(colors[ids[mask]] * shade[mask, None], 0, 1)
                collection.set_facecolors(rgba)
            fig.canvas.draw()
            images[members] = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            if number % 8 == 0:
                print(f"Rendered {number}/{len(unique)} distinct nodes", flush=True)
        plt.close(fig)
    return images


def frame(data, winners, images, si, gi):
    canvas = Image.new("RGB", (1000, 660), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 17)
    g, order = data["G"][gi], winners["orders"][si, gi]
    draw.text((26, 18), f"G = {g:.2f}     Seed {SEEDS[si]}     Maximum order = {order} ROI", font=font, fill="#222222")
    draw.text((26, 57), f"Maximum node Syn = {winners['peaks'][si, gi] * NATS_PER_BIT:.4f} nats     "
              f"Display rule met in {winners['active'][gi]}/8 seeds", font=small, fill="#555555")
    if winners["eligible"][si, gi]:
        canvas.paste(images[tuple(winners["nodes"][si][gi]["members"])], (0, 89))
    else:
        draw.text((175, 285), "No node shown: maximum order exceeds 10 ROI", font=font, fill="#777777")
    # Outside the cortical view/data region.
    for k, key in enumerate(COLORS):
        column, row = k % 4, k // 4
        x, y = 27 + column * 246, 559 + row * 29
        draw.rectangle((x, y + 4, x + 14, y + 16), fill=COLORS[key])
        name = {"SalVentAttn": "Salience / ventral attn.", "Cont": "Frontoparietal control"}.get(key, NETWORK_NAMES[key])
        draw.text((x + 21, y), name, font=small, fill="#333333")
    draw.text((27, 627), "Actual cached G samples; discrete changes; atlas row order is inferred.", font=small, fill="#666666")
    return canvas


def save_animation(data, winners, images, output):
    si = int(np.flatnonzero(SEEDS == DEMO_SEED)[0])
    frames = [frame(data, winners, images, si, gi) for gi in range(len(data["G"]))]
    # One frame per sampled G. Playback time is not a dynamical time coordinate.
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=[900] * (len(frames) - 1) + [1800], loop=0, disposal=2)


def save_explorer(data, winners, images, output):
    encoded = {}
    for members, image in images.items():
        stream = BytesIO()
        image.save(stream, format="PNG")
        encoded[",".join(map(str, members))] = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    payload = dict(G=data["G"].tolist(), seeds=SEEDS.tolist(), labels=winners["labels"].tolist(),
                   orders=winners["orders"].tolist(), syn=(winners["peaks"] * NATS_PER_BIT).tolist(),
                   nodes=winners["nodes"], active=winners["active"].tolist(), images=encoded,
                   frequency=winners["frequency"].tolist(), colors=COLORS, networkNames=NETWORK_NAMES)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>最大协同核随 G 变化</title><style>
body{font:16px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#24313d;background:#f5f6f8;margin:0;padding:26px}main{max-width:1060px;margin:auto;background:white;border-radius:12px;padding:25px}h1{font-size:25px;margin:0 0 10px}p{line-height:1.65;color:#596572}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:15px 0}input{flex:1;min-width:200px}button,select{font:inherit;padding:6px 12px;border:1px solid #cad0d7;border-radius:6px;background:white}button{cursor:pointer}#metrics{display:flex;gap:20px;flex-wrap:wrap;margin:10px 0}.value{font-size:25px;font-weight:650}#brain{width:100%;display:block}#empty{height:400px;display:flex;align-items:center;justify-content:center;color:#777;background:white;text-align:center}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:13px;margin:14px 0}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:7px;border-bottom:1px solid #e5e8eb}h2{font-size:18px;margin-top:24px}.note{font-size:13px}#seedStatus{display:flex;gap:8px;flex-wrap:wrap}.chip{font-size:13px;background:#eef2f7;border-radius:5px;padding:6px 9px}.chip.off{background:#f7f7f7;color:#868b91}.nav{display:flex;gap:8px}</style>
<main><h1>最大协同核随 G 变化</h1><p>每个 seed、每个 G 独立取树中 Syn 最大的节点。仅在它包含 ≤10 个 ROI 时显示完整成员；超过 10 阶时留空。</p>
<div class="controls"><label for="seed">随机种子</label><select id="seed"></select><div class="nav"><button id="prev" aria-label="上一个 G">←</button><button id="play">播放</button><button id="next" aria-label="下一个 G">→</button></div><label for="slider">G</label><input id="slider" type="range" min="0" max="54" value="6" step="1" aria-label="全局耦合 G"><strong id="g"></strong></div>
<div id="metrics" aria-live="polite"></div><img id="brain" alt="所选 seed 和 G 下最大协同节点的左右半球外侧及内侧皮层定位"><div id="empty" hidden></div><div class="legend" id="legend"></div>
<h2>八个 seed 的最大节点阶数</h2><div id="seedStatus"></div><h2>当前节点的完整成员</h2><div id="members"></div><h2>当前 G 的跨 seed 成员频数</h2><div id="frequency"></div>
<p class="note">白屏表示未满足 ≤10 阶显示条件，不表示整个系统没有协同。颜色表示 Yeo 网络；频数的分母始终是 8，未显示的 seed 不投成员票。图中的阶数是 SPT 父集合的 ROI 数，节点值依赖分解路径。动画依次展示实际采样的 G，没有插值；播放时间不是动力学时间。复用 440 棵既有树，fsaverage5 图谱行序沿用 inferred 状态。</p></main>
<script>const D=__DATA__;const $=id=>document.getElementById(id);let timer=null;
for(const [i,s] of D.seeds.entries()){$('seed').add(new Option('Seed '+s,i));}$('seed').value='1';$('slider').max=D.G.length-1;
for(const [k,c] of Object.entries(D.colors)){const a=document.createElement('span');a.innerHTML='<span class="swatch" style="background:'+c+'"></span>'+D.networkNames[k];$('legend').append(a);}
function render(){const si=+$('seed').value,gi=+$('slider').value,order=D.orders[si][gi],show=order<=10,n=D.nodes[si][gi];$('g').textContent=D.G[gi].toFixed(2);$('metrics').innerHTML='<div>最大阶数<br><span class="value">'+order+' ROI</span></div><div>最大节点 Syn<br><span class="value">'+D.syn[si][gi].toFixed(4)+' nats</span></div><div>满足门槛的 seed<br><span class="value">'+D.active[gi]+' / 8</span></div>';
$('brain').style.display=show?'block':'none';$('empty').style.display=show?'none':'flex';if(show){$('brain').src=D.images[n.members.join(',')];$('brain').alt='G='+D.G[gi].toFixed(2)+'，Seed '+D.seeds[si]+'，'+order+' 个 ROI 的最大 Syn 节点';}else{$('brain').removeAttribute('src');$('empty').textContent='最大节点为 '+order+' 阶，按 ≤10 阶规则不显示';}
$('seedStatus').innerHTML=D.seeds.map((s,i)=>'<span class="chip '+(D.orders[i][gi]>10?'off':'')+'">Seed '+s+': '+D.orders[i][gi]+' 阶'+(D.orders[i][gi]>10?' · 不显示':'')+'</span>').join('');
if(show){$('members').innerHTML='<table><thead><tr><th>ROI</th><th>所选二分中的子组</th></tr></thead><tbody>'+n.members.map(i=>'<tr><td>'+D.labels[i].replace('7Networks_','')+'</td><td>'+(n.child_members[0].includes(i)?'A':'B')+'</td></tr>').join('')+'</tbody></table>';}else{$('members').textContent='当前条件留空。';}
const f=D.frequency[gi];const rois=f.map((v,i)=>({v,i})).filter(x=>x.v>0).sort((a,b)=>b.v-a.v||a.i-b.i);$('frequency').innerHTML=rois.length?'<table><thead><tr><th>ROI</th><th>进入被显示节点的 seed 数</th></tr></thead><tbody>'+rois.map(x=>'<tr><td>'+D.labels[x.i].replace('7Networks_','')+'</td><td>'+x.v+' / 8</td></tr>').join('')+'</tbody></table>':'八个 seed 均未满足显示条件。';}
function step(delta){$('slider').value=Math.max(0,Math.min(D.G.length-1,+$('slider').value+delta));render();}function stop(){clearInterval(timer);timer=null;$('play').textContent='播放';}
$('prev').onclick=()=>{stop();step(-1)};$('next').onclick=()=>{stop();step(1)};$('slider').oninput=()=>{stop();render()};$('seed').onchange=render;$('play').onclick=()=>{if(timer){stop();return;}if(+$('slider').value===D.G.length-1)$('slider').value=0;$('play').textContent='暂停';render();timer=setInterval(()=>{if(+$('slider').value===D.G.length-1){stop();return;}step(1)},900)};render();</script></html>'''
    output.write_text(html.replace("__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    data = load_analysis(args.input_dir)
    winners = load_winners(data, args.input_dir)
    plot_summary(data, winners, args.output.with_suffix(".png"))
    images = render_core_images(winners)
    save_animation(data, winners, images, args.output.with_suffix(".gif"))
    save_explorer(data, winners, images, args.output.with_suffix(".html"))
    states = []
    for gi, g in enumerate(data["G"]):
        groups = Counter(tuple(winners["nodes"][si][gi]["members"]) for si in range(len(SEEDS)) if winners["eligible"][si, gi])
        states.append(dict(G=float(g), shown_seed_count=int(winners["active"][gi]),
                           full_node_groups=[dict(members=list(k), seed_count=v) for k, v in groups.most_common()]))
    payload = dict(definition="Per seed/G, maximum selected-node Syn; display entire winner only when ROI order <=10",
        cutoff_roi=CUTOFF, tie_rule="Exact Syn ties: smallest ROI order, then lexicographically smallest sorted members",
        peak_column_marker=peak_column(data, winners),
        blank_semantics="Winner order exceeds 10, not zero system synergy; no added uniformity or significance threshold",
        frequency_denominator=8, demo_seed=DEMO_SEED,
        G=data["G"].tolist(), seeds=SEEDS.tolist(), labels=winners["labels"].tolist(),
        winner_order=winners["orders"].tolist(), winner_syn_nats=(winners["peaks"] * NATS_PER_BIT).tolist(),
        actual_winner_nodes=winners["nodes"], displayed_members=winners["selected"].astype(int).tolist(),
        node_tie_counts=winners["node_ties"].tolist(), displayed_seed_counts=winners["active"].tolist(),
        member_seed_frequency=winners["frequency"].tolist(), states=states,
        distinct_displayed_nodes=len(images), audit=data["audit"],
        input_tree_fingerprint_sha256=data["fingerprint"], simulation=data["simulation"], search=data["search"],
        manuscript=dict(parent="P6UJCVG8", attachment="DXGC7JEA", explicit_revision=None,
            attachment_added="2026-10-02", pdf_creation_date="2026-09-30", relevant_locations="Methods pp15–17, Eqs5–12",
            evidence="Freshly retrieved full text, 19 pages; cited supplementary appendices unavailable"),
        figures=[str(args.output.with_suffix(s)) for s in (".png", ".gif", ".html")])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(states=[s for s in states if s["G"] in (.4, .5, .6, 1.18, 1.9, 2.6, 3.)],
                         distinct_displayed_nodes=len(images), audit=data["audit"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
