"""Split the current three-G ROI Shapley maps at a fixed SPT ROI order.

For the maximal tree blocks F with |F| < N, define
    v_low(S) = sum_F v(S intersect F), v_high(S) = v(S) - v_low(S).
At the full coalition, v_high is exactly the sum of original-tree split Syn
at nodes with >= N ROI. Shapley linearity makes the two maps add back to the
original attribution. This is a fixed-tree extension to subcoalitions, not a
pure-order PID decomposition, a joining-position filter, or equal node sharing.
Local E/I Xi belongs to the low band (one ROI block). All targets stay fixed.
Reuse cached topology, density and antithetic draws; no simulation or tree search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from scipy.linalg import cholesky
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.brain_surface_plot import _draw_surface
from scripts.compute_dmf_roi_shapley import permutation_contributions, prepare_game
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_main_independent_components import ROI, SURFACE, load_manifest
from scripts.plot_dmf_schaefer100_summary import load_schaefer100_surface_map
from scripts.plot_dmf_spt_peel_groups import PAIRED_SUMMARY
from scripts.run_dmf_subject_consistency import atomic_json, digest

TOL = 1e-8  # Native nats. Raw numerical negatives are retained and counted.
WIDTH, HEIGHT = 190., 116.
STARTS = [3., 62., 121.]
# Geometry is also used to embed unaltered cortex crops in the main plate.
VIEW_TOPS = [27., 76.]
VIEW_WIDTH, VIEW_HEIGHT, VIEW_DX, VIEW_DY = 28., 17.2, 29., 18.2


def output_paths(threshold):
    name = f'roi_order_split_N{threshold}'
    return (ROOT / 'results/dmf_schaefer100/main_independent_components_v2' / name,
            ROOT / 'fig/dmf_schaefer100/main_independent_components_v2' / name)


def frontier_blocks(tree, threshold):
    """Cut on original node ROI count, with N included in the high band."""
    if not isinstance(threshold, (int, np.integer)) or threshold < 2:
        raise ValueError('ROI-order threshold must be an integer >= 2')

    def visit(node):
        members, children = node['indices'], node['children']
        if not members or len(set(members)) != len(members):
            raise ValueError('Empty or duplicate tree members')
        if children:
            groups = [set(c['indices']) for c in children]
            if len(groups) != 2 or groups[0] & groups[1] or set.union(*groups) != set(members):
                raise ValueError('Tree children must form a binary partition')
            below = [f for child in children for f in visit(child)]
        elif len(members) != 1:
            raise ValueError('Expected singleton ROI leaves')
        else:
            below = [tuple(members)]
        return [tuple(members)] if len(members) < threshold else below

    return visit(tree)


def block_marginals(conditional, scalar_logs, permutation, blocks):
    """Prefix increments of the sum of induced within-frontier Xi games (nats)."""
    n = len(scalar_logs)
    p = np.asarray(permutation, int)
    if p.shape != (n,) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError('Expected a permutation of every ROI')
    flat = [i for block in blocks for i in block]
    if sorted(flat) != list(range(n)):
        raise ValueError('Frontier must partition all ROI exactly once')
    result = np.empty(n)
    for block in blocks:
        order = p[np.isin(p, block)]
        dims = np.column_stack((order, order+n)).ravel()
        factor = cholesky(conditional[np.ix_(dims, dims)], lower=True, check_finite=False)
        increments = 2*np.log(np.diag(factor)).reshape(-1, 2).sum(1)
        result[order] = .5*(scalar_logs[order]-increments)
    return result


def tree_budget(game, tree, threshold):
    """Re-evaluate every split in the CURRENT joint model; topology stays fixed."""
    raw, old_delta, high = [], [], []
    def visit(node):
        members = node['indices']
        value = game.v(members)
        old_delta.append(value-float(node['xi_bits'])*np.log(2))
        if node['children']:
            children = node['children']
            syn = value-sum(game.v(c['indices']) for c in children)
            raw.append(syn)
            if len(members) >= threshold:
                high.append(syn)
            for child in children:
                visit(child)
    blocks = frontier_blocks(tree, threshold)
    visit(tree)
    audit = audit_nonnegative(raw, TOL)
    total = game.v(range(game.n))
    low = sum(game.v(block) for block in blocks)
    closure = abs(low+sum(high)-total)
    if closure > TOL:
        raise ArithmeticError(f'Order-band budget does not close: {closure} nats')
    return blocks, np.array([low, sum(high)]), dict(
        split_syn=audit, maximum_node_change_from_legacy_nats=float(np.abs(old_delta).max()),
        budget_closure_error_nats=float(closure), frontier_roi_blocks=[list(b) for b in blocks],
        high_order_node_count=len(high), topology_rule='Cached spectral SPT; re-evaluated, not reoptimized')


def calculate(threshold):
    load_manifest()
    source = json.loads((ROI/'summary.json').read_text())
    trees = json.loads(PAIRED_SUMMARY.read_text())
    with np.load(ROI/'roi_shapley.npz') as a:
        g, seeds, labels = a['G'].copy(), a['seeds'].copy(), a['region_labels'].copy()
        original, xi = a['paired_samples_nats'].copy(), a['xi_nats'].copy()
        sampling = json.loads(str(a['sampling_json']))
    if g.tolist() != [0., 1.3, 10.] or seeds.tolist() != [3, 4, 5]:
        raise ValueError('Unexpected main-map conditions')
    pairs = len(original)
    low_samples = np.empty_like(original)
    totals = np.zeros((2, len(g), len(seeds)))
    conditions = []
    for gi, coupling in enumerate(g):
        for si, seed in enumerate(seeds):
            record = next(q for q in source['provenance']['conditions']
                          if q['G'] == coupling and q['seed'] == seed)
            density = ROOT/record['cache']
            if digest(density) != record['cache_sha256']:
                raise ValueError(f'Changed brain-map density: {density}')
            with np.load(density) as a:
                game = CommonTargetGame(a['covariance'])
                if str(a['source_sha256']) != record['source_sha256']:
                    raise ValueError('Intervention sample identity changed')
            np.testing.assert_allclose(game.v(range(game.n)), xi[gi, si], atol=TOL, rtol=0)
            if coupling == 0:
                # The corrected mechanism factorizes across ROI: no cross-ROI
                # split, no arbitrary hierarchy, and exactly zero high-order game.
                low_samples[:, gi, si] = original[:, gi, si]
                totals[0, gi, si] = xi[gi, si]
                conditions.append(dict(G=0., seed=int(seed), density_sha256=digest(density),
                    rule='Exact independent-ROI attribution; all local E/I Xi in low band',
                    cross_roi_raw_nats=game.totals()['cross_roi_nats']))
                continue
            row = next(q for q in trees['rows'] if q['coupling_g'] == coupling and q['seed'] == seed)
            tree_path = Path(row['cache'])
            tree_record = json.loads(tree_path.read_text())
            if (tree_record['seed'] != seed or tree_record['coupling_g'] != coupling
                    or row['covariance_provenance']['paired_input_sha256'] != record['source_sha256']):
                raise ValueError('Tree and density must use the same condition and intervention samples')
            blocks, totals[:, gi, si], audit = tree_budget(game, tree_record['tree'], threshold)
            c, logs, _ = prepare_game(game.conditional[None])
            rng = np.random.default_rng(sampling['rng_seed'])
            reconstruction = 0.
            for draw in range(pairs):
                order = rng.permutation(game.n)
                full, low = [], []
                for p in (order, order[::-1]):
                    full.append(permutation_contributions(c, logs, p)[0]*np.log(2))
                    low.append(block_marginals(c[0], logs[0], p, blocks))
                low_samples[draw, gi, si] = np.mean(low, axis=0)
                reconstruction = max(reconstruction, float(np.abs(
                    np.mean(full, axis=0)-original[draw, gi, si]).max()))
            if reconstruction > TOL:
                raise ArithmeticError(f'Original Shapley draws differ: {reconstruction} nats')
            conditions.append(dict(G=float(coupling), seed=int(seed),
                density_sha256=digest(density), tree_cache=str(tree_path.relative_to(ROOT)),
                tree_sha256=digest(tree_path), original_draw_reconstruction_max_nats=reconstruction, **audit))
            print(f'G={coupling:g}, seed={seed}: low/high {totals[:,gi,si]} nats', flush=True)
    samples = np.stack([low_samples, original-low_samples], axis=1)
    nonnegative = audit_nonnegative(samples, TOL)
    errors = np.abs(samples.sum(-1)-totals[None])
    if errors.max() > TOL:
        raise ArithmeticError(f'Band Shapley efficiency failed: {errors.max()} nats')
    recombination = float(np.abs(samples.sum(1)-original).max())
    if recombination > TOL:
        raise ArithmeticError(f'ROI recombination failed: {recombination} nats')
    attribution = samples.mean(0)
    pp_samples = 100*samples/xi[None, None, ..., None]
    pp = pp_samples.mean(0)
    condition_mc_se = pp_samples.std(0, ddof=1)/np.sqrt(pairs)
    mean_mc_se = pp_samples.mean(3).std(0, ddof=1)/np.sqrt(pairs)
    split_half = float(np.abs(pp_samples[:pairs//2].mean((0, 3))-
                              pp_samples[pairs//2:].mean((0, 3))).max())
    mc = dict(independent_antithetic_pairs=pairs, rng_seed=sampling['rng_seed'],
        maximum_condition_mc_se_pp=float(condition_mc_se.max()),
        maximum_seed_mean_mc_se_pp=float(mean_mc_se.max()),
        split_half_maximum_seed_mean_difference_pp=split_half,
        converged=bool(condition_mc_se.max() <= .05 and mean_mc_se.max() <= .03 and split_half <= .10),
        rule='Original fixed draw budget; error diagnostics do not test SPT topology stability')
    mean_pp = pp.mean(2)
    share = 100*totals/xi[None]
    rows = [dict(G=float(coupling),
        low_mean_nats=float(totals[0,i].mean()), high_mean_nats=float(totals[1,i].mean()),
        low_seed_sd_nats=float(totals[0,i].std(ddof=1)), high_seed_sd_nats=float(totals[1,i].std(ddof=1)),
        low_mean_share_percent=float(share[0,i].mean()), high_mean_share_percent=float(share[1,i].mean()))
        for i,coupling in enumerate(g)]
    data = dict(G=g, seeds=seeds, region_labels=labels, roi_attribution_nats=attribution,
        share_percent=pp, mean_share_percent=mean_pp, xi_nats=xi, band_totals_nats=totals,
        condition_mc_se_pp=condition_mc_se, mean_mc_se_pp=mean_mc_se)
    summary = dict(threshold_roi_order=threshold, G=g.tolist(), seeds=seeds.tolist(),
        definition='v_low(S)=sum_F v(S intersect F); v_high(S)=v(S)-v_low(S); F=maximal SPT blocks of size < N',
        order_unit='ROI blocks (paired E/I); local E/I Xi is retained in the low band',
        interpretation='SPT path-dependent ROI order; not pure-order PID atoms or permutation joining positions',
        normalization='Display raw band allocation in nats, then equal seed mean; no Xi division or per-band rescaling. Percentages are internal diagnostics only.',
        tolerance_nats=TOL, nonnegative_audit=nonnegative,
        negative_handling='Keep raw values; only [-tolerance,0) may display as zero; no significant negative projection',
        maximum_band_efficiency_error_nats=float(errors.max()),
        maximum_ROI_recombination_error_nats=recombination, mc=mc, by_G=rows, conditions=conditions,
        source_roi_sha256=digest(ROI/'roi_shapley.npz'), source_summary_sha256=digest(ROI/'summary.json'),
        tree_summary_sha256=digest(PAIRED_SUMMARY), surface_sha256=digest(SURFACE),
        estimator='Existing affine-TM Gaussian source-moment approximation; no density refit',
        manuscript_recheck=dict(parent='P6UJCVG8', body='DXGC7JEA', supplement='MWIWKSVG',
            checked_date='2026-10-16', body_pages='19/19', supplement_pages='28/28',
            locations='Methods (5)-(12), pp.15-17; Brain/Fig.2 pp.6-7; SI S1.2-S1.3 pp.3-5, S5 pp.8-9, S12.2.1 p.23',
            version_date=None, limitation='One body and one SI; no explicit manuscript date or revision',
            extension='Fixed-tree order-band coalition games are an explicit new visualization extension; not defined by SI S17 alone'))
    return data, summary


def band_percentages(attribution_nats, totals_nats, tolerance=TOL):
    """Normalize each seed by its own band budget; zero budgets stay undefined."""
    raw = np.asarray(attribution_nats, float)
    totals = np.asarray(totals_nats, float)
    if raw.ndim < 2 or raw.shape[:-1] != totals.shape:
        raise ValueError('Band totals must match every non-ROI attribution dimension')
    audits = dict(roi=audit_nonnegative(raw, tolerance),
                  totals=audit_nonnegative(totals, tolerance))
    error = float(np.abs(raw.sum(-1)-totals).max())
    if error > tolerance:
        raise ArithmeticError(f'Band allocation does not sum to its budget: {error} nats')
    valid = totals > tolerance
    if np.any(np.abs(raw[~valid]) > tolerance):
        raise ArithmeticError('Numerically zero band has nonzero ROI allocations')
    percent = np.full_like(raw, np.nan)
    np.divide(100*raw, totals[..., None], out=percent, where=valid[..., None])
    audits.update(maximum_efficiency_error_nats=error,
                  undefined_condition_count=int((~valid).sum()),
                  zero_budget_rule=f'Budget <= {tolerance:g} nats: undefined percentage')
    return percent, valid, audits


def draw(data, summary, output, dpi=350, *, high_only=False, band_percent=False, log_scale=False):
    if log_scale and band_percent:
        raise ValueError('The logarithmic view displays absolute nats')
    raw_audit = audit_nonnegative(data['roi_attribution_nats'], TOL)
    if band_percent:
        shares, valid, audit = band_percentages(data['roi_attribution_nats'], data['band_totals_nats'])
        counts = valid.sum(2)
        values = np.full(shares.shape[:2]+shares.shape[3:], np.nan)
        np.divide(np.where(valid[..., None], shares, 0.).sum(2), counts[..., None],
                  out=values, where=counts[..., None] > 0)
        summary['percentage_audit'] = dict(**audit, valid_seed_counts=counts.tolist(),
            mean_panel_sums_percent=[
                [float(values[b,g].sum()) if counts[b,g] else None for g in range(len(data['G']))]
                for b in range(2)])
    else:
        values = data['roi_attribution_nats'].mean(2)
    # Convert only audited numerical-zero raw attributions for the sequential map.
    display = values.copy()
    negative = display < 0
    summary['display_numerical_zero_count'] = int(negative.sum())
    display[negative] = 0.
    cmap = matplotlib.colormaps['viridis']
    if log_scale:
        if not np.all(np.isfinite(display)):
            raise ValueError('Logarithmic allocation maps require finite ROI values')
        positive = display[display > 0]
        if not positive.size:
            raise ValueError('Logarithmic allocation maps require positive ROI values')
        norm = LogNorm(float(positive.min()), float(positive.max()))
        # LogNorm masks zero. A dedicated color preserves zero-valued cortex;
        # medial-wall vertices retain the separate surface background color.
        zero_color = '#667788'
        cmap = cmap.with_extremes(bad=zero_color)
        summary['render_nonnegative_audit'] = raw_audit
    else:
        norm = Normalize(0., float(np.nanmax(display)))
    width = WIDTH + 12. if log_scale else WIDTH
    height = 66. if high_only else HEIGHT
    bands = [1] if high_only else [0,1]
    tops = [VIEW_TOPS[0]] if high_only else VIEW_TOPS
    summary['plot'] = dict(vmin=norm.vmin, vmax=norm.vmax, cmap='viridis',
        normalization='log10' if log_scale else 'linear',
        unit='percent' if band_percent else 'nats',
        values=('Normalize each seed by its own low/high band total, then equal seed mean'
                if band_percent else 'Equal mean of unnormalized per-seed ROI allocations'),
        width_mm=width, height_mm=height, starts_mm=STARTS, view_tops_mm=tops,
        view_width_mm=VIEW_WIDTH, view_height_mm=VIEW_HEIGHT, view_dx_mm=VIEW_DX, view_dy_mm=VIEW_DY,
        limits_policy=('One zero-based percentage scale across all G and both bands'
                       if band_percent else 'One zero-based nats scale across all G and both bands; high-only view retains the same limits'))
    if log_scale:
        summary['plot'].update(
            limits_policy='One shared logarithmic scale, actual positive ROI-mean minimum to maximum across all G and both bands',
            zero_policy='Zero remains zero; LogNorm masks it and the dedicated zero color is shown separately, without a pseudocount',
            zero_color=zero_color, zero_roi_mean_count=int(np.count_nonzero(display == 0)))
    threshold = summary['threshold_roi_order']
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':7, 'pdf.fonttype':42, 'savefig.facecolor':'white'}):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        def text(x,y,s,**kw):
            return fig.text(x/width,1-y/height,s,va='center',**kw)
        title = 'High-order ROI allocation' if high_only else 'ROI allocation by SPT order'
        units = 'within-band %' if band_percent else 'absolute nats'
        if log_scale:
            units += ' · log color scale'
        text(3,4,title+f' · {units} · mean across 3 seeds',fontsize=8,weight='bold')
        for gi, coupling in enumerate(data['G']):
            text(STARTS[gi]+28.5,11,rf'$G={coupling:g}$',ha='center',fontsize=9)
        for band, top in zip(bands,tops):
            heading = f'Low order: < {threshold} ROI' if band == 0 else f'High order: ≥ {threshold} ROI'
            text(3,top-9,heading,fontsize=7.6,weight='bold')
            for gi, x in enumerate(STARTS):
                row = summary['by_G'][gi]
                prefix = 'low' if band == 0 else 'high'
                note = f"Total: {row[prefix+'_mean_nats']:.3f} ± {row[prefix+'_seed_sd_nats']:.3f} nats"
                if band_percent:
                    note = ('N/A: high-order total = 0' if not counts[band,gi]
                            else f'Within-band sum: 100% · {counts[band,gi]} seeds')
                text(x+28.5,top-3,note,ha='center',fontsize=6.7)
                lm,rm,lv,rv = load_schaefer100_surface_map(SURFACE,data['region_labels'],display[band,gi])
                for vi,(mesh,vertices,azim) in enumerate([(lm,lv,180.),(rm,rv,0.),(lm,lv,0.),(rm,rv,180.)]):
                    y=top+(vi//2)*VIEW_DY
                    # Square 3D axes preserve the full mesh. Their projected
                    # cortex fits the centered 28 x 17.2 mm display slot.
                    ax=fig.add_axes([(x+(vi%2)*VIEW_DX)/width,
                                     1-(y+(VIEW_HEIGHT+VIEW_WIDTH)/2)/height,
                                     VIEW_WIDTH/width,VIEW_WIDTH/height],projection='3d')
                    _draw_surface(ax,mesh,vertices,cmap=cmap,norm=norm,
                                  elev=0.,azim=azim,background_color='#D7D7D7',zoom=1.40)
                    for collection in ax.collections:
                        collection.set_rasterized(True)
        cb_top,cb_height = (29.,32.) if high_only else (32.,64.)
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                       cax=fig.add_axes([179/width,1-(cb_top+cb_height)/height,1.6/width,cb_height/height]))
        if log_scale:
            decades = 10. ** np.arange(np.floor(np.log10(norm.vmin)), np.ceil(np.log10(norm.vmax))+1)
            ticks = [norm.vmin, *decades[(decades > norm.vmin*2.5) & (decades < norm.vmax/2.5)], norm.vmax]
            cb.set_ticks(ticks)
            cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, pos: f'{value:.3g}'))
            cb.minorticks_off()
            cb.set_label('ROI allocation (nats; log scale)',fontsize=7,rotation=270,labelpad=8)
            summary['plot']['ticks_nats'] = [float(t) for t in ticks]
            zero_top = cb_top + cb_height + 4.
            fig.add_artist(matplotlib.patches.Rectangle(
                (179/width,1-(zero_top+2)/height),2.8/width,2/height,
                transform=fig.transFigure,facecolor=zero_color,edgecolor='none'))
            text(183,zero_top+1,'0 nats',fontsize=6)
        else:
            cb.set_ticks(np.linspace(0.,norm.vmax,5))
            cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f' if band_percent else '%.3f'))
            cb.set_label('ROI share within band (%)' if band_percent else 'ROI allocation (nats)',
                         fontsize=7,rotation=270,labelpad=8 if band_percent else 5)
        cb.ax.tick_params(labelsize=6,length=2,pad=1.5)
        cb.outline.set_linewidth(.5)
        # A fixed external strip reserves all guide space, outside all mesh axes.
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        guide=cb.ax.get_tightbbox(renderer)
        if (band_percent or log_scale) and any(cb.ax.yaxis.label.get_window_extent(renderer).overlaps(
                tick.get_window_extent(renderer)) for tick in cb.ax.get_yticklabels()):
            raise ValueError('Colorbar label overlaps its tick labels')
        if (guide.x0 < 0 or guide.y0 < 0 or guide.x1 > fig.bbox.width or guide.y1 > fig.bbox.height
                or any(guide.overlaps(a.get_window_extent(renderer)) for a in fig.axes if a is not cb.ax)):
            raise ValueError('Order-band colorbar clips or overlaps a cortical view')
        output.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ['.png','.pdf']:
            fig.savefig(output.with_suffix(suffix),dpi=dpi,bbox_inches=None)
        plt.close(fig)


def draw_cached_view(threshold, dpi, *, log_scale=False):
    """Add a standalone display conversion without recomputing allocations."""
    cache, base = output_paths(threshold)
    source = json.loads((cache/'summary.json').read_text())
    if source['threshold_roi_order'] != threshold:
        raise ValueError('Cached ROI-order threshold does not match the request')
    for key, path in [('source_roi_sha256', ROI/'roi_shapley.npz'),
                      ('source_summary_sha256', ROI/'summary.json'),
                      ('tree_summary_sha256', PAIRED_SUMMARY), ('surface_sha256', SURFACE)]:
        if source[key] != digest(path):
            raise ValueError(f'Changed order-band source: {path}')
    with np.load(cache/'roi_order_attribution.npz') as archive:
        data = {key: archive[key].copy() for key in archive.files}
    summary = dict(threshold_roi_order=threshold, G=data['G'].tolist(), seeds=data['seeds'].tolist(),
        by_G=source['by_G'], tolerance_nats=TOL,
        normalization=('Equal mean of raw per-seed nats; logarithmic color mapping only' if log_scale else
                       '100 * per-seed ROI band allocation / same-seed band total; equal mean across valid seeds'),
        source_cache_sha256=digest(cache/'roi_order_attribution.npz'),
        source_summary_sha256=digest(cache/'summary.json'),
        manuscript_recheck=source['manuscript_recheck'],
        uncertainty='Display conversion only; existing MC diagnostics in overall-Xi percentage points are not within-band precision estimates')
    if log_scale:
        summary['manuscript_recheck'] = dict(source['manuscript_recheck'],
            checked_date='2026-10-16',
            locations='Methods (7)-(12), pp.16-17; SI S1.3, pp.4-5, (S16)-(S19)',
            scope='Requeried Zotero parent and all attachments; reread relevant full text; display conversion only')
    else:
        summary['undefined_policy'] = 'Zero band budget has no percentage distribution; display gray cortex and N/A'
    output = base.with_name(base.name+('_log_nats' if log_scale else '_band_percent'))
    draw(data, summary, output, dpi, band_percent=not log_scale, log_scale=log_scale)
    summary['figure_sha256'] = digest(output.with_suffix('.png'))
    atomic_json(cache/('summary_log_nats.json' if log_scale else 'summary_band_percent.json'), summary)
    print(json.dumps(dict(figure=str(output.with_suffix('.png')),
                          audit=summary.get('percentage_audit',summary.get('render_nonnegative_audit')),
                          plot=summary['plot'])), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold',type=int,default=50)
    parser.add_argument('--dpi',type=int,default=350)
    parser.add_argument('--view',choices=['absolute','absolute-log','band-percent'],default='absolute',
                        help='absolute-log and band-percent reuse cached order-band allocations without recomputation')
    args=parser.parse_args()
    if not 2 <= args.threshold <= 101:
        parser.error('Use an ROI threshold between 2 and 101')
    if args.view in ('absolute-log','band-percent'):
        draw_cached_view(args.threshold, args.dpi, log_scale=args.view == 'absolute-log')
        return
    with threadpool_limits(limits=1):
        data,summary=calculate(args.threshold)
        cache,output=output_paths(args.threshold)
        draw(data,summary,output,args.dpi)
        high_summary = json.loads(json.dumps(summary))
        high_output = output.with_name(output.name+'_high_nats')
        draw(data,high_summary,high_output,args.dpi,high_only=True)
        cache.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache/'roi_order_attribution.npz',**data)
        summary['figure_sha256']=digest(output.with_suffix('.png'))
        summary['high_only_plot']=high_summary['plot']
        summary['high_only_figure_sha256']=digest(high_output.with_suffix('.png'))
        atomic_json(cache/'summary.json',summary)
        print(json.dumps(dict(by_G=summary['by_G'],mc=summary['mc'],figure=str(output.with_suffix('.png')))),flush=True)


if __name__ == '__main__':
    main()
