"""Approved bounded A/B/C nonlinear-TM finite-data experiment.

Train all cases before marginalization, so a difficult integration does not
consume the entire budget before other fits. Reuse saved models with --resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import (SPTConfig, all_nonempty_subsets,
    build_spt_from_ei_table, flatten_nodes, nontrivial_bipartitions)
from scripts.spt_continuous_hierarchy_pilot import audit_table
from scripts.spt_continuous_phase_benchmark import SOURCES, score
from scripts.spt_finite_data_tm import sample_dataset
from scripts.spt_spline_tm import (ConditionalSplineTM, MarginalIntegrator,
    SYN_TOLERANCE, batched_log_prob, check_deadline, fit_spline, tensor)


def digest(x, y):
    return hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()


def mask(nodes):
    return sum(1 << (i-1) for i in nodes)


def named(values):
    return {','.join(str(i+1) for i in range(9) if m >> i & 1): value
            for m, value in values.items()}


def unnamed(values):
    return {mask(tuple(int(i) for i in key.split(','))): value for key,value in values.items()}


def build_checked_tree(values, supports):
    ei = {tuple(i+1 for i in range(9) if m >> i & 1): value
          for m, value in values.items()}
    audit = audit_table(ei, SYN_TOLERANCE)
    if audit['significant_violation_count']:
        return {'status': 'invalid_syn', 'audit': audit,
            'nonnegativity_error': f"Syn violation: minimum={audit['minimum_syn_nats']} nats; "
                f"threshold={-SYN_TOLERANCE}; affected_count={audit['significant_violation_count']}"}
    tree = build_spt_from_ei_table(SOURCES, ei,
                                  config=SPTConfig(syn_tolerance=SYN_TOLERANCE))
    nodes = flatten_nodes(tree.root)
    result = score({'root_split': tuple(c.sources for c in tree.root.children),
        'clades': [n.sources for n in nodes if 1<len(n.sources)<9],
        'status': 'resolved'}, supports)
    audit['closure_error_nats'] = tree.closure_error
    return {'status': 'resolved', 'audit': audit, 'spt': result,
            'xi_nats': tree.root.xi_value, 'root_syn_nats': tree.root.syn_value,
            'nodes': [{'sources': n.sources, 'children': [c.sources for c in n.children],
                       'xi_nats': n.xi_value, 'syn_nats': n.syn_value} for n in nodes]}


def heldout(model, source, target, truth):
    n = len(source)
    fitted = batched_log_prob(model, tensor(source.reshape(n,-1)), tensor(target.reshape(n,-1)))
    difference = truth-fitted
    return {'samples': n, 'nll_nats': float(-fitted.mean()),
            'oracle_nll_nats': float(-truth.mean()),
            'excess_nll_nats': float(difference.mean()),
            'excess_nll_iid_sem_nats': float(difference.std(ddof=1)/np.sqrt(n))}


def final_diagnostics(record, model, truth, supports):
    """Audit stored estimates; these controls never replace the reported tree."""
    levels=record.get('integration',{}).get('levels',[])
    if levels:
        values=unnamed(levels[-1]['ei_nats'])
        ei={tuple(i+1 for i in range(9) if m>>i&1):v for m,v in values.items()}
        candidates=[ei[s]-ei[l]-ei[r] for s in ei if len(s)>1
                    for l,r in nontrivial_bipartitions(s) if l in ei and r in ei]
        record['probe_syn_audit']={'tolerance_nats':SYN_TOLERANCE,
            'available_partition_count':len(candidates),
            'minimum_syn_nats':min(candidates),
            'tolerance_negative_count':sum(-SYN_TOLERANCE<=v<0 for v in candidates),
            'significant_violation_count':sum(v<-SYN_TOLERANCE for v in candidates),
            'scope':'fixed diagnostic queries only; not the complete 9330-partition audit',
            'clipping':False}
        record['probe_xi_nats']=values[511]-sum(values[1<<i] for i in range(9))
        baseline=json.loads((ROOT/'results/spt_finite_data_tm/summary.json').read_text())
        old=next(r for r in baseline['records'] if r['case']==record['case']
                 and r['n_train']==16384 and r['seed']==0)
        errors=[old['ei_nats'][s]-record['truth_ei_nats'][s] for s in levels[-1]['ei_nats']]
        record['affine_baseline']['probe_ei_rmse_nats']=float(np.sqrt(np.mean(np.square(errors))))
        record['paired_test_nll_gain_nats']=(record['affine_baseline']['heldout']['excess_nll_nats']
                                            -record['heldout']['excess_nll_nats'])
    if 'replicate_ei_nats' in record:
        record['replicate_spt_results']=[build_checked_tree(unnamed(table),supports)
                                          for table in record['replicate_ei_nats']]
    if record.get('status')=='invalid_syn':
        audit=record['audit']
        record['nonnegativity_error']=(f"Syn violation: minimum={audit['minimum_syn_nats']} nats; "
            f"threshold={-SYN_TOLERANCE}; affected_count={audit['significant_violation_count']}")
    if model.domain=='real' and record['fit']['selected_identity_spline']:
        from scripts.spt_finite_data_tm import FittedConditionalTM
        coefficient=model.coefficient.double().numpy()
        chol=model.cholesky.double().numpy()
        gaussian=FittedConditionalTM(coefficient,model.intercept.double().numpy(),
                                    chol@chol.T,16384,'real',record['node_width'])
        exact={mask(s):gaussian.info(s) for s in all_nonempty_subsets(SOURCES)}
        audit=build_checked_tree(exact,supports)
        audit['interpretation']='same selected identity-spline model; analytic integral control, NOT replacement for finite-RQMC tree'
        if 'ei_nats' in record:
            estimated=unnamed(record['ei_nats'])
            audit['finite_outer_ei_rmse_nats']=float(np.sqrt(np.mean([
                (estimated[m]-v)**2 for m,v in exact.items()])))
        audit['truth_ei_rmse_nats']=float(np.sqrt(np.mean([
            (v-truth[tuple(i+1 for i in range(9) if m>>i&1)])**2 for m,v in exact.items()])))
        record['identity_map_analytic_control']=audit


def integrate(model, width, truth, supports, record, save, deadline, outer=256,
              particles=(256,512,1024), complete_table=True, density_device='cpu'):
    subsets = all_nonempty_subsets(SOURCES)
    all_masks = [mask(s) for s in subsets]
    probes = list(dict.fromkeys([511, *(1 << i for i in range(9)), 31,480,63,448]))
    integrators = [MarginalIntegrator(model, node_width=width, outer=outer,
        max_particles=max(particles), seed=20261014+1009*i,
        deadline=deadline, density_device=density_device) for i in range(2)]
    existing={level['particles']:level for level in record.get('integration',{}).get('levels',[])}
    diagnostics, previous, selected = [], None, None
    for p in particles:
        if p in existing:
            level=existing[p]
            estimates=[unnamed(v) for v in level['replicate_ei_nats']]
            diagnostics.append(level)
            previous=estimates
            if level['maximum_nested_change_nats'] is not None and level['maximum_nested_change_nats']<=.01:
                selected=p
                break
            continue
        estimates, details = [], []
        for integrator in integrators:
            values, diagnostic = integrator.information(probes, p)
            estimates.append(values); details.append(diagnostic)
        change = (None if previous is None else max(
            abs(estimates[i][s]-previous[i][s]) for i in range(2) for s in probes))
        mean = {s: float(np.mean([v[s] for v in estimates])) for s in probes}
        errors = [mean[s]-truth[tuple(i+1 for i in range(9) if s >> i & 1)] for s in probes]
        diagnostics.append({'particles': p, 'ei_nats': named(mean),
            'replicate_ei_nats': [named(v) for v in estimates],
            'maximum_nested_change_nats': change,
            'maximum_scramble_difference_nats': max(abs(estimates[0][s]-estimates[1][s]) for s in probes),
            'probe_ei_rmse_nats': float(np.sqrt(np.mean(np.square(errors)))),
            'density_diagnostics': details})
        record.update(status='integration_diagnostic', integration={'outer_per_scramble': outer,
            'scrambles': 2, 'levels': diagnostics, 'convergence_threshold_nats': .01,
            'proposal': 'learned Gaussian posterior with exact spline/base weights' if model.domain=='real'
                        else 'registered uniform intervention',
            'density_device': density_device if model.domain=='phase' else 'cpu',
            'log_mixture_correction': 'none', 'clipping': False})
        save()
        print({'case': record['case'], 'particles': p,
               'max_nested_change_nats': change, 'probe_rmse': diagnostics[-1]['probe_ei_rmse_nats']}, flush=True)
        if change is not None and change <= .01:
            selected = p
            break
        previous = estimates
    if selected is None:
        record['status'] = 'integration_not_certified'
        record['integration']['reason'] = 'Highest two inner precisions differ by more than .01 nats; no tree'
        save()
        return
    record['integration']['selected_particles'] = selected
    if not complete_table:
        record['status']='integration_certified_pending_table';save();return
    record['integration']['table_density_device']=density_device if model.domain=='phase' else 'cpu'
    tables = ([unnamed(v) for v in record['partial_ei_nats']] if 'partial_ei_nats' in record
              else [{s: float(v) for s,v in previous_values.items()} for previous_values in estimates])
    pending = [s for s in all_masks if s not in tables[0]]
    for start in range(0, len(pending), 16):
        check_deadline(deadline)
        for i, integrator in enumerate(integrators):
            values, _ = integrator.information(pending[start:start+16], selected)
            tables[i].update(values)
        record['partial_ei_nats'] = [named(v) for v in tables]
        record['integration']['completed_source_subsets'] = len(tables[0])
        save()
    mean = {s: float(np.mean([v[s] for v in tables])) for s in all_masks}
    record.pop('partial_ei_nats', None)
    record['ei_nats'] = named(mean)
    record['replicate_ei_nats'] = [named(v) for v in tables]
    record['replicate_syn_audits'] = [audit_table(
        {tuple(i+1 for i in range(9) if m >> i & 1): v for m,v in table.items()}, SYN_TOLERANCE)
        for table in tables]
    record['replicate_spt_results']=[build_checked_tree(table,supports) for table in tables]
    record.update(build_checked_tree(mean, supports))
    errors = [mean[mask(s)]-truth[s] for s in subsets]
    record['error'] = {'ei_rmse_nats': float(np.sqrt(np.mean(np.square(errors)))),
                       'ei_max_abs_nats': float(np.max(np.abs(errors)))}
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/spt_finite_data_spline_tm')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--budget-seconds', type=int, default=2400)
    parser.add_argument('--cases', nargs='+', choices=list('ABC'), default=list('ABC'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--density-device', choices=('cpu','mps','auto'), default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic(); deadline = started+args.budget_seconds
    density_device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.density_device=='auto' else args.density_device
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output/'summary.json'
    if summary_path.exists() and not args.resume:
        raise FileExistsError('Existing result; use --resume to reuse fitted models')
    result = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else {
        'status': 'running', 'protocol': {'seed':0, 'train_samples':16384,
            'validation_samples':4096,'test_samples':8192,'validation_seed':101,
            'dataset_generation_max_train':65536,'variant':0,'labels':'identity',
            'domain':{'A':'real','B':'phase','C':'phase'},'horizon':1,
            'target':'complete future state','source_prior':'registered independent product',
            'model':'nonlinear conditional monotone autoregressive rational-quadratic spline TM',
            'hidden_layers':2,'hidden_units':128,'spline_bins':8,'maximum_epochs':args.epochs,
            'optimizer':'Adam, lr=5e-4, weight_decay=1e-6, gradient norm bound=10',
            'selection':'validation NLL including identity initial spline; no F1/truth selection',
            'real_preconditioner':'sample-fitted dense conditional affine TM, no planted blocks',
            'phase_context':'all source first/second Fourier harmonics; periodic earlier-output history',
            'phase_support':'bounded monotone spline; tied endpoint derivatives',
            'ei':'model-mediated MI, model-generated outer samples, one joint and shared denominator',
            'gaussian_proposal':'real domain only; exact nonlinear/affine likelihood-ratio correction',
            'ei_units':'nats','syn_tolerance_nats':SYN_TOLERANCE,'clipping':False,
            'budget_seconds':args.budget_seconds,'cases':args.cases,
            'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','supplement':'MWIWKSVG',
                'checked':'2026-10-14','version':'sole available pair, no explicit manuscript version/date',
                'locations':'Methods Eqs(10)-(12); SI S1.2 Eqs(S7)-(S15), S5.1'}}, 'records':[]}
    prior_elapsed = result.get('elapsed_seconds', 0.)
    deadline = started + max(0., args.budget_seconds-prior_elapsed)
    def save():
        result['elapsed_seconds'] = prior_elapsed+time.monotonic()-started
        temporary = summary_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(summary_path)
    save()
    affine_data = json.loads((ROOT/'results/spt_finite_data_tm/summary.json').read_text())
    records = {r['case']:r for r in result['records']}
    try:
        # First train all three cases; only then spend time on EI integration.
        for case in args.cases:
            check_deadline(deadline)
            if case in records and (args.output/f'{case}_model.pt').exists():
                continue
            x,y,tx,ty,true_log,truth,supports,domain = sample_dataset(case,65536,8192,0)
            vx,vy,*_ = sample_dataset(case,4096,1,101)
            x,y = x[:16384],y[:16384]
            baseline = next(r for r in affine_data['records'] if r['case']==case
                            and r['seed']==0 and r['n_train']==16384)
            if digest(x,y)!=baseline['training_prefix_digest'] or digest(tx,ty)!=baseline['test_digest']:
                raise ValueError('Actual training/test pairing differs from affine comparison')
            record = {'case':case,'status':'training','domain':domain,'dimension':9*x.shape[2],
                'node_width':x.shape[2], 'n_train':16384,'seed':0,
                'training_prefix_digest':digest(x,y),'validation_digest':digest(vx,vy),
                'test_digest':digest(tx,ty),'affine_baseline':{
                    'heldout':baseline['heldout'],'error':baseline['error'],
                    'spt':baseline['methods']['spt'],'xi_nats':baseline['xi_nats']}}
            result['records'].append(record); records[case]=record; save()
            print({'case':case,'stage':'fit','dimension':record['dimension']},flush=True)
            def progress(row):
                record['latest_training_metrics']=row; save()
            model, fit = fit_spline(x,y,vx,vy,domain=domain,seed=20261014,
                epochs=args.epochs,deadline=deadline,checkpoint_callback=progress)
            record['fit']=fit
            record['heldout']=heldout(model,tx,ty,true_log)
            record['truth_ei_nats']={','.join(map(str,s)):v for s,v in truth.items()}
            record['truth_xi_nats']=truth[SOURCES]-sum(truth[(i,)] for i in SOURCES)
            record['status']='fitted_ei_pending'
            torch.save(model.state_dict(),args.output/f'{case}_model.pt')
            record['model_digest']=hashlib.sha256((args.output/f'{case}_model.pt').read_bytes()).hexdigest()
            save()
            print({'case':case,'stage':'fitted','best_epoch':fit['best_epoch'],
                   'test_excess_nll':record['heldout']['excess_nll_nats']},flush=True)
            del x,y,tx,ty,vx,vy,model
        if not args.fit_only:
            for case in args.cases:
                check_deadline(deadline)
                record=records[case]
                if record['status'] in ('resolved','invalid_syn','integration_not_certified'):
                    continue
                model=ConditionalSplineTM(record['dimension'],record['domain'])
                model.load_state_dict(torch.load(args.output/f'{case}_model.pt',weights_only=True))
                model.eval()
                _,_,_,_,_,truth,supports,_=sample_dataset(case,1,1,0)
                print({'case':case,'stage':'integrate'},flush=True)
                integrate(model,record['node_width'],truth,supports,record,save,deadline,
                          complete_table=False,density_device=density_device)
            # Obtain every case's convergence diagnosis before spending the
            # remaining shared budget on complete tables.
            for case in args.cases:
                record=records[case]
                if record['status']!='integration_certified_pending_table':
                    continue
                check_deadline(deadline)
                model=ConditionalSplineTM(record['dimension'],record['domain'])
                model.load_state_dict(torch.load(args.output/f'{case}_model.pt',weights_only=True))
                model.eval()
                _,_,_,_,_,truth,supports,_=sample_dataset(case,1,1,0)
                print({'case':case,'stage':'complete_table'},flush=True)
                integrate(model,record['node_width'],truth,supports,record,save,deadline,
                          density_device=density_device)
            for case in args.cases:
                record=records[case]
                model=ConditionalSplineTM(record['dimension'],record['domain'])
                model.load_state_dict(torch.load(args.output/f'{case}_model.pt',weights_only=True))
                model.eval()
                _,_,_,_,_,truth,supports,_=sample_dataset(case,1,1,0)
                final_diagnostics(record,model,truth,supports)
        result['status']='fits_complete' if args.fit_only else 'complete'
    except TimeoutError as error:
        result['status']='budget_exhausted';result['budget_message']=str(error)
    except (ArithmeticError, ValueError, RuntimeError) as error:
        result['status']='failed';result['error']=str(error)
        raise
    finally:
        save()
    print({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']},flush=True)


if __name__=='__main__':
    main()
