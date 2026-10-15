"""A/seed0 comparison, bounded by one persisted 8 h budget.

Resume exact marginal caches and query records. Pair Xi is deliberately absent.
Active scope: SPT, SURD+maximum-weight tree, raw Block PhiR.
Integration repeats measure numerical stability, not independent data evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import linkage,to_tree
from scipy.spatial.distance import squareform

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_glv_finite_tm import RealConditionalTM,save_json,check_time,WallClockDeadline
from scripts.spt_glv_group_tm import RealGroupIntegrator,VERSION,columns
from scripts.spt_local_hyperedge_kuramoto import SOURCES,SUPPORTS
from scripts.spt_continuous_phase_benchmark import score

DIRECTORY=ROOT/'results/spt_local_glv_tm/a_comparison'
TOLERANCE=.01


def encode(nodes):
    return sum(1<<(int(i)-1) for i in nodes)


def labels(mask):
    return tuple((columns(mask)+1).tolist())


def splits(mask):
    anchor=mask&-mask
    left=(mask-1)&mask
    while left:
        if left&anchor:
            yield left,mask^left
        left=(left-1)&mask


def quality(root,clades):
    result=score(dict(status='resolved',root_split=root,clades=clades),SUPPORTS['A'])
    truth={frozenset(e) for e in SUPPORTS['A'] if len(e)<9}
    predicted={frozenset(e) for e in clades}
    result['strict_clade_f1']=2*len(truth&predicted)/(len(truth)+len(predicted))
    return result


def hac(matrix):
    distance=float(np.max(matrix))-matrix
    np.fill_diagonal(distance,0.)
    z=linkage(squareform(distance,checks=True),method='average')
    tree=to_tree(z)
    clades=[]
    def visit(node):
        if node.is_leaf(): return (node.id+1,)
        found=tuple(sorted((*visit(node.left),*visit(node.right))))
        if len(found)<9: clades.append(found)
        return found
    visit(tree)
    root=(tuple(sorted(i+1 for i in tree.left.pre_order())),tuple(sorted(i+1 for i in tree.right.pre_order())))
    return quality(root,clades)


def gates(d):
    result=[]
    if d.get('ess_minimum') is not None and d['ess_minimum']<10: result.append('ESS<10')
    if d.get('maximum_normalized_weight') is not None and d['maximum_normalized_weight']>=.25: result.append('weight>=.25')
    if d.get('max_projected_step',0)>=1e-3: result.append('proposal_step>=1e-3')
    return result


class Comparison:
    def __init__(self,directory,outer=1024,scope='surd-phi'):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        budget=json.loads((self.directory/'budget.json').read_text())
        self.started=budget['started_epoch']; self.deadline=WallClockDeadline(budget['deadline_epoch'])
        self.outer=outer
        self.scope=scope
        pilot=ROOT/'results/spt_local_glv_tm/pilot.json'
        source=json.loads(pilot.read_text())['records'][0]
        self.model=RealConditionalTM.load(source['model'])
        full=json.loads((ROOT/'results/spt_local_glv_tm/full_table.json').read_text())['records'][0]
        if full['case']!='A' or not full['screen_passed']: raise ValueError('A full EI screen unavailable')
        self.full=full
        self.protocol=dict(case='A',data_seed=0,pilot_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest(),
            version=VERSION,proposal='full missing coordinates in physical order',outer=outer,particles=256,max_particles=1024,scrambles=4,
            source='iid Uniform(-pi,pi)^9; u=pi log abundance',tau=.2,units='nats',
            syn_tolerance_nats=TOLERANCE,excluded=['pair_xi_hac'],statistical_test='none: single data seed',
            adaptation='concentrated outer rows use fresh per-row nested1024/4096/16384/65536; no F1 tuning')
        if scope=='surd-phi':
            self.protocol.update(methods=['spt','surd_maximum_weight','block_phi_r'],
                excluded=['pair_xi_hac','pair_phi_r_hac','future_mi_hac','block_phi_r_normalized'])
        self.path=self.directory/f'comparison_{outer}.json'
        self.query_path=self.directory/f'queries_{scope}_{outer}_{VERSION}.jsonl'
        old_queries=self.directory/f'queries_{outer}_{VERSION}.jsonl'
        if scope=='surd-phi' and not self.query_path.exists() and old_queries.exists():
            imported=[]
            compatible=[k for k in self.protocol if k not in ('excluded','methods')]
            for line in old_queries.read_text().splitlines():
                row=json.loads(line)
                if not row['key'].startswith('info:'): continue
                if any(row['protocol'].get(k)!=self.protocol[k] for k in compatible):
                    raise ValueError('Original information queries use a different scientific protocol')
                row.update(reused_from=old_queries.name,protocol=self.protocol)
                imported.append(json.dumps(row))
            self.query_path.write_text('\n'.join(imported)+'\n')
        self.queries={}
        if self.query_path.exists():
            for line in self.query_path.read_text().splitlines():
                row=json.loads(line)
                # Method cancellations do not change an information integral.
                # Preserve its original provenance and require all numerical
                # and scientific fields to match before reusing the query.
                comparable=set(row['protocol'])|set(self.protocol)
                comparable-={'methods','excluded'}
                if any(row['protocol'].get(k)!=self.protocol.get(k) for k in comparable):
                    raise ValueError('Query protocol mismatch')
                self.queries[row['key']]=row
        self.integrators=[RealGroupIntegrator(self.model,outer=outer,max_particles=1024,proposal='full',
            seed=20261015+1000*r,deadline=self.deadline,cache_dir=self.directory/'marginals') for r in range(4)]
        self.payload=dict(protocol=self.protocol,status='running',methods={},process_id=os.getpid(),
            independent_training_datasets=1,query_count=len(self.queries))
        self.used=set()
        self.save()

    def save(self):
        self.payload.update(elapsed_wall_seconds=time.time()-self.started,query_count=len(self.queries))
        save_json(self.path,self.payload)

    def query(self,a,b,*,future=False):
        check_time(self.deadline)
        key=f'{"future" if future else "info"}:{a}:{b}'
        self.used.add(key)
        if key in self.queries: return np.asarray(self.queries[key]['values_nats'])
        if not future and a==b==511 and self.outer==1024:
            values=np.array([r['ei_nats']['511'] for r in self.full['replicates']])
            row=dict(values_nats=values.tolist(),valid=True,from_full_target_cache=True,issues=[])
        else:
            values=[]; summaries=[]
            for r,e in enumerate(self.integrators):
                value,diagnostics=e.future_information(a,b,256) if future else e.information(a,b,256)
                particles=256
                issues=sorted(set(s for d in diagnostics for s in gates(d)))
                summaries.append(dict(scramble=r,particles=particles,issues=issues,
                    ess_minimum=min((d['ess_minimum'] for d in diagnostics if d.get('ess_minimum') is not None),default=None),
                    maximum_weight=max((d['maximum_normalized_weight'] for d in diagnostics if d.get('maximum_normalized_weight') is not None),default=None),
                    maximum_projected_step=max(d.get('max_projected_step',0) for d in diagnostics),
                    refined_row_count=sum(len(d.get('refined_rows',[])) for d in diagnostics),
                    maximum_row_particles=max((r['particles'] for d in diagnostics for r in d.get('refined_rows',[])),default=256),
                    prefix_changes_nats=[d['inner_prefix_change_nats'] for d in diagnostics]))
                values.append(value)
            values=np.asarray(values); sem=float(values.std(ddof=1)/2)
            issues=sorted(set(s for d in summaries for s in d['issues']))
            if sem>=.01: issues.append('information_SEM>=.01')
            if values.mean() < -TOLERANCE: issues.append('information_mean_below_negative_tolerance')
            row=dict(values_nats=values.tolist(),mean_nats=float(values.mean()),sem_nats=sem,
                     valid=not issues,issues=issues,diagnostics=summaries)
        row.update(key=key,protocol=self.protocol)
        with self.query_path.open('a') as stream: stream.write(json.dumps(row)+'\n')
        self.queries[key]=row
        if len(self.queries)%16==0:
            self.save()
            print(json.dumps(dict(stage=self.payload.get('stage'),queries=len(self.queries),last=key,
                elapsed_seconds=time.time()-self.started,invalid_queries=sum(not r['valid'] for r in self.queries.values()))),flush=True)
        return values

    def finalize_method(self,name,mean,repeats,used,*,extra=None):
        failed={k:self.queries[k]['issues'] for k in used if not self.queries[k]['valid']}
        result=dict(status='screen_passed' if not failed else 'numerically_unresolved',
            queried_information_count=len(used),failed_queries=failed,
            metrics=mean if not failed else None,integration_replicates=repeats if not failed else None)
        if not failed:
            root={frozenset(c) for c in mean['root_split']}
            result['all_replicate_roots_match_mean']=all({frozenset(c) for c in r['root_split']}==root for r in repeats)
            result['f1_range']=[min(r['clade_f1'] for r in repeats),max(r['clade_f1'] for r in repeats)]
        if extra: result.update(extra)
        self.payload['methods'][name]=result; self.save()
        print(json.dumps(dict(stage='method_complete',method=name,status=result['status'],
            f1=mean['clade_f1'] if not failed else None,failed_queries=len(failed))),flush=True)

    def pair_methods(self):
        self.payload['stage']='pairwise'; self.save()
        phi=np.zeros((4,9,9)); future=np.zeros_like(phi)
        self.used=set()
        self_info={i:self.query(1<<i,1<<i) for i in range(9)}
        for i,j in itertools.combinations(range(9),2):
            a,b=1<<i,1<<j
            values=self.query(a|b,a|b)-self_info[i]-self_info[j]+np.minimum.reduce([
                self_info[i],self_info[j],self.query(a,b),self.query(b,a)])
            phi[:,i,j]=phi[:,j,i]=values
        used=set(self.used)
        violations=int(np.count_nonzero(phi[:,np.triu_indices(9,1)[0],np.triu_indices(9,1)[1]] < -TOLERANCE))
        if violations:
            self.payload['methods']['pair_phi_r_hac']=dict(status='significant_phi_violation',minimum_nats=float(phi.min()),
                threshold_nats=-TOLERANCE,affected_count=violations); self.save()
        else:
            self.finalize_method('pair_phi_r_hac',hac(phi.mean(0)),[hac(p) for p in phi],used,
                extra=dict(matrix_nats=phi.mean(0).tolist(),replicate_matrices_nats=phi.tolist(),
                    tolerance_negative_count=int(np.count_nonzero(phi<0))))
        self.used=set()
        for i,j in itertools.combinations(range(9),2):
            future[:,i,j]=future[:,j,i]=self.query(i,j,future=True)
        self.finalize_method('future_mi_hac',hac(future.mean(0)),[hac(p) for p in future],self.used,
            extra=dict(matrix_nats=future.mean(0).tolist(),replicate_matrices_nats=future.tolist()))

    def block(self,normalize=False,repeat=None):
        clades=[]; candidates=[]; gaps=[]
        def info(a,b):
            values=self.query(a,b)
            return float(values.mean() if repeat is None else values[repeat])
        def visit(a):
            if a.bit_count()==1: return
            aa=info(a,a); rows=[]
            for l,r in splits(a):
                ll,rr=info(l,l),info(r,r)
                raw=aa-ll-rr+min(ll,rr,info(l,r),info(r,l))
                objective=raw/min(l.bit_count(),r.bit_count()) if normalize else raw
                rows.append((objective,labels(l),labels(r),l,r,raw)); candidates.append(raw)
            rows.sort(); best=rows[0]
            gaps.append(dict(block=labels(a),selected=[best[1],best[2]],objective_nats=best[0],
                runner_up_gap_nats=rows[1][0]-best[0] if len(rows)>1 else None))
            for child in best[3:5]:
                if child.bit_count()>1: clades.append(labels(child))
                visit(child)
            return best[1:3]
        root=visit(511)
        return quality(root,clades),dict(minimum_phi_nats=min(candidates),
            significant_phi_violation_count=sum(x < -TOLERANCE for x in candidates),
            tolerance_negative_count=sum(-TOLERANCE<=x<0 for x in candidates),selected_gaps=gaps)

    def run(self):
        try:
            spt=json.loads((ROOT/'results/spt_local_glv_tm/exploratory_spt.json').read_text())['records'][0]
            spt_mean=quality(spt['mean_table']['metrics']['root_split'],spt['mean_table']['metrics']['clades'])
            spt_reps=[quality(r['metrics']['root_split'],r['metrics']['clades']) for r in spt['integration_replicates']]
            self.payload['methods']['spt']=dict(status='existing_exploratory_screen_passed',metrics=spt_mean,
                integration_replicates=spt_reps,f1_range=[min(r['clade_f1'] for r in spt_reps),max(r['clade_f1'] for r in spt_reps)],
                all_replicate_roots_match_mean=spt['all_replicate_roots_match_mean'],
                all_replicate_evaluated_clades_match_mean=spt['all_replicate_evaluated_clades_match_mean'])
            if self.scope!='surd-phi': self.pair_methods()
            if self.scope=='surd-phi':
                from scripts.spt_glv_surd import pilot as surd_pilot,refine_pilot,refine_inner_pilot,full as surd_full
                self.payload['stage']='surd_specific_information'; self.save()
                pilot_result=surd_pilot(self.directory,self.deadline)
                initial_pilot=pilot_result
                pilot_result=refine_pilot(self.directory,self.deadline,pilot_result)
                pilot_result=refine_inner_pilot(self.directory,self.deadline,initial_pilot,pilot_result)
                surd=surd_full(self.directory,self.deadline,pilot_result)
                if surd['status']=='screen_passed':
                    tree=surd['tree']; mean=quality(tree['root_split'],tree['clades'])
                    repeats=[quality(t['root_split'],t['clades']) for t in surd['replicate_trees']]
                    self.finalize_method('surd_maximum_weight',mean,repeats,set(),extra=dict(
                        maximum_information_sem_nats=surd['maximum_information_sem_nats'],
                        tree_adaptation=surd['tree_adaptation'],grid=surd['grid']))
                else:
                    self.payload['methods']['surd_maximum_weight']=dict(status=surd['status'],
                        failed_checks=surd.get('failed_checks',[]),pilot_status=pilot_result['status'],
                        metrics=None,integration_replicates=None)
                    self.save()
            for normalize,name in [(False,'block_phi_r')]:
                self.payload['stage']=name; self.save(); self.used=set()
                mean,audit=self.block(normalize)
                repeats=[]; audits=[]
                for r in range(4):
                    value,check=self.block(normalize,repeat=r); repeats.append(value); audits.append(check)
                if audit['significant_phi_violation_count'] or any(a['significant_phi_violation_count'] for a in audits):
                    self.payload['methods'][name]=dict(status='significant_phi_violation',audit=audit,replicate_audits=audits)
                    self.save()
                else:
                    self.finalize_method(name,mean,repeats,self.used,extra=dict(audit=audit,replicate_audits=audits))
            self.payload['status']='complete_with_unresolved_numerics' if any(
                r['status'] not in ('screen_passed','existing_exploratory_screen_passed') for r in self.payload['methods'].values()) else 'comparison_complete'
        except TimeoutError as error:
            self.payload.update(status='budget_exhausted',error=str(error))
        except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
            self.payload.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
        except KeyboardInterrupt:
            self.payload.update(status='cancelled',error='Interrupted; cached completed queries retained')
        if time.time()>=self.deadline.epoch:
            self.payload['status']='budget_exhausted'
        self.save()
        print(json.dumps(dict(status=self.payload['status'],query_count=len(self.queries),seconds=time.time()-self.started)),flush=True)
        return self.payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DIRECTORY)
    parser.add_argument('--outer',type=int,default=1024)
    parser.add_argument('--scope',choices=['surd-phi','legacy-five'],default='surd-phi')
    args=parser.parse_args()
    Comparison(args.directory,args.outer,args.scope).run()
