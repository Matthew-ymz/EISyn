"""Paired, resumable partial-target baseline queries for the B-only follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_b_comparison import FOLDER,load_model,budget_deadline
from scripts.spt_kuramoto_group_tm import GroupTargetIntegrator
from scripts.spt_kuramoto_finite_tm import save_json,SYN_TOLERANCE
from scripts.spt import nontrivial_bipartitions
from scripts.spt_multilevel_parity_case import graph_tree
from scripts.spt_continuous_phase_benchmark import score


def mask(nodes): return sum(1<<(i-1) for i in nodes)
def nodes(a): return tuple(i+1 for i in range(9) if a>>i&1)


def scored(root,clades):
    from scripts.spt_local_hyperedge_kuramoto import SUPPORTS
    return score({'status':'exploratory_numerical','root_split':root,'clades':clades},SUPPORTS['B'])


class PersistentGroup(GroupTargetIntegrator):
    def __init__(self,model,*,folder,digest,outer,particles,scramble,deadline):
        super().__init__(model,outer=outer,particles=particles,max_particles=1024,
                         seed=20261015+1000*scramble,deadline=deadline)
        self.path=folder/f'group_n{outer}_p{particles}_r{scramble}.jsonl'
        protocol={'model_sha256':digest,'outer':outer,'particles':particles,'max_particles':1024,
            'scramble':scramble,'seed':20261015+1000*scramble,'version':3,
            'proposal':'GN if U empty; Picard4 and local Jacobian covariance otherwise, GN4 fallback above .25 noise SD; 80/15/5 normalized mixture',
            'diagonal_integration':'independent 1024 uniform pilot gate ESS<64 or maxweight>.05 selects fresh QMC1024 or annealed SMC1024; beta=(k/L)^2; N<=256 and informedP<=256 L8/M1, otherwise L16/M2; .1 uniform jump; fixed scale .7/sqrt(.3+beta*pilotFisher); compare MC prefix / independent SMC256 with same schedule; chunks32; SeedSequence(seed,A,B,200003,start,tag,P); valid completed v2 L16/M2 raw marginals reusable with provenance',
            'prior_only_particles':1024,'source':'independent uniform torus; same frozen product-VM joint'}
        if self.path.exists():
            stored=self.path.read_text().splitlines()
            previous=json.loads(stored[0])['protocol']
            if previous!=protocol and previous.get('version')==2:
                for key in ('model_sha256','outer','particles','max_particles','scramble','seed','proposal','prior_only_particles','source'):
                    if previous[key]!=protocol[key]: raise ValueError('v2 migration protocol mismatch')
                archive=self.path.with_name(self.path.name.replace('group_','group_v2_'))
                archive.write_text('\n'.join(stored)+'\n')
                with self.path.open('w') as output:
                    output.write(json.dumps({'protocol':protocol})+'\n')
                    for line in stored[1:]:
                        item=json.loads(line)
                        item['diagnostics']['reused_valid_raw_marginal_from_version']=2
                        item['diagnostics']['prior_cache_file']=archive.name
                        output.write(json.dumps(item)+'\n')
            with self.path.open() as f:
                if json.loads(next(f))['protocol']!=protocol: raise ValueError('Group cache protocol mismatch')
                for line in f:
                    item=json.loads(line)
                    self.cache[item['source'],item['target']]=(np.asarray(item['log_marginal']),item['diagnostics'])
        else:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.path.write_text(json.dumps({'protocol':protocol})+'\n')
            prefix=folder/f'group_n256_p{particles}_r{scramble}.jsonl'
            if outer==128 and prefix.exists():
                lines=prefix.read_text().splitlines(); previous=json.loads(lines[0])['protocol']
                if previous['model_sha256']!=digest or previous['seed']!=protocol['seed']:
                    raise ValueError('Outer prefix model/seed mismatch')
                for line in lines[1:]:
                    item=json.loads(line); item['log_marginal']=item['log_marginal'][:outer]
                    diag=item['diagnostics']
                    if 'prefix_log_marginal' in diag: diag['prefix_log_marginal']=diag['prefix_log_marginal'][:outer]
                    diag['reused_outer_prefix_from']=prefix.name
                    diag['diagnostic_scope']='original N256 extrema (conservative for prefix); only log-density arrays sliced to N128'
                    if 'smc_outer_count' in diag: diag['original_n256_smc_outer_count']=diag.pop('smc_outer_count')
                    if 'outer_count' in diag: diag['original_outer_count']=diag.pop('outer_count')
                    with self.path.open('a') as output: output.write(json.dumps(item)+'\n')
                    self.cache[item['source'],item['target']]=(np.asarray(item['log_marginal']),diag)
            old=folder/f'group_v1_n{outer}_p{particles}_r{scramble}.jsonl'
            if old.exists():
                with old.open() as f:
                    prior_protocol=json.loads(next(f))['protocol']
                    if prior_protocol['model_sha256']!=digest: raise ValueError('Old model mismatch')
                    for line in f:
                        item=json.loads(line); a,b=item['source'],item['target']; d=item['diagnostics']
                        # Only valid raw non-D0 marginals; all EI/Phi arithmetic
                        # is recomputed. Old diagonal evidence is never reused.
                        if (b & (511^a)) and (d.get('ess_minimum') or 0)>=10 and (d.get('maximum_normalized_weight') or 1)<.25:
                            item['diagnostics']['reused_valid_raw_marginal_from_version']=1
                            with self.path.open('a') as output: output.write(json.dumps(item)+'\n')
                            self.cache[a,b]=(np.asarray(item['log_marginal']),item['diagnostics'])

    def marginal(self,a,b):
        if (a,b) in self.cache: return self.cache[a,b]
        value,diagnostic=super().marginal(a,b)
        with self.path.open('a') as f:
            f.write(json.dumps({'source':a,'target':b,'log_marginal':value.tolist(),
                'diagnostics':diagnostic},allow_nan=False)+'\n')
        if len(self.cache)%32==0:
            print(json.dumps({'stage':'group marginals','cache':self.path.name,
                'completed':len(self.cache),'source':a,'target':b}),flush=True)
        return value,diagnostic


class Comparison:
    def __init__(self,folder,outer,particles):
        self.folder=folder; self.outer=outer; self.particles=particles
        model,_,digest=load_model()
        self.integrators=[PersistentGroup(model,folder=folder,digest=digest,outer=outer,
            particles=particles,scramble=r,deadline=budget_deadline(folder)) for r in range(4)]
        self.queries={}
        self.result={'protocol':{'case':'B','model_sha256':digest,'outer':outer,'particles':particles,
            'scrambles':4,'syn_tolerance_nats':SYN_TOLERANCE,'exact_unordered_bipartitions':True,
            'physical_datasets':1,'scope':'single-seed paired numerical comparison; no population significance'},
            'methods':{},'status':'running'}
        self.path=folder/f'group_results_n{outer}_p{particles}.json'

    def save(self):
        self.result['query_count']=len(self.queries)
        diagnostics=[d for q in self.integrators for _,d in q.cache.values()
                     if d.get('ess_minimum') is not None]
        if diagnostics:
            self.result['proposal_audit']={'minimum_ess':min(d['ess_minimum'] for d in diagnostics),
                'maximum_normalized_weight':max(d['maximum_normalized_weight'] for d in diagnostics),
                'low_ess_marginal_count':sum(d['ess_minimum']<10 for d in diagnostics),
                'large_weight_marginal_count':sum(d['maximum_normalized_weight']>=.25 for d in diagnostics)}
        save_json(self.path,self.result)

    def info(self,a,b):
        key=(a,b)
        if key not in self.queries:
            rows=[q.information(a,b) for q in self.integrators]
            values=np.array([r[0] for r in rows])
            prefix=[]; lower=[]
            for q,(_,pointwise,_) in zip(self.integrators,rows):
                numerator,nd=q.marginal(a,b); denominator,dd=q.marginal(0,b)
                # Full-target root uses a separately registered, nested stream.
                if b==511:
                    n64,_=q.full.marginal(a,min(64,self.particles))
                    d64,_=q.full.marginal(0,min(64,self.particles))
                else:
                    n64=np.asarray(nd.get('prefix_log_marginal',numerator))
                    d64=np.asarray(dd.get('prefix_log_marginal',denominator))
                prefix.append(float(np.mean(n64-d64)))
                lower.append(float(pointwise[:self.outer//2].mean()))
            self.queries[key]={'values':values,'prefix':np.array(prefix),'lower_outer':np.array(lower)}
        return self.queries[key]

    def phi(self,parent,left,right):
        whole=self.info(parent,parent)
        ll=self.info(left,left); rr=self.info(right,right)
        four=[self.info(a,b) for a in (left,right) for b in (left,right)]
        result={k:whole[k]-ll[k]-rr[k]+np.min([v[k] for v in four],axis=0)
                for k in ('values','prefix','lower_outer')}
        # Canonical point estimate: average each EI oracle value first, then
        # apply the nonlinear MMI minimum. Replicates remain uncertainty draws.
        for key,name in [('values','estimate'),('prefix','prefix_estimate'),('lower_outer','lower_outer_estimate')]:
            result[name]=float(whole[key].mean()-ll[key].mean()-rr[key].mean()+min(v[key].mean() for v in four))
        return result

    def pair_method(self):
        matrices=np.zeros((4,9,9)); prefix=np.zeros_like(matrices); lower=np.zeros_like(matrices)
        points=np.zeros((9,9)); point_prefix=np.zeros_like(points); point_lower=np.zeros_like(points)
        for i in range(9):
            for j in range(i+1,9):
                item=self.phi((1<<i)|(1<<j),1<<i,1<<j)
                for key,array in [('values',matrices),('prefix',prefix),('lower_outer',lower)]:
                    array[:,i,j]=array[:,j,i]=item[key]
                for key,array in [('estimate',points),('prefix_estimate',point_prefix),('lower_outer_estimate',point_lower)]:
                    array[i,j]=array[j,i]=item[key]
        off=matrices[:,np.triu_indices(9,1)[0],np.triu_indices(9,1)[1]]
        audit=self.audit(np.r_[off.ravel(),points[np.triu_indices(9,1)]])
        output={'matrix_nats':points.tolist(),'audit':audit,
            'aggregation':'mean each EI before MMI minimum; separate per-scramble objectives for uncertainty',
            'maximum_minimum_aggregation_difference_nats':float(abs(points-matrices.mean(0)).max()),
            'inner_change_nats':float(abs(points-point_prefix).max()),
            'outer_change_nats':float(abs(points-point_lower).max()),
            'maximum_sem_nats':float((matrices.std(0,ddof=1)/2).max())}
        if audit['significant_violation_count']:
            output.update(status='significant_phi_violation',error=
                f"PhiR minimum={audit['minimum_raw_nats']}, threshold={-SYN_TOLERANCE}, affected={audit['significant_violation_count']}")
        elif points.max()<=0:
            output['status']='no_resolved_pair_information'
        else:
            root,clades=graph_tree(points)
            output.update(status='exploratory_numerical',metrics=scored(root,clades),
                numerical_replicate_metrics=[scored(*graph_tree(m)) for m in matrices])
        self.result['methods']['pair_phi_r_hac']=output; self.save()
        print(json.dumps({'stage':'B pair PhiR','status':output['status'],
            'f1':output.get('metrics',{}).get('clade_f1')}),flush=True)

    @staticmethod
    def audit(values):
        values=np.asarray(values)
        return {'native_tolerance_nats':SYN_TOLERANCE,'minimum_raw_nats':float(values.min()),
            'tolerance_negative_count':int(np.count_nonzero((values<0)&(values>=-SYN_TOLERANCE))),
            'significant_violation_count':int(np.count_nonzero(values < -SYN_TOLERANCE))}

    def block_method(self,normalize=False):
        candidates={}; diagnostics=[]; all_raw=[]
        def visit(parent,replicate=None):
            if parent.bit_count()==1: return nodes(parent),[],None
            if parent not in candidates:
                rows=[]
                for left,right in nontrivial_bipartitions(nodes(parent)):
                    l,r=mask(left),mask(right)
                    item=self.phi(parent,l,r); all_raw.extend(item['values']); all_raw.append(item['estimate'])
                    divisor=min(len(left),len(right)) if normalize else 1
                    rows.append((item,l,r,divisor))
                candidates[parent]=rows
                print(json.dumps({'stage':'B block candidates','normalized':normalize,
                    'parent':parent,'candidates':len(rows)}),flush=True)
            rows=candidates[parent]
            def value(row):
                v=row[0]['values']; return (row[0]['estimate'] if replicate is None else float(v[replicate]))/row[3]
            ranked=sorted(rows,key=lambda row:(value(row),nodes(row[1]),nodes(row[2])))
            best=ranked[0]; _,l,r,divisor=best
            if replicate is None:
                delta=ranked[1][0]['values']/ranked[1][3]-best[0]['values']/divisor if len(ranked)>1 else None
                diagnostics.append({'parent':parent,'left':l,'right':r,'candidate_count':len(rows),
                    'gap_nats':value(ranked[1])-value(best) if delta is not None else None,
                    'gap_sem_nats':float(delta.std(ddof=1)/2) if delta is not None else None,
                    'maximum_inner_change_nats':max(abs(z[0]['estimate']-z[0]['prefix_estimate'])/z[3] for z in rows),
                    'maximum_outer_change_nats':max(abs(z[0]['estimate']-z[0]['lower_outer_estimate'])/z[3] for z in rows),
                    'objective_units':'nats/node' if normalize else 'nats',
                    'maximum_minimum_aggregation_difference_nats':max(abs(z[0]['estimate']-z[0]['values'].mean()) for z in rows)})
            _,lc,_=visit(l,replicate); _,rc,_=visit(r,replicate)
            children=[nodes(k) for k in (l,r) if k.bit_count()>1]
            return nodes(parent),children+lc+rc,(nodes(l),nodes(r))
        _,clades,root=visit(511)
        metrics=scored(root,clades)
        replicate_metrics=[]
        for r in range(4):
            _,clades,root=visit(511,r); replicate_metrics.append(scored(root,clades))
        audit=self.audit(all_raw)
        status='significant_phi_violation' if audit['significant_violation_count'] else 'exploratory_numerical'
        output={'status':status,'audit':audit,'selection_diagnostics':diagnostics}
        if status=='exploratory_numerical': output.update(metrics=metrics,numerical_replicate_metrics=replicate_metrics)
        else: output['error']=f"PhiR minimum={audit['minimum_raw_nats']}, threshold={-SYN_TOLERANCE}, affected={audit['significant_violation_count']}"
        name='normalized_block_phi_r' if normalize else 'block_phi_r'
        self.result['methods'][name]=output; self.save()
        print(json.dumps({'stage':'B '+name,'status':status,'f1':output.get('metrics',{}).get('clade_f1')}),flush=True)

    def run(self):
        started=time.monotonic()
        try:
            self.block_method(False)
            self.block_method(True)
            self.result['status']='complete'
        except (TimeoutError,ArithmeticError,np.linalg.LinAlgError) as error:
            self.result.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'numerical_failure',error=str(error))
        self.result['elapsed_seconds']=time.monotonic()-started
        self.save(); return self.result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer',type=int,default=256)
    parser.add_argument('--particles',type=int,default=256)
    args=parser.parse_args()
    result=Comparison(FOLDER,args.outer,args.particles).run()
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}))
