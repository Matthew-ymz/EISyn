"""One prediction-selected feature amendment, inside the original absolute budget.

No SPT score had been computed when this amendment was registered. All original
36-feature failures remain intact. A is a fit control; only B_2 can attain the
declared gap because all other discovery configurations have a baseline F1=1.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_kuramoto_six_topology_search as run
from scripts.kuramoto_six_topology_core import configuration,fit_mlp,save,check,dataset,channel_diagnostics

BASE=ROOT/'results/kuramoto_six_topology_search'
LOGBASE=ROOT/'docs/log/kuramoto_six_topology_search'


def configure(expanded=False):
    arm='generic_fourier_x4' if expanded else 'generic_fourier'
    run.FOLDER=BASE/arm;run.FOLDER.mkdir(exist_ok=True)
    run.LOG=LOGBASE/arm;run.LOG.mkdir(exist_ok=True)
    split=(24576,4096,4096) if expanded else (6144,1024,1024)
    run.fit_mlp=lambda x,y,seed,deadline:fit_mlp(x,y,seed,deadline,feature_kind='universal_fourier',split=split)
    if expanded:
        run.dataset=lambda cfg,seed:dataset(cfg,seed,blocks=4)
        run.channel_diagnostics=lambda model,x,y,cfg,seed:channel_diagnostics(model,x,y,cfg,seed,split=split)
    run.budget=lambda:json.loads((BASE/'budget.json').read_text())
    shutil.copyfile(BASE/'integration_protocol.json',run.FOLDER/'integration_protocol.json')


def worker(expanded=False):
    configure(expanded);deadline=run.budget()['deadline_epoch']
    summaries=[]
    for cfg in (configuration('A',.5),configuration('B',2)):
        check(deadline)
        folder=run.FOLDER/f"{cfg['id']}_seed0";folder.mkdir(exist_ok=True)
        baseline=BASE/f"{cfg['id']}_seed0"/'baselines_raw.json'
        if not expanded and not (folder/'baselines_raw.json').exists():shutil.copyfile(baseline,folder/'baselines_raw.json')
        summaries.append(run.screen(cfg,0,deadline))
        save(run.FOLDER/'screen_summary.json',summaries)
    eligible=[s for s in summaries if s['id']=='B_2' and s['diagnostics']['fit_eligible']]
    results=[]
    for s in eligible:
        cfg=configuration('B',2)
        r=run.evaluate(cfg,0,deadline);results.append(dict(id='B_2',result=r))
    save(run.FOLDER/'discovery_results.json',results)
    qualified=[r for r in results if r['result']['meets_search_target']]
    if qualified:
        save(run.FOLDER/'frozen_selection.json',qualified[0]);validations=[]
        for seed in (1,2,3):
            check(deadline);cfg=configuration('B',2)
            screen=run.screen(cfg,seed,deadline)
            if screen['diagnostics']['fit_eligible']:
                result=run.evaluate(cfg,seed,deadline)
            else:result=dict(status='failed_model_fit',diagnostics=screen['diagnostics'])
            validations.append(dict(seed=seed,result=result));save(run.FOLDER/'validation_results.json',validations)
    run.boundary('complete',screened=len(summaries),evaluated=len(results),qualified=len(qualified))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--expanded',action='store_true');args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=2):worker(args.expanded)
        return
    budget=json.loads((BASE/'budget.json').read_text());remaining=budget['deadline_epoch']-time.time()
    if remaining<=0:raise TimeoutError('Original absolute budget exhausted')
    if (BASE/'exit.json').exists():
        old=json.loads((BASE/'exit.json').read_text())
        (BASE/'exit.json').replace(BASE/f"exit_{old.get('arm','initial_screen')}.json")
    log=(LOGBASE/'run.log').open('a')
    arguments=[sys.executable,__file__,'--worker']+(['--expanded'] if args.expanded else [])
    process=subprocess.Popen(arguments,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    arm='generic_fourier_x4' if args.expanded else 'generic_fourier'
    save(BASE/'status.json',dict(phase='running',arm=arm,updated_epoch=time.time()))
    save(BASE/'process.json',dict(parent_pid=os.getpid(),worker_pid=process.pid,log=str(LOGBASE/'run.log'),deadline_epoch=budget['deadline_epoch'],arm=arm))
    try:code=process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        code=124
    save(BASE/'exit.json',dict(code=code,arm=arm,finished_epoch=time.time(),elapsed_from_budget_start=time.time()-budget['started_epoch']))
    log.close()
    if code:sys.exit(code)


if __name__=='__main__':main()
