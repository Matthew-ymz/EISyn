"""Two-lane experiment orchestration, one approved eight-hour wall-clock end."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.spt_matched_common import FOLDER,read,state


def worker_alive(stage, launch):
    pid=launch.get('child_pid')
    if not pid:return False
    script='spt_matched_full_spt.py' if stage=='spt' else 'spt_matched_kuramoto_wms.py' if stage.startswith('kuramoto_wms_') else f'spt_matched_{stage}.py'
    command=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
    return f'{script} --worker' in command


def run(resume=False):
    budget=read(FOLDER/'budget.json');end=budget['deadline_epoch'];lock=threading.Lock();children=[]
    result=dict(status='running',process_id=os.getpid(),deadline_epoch=end,
                lane_primary=['existing_spt','block','kuramoto_wms_A','kuramoto_wms_C'],
                lane_secondary=['existing_natural','surd'],completed_stages=[])
    path=FOLDER/'workflow_launch.json'
    attempt=dict(process_id=os.getpid(),deadline_epoch=end,started_epoch=time.time())
    if path.exists():
        if not resume:raise ValueError('Workflow already launched; use cache-preserving --resume')
        old=read(path)
        command=subprocess.run(['ps','-p',str(old['process_id']),'-o','args='],capture_output=True,text=True).stdout
        if 'run_spt_matched_comparison.py' in command:raise RuntimeError('Existing workflow is still alive')
        attempt['previous_attempt']=old
        from scripts.spt_kuramoto_finite_tm import save_json
        save_json(path,attempt)
    else:
        with path.open('x') as stream:json.dump(attempt,stream)
    state('workflow',result)

    def record(stage):
        payload=read(FOLDER/f'{stage}.json')
        with lock:
            launch=read(FOLDER/f'{stage}_launch.json')
            status=payload.get('status')
            error=payload.get('error')
            if status=='running' and 'finished_epoch' in launch:
                status='failed';error=error or f"Worker exited {launch.get('exit_code')} with stale running checkpoint"
            result['completed_stages'].append({'stage':stage,'status':status,'error':error})
            state('workflow',result)

    def wait_existing(stage):
        # Dependency scheduling only. Hourly scientific supervision is separate.
        launch_path=FOLDER/f'{stage}_launch.json'
        while time.time()<end:
            launch=read(launch_path)
            if 'finished_epoch' in launch:
                if not worker_alive(stage,launch):
                    record(stage);return
            time.sleep(min(60,max(.01,end-time.time())))
        raise TimeoutError(f'Shared deadline while waiting for {stage}')

    def launch(stage,script,extra=()):
        if time.time()>=end:raise TimeoutError('Shared eight-hour deadline reached')
        if (FOLDER/f'{stage}_launch.json').exists():
            wait_existing(stage);return
        required=ROOT/'scripts'/script
        if not required.exists():raise FileNotFoundError(required)
        with (FOLDER/f'{stage}_launcher.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(required),*extra],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:children.append(child)
            try:child.wait(timeout=max(.01,end-time.time()+5))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError(f'{stage} parent exceeded the shared cutoff')
        record(stage)

    def primary():
        wait_existing('spt')
        launch('block','spt_matched_block.py')
        for case in 'AC':launch(f'kuramoto_wms_{case}','spt_matched_kuramoto_wms.py',('--case',case))

    def secondary():
        wait_existing('natural')
        launch('surd','spt_matched_surd.py')

    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(primary),pool.submit(secondary)]
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:errors.append(f'{type(error).__name__}: {error}')
    result.update(status='budget_exhausted' if time.time()>=end else 'ended',errors=errors,finished_epoch=time.time())
    state('workflow',result)
    print(json.dumps({'status':result['status'],'stages':result['completed_stages'],'errors':errors}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args();run(args.resume)
