#!/usr/bin/env python3
"""User-authorized all-93 exploratory extension; reuse the frozen scientific kernel."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE, initialize, job, prepare
from scripts.run_dmf_subject_consistency import atomic_json, digest


def extension_contract(base):
    """Do not change the old hashed kernel, its caches, or its discovery record."""
    c = prepare(base)
    amended = dict(
        version='all93-exploratory-extension-v1',
        scientific_contract_sha256=digest(base / 'contract.json'),
        launcher_sha256=digest(Path(__file__)),
        authorization_date='2026-10-07',
        authorization='User explicitly requested all 93 subjects, irrespective of discovery gate; time cost accepted',
        subject_ids=c['subject_ids'], G=c['G'], seeds=c['seeds'],
        total_conditions=len(c['subject_ids']) * len(c['G']) * len(c['seeds']),
        superseded_rules=['8-subject gate as a prerequisite for expansion', '85-subject untouched candidate validation'],
        preserved_rules='All interventions, TM approximation, natural SPT, exact members, background, windows, units and tolerances',
        readouts=dict(seed_repeat=2, window_hits=[1, 3, 4],
            strength='raw node Syn strictly above the frozen order-specific G=0 reference',
            low_and_high='any repeated point and stage>=2/4, separately',
            specificity='all repeated points outside the six-point window; never infer exclusivity from endpoint absence',
            fractions='critical hits/6, endpoint hits/4, outside hits/35 per individual, then mean across 93'),
        ranking=dict(strict='critical>=4/6 subjects descending, endpoint-any leaks ascending, raw critical count descending, order and exact members',
            relaxed='critical>=1/6 subjects descending, endpoint-any leaks ascending, >=3/6 then >=4/6 counts descending, order and members',
            window_exclusive='>=4/6 and no outside-window hit count descending, strict count descending, endpoint leaks ascending',
            preferred_orders=[3,10], search_orders=[3,99], pair_reference=[2]),
        inference='All 93 subjects participate in candidate selection. Selected estimates are descriptive, not held-out confirmation; no post-selection uncorrected p values',
        manuscript=dict(parent='P6UJCVG8', attachments=['DXGC7JEA','MWIWKSVG'],
            checked_date='2026-10-07', indexed_pages=[19,28], explicit_version_or_date=None,
            ambiguity='One main text and one supplement, neither with an explicit revision/date',
            relevant_locations='Main Brain pp6-7; Methods Eqs5-12 pp15-17; Supplement S1.2 pp3-4, S5/AlgorithmS1 pp8-9, S12.2.1/EqsS82-S87 p23'))
    path = base / 'all93_extension_contract.json'
    if path.exists() and json.loads(path.read_text()) != amended:
        raise ValueError('Frozen all93 extension changed; preserve its protocol before resuming')
    if not path.exists():
        atomic_json(path, amended)
    return c, amended


def run(base, workers=4, limit=None, postprocess=True):
    c, amended = extension_contract(base)
    sha = digest(base / 'contract.json')
    with (base / 'all93_run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = []
        reused = 0
        for p in c['subject_ids']:
            for gi, g in enumerate(c['G']):
                for seed in c['seeds']:
                    path = base / 'trees' / f'{p}_G{g:.2f}_seed{seed}.json'
                    if path.exists():
                        r = json.loads(path.read_text())
                        if (r['contract_sha256'] != sha or r['subject'] != p
                                or r['G'] != g or r['seed'] != seed):
                            raise ValueError(f'Existing condition provenance mismatch: {path}')
                        reused += 1
                    else:
                        tasks.append((p, gi, seed, False))
        if limit is not None:
            tasks = tasks[:limit]
        prefix = 'all93_smoke' if limit is not None else 'all93'
        started = time.perf_counter()
        record = dict(pid=os.getpid(), workers=workers, scheduled_new_conditions=len(tasks),
            reused_conditions=reused, total_conditions=amended['total_conditions'],
            started_utc=datetime.now(timezone.utc).isoformat(), limit=limit,
            scientific_contract_sha256=sha,
            extension_contract_sha256=digest(base / 'all93_extension_contract.json'))
        atomic_json(base / (prefix + '_launch.json'), record)
        print(json.dumps(dict(event='launch', **record)), flush=True)
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize, initargs=(str(base),)) as pool:
                futures = {pool.submit(job, t): t for t in tasks}
                for f in as_completed(futures):
                    result = f.result()
                    completed += 1
                    print(json.dumps(dict(completed=completed, scheduled=len(tasks), **result)), flush=True)
            atomic_json(base / (prefix + '_simulation_complete.json'), dict(status='complete',
                new_conditions=completed, reused_conditions=reused,
                elapsed_seconds=time.perf_counter() - started, **{
                    k: record[k] for k in ['scientific_contract_sha256','extension_contract_sha256']}))
            if limit is None and postprocess:
                for script in ['analyze_dmf_critical_coalitions_all93.py',
                               'plot_dmf_critical_coalitions_all93.py',
                               'report_dmf_critical_coalitions_all93.py']:
                    print(json.dumps(dict(event='postprocess', script=script)), flush=True)
                    subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                        '--output-dir', str(base)], cwd=ROOT, check=True)
                atomic_json(base / 'all93_pipeline_complete.json', dict(status='computation_analysis_exports_complete',
                    elapsed_seconds=time.perf_counter()-started,
                    visual_review='pending agent inspection of exported figure',
                    scientific_contract_sha256=sha,
                    extension_contract_sha256=record['extension_contract_sha256']))
        except BaseException as exc:
            atomic_json(base / (prefix + '_failure.json'), dict(status='failed',
                exception=type(exc).__name__, message=str(exc), completed_new_conditions=completed,
                elapsed_seconds=time.perf_counter()-started, launch=record))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=BASE)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--no-postprocess', action='store_true')
    a = p.parse_args()
    if a.prepare_only:
        _, c = extension_contract(a.output_dir)
        print(json.dumps(dict(total_conditions=c['total_conditions'], subjects=len(c['subject_ids']))))
    else:
        run(a.output_dir, a.workers, a.limit, not a.no_postprocess)


if __name__ == '__main__':
    main()
