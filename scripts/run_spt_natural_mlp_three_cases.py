"""Approved 30-minute three-case natural MLP -> complete EI -> standard SPT.

Only SPT is recomputed. Preserved network/score, independent-chain splits, one
fixed MLP per case, and a preregistered two-level/four-scramble integration grid.
The parent kills its own worker process group at the persisted absolute deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spt_kuramoto_finite_tm import FullTargetIntegrator, check_time, save_json, digest
from scripts.spt_glv_finite_tm import RealFullTargetIntegrator
from scripts.spt_kuramoto_natural_mlp import NaturalMLPConditionalTM, chain_partitions
from scripts.spt_real_natural_mlp import NaturalRealMLPConditionalTM
from scripts.analyze_spt_glv_full_table import tree_record
from scripts.spt_local_hyperedge_kuramoto import wrap, SUPPORTS

FOLDER = ROOT / 'results/spt_natural_mlp_three_cases'
REPORT = ROOT / 'docs/reports/spt_natural_mlp_three_cases.md'
CASES = [('Kuramoto A', 'kuramoto_A', 'A', 'circular', 'results/spt_local_kuramoto_tm/ac_comparison/A/natural_chains.npz'),
         ('HO-LV A', 'glv_A', 'A', 'real', 'results/spt_matched_three_case/full_comparison/glv_natural_chains.npz'),
         ('Kuramoto B', 'kuramoto_B', 'C', 'circular', 'results/spt_local_kuramoto_tm/ac_comparison/C/natural_chains.npz')]
LAYOUTS = [(128, 64), (512, 128)]
TOLERANCE = .01
NET_INITIALIZATION_LOCK = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text())


def protocol():
    return dict(cases=[a[0] for a in CASES], layouts=LAYOUTS, scrambles=4,
        natural_split='fixed independent chains 48/8/8; same seed and state-pair indices',
        mlp='two64 tanh hidden layers; AdamW lr.003 wd1e-5; batch1024; max250; patience25; validation only',
        intervention='iid Uniform(-pi,pi)^9 in native phase/log-abundance coordinates AFTER freezing natural fit',
        target='complete nine-coordinate future', horizon=.2, units='nats',
        syn_tolerance_nats=TOLERANCE, clipping=False, estimator='conditional TM with proposal-corrected RQMC marginal integration',
        construction='all511 nonempty EI; audit all9330 splits; standard exact-node SPT; singleton stopping',
        precision_gates=dict(max_ei_sem_nats=.01, max_grid_change_nats=.01, minimum_ess=10,
                             maximum_weight=.25, maximum_remaining_step=1e-3),
        score='unchanged module F1 against frozen SUPPORTS; primitive internal binary refinements ignored',
        inference='one natural dataset per case; scrambles are numerical repeats, not independent system samples',
        manuscript=dict(parent='P6UJCVG8', main='DXGC7JEA', si='MWIWKSVG', checked='2026-10-16',
            version='only available attachment pair; no explicit manuscript date/version',
            locations='Methods(10)-(12); SI S1.2(S7)-(S15), S5.1(S49)-(S50), S7; general conditional TM differs from affine SI path'))


def diagnostics(model, data, circular):
    x, y = data
    residual = y - model.mean(x)
    if circular:
        residual = wrap(residual)
    return dict(samples=len(x), rmse_native=float(np.sqrt(np.mean(residual ** 2))),
        conditional_nll_nats=float(-model.log_prob(x, y).mean()), data_sha256=digest(x, y))


def prepare(case, end):
    label, name, native, family, pool = case
    directory = FOLDER / name
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / pool, allow_pickle=False) as data:
        states, meta = data['states'].copy(), json.loads(str(data['meta']))
    if meta.get('case') not in (native, 'matched_simplified_A') or meta.get('tau') != .2 or meta.get('dt') != .0025:
        raise ValueError('Natural pool does not match the registered case/horizon')
    data, manifest = chain_partitions(states)
    manifest.update(pool_path=pool, pool_file_sha256=hashlib.sha256((ROOT / pool).read_bytes()).hexdigest(), source_metadata=meta)
    save_json(directory / 'shared_data.json', manifest)
    cls = NaturalMLPConditionalTM if family == 'circular' else NaturalRealMLPConditionalTM
    path = directory / 'model.json'
    reused = name == 'kuramoto_A'
    if reused:
        pilot = ROOT / 'results/spt_natural_mlp_pilot/kuramoto_A'
        previous = read(pilot / 'shared_data.json')
        if previous != manifest:
            raise ValueError('Frozen Kuramoto A natural split/pool changed')
        model = cls.load(read(pilot / 'model.json'))
        fit = read(pilot / 'pilot.json')['fit']
    elif path.exists():
        model, fit = cls.load(read(path)), read(directory / 'fit.json')
    else:
        check_time(end)
        print(json.dumps(dict(case=label, stage='natural_mlp_fit')), flush=True)
        if family == 'circular':
            model, fit = cls.fit(*data['train'], *data['validation'], deadline=end,
                                progress=lambda a: print(json.dumps(dict(case=label, **a)), flush=True))
        else:
            model, fit = cls.fit(*data['train'], *data['validation'], deadline=end)
    save_json(path, model.payload())
    save_json(directory / 'fit.json', fit)
    return model, dict(case=label, name=name, native_reference=native, family=family, status='fitted',
        reused_pilot_model=reused, model_sha256=model.sha256(), natural_data=manifest,
        fit={k: v for k, v in fit.items() if k != 'history'},
        natural_test=diagnostics(model, data['test'], family == 'circular'), levels=[])


def split_signature(metrics, native):
    primitive = [e for e in SUPPORTS[native] if not any(set(g) < set(e) for g in SUPPORTS[native])]
    return {tuple(c) for c in metrics['clades'] if not any(set(c) < set(e) for e in primitive)}


def evaluated_tree(values, native):
    tree = tree_record({str(i + 1): float(v) for i, v in enumerate(values)}, native, TOLERANCE)
    if tree['status'] != 'exploratory_only':
        audit = tree.get('audit', {})
        minimum = min(audit.get('minimum_syn_nats', float('inf')), audit.get('minimum_xi_nats', float('inf')))
        count = audit.get('significant_violation_count', 0) + audit.get('significant_xi_violation_count', 0)
        tree['error'] = f"SPT blocked: minimum={minimum} nats; threshold={-TOLERANCE}; affected={count}; status={tree['status']}"
        return tree
    truth = {frozenset(e) for e in SUPPORTS[native] if len(e) < 9}
    found = {frozenset(e) for e in tree['metrics']['clades']}
    tree['metrics']['strict_clade_f1'] = 2 * len(truth & found) / (len(truth) + len(found))
    return tree


def make_integrator(model, family, outer, particles, repeat, end):
    factory = FullTargetIntegrator if family == 'circular' else RealFullTargetIntegrator
    # Serialize the shared Sobol direction-table initialization. Sampling and
    # all actual marginal queries remain paired and independent across cases.
    with NET_INITIALIZATION_LOCK:
        return factory(model, outer=outer, max_particles=particles,
                       seed=20261016 + 1000 * repeat, deadline=end)


def compute_level(model, record, outer, particles, end):
    directory = FOLDER / record['name']
    path = directory / f'ei_n{outer}_p{particles}.json'
    level = read(path) if path.exists() else dict(outer=outer, particles=particles, model_sha256=model.sha256(), replicates=[])
    if level['model_sha256'] != model.sha256():
        raise ValueError('Cannot reuse EI from a different model')
    for repeat in range(4):
        check_time(end)
        row = next((r for r in level['replicates'] if r['scramble'] == repeat), None)
        if row is None:
            row = dict(scramble=repeat, information_nats={}, diagnostics={})
            level['replicates'].append(row)
        if len(row['information_nats']) == 511:
            continue
        started = time.monotonic()
        integral = make_integrator(model, record['family'], outer, particles, repeat, end)
        denominator, d = integral.marginal(0, particles)
        row['denominator_diagnostic'] = d
        for mask in range(1, 512):
            if str(mask) in row['information_nats']:
                continue
            numerator, diagnostic = integral.marginal(mask, particles)
            value = float(np.mean(numerator - denominator))
            if not np.isfinite(value):
                raise ArithmeticError('Nonfinite EI estimate')
            row['information_nats'][str(mask)], row['diagnostics'][str(mask)] = value, diagnostic
            if mask % 64 == 0:
                save_json(path, level)
                print(json.dumps(dict(case=record['case'], outer=outer, particles=particles,
                                     scramble=repeat, subsets=mask, seconds=time.monotonic() - started)), flush=True)
        row['elapsed_seconds'] = time.monotonic() - started
        save_json(path, level)
    samples = np.array([[r['information_nats'][str(m)] for m in range(1, 512)] for r in level['replicates']])
    mean = samples.mean(0)
    level['mean_information_nats'] = {str(i + 1): float(v) for i, v in enumerate(mean)}
    level['maximum_ei_sem_nats'] = float(samples.std(0, ddof=1).max() / 2)
    level['mean_tree'] = evaluated_tree(mean, record['native_reference'])
    level['replicate_trees'] = [evaluated_tree(a, record['native_reference']) for a in samples]
    diagnostic = [d for r in level['replicates'] for d in [r['denominator_diagnostic'], *r['diagnostics'].values()] if d.get('ess_minimum') is not None]
    level['minimum_ess'] = min(d['ess_minimum'] for d in diagnostic)
    level['maximum_weight'] = max(d['maximum_normalized_weight'] for d in diagnostic)
    step_key = 'max_remaining_step_rad' if record['family'] == 'circular' else 'max_projected_remaining_step_coordinate'
    level['maximum_remaining_step'] = max(d.get(step_key, 0) for d in diagnostic)
    level['complete'] = True
    save_json(path, level)
    return {k: v for k, v in level.items() if k not in ('replicates', 'mean_information_nats')}


def assess(record):
    if not record['levels']:
        return
    high = record['levels'][-1]
    mean = high['mean_tree']
    issues = []
    if mean['status'] != 'exploratory_only':
        record.update(status='nonnegativity_failed', error=mean.get('error'), metrics=None)
        return
    if any(r['status'] != 'exploratory_only' for r in high['replicate_trees']):
        issues.append('one_or_more_numerical_repeat_Syn_audits_failed')
    if high['maximum_ei_sem_nats'] >= .01:
        issues.append('maximum_EI_SEM>=.01_nats')
    if high['minimum_ess'] < 10 or high['maximum_weight'] >= .25:
        issues.append('importance_weights')
    if high['maximum_remaining_step'] >= 1e-3:
        issues.append('posterior_proposal_remaining_step')
    if len(record['levels']) != 2:
        issues.append('high_grid_not_completed')
    else:
        lo = read(FOLDER / record['name'] / 'ei_n128_p64.json')['mean_information_nats']
        hi = read(FOLDER / record['name'] / 'ei_n512_p128.json')['mean_information_nats']
        change = max(abs(lo[str(m)] - hi[str(m)]) for m in range(1, 512))
        record['maximum_grid_change_nats'] = change
        if change >= .01:
            issues.append('maximum_grid_change>=.01_nats')
    target = split_signature(mean['metrics'], record['native_reference'])
    repeats = [r['metrics'] for r in high['replicate_trees'] if r['status'] == 'exploratory_only']
    agreement = sum(split_signature(r, record['native_reference']) == target for r in repeats)
    if agreement != 4:
        issues.append('scored_module_tree_varies_across_scrambles')
    record.update(status='numerically_unresolved' if issues else 'screen_passed', failed_checks=issues,
        metrics=mean['metrics'], diagnostic_only=bool(issues), scored_tree_agreement=f'{agreement}/4',
        numerical_f1_range=[min(r['clade_f1'] for r in repeats), max(r['clade_f1'] for r in repeats)] if repeats else None)


def worker():
    budget = read(FOLDER / 'budget.json')
    end = time.monotonic() + budget['deadline_epoch'] - time.time()
    definition = protocol()
    save_json(FOLDER / 'protocol.json', definition)
    result = dict(status='running', budget=budget, protocol=definition, records=[])
    models = {}
    result_lock = threading.RLock()
    def save():
        with result_lock:
            result['elapsed_seconds'] = time.time() - budget['started_epoch']
            save_json(FOLDER / 'summary.json', result)
    save()
    try:
        for case in CASES:
            try:
                model, record = prepare(case, end)
                models[record['name']] = model
                result['records'].append(record)
            except Exception as error:
                result['records'].append(dict(case=case[0], name=case[1], status='fit_failed', error=f'{type(error).__name__}: {error}', levels=[]))
            save()
        def run_level(record, outer, particles):
            check_time(end)
            try:
                level = compute_level(models[record['name']], record, outer, particles, end)
                with result_lock:
                    record['levels'].append(level)
                    assess(record)
                    print(json.dumps(dict(case=record['case'], stage='level_complete', outer=outer,
                        status=record['status'], f1=(record.get('metrics') or {}).get('clade_f1'))), flush=True)
                    save()
            except TimeoutError:
                raise
            except Exception as error:
                with result_lock:
                    record.update(status='integration_failed', error=f'{type(error).__name__}: {error}')
                    save()
        # Cases have separate deterministic nets, models, and checkpoint files.
        # Three threads share the SAME process group and absolute deadline.
        # Existing full or partial replicate tables resume without new draws.
        with ThreadPoolExecutor(max_workers=3) as executor:
            for outer, particles in LAYOUTS:
                result['stage'] = f'three cases: N{outer}/P{particles}'
                save()
                futures = [executor.submit(run_level, record, outer, particles)
                           for record in result['records'] if record['name'] in models]
                for future in as_completed(futures):
                    future.result()
        result.update(status='complete', stage='ended')
        save()
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'failed', error=f'{type(error).__name__}: {error}')
        save()
        if not isinstance(error, TimeoutError):
            raise


def report():
    data = read(FOLDER / 'summary.json')
    records = {r['case']: r for r in data['records']}
    lines = ['# 三案例：共同自然数据 MLP 后的 SPT F1', '',
        '2026-10-16。用户批准30分钟总上限，仅补算Kuramoto A、HO-LV A、Kuramoto B的自然MLP和SPT，不重跑其他基线，不调参或按F1选择模型。', '',
        f"任务状态：`{data['status']}`；固定预算起点后累计{data.get('elapsed_seconds', 0):.1f}秒。", '',
        '| 案例 | 旧非MLP模块F1 | 自然MLP后模块F1 | 数值重复范围 | 状态 |', '|---|---:|---:|---|---|']
    for case in CASES:
        r = records.get(case[0], {})
        score = (r.get('metrics') or {}).get('clade_f1')
        cell = f"{score:.4f}" + ('*' if r.get('diagnostic_only') else '') if score is not None else '未得到有效树'
        valid_count = sum(t['status'] == 'exploratory_only' for t in r.get('levels', [{}])[-1].get('replicate_trees', []))
        spread = str(r.get('numerical_f1_range', '—')) + f'（{valid_count}/4可评分）'
        lines.append(f"| {case[0]} | 1.0000 | {cell} | {spread} | {r.get('status', '未启动')} |")
    if all(len(records.get(case[0], {}).get('levels', [])) == 2 for case in CASES):
        lines += ['', '以上新分数来自完整高档均值EI表，而非复用旧图。星号数值仍受积分档位变化和/或树稳定性限制；本轮只能判断是否重现旧图表现，不能把诊断值当作已充分收敛的真实机制恢复率。']
    lines += ['', '* 表示完整均值EI表通过Syn审计并生成的诊断树，但积分精度或树稳定性尚未通过；不能作为已证实的可靠F1。显著Syn非负违规的重复不构树、不补分，重复范围仅包含可评分的重复；均值EI仍使用全部四份完整表，不按审计或F1选择重复。旧1.0来自字典模型，不是本次MLP结果。', '',
        '## 固定比较口径', '',
        '- 使用三个案例各自原WMS/SHAP自然轨迹池。每例按同一固定链索引48/8/8划分训练/验证/测试，24576/4096/4096相邻状态对，无跨集合共享状态。Kuramoto B仍映射原缓存键C。',
        '- Kuramoto A复用已批准试算的冻结MLP；其余采用同样两层64单元tanh和AdamW单一设置，验证集仅选轮次。Kuramoto用sin/cos输入和圆周损失；LV用训练统计量标准化的实数坐标和MSE。这些状态域适配分别声明。',
        '- 自然训练后冻结MLP与训练残差条件密度；Kuramoto为独立恒定von Mises，LV为独立恒定Gaussian条件TM，无噪声下限。后续干预均为原生坐标独立Uniform(-pi,pi)^9，target固定完整未来九坐标，时距0.2，信息单位nats。',
        '- 固定N128/P64及N512/P128两档，每档四scramble；同一联合下查询全部511非空源子集，审计全部9330二分，再执行标准逐节点精确SPT至单节点。原生Syn容差0.01 nats，容差内负值数量显式记录；显著负值失败，无静默裁剪。',
        '- 评分沿用旧图的耦合模块参考，忽略primitive模块内部任意二叉细化，并保留严格clade F1。分数不等于全部物理超边恢复率。每例仅一个自然数据集，四scramble仅为数值重复。',
        '- 0.01 nats的EI SEM/两档变化、ESS≥10、最大权重<.25、后验剩余步<1e-3及四次评分树一致共同决定数值标签。未过检查可展示完整合法均值树的诊断F1，但不能据此宣布可靠恢复。', '',
        '## 各例审计', '']
    for case in CASES:
        r = records.get(case[0], {})
        lines += [f"### {case[0]}", '', f"状态：`{r.get('status', '未启动')}`。"]
        if r.get('error'):
            lines += [f"限制：{r['error']}"]
        if r.get('fit'):
            lines += [f"训练轮次{r['fit'].get('epochs')}，选中轮次{r['fit'].get('selected_epoch')}，模型摘要`{r['model_sha256']}`。自然测试RMSE={r['natural_test']['rmse_native']:.6f}（本例原生坐标单位）。"]
        if r.get('metrics'):
            m = r['metrics']
            lines += [f"模块F1={m['clade_f1']:.4f}；严格clade F1={m['strict_clade_f1']:.4f}；根切分={m['root_split']}；评分树一致{r['scored_tree_agreement']}。"]
        lines += [f"未通过检查：`{r.get('failed_checks', [])}`。"]
        for level in r.get('levels', []):
            a = level['mean_tree'].get('audit', {})
            level_score = (level['mean_tree'].get('metrics') or {}).get('clade_f1')
            lines += [f"- N{level['outer']}/P{level['particles']}：均值树`{level['mean_tree']['status']}`；最大EI SEM={level['maximum_ei_sem_nats']:.6g} nats；min ESS={level['minimum_ess']:.4g}；max weight={level['maximum_weight']:.4g}。",
                f"  该档模块F1={level_score}；raw Syn最小估计值={a.get('minimum_syn_nats')} nats；阈值=-0.01 nats；容差内负值数={a.get('tolerance_negative_count')}；显著违规数={a.get('significant_violation_count')}。"]
            values = read(FOLDER / r['name'] / f"ei_n{level['outer']}_p{level['particles']}.json").get('mean_information_nats')
            if values:
                root_xi = values['511'] - sum(values[str(1 << i)] for i in range(9))
                lines += [f"  全源raw Ξ估计={root_xi:.8g} nats；容差内负估计仅作数值零处理，不是负Syn的证据。"]
        lines += ['']
    lines += ['## 解释边界与复现', '',
        '自然数据上的低预测误差不能保证均匀干预域可信。此前Kuramoto A试算已显示明显干预外推误差，因此即使新F1高，也只说明该冻结MLP诱导联合的模块恢复，不能自动证明学到了真实机制。若F1下降，不能直接断言是SPT算法失效；模型表达、自然数据覆盖与积分近似均可能参与，本轮未做消融。', '',
        '本次重新通过Zotero核对父项P6UJCVG8题名 *Emergent hierarchical organization of causal interactions in complex systems*，列举唯一主文DXGC7JEA（19/19页）及SI MWIWKSVG（28/28页），读取Methods(10)–(12)、SI S1.2、S5.1、S7。附件无明确稿件日期/版本，元数据日期不构成新稿证据。一般条件TM和模型介导信息积分是明确的估计器适配，区别于SI S1.2仿射/Wishart路径；本例完整未来坐标亦不同于SI S7增量target。', '',
        '`scripts/run_spt_natural_mlp_three_cases.py`；内部预算、协议、共用自然划分、模型、全EI及审计缓存位于`results/spt_natural_mlp_three_cases/`，全部模型摘要隔离，未复用旧字典模型EI。父进程按持久化绝对截止终止自己的工作进程组，重启不重置预算。', '',
        '执行修复记录：依据实测吞吐将案例积分改为同一进程内三案例并行，科学设置和绝对截止均未改变。首次并行出现SciPy Sobol方向表初始化线程冲突，已停止该进程并隔离其新高档结果；仅保留冲突前经摘要核验的正确积分前缀。随后串行化采样器构造、保留积分并行，串并行采样逐点一致测试通过。两次修复日志、被排除缓存与原始正确前缀摘要均保留；报告采用的是修复后结果。', '',
        '[原MLP试算](spt_natural_mlp_pilot.md) · [旧字典模型比较](spt_matched_three_case_comparison.md)。']
    verification = FOLDER / 'final_verification.json'
    if verification.exists():
        v = read(verification)
        lines += ['', f"最终核验：{v['software_tests_passed']}项软件测试通过；{v['preserved_prefixes_verified']}份正确缓存前缀逐值摘要一致；三个自然池与原网络匹配；科学核心文件未改变，模型摘要与EI缓存一致；短期监督已结束。"]
    REPORT.write_text('\n'.join(lines) + '\n')


def launch():
    budget = read(FOLDER / 'budget.json')
    if budget['budget_seconds'] != 1800 or budget['deadline_epoch'] - budget['started_epoch'] > 1800:
        raise ValueError('Expected approved shared 30-minute budget')
    if time.time() >= budget['deadline_epoch']:
        raise TimeoutError('Absolute budget passed; cannot reset')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (FOLDER / 'run.log').open('a') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save_json(FOLDER / 'process.json', dict(pid=child.pid, deadline_epoch=budget['deadline_epoch']))
        try:
            code = child.wait(timeout=max(.01, budget['deadline_epoch'] - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            result = read(FOLDER / 'summary.json')
            result.update(status='budget_exhausted', error='Parent absolute deadline watchdog stopped its worker',
                          elapsed_seconds=time.time() - budget['started_epoch'])
            save_json(FOLDER / 'summary.json', result)
            code = 124
    report()
    print(json.dumps(dict(exit_code=code, report=str(REPORT))))
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.report_only:
        report()
    elif args.worker:
        worker()
    else:
        raise SystemExit(launch())
