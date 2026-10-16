"""Bounded, approved Kuramoto A natural-MLP fit and SPT/SHAP interface pilot.

Requires a persisted budget.json. A parent watchdog enforces its absolute deadline;
relaunching cannot reset it. No full EI table, Syn estimate, tree, or F1 is computed.
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
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtri
from scipy.stats import vonmises

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spt_kuramoto_finite_tm import FullTargetIntegrator, PeriodicConditionalTM, check_time, digest, save_json
from scripts.spt_kuramoto_natural_baselines import natural_data, shap_profile
from scripts.spt_kuramoto_natural_mlp import NaturalMLPConditionalTM, chain_partitions
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_local_hyperedge_kuramoto import wrap

DEFAULT = ROOT / 'results/spt_natural_mlp_pilot/kuramoto_A'
POOL = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison/A'
REPORT = ROOT / 'docs/reports/spt_natural_mlp_pilot.md'


def load_budget(folder):
    budget = json.loads((folder / 'budget.json').read_text())
    if not 0 < budget['budget_seconds'] <= 900 or budget['deadline_epoch'] - budget['started_epoch'] > 900:
        raise ValueError('Pilot must have a persisted total budget of at most 15 minutes')
    return budget


def predictive_diagnostics(model, source, target, train_offset):
    residual = wrap(target - model.mean(source))
    return dict(pairs=len(source), paired_arrays_sha256=digest(source, target),
        rmse_rad=float(np.sqrt(np.mean(residual ** 2))),
        persistence_rmse_rad=float(np.sqrt(np.mean(wrap(target - source) ** 2))),
        constant_increment_rmse_rad=float(np.sqrt(np.mean(wrap(target - source - train_offset) ** 2))),
        conditional_nll_nats=float(-model.log_prob(source, target).mean()),
        maximum_absolute_residual_correlation=float(abs(np.corrcoef(residual, rowvar=False) - np.eye(9)).max()))


def worker(folder):
    budget = load_budget(folder)
    end = time.monotonic() + budget['deadline_epoch'] - time.time()
    result = dict(status='running', stage='shared_natural_data', budget=budget,
        scope='one Kuramoto A dataset; fit and interfaces only',
        syn_audit=dict(tolerance_nats=.01, estimated_syn_count=0, applied=False,
                       reason='No EI difference or Syn is computed in this pilot', clipping=False))
    def save():
        result['elapsed_total_seconds'] = time.time() - budget['started_epoch']
        save_json(folder / 'pilot.json', result)
    def progress(value):
        print(json.dumps(value), flush=True)
        result['training_progress'] = value
        save()
    save()
    try:
        check_time(end)
        if not (POOL / 'natural_chains.npz').exists():
            raise FileNotFoundError('Existing shared natural pool required; pilot will not silently generate another')
        states, metadata = natural_data(SimpleNamespace(folder=POOL, case='A', deadline=lambda: end))
        data, manifest = chain_partitions(states)
        manifest.update(pool_path=str((POOL / 'natural_chains.npz').relative_to(ROOT)),
            pool_file_sha256=hashlib.sha256((POOL / 'natural_chains.npz').read_bytes()).hexdigest(),
            source_metadata=metadata)
        save_json(folder / 'shared_data.json', manifest)
        result['shared_data'] = manifest
        result['stage'] = 'natural_mlp_fit'
        save()
        model, fit = NaturalMLPConditionalTM.fit(*data['train'], *data['validation'], deadline=end, progress=progress)
        save_json(folder / 'model.json', model.payload())
        result.update(fit=fit, model_sha256=model.sha256(), stage='natural_test')
        train_x, train_y = data['train']
        offset = np.angle(np.mean(np.exp(1j * wrap(train_y - train_x)), axis=0))
        result['natural_test'] = predictive_diagnostics(model, *data['test'], offset)
        # Both adapters receive the SAME reloaded frozen model object and manifest.
        shared = PeriodicConditionalTM.load(json.loads((folder / 'model.json').read_text()))
        result['reload_maximum_error_rad'] = float(abs(shared.mean(data['test'][0][:32]) - model.mean(data['test'][0][:32])).max())
        frozen_sha = shared.sha256()
        result['stage'] = 'spt_interface'
        save()
        integral = FullTargetIntegrator(shared, outer=8, max_particles=32, seed=20261016, deadline=end)
        queries = {}
        for mask in (511, 510, 0):
            value, diagnostic = integral.marginal(mask, 32)
            if value.shape != (8,) or not np.isfinite(value).all():
                raise ArithmeticError('SPT marginal interface returned nonfinite or mis-shaped output')
            queries[str(mask)] = dict(finite=True, shape=list(value.shape), diagnostic=diagnostic)
        result['spt_interface'] = dict(passed=True, source='iid Uniform(-pi,pi)^9 AFTER natural-only fit',
            full_future_target=True, model_sha256=shared.sha256(), marginal_queries=queries,
            precision_certified=False, information_values_computed=False)
        rng = np.random.default_rng(20261017)
        sx = rng.uniform(-np.pi, np.pi, (1024, 9))
        sy = shared.sample(sx, rng.uniform(size=(1024, 9)))
        normal = ndtri(vonmises.cdf(wrap(sy - shared.mean(sx)), shared.kappa))
        result['conditional_tm_check'] = dict(finite=bool(np.isfinite(normal).all()),
            maximum_absolute_normal_mean=float(abs(normal.mean(0)).max()),
            maximum_absolute_normal_second_moment_error=float(abs((normal ** 2).mean(0) - 1).max()),
            scope='conditional sampler/CDF consistency, not a true-system goodness-of-fit test')
        result['stage'] = 'shap_interface'
        save()
        background = train_x[rng.choice(len(train_x), 8, replace=False)]
        evaluation = data['test'][0][rng.choice(len(data['test'][0]), 8, replace=False)]
        profiles, audit = shap_profile(shared, evaluation, background, lambda: end)
        result['shap_interface'] = dict(passed=bool(np.isfinite(profiles).all()),
            model_sha256=shared.sha256(), background_source='shared natural training partition',
            evaluation_source='shared natural test partition', profile_shape=list(profiles.shape), audit=audit)
        result['shared_model_verified'] = frozen_sha == shared.sha256() == result['spt_interface']['model_sha256'] == result['shap_interface']['model_sha256']
        result['wms_data_adapter'] = dict(manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            training_pairs=len(train_x), test_pairs=len(data['test'][0]),
            scope='same natural arrays exposed; WMS not recomputed', natural_dependencies_preserved=True)
        result['stage'] = 'heldout_intervention_diagnostic'
        save()
        check_time(end)
        truth = SimplifiedKuramoto('A')
        ix = rng.uniform(-np.pi, np.pi, (1024, 9))
        iy = truth.step(ix, rng, dt=.0025)
        result['intervention_diagnostic'] = predictive_diagnostics(shared, ix, iy, offset)
        result['intervention_diagnostic']['used_for_training_or_selection'] = False
        result['fit_support_limitation'] = 'Natural-only fit need not generalize to independent interventions; no true-system causal recovery is certified.'
        check_time(end)
        result.update(status='complete', stage='ended')
        if not result['shared_model_verified'] or not result['shap_interface']['passed'] or not result['conditional_tm_check']['finite']:
            raise ArithmeticError('Shared frozen-model or density interface check failed')
        save()
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        save()
        raise


def write_report(folder):
    result = json.loads((folder / 'pilot.json').read_text())
    lines = ['# Kuramoto A：共同自然数据 MLP 可行性试算', '',
        '2026-10-16。用户批准范围：总墙钟不超过15分钟，单案例拟合与SPT/SHAP共用模型接口；不计算完整EI表、Syn、层级树或F1。', '',
        f"状态：`{result['status']}`；从固定预算起点累计 {result['elapsed_total_seconds']:.1f} 秒。"]
    if 'error' in result:
        lines += ['', f"限制：{result['error']}"]
    if 'fit' in result:
        fit = result['fit']
        lines += ['', '## 共同训练输入', '',
            '复用旧WMS/SHAP的原始自然轨迹池（64链，各512个相邻转移），按独立链固定划分48/8/8链，分别为24576/4096/4096对训练/验证/测试。链间不共享状态，未逐节点打乱。旧基线使用的时间切分与本次新链切分不同，因此只复用原始池，不宣称旧分数已满足新划分。', '',
            f"固定18–64–64–9 tanh MLP，sin/cos相位输入与恒等残差连接；AdamW单一配置，完成{fit['epochs']}轮，验证损失选择第{fit['selected_epoch']}轮。没有按真实超边、测试误差或恢复分数选参。", '',
            '用训练残差拟合独立von Mises噪声，条件CDF构成可归一化、可采样和可求密度的TM。其恒定集中度及输出条件独立是假设，不能据接口通过声称真实动力学密度拟合充分。', '',
            '## 预测诊断', '', '| 测试分布 | MLP RMSE / rad | 保持当前状态 / rad | 固定训练平均增量 / rad | 条件NLL / nats |', '|---|---:|---:|---:|---:|']
        for key, label in [('natural_test', '独立自然链'), ('intervention_diagnostic', '独立均匀干预（只作诊断）')]:
            if key in result:
                d = result[key]
                lines.append(f"| {label} | {d['rmse_rad']:.6f} | {d['persistence_rmse_rad']:.6f} | {d['constant_increment_rmse_rad']:.6f} | {d['conditional_nll_nats']:.4f} |")
        lines += ['', 'RMSE针对含扩散噪声的完整未来相位，差值按圆周取主值；两种测试分布不同，表格用于检查干预外推，不将它们视为配对重复。每例仅一个物理数据集，不能报告总体显著性。']
        if 'intervention_diagnostic' in result:
            natural, intervention = result['natural_test'], result['intervention_diagnostic']
            ratio = intervention['rmse_rad'] / natural['rmse_rad']
            gain = 100 * (1 - intervention['rmse_rad'] / intervention['constant_increment_rmse_rad'])
            lines += ['', f"干预测试RMSE为自然测试的{ratio:.2f}倍，对固定训练平均增量基线的RMSE改善仅{gain:.2f}%。这表明当前MLP虽然接通了干预计算接口，其干预域预测仍弱；不能据此宣称可靠恢复真实系统的SPT。自然数据覆盖不足与固定模型表达能力可能共同影响这一结果，本轮没有通过额外消融区分原因。",
                f"自然测试与干预测试的最大绝对残差相关分别为{natural['maximum_absolute_residual_correlation']:.3f}和{intervention['maximum_absolute_residual_correlation']:.3f}，提示恒定独立残差模型在干预域也存在明显失配。后续若继续，应优先核验干预域模型可信度；不能将增加EI积分精度当作修复预测失配。"]
    if 'spt_interface' in result:
        lines += ['', '## 共用模型与计算边界', '',
            '- SPT适配器在自然拟合完成后生成独立均匀干预；只检查空集、缺一源和完整源的边缘对数密度接口。outer=8、particles=32仅为软件接通检查，未验证信息积分精度。',
            f"- SHAP接口：`{result.get('shap_interface', {}).get('passed', False)}`；共享冻结模型：`{result.get('shared_model_verified', False)}`。SHAP采用8条自然测试评价行、8条自然训练背景行及全部512联盟；只用于接口和效率恒等式检查，不构树。",
            '- WMS可复用同一自然状态对与划分清单，但本轮未重算WMS。旧五法分数保留为历史结果，不能替换成新口径的已完成比较。',
            '- 本轮未估计Syn；预声明容差0.01 nats未实际应用，也未裁剪。',
            '- 条件模型、训练数据和划分改变，旧EI及归因缓存不适用于本次模型。']
    lines += ['', '## 手稿与复现', '',
        '本次重新通过Zotero父项P6UJCVG8核对题名 *Emergent hierarchical organization of causal interactions in complex systems*，列出附件并读取正文DXGC7JEA（19/19页）Methods式（10）–（12）和SI MWIWKSVG（28/28页）S1.2、S5.1、S9.4相关全文。仍无明确稿件日期/版本，附件元数据时间不代表版本日期。共同自然MLP为用户新增约束；本次条件非线性TM不同于SI的仿射估计/Wishart修正路径。', '',
        '`scripts/run_spt_natural_mlp_pilot.py`读取固定绝对截止，父进程独立强制终止超时工作进程，重启不重置预算。内部缓存位于 `results/spt_natural_mlp_pilot/kuramoto_A/`：budget.json、shared_data.json、model.json、pilot.json和run.log。`shared_data.json`含原池、状态对和划分摘要；成本高的模型保留，未复制原始轨迹或生成CSV。', '',
        '[共同输入约束](../log/spt_natural_mlp_training_contract.md)。']
    verification = folder / 'verification.json'
    if verification.exists():
        check = json.loads(verification.read_text())
        lines += ['', f"留存软件验证：{check['tests_passed']}项相关测试通过，涵盖MLP数值Jacobian、圆周密度归一化、无泄漏链划分、序列化、绝对截止实际强制终止、已完成模型复用及既有TM/自然基线回归。格式检查通过。验证摘要见试算目录的verification.json；软件验证不等于干预域模型有效性验证。"]
    REPORT.write_text('\n'.join(lines) + '\n')


def launch(folder):
    budget = load_budget(folder)
    cached = folder / 'pilot.json'
    if cached.exists() and json.loads(cached.read_text()).get('status') == 'complete':
        write_report(folder)
        print(json.dumps(dict(exit_code=0, reused_completed_pilot=True, report=str(REPORT))))
        return 0
    remaining = budget['deadline_epoch'] - time.time()
    if remaining <= 0:
        raise TimeoutError('Persisted pilot deadline has already passed; budget cannot be reset')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (folder / 'run.log').open('a') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(folder)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save_json(folder / 'process.json', dict(pid=child.pid, deadline_epoch=budget['deadline_epoch']))
        try:
            code = child.wait(timeout=max(.01, budget['deadline_epoch'] - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            path = folder / 'pilot.json'
            result = json.loads(path.read_text()) if path.exists() else {}
            result.update(status='budget_exhausted', error='Absolute wall-clock watchdog stopped the worker',
                          elapsed_total_seconds=time.time() - budget['started_epoch'])
            save_json(path, result)
            code = 124
    write_report(folder)
    print(json.dumps(dict(exit_code=code, report=str(REPORT), result=str(folder / 'pilot.json'))))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
    else:
        raise SystemExit(launch(args.output))
