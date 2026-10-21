"""Learn the complete endpoint channel without a dynamics-family prior.

The default is a single, inexpensive seed-0 pilot. Fitting functions receive only
transition counts and the binary state dictionary; they never use a Hamiltonian,
generator, interaction mask, module labels, oracle probabilities, or SPT scores.
The simulator and cached CTMC fit are used only by the evaluation driver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/six_node_data_driven_tpm"
CONFIRMATION_ID = "data_only_tpm_paired_confirmation_v1"
CONFIRMATION_SEEDS = (1, 2)
SAMPLE_BUDGETS = (4800, 48000)
STATE_COUNT = 64
STATES = (2 * ((np.arange(STATE_COUNT)[:, None] >> np.arange(6)) & 1) - 1).astype(float)


@dataclass(frozen=True)
class MLPConfig:
    width: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_epochs: int = 3000
    validation_interval: int = 10
    patience_checks: int = 30
    minimum_improvement: float = 1e-5


def validate_counts(counts):
    values = np.asarray(counts, dtype=float)
    if values.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("Expected a complete 64-by-64 transition-count table")
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Transition counts must be finite, nonnegative and nonempty")
    return values


def validate_kernel(kernel):
    values = np.asarray(kernel, dtype=float)
    if values.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("Expected a complete 64-by-64 TPM")
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ArithmeticError("Invalid TPM; no probability projection is applied")
    return values


def transition_counts(initial, future):
    initial, future = np.asarray(initial), np.asarray(future)
    if initial.shape != future.shape or initial.ndim != 1 or initial.size == 0:
        raise ValueError("Endpoint codes must be nonempty, aligned one-dimensional arrays")
    if (not np.issubdtype(initial.dtype, np.integer)
            or not np.issubdtype(future.dtype, np.integer)
            or np.any(initial < 0) or np.any(initial >= STATE_COUNT)
            or np.any(future < 0) or np.any(future >= STATE_COUNT)):
        raise ValueError("Endpoint codes must be integers in [0, 64)")
    return np.bincount(STATE_COUNT * initial + future, minlength=STATE_COUNT**2).reshape(
        STATE_COUNT, STATE_COUNT)


def split_counts(initial, future, seed):
    """80/20 split of endpoints, before any fitting or information scoring."""
    indices = np.random.default_rng(31000 + seed).permutation(len(initial))
    ntrain = int(0.8 * len(indices))
    train, validation = indices[:ntrain], indices[ntrain:]
    return (transition_counts(initial[train], future[train]),
            transition_counts(initial[validation], future[validation]),
            dict(train_indices=train.tolist(), validation_indices=validation.tolist()))


def negative_log_likelihood(kernel, counts):
    """Nats per endpoint pair; an observed zero probability is infinite, not floored."""
    probabilities, counts = validate_kernel(kernel), validate_counts(counts)
    observed = counts > 0
    if np.any(probabilities[observed] == 0):
        return math.inf
    return float(-np.sum(counts[observed] * np.log(probabilities[observed])) / counts.sum())


def direct_tpm(counts, shrinkage=0.0):
    """Row frequencies shrunk toward the training-only joint future marginal.

    Shrinkage is a total pseudocount per row, not a per-cell pseudocount. A fixed
    half-count on the marginal's 64 outcomes ensures a well-defined prior even
    for an unseen output. With zero shrinkage, an unvisited source fails rather
    than silently receiving an invented row.
    """
    counts = validate_counts(counts)
    if not np.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("Shrinkage must be finite and nonnegative")
    visits = counts.sum(axis=1, keepdims=True)
    if shrinkage == 0 and np.any(visits == 0):
        raise ValueError("An unvisited source has no empirical conditional distribution")
    prior = (counts.sum(axis=0) + 0.5) / (counts.sum() + STATE_COUNT * 0.5)
    return validate_kernel((counts + shrinkage * prior) / (visits + shrinkage))


def select_shrinkage(train, validation):
    candidates = (1., 3., 10., 30., 100., 300., 1000.)
    scores = [dict(shrinkage=value,
                   validation_nll_nats=negative_log_likelihood(direct_tpm(train, value), validation))
              for value in candidates]
    selected = min(scores, key=lambda row: (row["validation_nll_nats"], row["shrinkage"]))
    return selected["shrinkage"], scores


def count_cross_entropy(logits, counts):
    """Exactly the samplewise multiclass cross entropy, aggregated by input state."""
    import torch
    return -(counts * torch.log_softmax(logits, dim=1)).sum() / counts.sum()


def make_mlp(counts, seed, config):
    import torch
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(6, config.width), torch.nn.SiLU(),
        torch.nn.Linear(config.width, config.width), torch.nn.SiLU(),
        torch.nn.Linear(config.width, STATE_COUNT))
    # Data-only initialization; all 64 future states remain joint output classes.
    prior = (counts.sum(axis=0) + .5) / (counts.sum() + STATE_COUNT * .5)
    with torch.no_grad():
        model[-1].bias.copy_(torch.tensor(np.log(prior), dtype=torch.float32))
    return model


def extract_tpm(model):
    import torch
    with torch.no_grad():
        logits = model(torch.tensor(STATES, dtype=torch.float32)).double()
        kernel = torch.softmax(logits, dim=1).numpy()
    return validate_kernel(kernel)


def train_mlp(train, *, seed, config, validation=None, epochs=None):
    """Select a checkpoint on validation CE, or refit for a fixed epoch count."""
    import torch
    train = validate_counts(train)
    if validation is not None:
        validation = validate_counts(validation)
    model = make_mlp(train, seed, config)
    inputs = torch.tensor(STATES, dtype=torch.float32)
    counts = torch.tensor(train, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best_score, best_epoch, stale = math.inf, 0, 0
    best_kernel = None
    history = []
    limit = config.max_epochs if epochs is None else int(epochs)
    if limit < 1:
        raise ValueError("Epoch count must be positive")
    started = time.monotonic()
    for epoch in range(1, limit + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = count_cross_entropy(model(inputs), counts)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MLP training cross entropy")
        loss.backward()
        optimizer.step()
        if validation is not None and (epoch % config.validation_interval == 0 or epoch == limit):
            kernel = extract_tpm(model)
            score = negative_log_likelihood(kernel, validation)
            history.append(dict(epoch=epoch, train_nll_nats=negative_log_likelihood(kernel, train),
                                validation_nll_nats=score))
            if score < best_score - config.minimum_improvement:
                best_score, best_epoch, best_kernel, stale = score, epoch, kernel.copy(), 0
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    final_kernel = extract_tpm(model)
    return (best_kernel if validation is not None else final_kernel,
            dict(selected_epoch=best_epoch if validation is not None else limit,
                 executed_epochs=epoch, validation_nll_nats=None if validation is None else best_score,
                 parameter_count=sum(p.numel() for p in model.parameters()),
                 elapsed_seconds=time.monotonic()-started, history=history))


def information_summary(kernel, test_counts, oracle):
    """Evaluate only after fitting; truth is used for reporting, never training."""
    from scripts import search_six_node_mixed_dynamics as benchmark
    from scripts.compose_spt_three_examples import module_f1
    result = benchmark.spt(validate_kernel(kernel))
    oracle_table = benchmark.exact_ei(oracle)
    errors = [abs(value-oracle_table[key]) for key, value in benchmark.exact_ei(kernel).items()]
    nll = negative_log_likelihood(kernel, test_counts)
    # JSON null and an explicit status preserve an infinite score without NaN JSON.
    return dict(spt=result, module_f1=module_f1(result["tree"], [[1, 2, 3], [4, 5, 6]]),
                heldout_nll_nats=nll if math.isfinite(nll) else None,
                heldout_nll_status="finite" if math.isfinite(nll) else "infinite_zero_probability",
                mean_absolute_subset_ei_error_bits=float(np.mean(errors)),
                maximum_absolute_subset_ei_error_bits=float(np.max(errors)),
                root_xi_bits=result["tree"]["xi_bits"],
                root_syn_bits=result["tree"]["syn_bits"])


def run_seed(seed=0, sample_count=4800, config=MLPConfig(), output=OUT,
             *, confirmation=False, include_ctmc=False):
    import torch
    from threadpoolctl import threadpool_limits
    from scripts import search_six_node_mixed_dynamics as benchmark
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((benchmark.OUT/"selected_ctmc_population.json").read_text())
    with threadpool_limits(limits=1):
        oracle = benchmark.spin_kernel(**selected["config"])
    initial, future, counts = benchmark.draw_joint(oracle, seed, sample_count)
    train, validation, split = split_counts(initial, future, seed)
    assert np.array_equal(train + validation, counts)
    source_hash, future_hash = benchmark.digest(initial), benchmark.digest(future)
    cached = None
    cache_path = benchmark.OUT/"ctmc_confirmation.json"
    if sample_count == 4800 and cache_path.exists():
        cached = next((row for row in json.loads(cache_path.read_text())["conditions"]
                       if row["seed"] == seed), None)
        if cached is not None:
            if (source_hash != cached["source_sha256"] or future_hash != cached["future_sha256"]):
                raise RuntimeError("Endpoint hashes do not match the CTMC comparison")

    shrinkage, shrinkage_scores = select_shrinkage(train, validation)
    _, selection = train_mlp(train, seed=42000+seed, config=config, validation=validation)
    learned, refit = train_mlp(counts, seed=42000+seed, config=config,
                             epochs=selection["selected_epoch"])
    # The independent test set is first generated after all model selection/refitting.
    _, _, test_counts = benchmark.draw_joint(oracle, 20000+seed, sample_count)
    kernels = dict(empirical=direct_tpm(counts), smoothed=direct_tpm(counts, shrinkage), mlp=learned)
    ctmc_fit = None
    if cached is not None:
        theta = np.array([cached["fit"]["coefficients"]["".join(map(str, term))]
                          for term in benchmark.TERMS])
        with threadpool_limits(limits=1):
            from scipy.linalg import expm
            kernels["ctmc_prior"] = expm(selected["config"]["tau"] * benchmark.generic_generator(theta)[0])
        ctmc_fit = dict(source="hash-verified original cached fit", **cached["fit"])
    elif include_ctmc:
        started = time.monotonic()
        kernels["ctmc_prior"], fit = benchmark.fit_ctmc(counts, tau=selected["config"]["tau"], penalty=10.)
        ctmc_fit = dict(source="refit all endpoint pairs", elapsed_seconds=time.monotonic()-started, **fit)

    methods = {name: information_summary(kernel, test_counts, oracle) for name, kernel in kernels.items()}
    methods["oracle"] = information_summary(oracle, test_counts, oracle)
    if cached is not None:
        actual = methods["ctmc_prior"]
        if (actual["spt"]["groups"] != cached["fitted_spt"]["groups"]
                or abs(actual["heldout_nll_nats"] - cached["heldout_nll_nats"]) > 1e-10):
            raise RuntimeError("Reconstructed CTMC does not reproduce cached predictions")
    payload = dict(schema_version=2, seed=seed, sample_count=sample_count,
                   protocol=CONFIRMATION_ID if confirmation else "single-seed pilot; no confirmation claim",
                   role="confirmation" if confirmation else "pilot",
                   configuration=selected["config"],
                   sample_budget_role=("original matched endpoint budget" if sample_count == 4800
                                       else "larger-sample budget selected on seed-0 pilot; frozen before confirmation"),
                   source_intervention="independent uniform six-bit input q(x)=1/64",
                   target="complete joint six-spin endpoint at lag tau=2",
                   ei_units="bits", loss_units="nats per transition", syn_tolerance_bits=benchmark.TOL,
                   ei_estimator="exact finite-state summation; no TM needed on finite support",
                   spt_rule="raw residual, exhaustive bipartitions, complete to singletons",
                   mechanism_prior_in_data_driven_fits=False, conditional_output_factorization=False,
                   source_sha256=source_hash, future_sha256=future_hash,
                   covered_source_states=int((counts.sum(axis=1)>0).sum()),
                   unseen_transition_cells=int((counts==0).sum()),
                   training_count=int(train.sum()), validation_count=int(validation.sum()),
                   split_sha256=benchmark.digest(np.array(split["train_indices"], dtype=np.int64)),
                   final_refit_count=int(counts.sum()),
                   shrinkage=dict(selected_total_pseudocount=shrinkage, candidates=shrinkage_scores,
                                  prior="training-only future marginal with half-count smoothing"),
                   mlp=dict(architecture=[6, config.width, config.width, 64], activation="SiLU",
                            config=asdict(config), initialization_seed=42000+seed,
                            selection=selection, refit=refit), ctmc_fit=ctmc_fit, methods=methods,
                   manuscript=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG",
                                   checked_on="2026-10-21", explicit_version=None,
                                   relevant="main Methods (5)-(12); SI S1.1 (S1)-(S3), S5/Algorithm S1",
                                   limitation="Unique indexed attachments have no explicit draft date; SI lacks new Example II. Local 2026-10-20 Overleaf snapshot and existing code supply its settings."))
    stem = output/f"n{sample_count}_seed{seed}"
    np.savez_compressed(stem.with_suffix(".npz"), initial=initial, future=future,
                        train_indices=np.array(split["train_indices"]),
                        validation_indices=np.array(split["validation_indices"]),
                        test_counts=test_counts, oracle=oracle, **kernels)
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    for name, result in methods.items():
        print(name, "root", result["spt"]["groups"], "F1", result["module_f1"],
              "NLL", result["heldout_nll_nats"], "Xi", result["root_xi_bits"], flush=True)
    print(stem.with_suffix(".json"), flush=True)
    return payload


def numeric_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A summary requires finite, nonempty seed-level measurements")
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values)>1 else None,
                minimum=float(values.min()), maximum=float(values.max()), values=values.tolist())


def summarize_confirmation(records):
    """Independent replication unit is the data seed; seed 0 is excluded."""
    methods = ("empirical", "smoothed", "mlp", "ctmc_prior", "oracle")
    summary = dict(protocol=CONFIRMATION_ID, seeds=list(CONFIRMATION_SEEDS),
                   sample_budgets=list(SAMPLE_BUDGETS), pilot_seed_excluded=0,
                   replication_unit="independent endpoint dataset / data seed",
                   pairing="same endpoints and test counts for all methods within each seed and budget",
                   between_budget_pairing="different, non-nested endpoint draws; no paired budget-effect claim",
                   model_selection="validation NLL only; fixed architecture and optimizer; final full-data refit",
                   recovery_interval="95% Wilson score interval for seed-level binary outcomes; descriptive at n=2",
                   conditions=[])
    for sample_count in SAMPLE_BUDGETS:
        rows = sorted([row for row in records if row["sample_count"] == sample_count], key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != list(CONFIRMATION_SEEDS):
            raise ValueError(f"Missing, duplicated or unexpected confirmation seeds at n={sample_count}")
        for row in rows:
            if row["protocol"] != CONFIRMATION_ID or row["mlp"]["config"] != asdict(MLPConfig()):
                raise ValueError("Confirmation protocol/configuration mismatch")
            if set(row["methods"]) != set(methods) or row["final_refit_count"] != sample_count:
                raise ValueError("Unmatched method availability or endpoint budget")
        result = dict(sample_count=sample_count, methods={}, paired_mlp_minus_ctmc={})
        for name in methods:
            scores = [row["methods"][name] for row in rows]
            success = sum(score["spt"]["exact_original_two_groups"] for score in scores)
            n = len(scores)
            z, rate = 1.959963984540054, success/n
            denominator = 1 + z*z/n
            center = (rate+z*z/(2*n))/denominator
            half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/denominator
            finite = [score["heldout_nll_nats"] is not None for score in scores]
            if not all(finite) and any(finite):
                # Do not average a finite subset of a method with infinite losses.
                nll = dict(status="mixed_finite_and_infinite", finite_count=sum(finite), infinite_count=n-sum(finite))
            elif all(finite):
                nll = dict(status="finite", **numeric_summary([
                    score["heldout_nll_nats"]-row["methods"]["oracle"]["heldout_nll_nats"]
                    for row, score in zip(rows, scores)]))
            else:
                nll = dict(status="all_infinite_zero_probability", finite_count=0, infinite_count=n)
            result["methods"][name] = dict(
                root_recovery_count=int(success), total=n, root_recovery_rate=rate,
                root_recovery_wilson_95=[center-half, center+half],
                module_f1=numeric_summary([score["module_f1"] for score in scores]),
                subset_ei_mae_bits=numeric_summary([score["mean_absolute_subset_ei_error_bits"] for score in scores]),
                excess_test_nll_nats=nll,
                root_xi_bits=numeric_summary([score["root_xi_bits"] for score in scores]),
                root_syn_bits=numeric_summary([score["root_syn_bits"] for score in scores]),
                actual_root_partitions=[score["spt"]["groups"] for score in scores],
                minimum_candidate_syn_bits=min(score["spt"]["syn_audit"]["minimum_syn_bits"] for score in scores),
                tolerance_band_negative_count=sum(score["spt"]["syn_audit"]["small_negative_count"] for score in scores),
                significant_violation_count=sum(score["spt"]["syn_audit"]["violation_count"] for score in scores))
        for key in ("module_f1", "heldout_nll_nats", "mean_absolute_subset_ei_error_bits"):
            result["paired_mlp_minus_ctmc"][key] = numeric_summary([
                row["methods"]["mlp"][key] - row["methods"]["ctmc_prior"][key] for row in rows])
        summary["conditions"].append(result)
    return summary


def checked_cached_confirmation(path, seed, sample_count, configuration):
    """Reject stale/mismatched support artifacts instead of trusting file presence."""
    from scripts import search_six_node_mixed_dynamics as benchmark
    row = json.loads(path.read_text())
    if (row["protocol"] != CONFIRMATION_ID or row["mlp"]["config"] != asdict(MLPConfig())
            or row["seed"] != seed or row["sample_count"] != sample_count
            or row["configuration"] != configuration):
        raise ValueError("Cached configuration differs from the frozen confirmation protocol")
    with np.load(path.with_suffix(".npz")) as arrays:
        initial, future = arrays["initial"], arrays["future"]
        if (len(initial) != sample_count or benchmark.digest(initial) != row["source_sha256"]
                or benchmark.digest(future) != row["future_sha256"]):
            raise ValueError("Cached endpoint arrays differ from their recorded hashes")
        train_indices, validation_indices = arrays["train_indices"], arrays["validation_indices"]
        if (sorted(np.r_[train_indices, validation_indices].tolist()) != list(range(sample_count))
                or benchmark.digest(train_indices) != row["split_sha256"]):
            raise ValueError("Cached training/validation split is incomplete, overlapping or changed")
        if "oracle" not in arrays:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                oracle = benchmark.spin_kernel(**configuration)
        for name, result in row["methods"].items():
            kernel = validate_kernel(oracle if name == "oracle" and "oracle" not in arrays else arrays[name])
            nll = negative_log_likelihood(kernel, arrays["test_counts"])
            if result["heldout_nll_nats"] is None:
                matches = math.isinf(nll) and result["heldout_nll_status"] == "infinite_zero_probability"
            else:
                matches = abs(nll-result["heldout_nll_nats"]) < 1e-12
            ei = benchmark.exact_ei(kernel)
            if not matches or any(abs(value-result["spt"]["ei_bits"]["".join(map(str, subset))])>1e-12
                                  for subset, value in ei.items()):
                raise ValueError("Cached TPM does not reproduce saved EI or predictive loss")
    return row


def run_confirmation(output=OUT/"confirmation"):
    """Paired comparison, reduced by the user to fixed first two seeds 1 and 2."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from scripts import search_six_node_mixed_dynamics as benchmark
    configuration = json.loads((benchmark.OUT/"selected_ctmc_population.json").read_text())["config"]
    records, failures = [], []
    for count in SAMPLE_BUDGETS:
        for seed in CONFIRMATION_SEEDS:
            print(f"CONFIRMATION n={count} seed={seed}", flush=True)
            try:
                saved = output/f"n{count}_seed{seed}.json"
                if saved.exists() and saved.with_suffix(".npz").exists():
                    records.append(checked_cached_confirmation(saved, seed, count, configuration))
                    print("Reused matching saved dataset and fitted channels", flush=True)
                else:
                    records.append(run_seed(seed, count, output=output, confirmation=True, include_ctmc=True))
            except Exception as exc:
                failure = dict(sample_count=count, seed=seed, error_type=type(exc).__name__, message=str(exc))
                failures.append(failure)
                (output/f"failure_n{count}_seed{seed}.json").write_text(json.dumps(failure, indent=2)+"\n")
                print("FAILED", failure, flush=True)
    if failures:
        raise RuntimeError(f"Confirmation incomplete: {len(failures)} failed datasets; all failure records retained")
    summary = summarize_confirmation(records)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def evaluate_clustering_baselines(output=OUT/"confirmation"):
    """Re-evaluate the original endpoint-block baselines; never retrain a model."""
    from scripts import search_six_node_mixed_dynamics as benchmark
    from scripts.compose_spt_three_examples import module_f1
    output = Path(output)
    configuration = json.loads((benchmark.OUT/"selected_ctmc_population.json").read_text())["config"]
    records = []
    names = ("mic", "mi_hac", "dcor_hac", "correlation_hac", "ward")
    for count in SAMPLE_BUDGETS:
        for seed in CONFIRMATION_SEEDS:
            stem = output/f"n{count}_seed{seed}"
            fitted = checked_cached_confirmation(stem.with_suffix(".json"), seed, count, configuration)
            with np.load(stem.with_suffix(".npz")) as arrays:
                counts = transition_counts(arrays["initial"], arrays["future"])
                layers = {}
                for layer, probability in (("data", counts.ravel()/counts.sum()),
                                           ("shared_mlp", arrays["mlp"].ravel()/64)):
                    result = benchmark.discrete_methods(probability, "transition")
                    for method in result["methods"].values():
                        method["module_f1"] = (module_f1(method["tree"], [[1, 2, 3], [4, 5, 6]])
                                               if method["kind"] == "hierarchy" else None)
                    layers[layer] = result
            records.append(dict(sample_count=count, seed=seed, view="complete initial-future node blocks",
                                source_sha256=fitted["source_sha256"], future_sha256=fitted["future_sha256"],
                                layers=layers, spt_mlp=fitted["methods"]["mlp"]))
    summary = []
    for count in SAMPLE_BUDGETS:
        rows = [row for row in records if row["sample_count"] == count]
        condition = dict(sample_count=count, layers={})
        for layer in ("data", "shared_mlp"):
            methods = {}
            for name in names:
                scores = [row["layers"][layer]["methods"][name] for row in rows]
                methods[name] = dict(root_recovery_count=sum(score["exact_original_two_groups"] for score in scores),
                                     total=len(rows), module_f1=numeric_summary([score["module_f1"] for score in scores]),
                                     actual_root_partitions=[score["groups"] for score in scores])
            condition["layers"][layer] = methods
        summary.append(condition)
    payload = dict(seeds=list(CONFIRMATION_SEEDS),
                   source="same saved endpoints and MLP channels; no new transitions or fitting",
                   primary_view="complete per-node initial-future endpoint blocks, fixed before evaluation",
                   shared_fit="q(x)P_mlp(y|x), q(x)=1/64; exactly the SPT learned joint",
                   conditions=records, summary=summary)
    (output/"clustering_baselines_two_seeds.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    for condition in summary:
        for layer, methods in condition["layers"].items():
            print(condition["sample_count"], layer,
                  {name: (f"{result['root_recovery_count']}/{result['total']}", result["module_f1"]["mean"])
                   for name, result in methods.items()}, flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4800)
    parser.add_argument("--max-epochs", type=int, default=MLPConfig.max_epochs)
    parser.add_argument("--output", type=Path, default=OUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirmation", action="store_true", help="Run/reuse fixed seeds 1 and 2 at both sample budgets")
    mode.add_argument("--baselines-only", action="store_true", help="Evaluate original clustering methods on saved data/channels")
    args = parser.parse_args()
    if args.confirmation:
        if args.seed != 0 or args.samples != 4800 or args.max_epochs != MLPConfig.max_epochs:
            parser.error("Confirmation uses the frozen seeds, budgets and MLP config; omit single-seed overrides")
        run_confirmation(args.output/"confirmation")
    elif args.baselines_only:
        evaluate_clustering_baselines(args.output/"confirmation")
    else:
        run_seed(args.seed, args.samples, MLPConfig(max_epochs=args.max_epochs), args.output)


if __name__ == "__main__":
    main()
