"""Controlled MLP-versus-reweighting effective-information experiment."""

from .reweighted_ei_experiment import (
    ExperimentConfig,
    aggregate_full_validation,
    aggregate_runtime_benchmark,
    benchmark_method_runtimes,
    build_interpretation,
    build_full_interpretation,
    evaluate_between_relation,
    observational_mi_quadrature,
    oracle_convergence_check,
    plot_combined_validation_and_runtime,
    plot_comparison,
    plot_full_validation,
    plot_runtime_benchmark,
    run_experiment,
    summarize_full_agreement,
    summarize_results,
)

__all__ = [
    "ExperimentConfig",
    "aggregate_full_validation",
    "aggregate_runtime_benchmark",
    "benchmark_method_runtimes",
    "build_full_interpretation",
    "build_interpretation",
    "evaluate_between_relation",
    "observational_mi_quadrature",
    "oracle_convergence_check",
    "plot_combined_validation_and_runtime",
    "plot_comparison",
    "plot_full_validation",
    "plot_runtime_benchmark",
    "run_experiment",
    "summarize_full_agreement",
    "summarize_results",
]
