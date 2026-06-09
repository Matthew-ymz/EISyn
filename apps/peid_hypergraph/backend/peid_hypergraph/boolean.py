from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np

from .models import Rule, parse_rule


BOOLEAN_ARITY = {
    "copy": 1,
    "not": 1,
    "and": 2,
    "or": 2,
    "xor": 2,
    "majority": 3,
    "mux": 3,
}


def _entropy_bits(probabilities: Iterable[float]) -> float:
    total = 0.0
    for probability in probabilities:
        p = float(probability)
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def _eval_rule(rule: Rule, state: dict[str, int]) -> int:
    values = [int(state[name]) for name in rule.inputs]
    if rule.rule_type == "copy":
        return values[0]
    if rule.rule_type == "not":
        return 1 - values[0]
    if rule.rule_type == "and":
        return int(values[0] and values[1])
    if rule.rule_type == "or":
        return int(values[0] or values[1])
    if rule.rule_type == "xor":
        return int(values[0] ^ values[1])
    if rule.rule_type == "majority":
        return int(sum(values) >= 2)
    if rule.rule_type == "mux":
        return int(values[1] if values[0] else values[2])
    raise ValueError(f"Unsupported Boolean rule type: {rule.rule_type}")


def _all_states(variables: tuple[str, ...]) -> list[dict[str, int]]:
    return [dict(zip(variables, bits)) for bits in itertools.product((0, 1), repeat=len(variables))]


def _validate_boolean_payload(payload: dict[str, object]) -> tuple[tuple[str, ...], dict[str, Rule], float, int]:
    variables_raw = payload.get("variables")
    rules_raw = payload.get("update_rules")
    if not isinstance(variables_raw, list) or not variables_raw or not all(isinstance(item, str) for item in variables_raw):
        raise ValueError("variables must be a nonempty list of names.")
    variables = tuple(variables_raw)
    if len(set(variables)) != len(variables):
        raise ValueError("variables must be unique.")
    if not isinstance(rules_raw, dict):
        raise ValueError("update_rules must be an object keyed by target name.")

    rules: dict[str, Rule] = {}
    for variable in variables:
        if variable not in rules_raw:
            raise ValueError(f"Missing update rule for {variable}.")
        rule = parse_rule(rules_raw[variable])
        if rule.rule_type not in BOOLEAN_ARITY:
            raise ValueError(f"Unsupported Boolean rule type: {rule.rule_type}")
        if len(rule.inputs) != BOOLEAN_ARITY[rule.rule_type]:
            raise ValueError(f"{rule.rule_type} expects {BOOLEAN_ARITY[rule.rule_type]} inputs.")
        unknown = [name for name in rule.inputs if name not in variables]
        if unknown:
            raise ValueError(f"Unknown rule inputs: {unknown}")
        rules[variable] = rule

    noise = float(payload.get("noise", 0.0))
    if noise < 0.0 or noise > 0.5:
        raise ValueError("noise must be between 0 and 0.5.")
    max_source_order = int(payload.get("max_source_order", 2))
    if max_source_order < 1 or max_source_order > len(variables):
        raise ValueError("max_source_order must be between 1 and the node count.")
    return variables, rules, noise, max_source_order


def _target_probability_by_source(
    variables: tuple[str, ...],
    rules: dict[str, Rule],
    source_set: tuple[str, ...],
    target: str,
    noise: float,
) -> dict[tuple[int, ...], np.ndarray]:
    states = _all_states(variables)
    grouped: dict[tuple[int, ...], list[float]] = {}
    for state in states:
        source_value = tuple(state[name] for name in source_set)
        deterministic = _eval_rule(rules[target], state)
        p_one = (1.0 - noise) * deterministic + noise * (1 - deterministic)
        grouped.setdefault(source_value, []).append(float(p_one))

    probabilities: dict[tuple[int, ...], np.ndarray] = {}
    for source_value, values in grouped.items():
        p_one = float(np.mean(values))
        probabilities[source_value] = np.array([1.0 - p_one, p_one], dtype=float)
    return probabilities


def effective_information_bits(
    variables: tuple[str, ...],
    rules: dict[str, Rule],
    source_set: tuple[str, ...],
    target: str,
    noise: float,
) -> float:
    conditional = _target_probability_by_source(variables, rules, source_set, target, noise)
    source_count = float(len(conditional))
    target_marginal = sum(conditional.values(), np.zeros(2, dtype=float)) / source_count
    target_entropy = _entropy_bits(target_marginal)
    conditional_entropy = sum(_entropy_bits(probs) for probs in conditional.values()) / source_count
    value = target_entropy - conditional_entropy
    return float(max(value, 0.0))


def mobius_interaction(
    variables: tuple[str, ...],
    rules: dict[str, Rule],
    source_set: tuple[str, ...],
    target: str,
    noise: float,
) -> float:
    order = len(source_set)
    total = 0.0
    for size in range(1, order + 1):
        sign = (-1) ** (order - size)
        for subset in itertools.combinations(source_set, size):
            total += sign * effective_information_bits(variables, rules, tuple(subset), target, noise)
    return float(total)


def compute_boolean_graph(payload: dict[str, object]) -> dict[str, object]:
    variables, rules, noise, max_source_order = _validate_boolean_payload(payload)
    nodes = [{"id": name, "label": name} for name in variables]

    pairwise_edges: list[dict[str, object]] = []
    ei_lookup: dict[tuple[tuple[str, ...], str], float] = {}
    for source in variables:
        for target in variables:
            ei = effective_information_bits(variables, rules, (source,), target, noise)
            ei_lookup[((source,), target)] = ei
            pairwise_edges.append({"source": source, "target": target, "ei": ei})

    hyperedges: list[dict[str, object]] = []
    signed_interactions: list[dict[str, object]] = []
    for order in range(2, max_source_order + 1):
        for source_set in itertools.combinations(variables, order):
            for target in variables:
                joint_ei = effective_information_bits(variables, rules, tuple(source_set), target, noise)
                singleton_sum = sum(ei_lookup[((source,), target)] for source in source_set)
                if order == 2:
                    interaction = joint_ei - singleton_sum
                    label = "synergy"
                else:
                    interaction = mobius_interaction(variables, rules, tuple(source_set), target, noise)
                    label = "signed_interaction"
                row = {
                    "sources": list(source_set),
                    "target": target,
                    "source_order": order,
                    "joint_ei": joint_ei,
                    "single_ei_sum": singleton_sum,
                    "synergy": interaction,
                    "display_value": max(interaction, 0.0),
                    "interaction_type": label,
                }
                if interaction > 1.0e-12:
                    hyperedges.append(row)
                elif order >= 3:
                    signed_interactions.append(row)

    return {
        "nodes": nodes,
        "pairwise_edges": pairwise_edges,
        "hyperedges": hyperedges,
        "diagnostics": {
            "signed_interactions": signed_interactions,
            "estimator": "exact",
            "source_intervention": "maximum_entropy_independent",
            "state_family": "boolean",
        },
        "metadata": {
            "mode": "boolean",
            "noise": noise,
            "max_source_order": max_source_order,
        },
    }
