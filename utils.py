from __future__ import annotations

import html
import importlib
import math
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def enumerate_binary_states(n: int) -> np.ndarray:
    """Enumerate all binary states in lexicographic order."""

    if n < 0:
        raise ValueError("Number of nodes must be non-negative.")
    if n == 0:
        return np.zeros((1, 0), dtype=int)
    return np.array(list(product([0, 1], repeat=n)), dtype=int)


def build_probabilistic_boolean_tpm(
    adjacency: np.ndarray,
    node_specs: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Build a system TPM from independent Bernoulli node updates.

    `adjacency[i, j]` denotes the weight from node `i` at time `t` to node `j`
    at time `t+1`. Each node specification can provide:

    - `bias`: scalar bias term
    - `alpha`, `beta`, `gamma`: coefficients for copy / coop / parity terms
    - `copy_weights`: optional explicit weights overriding the adjacency column
    - `coop_sources`: optional list of source indices for one centered product term
    - `coop_pairs`: optional list of `(i, k)` or `(i, k, weight)`
    - `parity_sources`: optional list of source indices used in parity
    - `eta`: optional scaling on the parity term
    """

    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")
    n_nodes = matrix.shape[0]
    if len(node_specs) != n_nodes:
        raise ValueError("Need one node specification per node.")

    current_states = enumerate_binary_states(n_nodes)
    next_states = current_states
    tpm = np.zeros((len(current_states), len(next_states)), dtype=float)

    for row_index, state in enumerate(current_states):
        probabilities = np.array(
            [_node_activation_probability(state, matrix, node_specs, j) for j in range(n_nodes)],
            dtype=float,
        )
        tpm[row_index] = np.prod(
            np.where(next_states == 1, probabilities, 1.0 - probabilities),
            axis=1,
        )

    return tpm


def build_deterministic_boolean_tpm(
    n_nodes: int,
    update_rule: Callable[[tuple[int, ...]], Sequence[int]],
) -> np.ndarray:
    """Build a deterministic Boolean TPM from an explicit update rule."""

    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative.")

    states = enumerate_binary_states(n_nodes)
    state_to_index = {tuple(state.tolist()): index for index, state in enumerate(states)}
    tpm = np.zeros((len(states), len(states)), dtype=float)

    for row_index, state in enumerate(states):
        state_tuple = tuple(int(bit) for bit in state.tolist())
        next_state = tuple(int(bit) for bit in update_rule(state_tuple))
        if len(next_state) != n_nodes:
            raise ValueError("update_rule must return one bit per node.")
        if any(bit not in (0, 1) for bit in next_state):
            raise ValueError("update_rule must return binary states.")
        tpm[row_index, state_to_index[next_state]] = 1.0

    return tpm


def build_marshall_example1_tpm() -> np.ndarray:
    """Build the 4-node micro TPM from Marshall et al. (2024) Example 1."""

    states = enumerate_binary_states(4)
    tpm = np.zeros((len(states), len(states)), dtype=float)

    for row_index, state in enumerate(states):
        a, b, c, d = (int(bit) for bit in state.tolist())
        probabilities = np.array(
            [
                0.05 + 0.01 * a + 0.1 * b + 0.8 * int(c and d),
                0.05 + 0.01 * b + 0.1 * a + 0.8 * int(c and d),
                0.05 + 0.01 * c + 0.1 * d + 0.8 * int(a and b),
                0.05 + 0.01 * d + 0.1 * c + 0.8 * int(a and b),
            ],
            dtype=float,
        )
        tpm[row_index] = np.prod(
            np.where(states == 1, probabilities, 1.0 - probabilities),
            axis=1,
        )

    return tpm


def enumerate_pair_partitions(indices: Sequence[int]) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate all perfect pair partitions of an even-sized collection."""

    ordered = tuple(int(index) for index in indices)
    if len(set(ordered)) != len(ordered):
        raise ValueError("indices must be unique.")
    if len(ordered) % 2 != 0:
        raise ValueError("indices must have even cardinality.")
    if not ordered:
        return [tuple()]

    partitions: list[tuple[tuple[int, int], ...]] = []

    def _recurse(remaining: tuple[int, ...], blocks: list[tuple[int, int]]) -> None:
        if not remaining:
            normalized = tuple(sorted(tuple(sorted(block)) for block in blocks))
            partitions.append(normalized)
            return

        head = remaining[0]
        for offset in range(1, len(remaining)):
            pair = (head, remaining[offset])
            tail = remaining[1:offset] + remaining[offset + 1 :]
            blocks.append(pair)
            _recurse(tail, blocks)
            blocks.pop()

    _recurse(ordered, [])
    deduped: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for partition in partitions:
        if partition in seen:
            continue
        seen.add(partition)
        deduped.append(partition)
    return deduped


def enumerate_surjective_binary_mappings(n_constituents: int) -> list[tuple[int, ...]]:
    """Enumerate all surjective binary state-to-state mappings for `n_constituents` bits."""

    if n_constituents <= 0:
        raise ValueError("n_constituents must be positive.")

    n_micro_states = 2 ** n_constituents
    mappings = [
        tuple(int(bit) for bit in labels)
        for labels in product((0, 1), repeat=n_micro_states)
        if any(bit == 0 for bit in labels) and any(bit == 1 for bit in labels)
    ]
    return mappings


def enumerate_partitions_fixed_blocks(
    indices: Sequence[int],
    n_blocks: int,
) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate all set partitions of `indices` into exactly `n_blocks` blocks."""

    ordered = tuple(int(index) for index in indices)
    if len(set(ordered)) != len(ordered):
        raise ValueError("indices must be unique.")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")
    if n_blocks > len(ordered):
        raise ValueError("n_blocks cannot exceed the number of indices.")
    if not ordered:
        return [tuple()] if n_blocks == 0 else []

    partitions: list[tuple[tuple[int, ...], ...]] = []

    def _recurse(
        position: int,
        blocks: list[list[int]],
    ) -> None:
        if position == len(ordered):
            if len(blocks) == n_blocks:
                normalized = tuple(sorted(tuple(block) for block in blocks))
                partitions.append(normalized)
            return

        remaining = len(ordered) - position
        if len(blocks) > n_blocks or len(blocks) + remaining < n_blocks:
            return

        value = ordered[position]
        for block_index, block in enumerate(blocks):
            block.append(value)
            _recurse(position + 1, blocks)
            block.pop()
            if len(block) == 0:
                del blocks[block_index]
                break

        if len(blocks) < n_blocks:
            blocks.append([value])
            _recurse(position + 1, blocks)
            blocks.pop()

    _recurse(0, [])
    deduped: list[tuple[tuple[int, ...], ...]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for partition in partitions:
        if partition in seen:
            continue
        seen.add(partition)
        deduped.append(partition)
    return deduped


def coarse_grain_tpm_by_state_labels(
    tpm: np.ndarray,
    *,
    input_state_labels: Sequence[int],
    output_state_labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Coarse-grain a TPM by averaging rows and aggregating columns."""

    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")

    input_labels = np.asarray(input_state_labels, dtype=int).reshape(-1)
    if input_labels.shape[0] != matrix.shape[0]:
        raise ValueError("input_state_labels must match the number of TPM rows.")
    output_labels = np.asarray(
        output_state_labels if output_state_labels is not None else input_state_labels,
        dtype=int,
    ).reshape(-1)
    if output_labels.shape[0] != matrix.shape[1]:
        raise ValueError("output_state_labels must match the number of TPM columns.")
    if np.any(input_labels < 0) or np.any(output_labels < 0):
        raise ValueError("State labels must be non-negative integers.")

    n_input_macro = int(input_labels.max()) + 1 if input_labels.size else 0
    macro_rows = np.zeros((n_input_macro, matrix.shape[1]), dtype=float)
    for macro_state in range(n_input_macro):
        member_rows = np.flatnonzero(input_labels == macro_state)
        if len(member_rows) == 0:
            continue
        macro_rows[macro_state] = matrix[member_rows].mean(axis=0)

    return coarse_grain_output_tpm(macro_rows, output_labels)


def coarse_grain_binary_or_tpm(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Coarse-grain a binary system by OR-pooling node groups into macro states."""

    normalized_groups = tuple(
        _normalize_indices(group, n_nodes, "groups block")
        for group in groups
    )
    covered = tuple(sorted(index for group in normalized_groups for index in group))
    if covered != tuple(range(n_nodes)):
        raise ValueError("groups must form a partition of all nodes.")
    if len(set(covered)) != len(covered):
        raise ValueError("groups must be disjoint.")

    states = enumerate_binary_states(n_nodes)
    labels = np.zeros(len(states), dtype=int)
    for row_index, state in enumerate(states):
        macro_bits = [int(np.any(state[list(group)])) for group in normalized_groups]
        labels[row_index] = int(_encode_binary_states(np.asarray(macro_bits, dtype=int).reshape(1, -1))[0])

    return coarse_grain_tpm_by_state_labels(system_tpm, input_state_labels=labels)


def effective_information_from_tpm(
    tpm: np.ndarray,
    *,
    log_base: float = 2.0,
    atol: float = 1e-12,
) -> float:
    """Compute effective information from a TPM under a maximum-entropy input.

    The TPM is assumed to be row-stochastic, where rows correspond to input states
    and columns correspond to output states. The input intervention is the uniform
    distribution over all input states, which is the maximum-entropy distribution
    for a finite discrete state space.
    """

    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("TPM must be non-empty.")
    if np.any(matrix < -atol):
        raise ValueError("TPM entries must be non-negative.")

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Each TPM row must sum to 1.")

    matrix = np.clip(matrix, 0.0, None)
    n_inputs = matrix.shape[0]
    output_marginal = matrix.mean(axis=0)

    positive_mask = matrix > 0.0
    denominator = output_marginal[np.newaxis, :]
    ratios = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > 0.0,
    )
    log_term = np.zeros_like(matrix)
    log_term[positive_mask] = np.log(ratios[positive_mask]) / math.log(log_base)

    ei = np.sum(matrix[positive_mask] * log_term[positive_mask]) / n_inputs
    return float(ei)


def source_subset_tpm(
    system_tpm: np.ndarray,
    n_nodes: int,
    source_indices: Sequence[int],
) -> np.ndarray:
    """Marginalize a full system TPM to a source subset under uniform intervention."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    indices = tuple(sorted(source_indices))
    if not indices:
        raise ValueError("source_indices must be non-empty.")
    if len(set(indices)) != len(indices):
        raise ValueError("source_indices must be unique.")
    if any(i < 0 or i >= n_nodes for i in indices):
        raise ValueError("source_indices contain out-of-range values.")

    full_states = enumerate_binary_states(n_nodes)
    subset_states = full_states[:, indices]
    subset_ids = _encode_binary_states(subset_states)
    n_subset_states = 2 ** len(indices)
    averaged_tpm = np.zeros((n_subset_states, system.shape[1]), dtype=float)

    for row, subset_id in enumerate(subset_ids):
        averaged_tpm[subset_id] += system[row]

    averaged_tpm /= 2 ** (n_nodes - len(indices))
    return averaged_tpm


def coarse_grain_output_tpm(
    tpm: np.ndarray,
    output_labels: Sequence[int],
) -> np.ndarray:
    """Aggregate TPM columns according to a discrete output labeling."""

    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")

    labels = np.asarray(output_labels, dtype=int).reshape(-1)
    if labels.shape[0] != matrix.shape[1]:
        raise ValueError("output_labels must match the number of TPM columns.")

    unique_labels, inverse = np.unique(labels, return_inverse=True)
    coarse_tpm = np.zeros((matrix.shape[0], len(unique_labels)), dtype=float)
    for column_index, label_index in enumerate(inverse):
        coarse_tpm[:, label_index] += matrix[:, column_index]
    return coarse_tpm


def target_subset_tpm(
    system_tpm: np.ndarray,
    n_nodes: int,
    target_indices: Sequence[int],
) -> np.ndarray:
    """Marginalize a full system TPM to a target subset."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    indices = _normalize_indices(target_indices, n_nodes, "target_indices")
    if not indices:
        raise ValueError("target_indices must be non-empty.")

    full_states = enumerate_binary_states(n_nodes)
    target_states = full_states[:, indices]
    target_ids = _encode_binary_states(target_states)
    return coarse_grain_output_tpm(system, target_ids)


def source_target_tpm(
    system_tpm: np.ndarray,
    n_nodes: int,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> np.ndarray:
    """Return the TPM from a source subset to a target subset."""

    indices = _normalize_indices(target_indices, n_nodes, "target_indices")
    if not indices:
        raise ValueError("target_indices must be non-empty.")

    source_tpm = source_subset_tpm(system_tpm, n_nodes=n_nodes, source_indices=source_indices)
    full_states = enumerate_binary_states(n_nodes)
    target_states = full_states[:, indices]
    target_ids = _encode_binary_states(target_states)
    return coarse_grain_output_tpm(source_tpm, target_ids)


def discrete_effective_information(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    log_base: float = 2.0,
) -> float:
    """Compute EI from a source subset to a target subset in a discrete system."""

    return effective_information_from_tpm(
        source_target_tpm(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=source_indices,
            target_indices=target_indices,
        ),
        log_base=log_base,
    )


def discrete_state_backward_repertoire(
    tpm: np.ndarray,
    *,
    current_state_index: int,
) -> np.ndarray:
    """Return the posterior over previous states given a current state."""

    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")
    if current_state_index < 0 or current_state_index >= matrix.shape[1]:
        raise ValueError("current_state_index is out of range.")

    column = np.clip(matrix[:, current_state_index], 0.0, None)
    total = float(np.sum(column))
    if total <= 0.0:
        raise ValueError("Current state has zero probability under the uniform intervention.")
    return column / total


def discrete_state_effective_information(
    tpm: np.ndarray,
    *,
    current_state_index: int,
    log_base: float = 2.0,
) -> float:
    """State-dependent effective information under a uniform prior."""

    posterior = discrete_state_backward_repertoire(tpm, current_state_index=current_state_index)
    n_states = posterior.shape[0]
    prior = np.full(n_states, 1.0 / n_states, dtype=float)
    positive = posterior > 0.0
    return float(
        np.sum(
            posterior[positive]
            * (np.log(posterior[positive] / prior[positive]) / math.log(log_base))
        )
    )


def all_set_partitions(indices: Sequence[int]) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate all set partitions of a collection of indices."""

    ordered = tuple(int(index) for index in indices)
    if len(set(ordered)) != len(ordered):
        raise ValueError("indices must be unique.")
    if not ordered:
        return [tuple()]

    partitions: list[tuple[tuple[int, ...], ...]] = []

    def _recurse(remaining: tuple[int, ...], blocks: list[list[int]]) -> None:
        if not remaining:
            normalized = tuple(
                tuple(sorted(block))
                for block in sorted((tuple(block) for block in blocks), key=lambda block: (len(block), block))
            )
            partitions.append(normalized)
            return

        head = remaining[0]
        tail = remaining[1:]
        for block_index in range(len(blocks)):
            blocks[block_index].append(head)
            _recurse(tail, blocks)
            blocks[block_index].pop()
        blocks.append([head])
        _recurse(tail, blocks)
        blocks.pop()

    _recurse(ordered, [])
    deduped: list[tuple[tuple[int, ...], ...]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for partition in partitions:
        if partition in seen:
            continue
        seen.add(partition)
        deduped.append(partition)
    return deduped


def discrete_partitioned_effective_information(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    subset_indices: Sequence[int],
    current_state: Sequence[int],
    partition: Sequence[Sequence[int]],
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute partitioned effective information for a subset and state."""

    subset = _normalize_indices(subset_indices, n_nodes, "subset_indices")
    if not subset:
        raise ValueError("subset_indices must be non-empty.")

    partition_blocks = tuple(
        _normalize_indices(block, n_nodes, "partition block")
        for block in partition
    )
    covered = tuple(sorted(index for block in partition_blocks for index in block))
    if covered != tuple(subset):
        raise ValueError("partition must cover the subset exactly.")
    if len(set(covered)) != len(covered):
        raise ValueError("partition blocks must be disjoint.")

    state_bits = tuple(int(bit) for bit in current_state)
    if len(state_bits) != n_nodes:
        raise ValueError("current_state must provide one bit per node.")

    subset_tpm = source_target_tpm(
        system_tpm,
        n_nodes=n_nodes,
        source_indices=subset,
        target_indices=subset,
    )
    subset_state_index = _encode_binary_states(
        np.asarray([state_bits[index] for index in subset], dtype=int).reshape(1, -1)
    )[0]
    whole_posterior = discrete_state_backward_repertoire(
        subset_tpm,
        current_state_index=int(subset_state_index),
    )

    subset_states = enumerate_binary_states(len(subset))
    local_position = {node: position for position, node in enumerate(subset)}
    product_posterior = np.ones(len(subset_states), dtype=float)
    part_posteriors: list[np.ndarray] = []
    for block in partition_blocks:
        part_tpm = source_target_tpm(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=block,
            target_indices=block,
        )
        part_state_index = _encode_binary_states(
            np.asarray([state_bits[index] for index in block], dtype=int).reshape(1, -1)
        )[0]
        part_posterior = discrete_state_backward_repertoire(
            part_tpm,
            current_state_index=int(part_state_index),
        )
        part_posteriors.append(part_posterior)

        block_positions = [local_position[node] for node in block]
        block_ids = _encode_binary_states(subset_states[:, block_positions])
        product_posterior *= part_posterior[block_ids]

    positive = whole_posterior > 0.0
    partition_ei = float(
        np.sum(
            whole_posterior[positive]
            * (np.log(whole_posterior[positive] / product_posterior[positive]) / math.log(log_base))
        )
    )

    entropy_scale = math.log(2.0) / math.log(log_base)
    normalization = (len(partition_blocks) - 1) * min(len(block) for block in partition_blocks) * entropy_scale
    normalized_ei = partition_ei / normalization if normalization > 0 else float("inf")
    return {
        "partition": partition_blocks,
        "partition_ei": partition_ei,
        "normalized_partition_ei": normalized_ei,
        "whole_posterior": whole_posterior,
        "product_posterior": product_posterior,
        "part_posteriors": part_posteriors,
    }


def discrete_integrated_information(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    subset_indices: Sequence[int],
    current_state: Sequence[int],
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute state-dependent integrated information for a discrete subset."""

    subset = _normalize_indices(subset_indices, n_nodes, "subset_indices")
    if not subset:
        raise ValueError("subset_indices must be non-empty.")

    subset_tpm = source_target_tpm(
        system_tpm,
        n_nodes=n_nodes,
        source_indices=subset,
        target_indices=subset,
    )
    state_bits = tuple(int(bit) for bit in current_state)
    subset_state = tuple(state_bits[index] for index in subset)
    subset_state_index = int(
        _encode_binary_states(np.asarray(subset_state, dtype=int).reshape(1, -1))[0]
    )
    whole_ei = discrete_state_effective_information(
        subset_tpm,
        current_state_index=subset_state_index,
        log_base=log_base,
    )

    candidate_partitions = [
        partition for partition in all_set_partitions(subset)
        if len(partition) >= 2
    ]
    if not candidate_partitions:
        return {
            "subset_indices": tuple(subset),
            "subset_state": subset_state,
            "state_index": subset_state_index,
            "whole_ei": whole_ei,
            "phi": 0.0,
            "mip_partition": tuple(),
            "mip_normalized_ei": float("inf"),
            "partition_results": [],
        }

    partition_results = [
        discrete_partitioned_effective_information(
            system_tpm,
            n_nodes=n_nodes,
            subset_indices=tuple(subset),
            current_state=state_bits,
            partition=partition,
            log_base=log_base,
        )
        for partition in candidate_partitions
    ]
    mip = min(
        partition_results,
        key=lambda item: (item["normalized_partition_ei"], item["partition_ei"]),
    )
    return {
        "subset_indices": tuple(subset),
        "subset_state": subset_state,
        "state_index": subset_state_index,
        "whole_ei": whole_ei,
        "phi": float(mip["partition_ei"]),
        "mip_partition": mip["partition"],
        "mip_normalized_ei": float(mip["normalized_partition_ei"]),
        "partition_results": partition_results,
    }


def find_discrete_complexes(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    current_state: Sequence[int],
    log_base: float = 2.0,
) -> list[dict[str, Any]]:
    """Enumerate and rank complexes for a discrete system and current state."""

    state_bits = tuple(int(bit) for bit in current_state)
    if len(state_bits) != n_nodes:
        raise ValueError("current_state must provide one bit per node.")

    all_results = [
        discrete_integrated_information(
            system_tpm,
            n_nodes=n_nodes,
            subset_indices=subset,
            current_state=state_bits,
            log_base=log_base,
        )
        for subset_size in range(1, n_nodes + 1)
        for subset in combinations(range(n_nodes), subset_size)
    ]

    phi_by_subset = {tuple(item["subset_indices"]): float(item["phi"]) for item in all_results}
    complexes: list[dict[str, Any]] = []
    for item in all_results:
        subset = tuple(item["subset_indices"])
        phi = float(item["phi"])
        if phi <= 0.0:
            continue
        if any(
            set(subset).issubset(other_subset) and set(subset) != set(other_subset) and other_phi > phi
            for other_subset, other_phi in phi_by_subset.items()
        ):
            continue
        is_main_complex = all(
            phi_by_subset[other_subset] < phi
            for other_subset in phi_by_subset
            if set(other_subset).issubset(subset) and set(other_subset) != set(subset)
        )
        complexes.append(
            {
                **item,
                "is_main_complex": is_main_complex,
            }
        )

    complexes.sort(
        key=lambda item: (-float(item["phi"]), -len(item["subset_indices"]), tuple(item["subset_indices"]))
    )
    return complexes


def discrete_subset_synergy(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    subset_indices: Sequence[int],
    target_indices: Sequence[int] | None = None,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute a simple source-side EI synergy score for a subset.

    The default target is the same subset at the next time step. The source-side
    synergy is defined against the singleton partition:

        Syn(S -> T) = EI(S -> T) - sum_i EI({i} -> T),   i in S
    """

    subset = tuple(_normalize_indices(subset_indices, n_nodes, "subset_indices"))
    if not subset:
        raise ValueError("subset_indices must be non-empty.")
    target = tuple(_normalize_indices(target_indices or subset, n_nodes, "target_indices"))
    if not target:
        raise ValueError("target_indices must be non-empty.")

    whole_ei = discrete_effective_information(
        system_tpm,
        n_nodes=n_nodes,
        source_indices=subset,
        target_indices=target,
        log_base=log_base,
    )
    singleton_eis = {
        source: discrete_effective_information(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=[source],
            target_indices=target,
            log_base=log_base,
        )
        for source in subset
    }
    synergy = whole_ei - sum(singleton_eis.values())
    return {
        "subset_indices": subset,
        "target_indices": target,
        "whole_ei": float(whole_ei),
        "singleton_eis": singleton_eis,
        "synergy": float(synergy),
    }


def discrete_downward_causation(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    target_index: int,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute the target-specific downward-causation score `DC_j` in Eq. (5.1)."""

    if target_index < 0 or target_index >= n_nodes:
        raise ValueError("target_index must be within `[0, n_nodes)`.")

    target = [int(target_index)]
    all_sources = list(range(n_nodes))
    environment = [index for index in all_sources if index != target_index]

    ei_full = discrete_effective_information(
        system_tpm,
        n_nodes=n_nodes,
        source_indices=all_sources,
        target_indices=target,
        log_base=log_base,
    )
    ei_singles = [
        discrete_effective_information(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=[source],
            target_indices=target,
            log_base=log_base,
        )
        for source in all_sources
    ]
    self_ei = float(ei_singles[target_index])
    environment_ei = (
        discrete_effective_information(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=environment,
            target_indices=target,
            log_base=log_base,
        )
        if environment
        else 0.0
    )
    environment_budget = float(sum(ei_singles[index] for index in environment))
    joint_term = float(ei_full - self_ei - environment_ei)
    environment_synergy = float(environment_ei - environment_budget)
    dc = float(ei_full - sum(ei_singles))

    return {
        "target_index": int(target_index),
        "ei_full": float(ei_full),
        "ei_singles": [float(value) for value in ei_singles],
        "self_ei": self_ei,
        "environment_ei": float(environment_ei),
        "joint_term": joint_term,
        "environment_synergy": environment_synergy,
        "dc": dc,
    }


def discrete_downward_causation_all_targets(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    log_base: float = 2.0,
) -> list[dict[str, Any]]:
    """Compute `DC_j` for every target node in a discrete system."""

    return [
        discrete_downward_causation(
            system_tpm,
            n_nodes=n_nodes,
            target_index=target_index,
            log_base=log_base,
        )
        for target_index in range(n_nodes)
    ]


def discrete_synergy(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    log_base: float = 2.0,
) -> float:
    """Compute source-side synergy as joint EI minus the sum of singleton EIs."""

    sources = _normalize_indices(source_indices, n_nodes, "source_indices")
    if not sources:
        raise ValueError("source_indices must be non-empty.")

    joint_ei = discrete_effective_information(
        system_tpm,
        n_nodes=n_nodes,
        source_indices=sources,
        target_indices=target_indices,
        log_base=log_base,
    )
    single_budget = sum(
        discrete_effective_information(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=[source],
            target_indices=target_indices,
            log_base=log_base,
        )
        for source in sources
    )
    return float(joint_ei - single_budget)


def discrete_causal_graph(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    synergy_order: int = 2,
    target_orders: Sequence[int] | None = None,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Summarize pairwise EI edges and fixed-order source hyperedges for all targets."""

    if synergy_order < 2:
        raise ValueError("synergy_order must be at least 2.")
    if target_orders is None:
        normalized_target_orders = (1,)
    else:
        normalized_target_orders = tuple(sorted({int(order) for order in target_orders}))
        if not normalized_target_orders:
            raise ValueError("target_orders must be non-empty.")
        if any(order < 1 for order in normalized_target_orders):
            raise ValueError("target_orders must be positive integers.")

    system = _validate_system_tpm(system_tpm, n_nodes)
    pairwise_ei = np.zeros((n_nodes, n_nodes), dtype=float)
    for source in range(n_nodes):
        for target in range(n_nodes):
            pairwise_ei[source, target] = discrete_effective_information(
                system,
                n_nodes=n_nodes,
                source_indices=[source],
                target_indices=[target],
                log_base=log_base,
            )

    hyperedges_by_target_order: dict[int, list[dict[str, Any]]] = {}
    for target_order in normalized_target_orders:
        hyperedges: list[dict[str, Any]] = []
        for targets in combinations(range(n_nodes), target_order):
            for sources in combinations(range(n_nodes), synergy_order):
                joint_ei = discrete_effective_information(
                    system,
                    n_nodes=n_nodes,
                    source_indices=sources,
                    target_indices=targets,
                    log_base=log_base,
                )
                single_eis = [
                    float(
                        discrete_effective_information(
                            system,
                            n_nodes=n_nodes,
                            source_indices=[source],
                            target_indices=targets,
                            log_base=log_base,
                        )
                    )
                    for source in sources
                ]
                synergy = float(joint_ei - sum(single_eis))
                row = {
                    "sources": tuple(int(source) for source in sources),
                    "targets": tuple(int(target) for target in targets),
                    "value": synergy,
                    "synergy": synergy,
                    "ei_joint": float(joint_ei),
                    "single_eis": single_eis,
                }
                if target_order == 1:
                    row["target"] = int(targets[0])
                hyperedges.append(row)
        hyperedges_by_target_order[target_order] = hyperedges

    return {
        "pairwise_ei": pairwise_ei,
        "hyperedges": hyperedges_by_target_order.get(1, []),
        "hyperedges_by_target_order": hyperedges_by_target_order,
    }


def sample_tpm_rollout(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    n_steps: int,
    burn_in: int = 0,
    initial_state: Sequence[int] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `(X_t, X_{t+1})` transitions from a discrete system TPM."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")

    rng = np.random.default_rng(seed)
    states = enumerate_binary_states(n_nodes)
    if initial_state is None:
        current_index = int(rng.integers(len(states)))
    else:
        initial = np.asarray(initial_state, dtype=int).reshape(-1)
        if initial.shape[0] != n_nodes:
            raise ValueError("initial_state must have length n_nodes.")
        if not np.all((initial == 0) | (initial == 1)):
            raise ValueError("initial_state must be binary.")
        current_index = int(_encode_binary_states(initial)[0])

    past_states = np.zeros((n_steps, n_nodes), dtype=int)
    future_states = np.zeros((n_steps, n_nodes), dtype=int)
    total_steps = burn_in + n_steps
    for step in range(total_steps):
        next_index = int(rng.choice(len(states), p=system[current_index]))
        if step >= burn_in:
            sample_index = step - burn_in
            past_states[sample_index] = states[current_index]
            future_states[sample_index] = states[next_index]
        current_index = next_index

    return past_states, future_states


def pairwise_observed_mutual_information(
    past_states: np.ndarray,
    future_states: np.ndarray,
    *,
    log_base: float = 2.0,
) -> np.ndarray:
    """Estimate `I(X_t^(i); X_{t+1}^(j))` from sampled state transitions."""

    past = np.asarray(past_states, dtype=int)
    future = np.asarray(future_states, dtype=int)
    if past.ndim != 2 or future.ndim != 2 or past.shape != future.shape:
        raise ValueError("past_states and future_states must be 2D arrays of matching shape.")

    n_samples, n_nodes = past.shape
    if n_samples == 0:
        raise ValueError("Need at least one sampled transition.")

    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for source in range(n_nodes):
        for target in range(n_nodes):
            matrix[source, target] = _empirical_discrete_mutual_information(
                past[:, source],
                future[:, target],
                log_base=log_base,
            )
    return matrix


def observed_mutual_information_graph(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    n_steps: int,
    burn_in: int = 0,
    initial_state: Sequence[int] | None = None,
    seed: int | None = None,
    log_base: float = 2.0,
) -> np.ndarray:
    """Sample a discrete rollout and estimate the pairwise lagged MI matrix."""

    past_states, future_states = sample_tpm_rollout(
        system_tpm,
        n_nodes=n_nodes,
        n_steps=n_steps,
        burn_in=burn_in,
        initial_state=initial_state,
        seed=seed,
    )
    return pairwise_observed_mutual_information(
        past_states,
        future_states,
        log_base=log_base,
    )


def joint_ei_decomposition(
    system_tpm: np.ndarray,
    n_nodes: int,
    *,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute joint-state EI, single-node EI budget, and highest-order synergy."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    ei_full = effective_information_from_tpm(system, log_base=log_base)
    ei_singles = [
        effective_information_from_tpm(
            source_subset_tpm(system, n_nodes=n_nodes, source_indices=[node]),
            log_base=log_base,
        )
        for node in range(n_nodes)
    ]
    un_budget = float(sum(ei_singles))
    syn_high = float(ei_full - un_budget)
    rho_high = 0.0 if abs(ei_full) <= 1e-12 else float(syn_high / ei_full)
    return {
        "ei_full": float(ei_full),
        "ei_singles": ei_singles,
        "un_budget": un_budget,
        "syn_high": syn_high,
        "rho_high": rho_high,
    }


def target_ei_decomposition(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute target-wise EI budgets together with total source-side synergy."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    ei_full = effective_information_from_tpm(system, log_base=log_base)
    targets: list[dict[str, Any]] = []
    syn_budget = 0.0
    total_target_ei = 0.0
    all_sources = list(range(n_nodes))

    for target_index in range(n_nodes):
        target = [target_index]
        full_target_ei = discrete_effective_information(
            system,
            n_nodes=n_nodes,
            source_indices=all_sources,
            target_indices=target,
            log_base=log_base,
        )
        single_target_eis = [
            discrete_effective_information(
                system,
                n_nodes=n_nodes,
                source_indices=[source_index],
                target_indices=target,
                log_base=log_base,
            )
            for source_index in range(n_nodes)
        ]
        synergy = float(full_target_ei - sum(single_target_eis))
        targets.append(
            {
                "target_index": int(target_index),
                "full_ei": float(full_target_ei),
                "single_eis": [float(value) for value in single_target_eis],
                "synergy": synergy,
            }
        )
        syn_budget += synergy
        total_target_ei += float(full_target_ei)

    # This remains a target-wise auxiliary summary, not the system-level Syn used in RQ3.
    rho_syn = 0.0 if abs(ei_full) <= 1e-12 else float(syn_budget / ei_full)
    return {
        "ei_full": float(ei_full),
        "target_ei_sum": float(total_target_ei),
        "syn_budget": float(syn_budget),
        "rho_syn": rho_syn,
        "targets": targets,
    }


def system_ei_decomposition(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Compute system-level Syn using the full next-state as the target."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    ei_full = effective_information_from_tpm(system, log_base=log_base)
    single_eis = [
        effective_information_from_tpm(
            source_subset_tpm(system, n_nodes=n_nodes, source_indices=[source_index]),
            log_base=log_base,
        )
        for source_index in range(n_nodes)
    ]
    syn = float(ei_full - sum(single_eis))
    rho_syn = 0.0 if abs(ei_full) <= 1e-12 else float(syn / ei_full)
    return {
        "ei_full": float(ei_full),
        "single_eis": [float(value) for value in single_eis],
        "syn": syn,
        "rho_syn": rho_syn,
    }


def search_binary_or_pair_coarse_grainings(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    log_base: float = 2.0,
) -> list[dict[str, Any]]:
    """Search all pairwise OR coarse-grainings of an even-sized binary system."""

    if n_nodes % 2 != 0:
        raise ValueError("search_binary_or_pair_coarse_grainings requires an even number of nodes.")

    results: list[dict[str, Any]] = []
    for groups in enumerate_pair_partitions(range(n_nodes)):
        macro_tpm = coarse_grain_binary_or_tpm(
            system_tpm,
            n_nodes=n_nodes,
            groups=groups,
        )
        summary = target_ei_decomposition(
            macro_tpm,
            n_nodes=len(groups),
            log_base=log_base,
        )
        system_summary = system_ei_decomposition(
            macro_tpm,
            n_nodes=len(groups),
            log_base=log_base,
        )
        results.append(
            {
                "groups": groups,
                "macro_tpm": macro_tpm,
                "ei": float(system_summary["ei_full"]),
                "syn": float(system_summary["syn"]),
                "rho_syn": float(system_summary["rho_syn"]),
                "targets": summary["targets"],
                "target_syn_budget": float(summary["syn_budget"]),
            }
        )

    results.sort(
        key=lambda item: (
            -float(item["ei"]),
            -float(item["syn"]),
            -float(item["rho_syn"]),
            tuple(item["groups"]),
        )
    )
    return results


def search_marshall_example1_macro_mappings(
    system_tpm: np.ndarray | None = None,
    *,
    groups: Sequence[Sequence[int]] = ((0, 1), (2, 3)),
    log_base: float = 2.0,
) -> list[dict[str, Any]]:
    """Search all 14 x 14 macro mappings for Marshall et al. (2024) Example 1."""

    micro_tpm = build_marshall_example1_tpm() if system_tpm is None else _validate_system_tpm(system_tpm, 4)
    normalized_groups = tuple(
        tuple(_normalize_indices(group, 4, "groups block"))
        for group in groups
    )
    if len(normalized_groups) != 2:
        raise ValueError("groups must contain exactly two macro blocks.")
    covered = tuple(sorted(index for group in normalized_groups for index in group))
    if covered != (0, 1, 2, 3):
        raise ValueError("groups must partition the four micro nodes.")

    states = enumerate_binary_states(4)
    alpha_ids = _encode_binary_states(states[:, list(normalized_groups[0])])
    beta_ids = _encode_binary_states(states[:, list(normalized_groups[1])])
    mappings = enumerate_surjective_binary_mappings(2)
    paper_mapping = (0, 0, 0, 1)

    results: list[dict[str, Any]] = []
    for alpha_mapping in mappings:
        alpha_bits = np.asarray(alpha_mapping, dtype=int)[alpha_ids]
        for beta_mapping in mappings:
            beta_bits = np.asarray(beta_mapping, dtype=int)[beta_ids]
            macro_labels = _encode_binary_states(np.column_stack([alpha_bits, beta_bits]))
            macro_tpm = coarse_grain_tpm_by_state_labels(
                micro_tpm,
                input_state_labels=macro_labels,
                output_state_labels=macro_labels,
            )
            system_summary = system_ei_decomposition(
                macro_tpm,
                n_nodes=2,
                log_base=log_base,
            )
            results.append(
                {
                    "groups": normalized_groups,
                    "alpha_mapping": tuple(alpha_mapping),
                    "beta_mapping": tuple(beta_mapping),
                    "paper_mapping": bool(
                        tuple(alpha_mapping) == paper_mapping and tuple(beta_mapping) == paper_mapping
                        and normalized_groups == ((0, 1), (2, 3))
                    ),
                    "macro_tpm": macro_tpm,
                    "ei": float(system_summary["ei_full"]),
                    "syn": float(system_summary["syn"]),
                    "rho_syn": float(system_summary["rho_syn"]),
                }
            )

    results.sort(
        key=lambda item: (
            -float(item["ei"]),
            -float(item["syn"]),
            tuple(int(v) for v in item["alpha_mapping"]),
            tuple(int(v) for v in item["beta_mapping"]),
        )
    )
    return results


def search_marshall_example1_all_pair_partitions(
    system_tpm: np.ndarray | None = None,
    *,
    log_base: float = 2.0,
) -> list[dict[str, Any]]:
    """Search Marshall et al. (2024) Example 1 over all 2+2 partitions and mappings."""

    micro_tpm = build_marshall_example1_tpm() if system_tpm is None else _validate_system_tpm(system_tpm, 4)
    results: list[dict[str, Any]] = []
    for groups in enumerate_pair_partitions(range(4)):
        results.extend(
            search_marshall_example1_macro_mappings(
                micro_tpm,
                groups=groups,
                log_base=log_base,
            )
        )

    results.sort(
        key=lambda item: (
            -float(item["ei"]),
            -float(item["syn"]),
            tuple(tuple(int(index) for index in block) for block in item["groups"]),
            tuple(int(v) for v in item["alpha_mapping"]),
            tuple(int(v) for v in item["beta_mapping"]),
        )
    )
    return results


def search_binary_or_fixed_macro_dim_coarse_grainings(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    n_macro: int,
    log_base: float = 2.0,
) -> list[dict[str, Any]]:
    """Search all OR coarse-grainings with a fixed number of macro variables."""

    if n_macro <= 0:
        raise ValueError("n_macro must be positive.")
    if n_macro > n_nodes:
        raise ValueError("n_macro cannot exceed n_nodes.")

    subset_scores = subset_synergy_scores(
        system_tpm,
        n_nodes=n_nodes,
        log_base=log_base,
    )
    results: list[dict[str, Any]] = []
    for groups in enumerate_partitions_fixed_blocks(range(n_nodes), n_macro):
        macro_tpm = coarse_grain_binary_or_tpm(
            system_tpm,
            n_nodes=n_nodes,
            groups=groups,
        )
        summary = target_ei_decomposition(
            macro_tpm,
            n_nodes=len(groups),
            log_base=log_base,
        )
        system_summary = system_ei_decomposition(
            macro_tpm,
            n_nodes=len(groups),
            log_base=log_base,
        )
        compactness = coarse_graining_subset_synergy_compactness(
            groups=groups,
            subset_scores=subset_scores,
        )
        results.append(
            {
                "groups": groups,
                "macro_tpm": macro_tpm,
                "ei": float(system_summary["ei_full"]),
                "syn": float(system_summary["syn"]),
                "rho_syn": float(system_summary["rho_syn"]),
                "compact_intra": float(compactness["intra_score"]),
                "compact_inter": float(compactness["inter_score"]),
                "compact_total": float(compactness["total_score"]),
                "compact_fraction": float(compactness["compact_fraction"]),
                "targets": summary["targets"],
                "target_syn_budget": float(summary["syn_budget"]),
            }
        )

    results.sort(
        key=lambda item: (
            -float(item["ei"]),
            -float(item["syn"]),
            -float(item["rho_syn"]),
            tuple(item["groups"]),
        )
    )
    return results


def planted_module_pack_score(
    *,
    groups: Sequence[Sequence[int]],
    planted_groups: Sequence[Sequence[int]],
    module_scores: Sequence[float],
) -> dict[str, Any]:
    """Score how much planted synergistic module mass is absorbed within coarse blocks."""

    normalized_groups = tuple(tuple(sorted(int(index) for index in group)) for group in groups)
    normalized_planted = tuple(
        tuple(sorted(int(index) for index in group))
        for group in planted_groups
    )
    scores = tuple(float(score) for score in module_scores)
    if len(normalized_planted) != len(scores):
        raise ValueError("planted_groups and module_scores must have the same length.")

    packed_modules: list[bool] = []
    packed_score = 0.0
    for planted_group, score in zip(normalized_planted, scores):
        planted_set = set(planted_group)
        packed = any(planted_set.issubset(set(group)) for group in normalized_groups)
        packed_modules.append(bool(packed))
        if packed:
            packed_score += float(score)

    total_score = float(sum(scores))
    packed_fraction = 0.0 if abs(total_score) <= 1e-12 else float(packed_score / total_score)
    return {
        "packed_score": float(packed_score),
        "packed_fraction": packed_fraction,
        "packed_modules": tuple(packed_modules),
    }


def subset_synergy_scores(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    min_subset_size: int = 2,
    log_base: float = 2.0,
) -> dict[tuple[int, ...], float]:
    """Compute `Phi^EID(X_t^A)` with target restricted to the same future subset `A`."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    if min_subset_size < 1:
        raise ValueError("min_subset_size must be at least 1.")

    scores: dict[tuple[int, ...], float] = {}
    for subset_size in range(max(2, min_subset_size), n_nodes + 1):
        for subset in combinations(range(n_nodes), subset_size):
            target_indices = tuple(int(index) for index in subset)
            joint_ei = float(
                discrete_effective_information(
                    system,
                    n_nodes=n_nodes,
                    source_indices=subset,
                    target_indices=target_indices,
                    log_base=log_base,
                )
            )
            single_budget = float(
                sum(
                    discrete_effective_information(
                        system,
                        n_nodes=n_nodes,
                        source_indices=[source_index],
                        target_indices=target_indices,
                        log_base=log_base,
                    )
                    for source_index in subset
                )
            )
            scores[tuple(int(index) for index in subset)] = float(joint_ei - single_budget)
    return scores


def coarse_graining_subset_synergy_compactness(
    *,
    groups: Sequence[Sequence[int]],
    subset_scores: Mapping[Sequence[int], float],
) -> dict[str, Any]:
    """Split total subset synergy into the groups selected by a coarse-graining and the remainder."""

    normalized_groups = tuple(
        tuple(sorted(int(index) for index in group))
        for group in groups
    )
    normalized_scores = {
        tuple(sorted(int(index) for index in subset)): float(score)
        for subset, score in subset_scores.items()
    }

    selected = set(normalized_groups)
    intra_score = float(sum(score for subset, score in normalized_scores.items() if subset in selected))
    total_score = float(sum(normalized_scores.values()))
    inter_score = float(total_score - intra_score)
    compact_fraction = 0.0 if abs(total_score) <= 1e-12 else float(intra_score / total_score)
    return {
        "intra_score": intra_score,
        "inter_score": inter_score,
        "total_score": total_score,
        "compact_fraction": compact_fraction,
        "selected_subsets": tuple(subset for subset in normalized_groups if subset in normalized_scores),
    }


def solve_stationary_covariance(
    coupling: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Solve the stationary covariance of a stable linear-Gaussian AR(1) system."""

    matrix = np.asarray(coupling, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coupling must be a square matrix.")
    if noise.shape != matrix.shape:
        raise ValueError("noise_covariance must match coupling shape.")
    if not np.allclose(noise, noise.T, atol=atol):
        raise ValueError("noise_covariance must be symmetric.")

    spectral_radius = max(abs(value) for value in np.linalg.eigvals(matrix))
    if spectral_radius >= 1.0 - atol:
        raise ValueError("Linear-Gaussian system must be stable (spectral radius < 1).")

    n_nodes = matrix.shape[0]
    lhs = np.eye(n_nodes * n_nodes) - np.kron(matrix, matrix)
    rhs = noise.reshape(-1, order="F")
    covariance = np.linalg.solve(lhs, rhs).reshape((n_nodes, n_nodes), order="F")
    return 0.5 * (covariance + covariance.T)


def gaussian_conditional_covariance(
    covariance: np.ndarray,
    *,
    target_indices: Sequence[int],
    given_indices: Sequence[int],
    atol: float = 1e-12,
) -> np.ndarray:
    """Return the Gaussian conditional covariance Σ(target | given)."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")

    target = _normalize_indices(target_indices, matrix.shape[0], "target_indices")
    given = _normalize_indices(given_indices, matrix.shape[0], "given_indices")
    target_block = matrix[np.ix_(target, target)]
    if not given:
        return target_block.copy()

    cross = matrix[np.ix_(target, given)]
    given_block = matrix[np.ix_(given, given)]
    solve_term = np.linalg.pinv(given_block, rcond=atol) @ matrix[np.ix_(given, target)]
    conditional = target_block - cross @ solve_term
    return 0.5 * (conditional + conditional.T)


def gaussian_mutual_information(
    covariance: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    log_base: float = math.e,
    atol: float = 1e-12,
) -> float:
    """Return Gaussian mutual information I(source ; target)."""

    matrix = np.asarray(covariance, dtype=float)
    source = _normalize_indices(source_indices, matrix.shape[0], "source_indices")
    target = _normalize_indices(target_indices, matrix.shape[0], "target_indices")
    if not source or not target:
        return 0.0

    source_block = matrix[np.ix_(source, source)]
    conditional = gaussian_conditional_covariance(
        matrix,
        target_indices=source,
        given_indices=target,
        atol=atol,
    )
    return 0.5 * (
        _safe_logdet(source_block, atol=atol) - _safe_logdet(conditional, atol=atol)
    ) / math.log(log_base)


def dimension_averaged_effective_information_for_dynamics(
    dynamics: Callable[[Any], Any],
    *,
    state_dim: int,
    output_covariance: np.ndarray | None,
    intervention_bound: float | Sequence[float] | np.ndarray,
    n_mc_samples: int = 4096,
    seed: int | None = None,
    log_base: float = math.e,
    finite_difference_step: float = 1e-6,
    backend: str = "auto",
    atol: float = 1e-12,
) -> float:
    """Estimate dimension-averaged EI for a square continuous dynamics map.

    The callable can be a plain Python/numpy function or a torch-callable object
    such as ``torch.nn.Module``. The map is treated as the mean function of a
    conditional Gaussian output model with the explicitly provided covariance.
    """

    if state_dim <= 0:
        raise ValueError("state_dim must be positive.")
    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be positive.")

    covariance = _validate_explicit_output_covariance(
        output_covariance,
        state_dim=state_dim,
        atol=atol,
    )
    lower, upper = _normalize_intervention_bounds(intervention_bound, state_dim=state_dim)
    widths = upper - lower
    if np.any(widths <= 0.0):
        raise ValueError("Each intervention interval must have positive width.")

    rng = np.random.default_rng(seed)
    samples = rng.uniform(lower, upper, size=(n_mc_samples, state_dim))
    jacobian_backend = _select_dynamics_backend(dynamics, state_dim=state_dim, backend=backend)
    logdet_sum = 0.0
    for sample in samples:
        jacobian = _compute_dynamics_jacobian(
            dynamics,
            sample,
            state_dim=state_dim,
            backend=jacobian_backend,
            finite_difference_step=finite_difference_step,
            atol=atol,
        )
        sign, logabsdet = np.linalg.slogdet(jacobian)
        if sign == 0:
            raise ValueError("Dynamics Jacobian is singular at a sampled intervention point.")
        logdet_sum += float(logabsdet)

    log_input_volume = float(np.log(widths).sum())
    mean_logdet = logdet_sum / float(n_mc_samples)
    noise_entropy = 0.5 * (
        state_dim * math.log(2.0 * math.pi * math.e)
        + _safe_logdet(covariance, atol=atol)
    )
    ei = log_input_volume + mean_logdet - noise_entropy
    return float(ei / state_dim / math.log(log_base))


def dimension_averaged_causal_emergence_for_dynamics(
    *,
    macro_dynamics: Callable[[Any], Any],
    micro_dynamics: Callable[[Any], Any],
    macro_dim: int,
    micro_dim: int,
    macro_output_covariance: np.ndarray | None,
    micro_output_covariance: np.ndarray | None,
    intervention_bound: float | Sequence[float] | np.ndarray | None = None,
    macro_intervention_bound: float | Sequence[float] | np.ndarray | None = None,
    micro_intervention_bound: float | Sequence[float] | np.ndarray | None = None,
    n_mc_samples: int = 4096,
    seed: int | None = None,
    log_base: float = math.e,
    finite_difference_step: float = 1e-6,
    backend: str = "auto",
    atol: float = 1e-12,
) -> float:
    """Compute dCE = dEI(macro) - dEI(micro) for two square dynamics maps."""

    shared_bound = intervention_bound
    macro_bound = macro_intervention_bound if macro_intervention_bound is not None else shared_bound
    micro_bound = micro_intervention_bound if micro_intervention_bound is not None else shared_bound
    if macro_bound is None or micro_bound is None:
        raise ValueError("An intervention bound must be provided for both macro and micro dynamics.")

    macro_ei = dimension_averaged_effective_information_for_dynamics(
        macro_dynamics,
        state_dim=macro_dim,
        output_covariance=macro_output_covariance,
        intervention_bound=macro_bound,
        n_mc_samples=n_mc_samples,
        seed=seed,
        log_base=log_base,
        finite_difference_step=finite_difference_step,
        backend=backend,
        atol=atol,
    )
    micro_ei = dimension_averaged_effective_information_for_dynamics(
        micro_dynamics,
        state_dim=micro_dim,
        output_covariance=micro_output_covariance,
        intervention_bound=micro_bound,
        n_mc_samples=n_mc_samples,
        seed=seed,
        log_base=log_base,
        finite_difference_step=finite_difference_step,
        backend=backend,
        atol=atol,
    )
    return float(macro_ei - micro_ei)


def jacobian_uniform_box_total_synergy_for_dynamics(
    dynamics: Callable[[Any], Any],
    *,
    state_dim: int,
    source_partition: Sequence[Sequence[int]],
    target_indices: Sequence[int],
    output_covariance: np.ndarray | None,
    intervention_bound: float | Sequence[float] | np.ndarray,
    n_mc_samples: int = 4096,
    seed: int | None = None,
    log_base: float = math.e,
    finite_difference_step: float = 1e-6,
    backend: str = "auto",
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Estimate the Jacobian-volume total synergy under a uniform-box intervention.

    This implements the Monte Carlo analogue of the geometric-gain expression in
    Appendix D: the whole source block is compared against the product over the
    blocks in ``source_partition``. Effective target noise for each block is
    evaluated pointwise: for every sampled state, the omitted intervention
    directions contribute a local covariance term under linearization, and that
    pointwise effective noise enters the log-pseudo-determinant before the final
    Monte Carlo average is taken.
    """

    if state_dim <= 0:
        raise ValueError("state_dim must be positive.")
    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be positive.")
    if not source_partition:
        raise ValueError("source_partition must contain at least one block.")

    covariance = _validate_explicit_output_covariance(
        output_covariance,
        state_dim=state_dim,
        atol=atol,
    )
    targets = _normalize_indices(target_indices, state_dim, "target_indices")
    if not targets:
        raise ValueError("target_indices must contain at least one entry.")

    normalized_partition = [
        _normalize_indices(block, state_dim, "source_partition")
        for block in source_partition
    ]
    for block in normalized_partition:
        if not block:
            raise ValueError("Each source_partition block must be non-empty.")
    if len({index for block in normalized_partition for index in block}) != sum(len(block) for block in normalized_partition):
        raise ValueError("source_partition blocks must be disjoint.")

    whole_sources = sorted({index for block in normalized_partition for index in block})
    if not whole_sources:
        raise ValueError("source_partition must cover at least one source index.")

    lower, upper = _normalize_intervention_bounds(intervention_bound, state_dim=state_dim)
    widths = upper - lower
    if np.any(widths <= 0.0):
        raise ValueError("Each intervention interval must have positive width.")
    intervention_variances = (widths ** 2) / 12.0

    rng = np.random.default_rng(seed)
    samples = rng.uniform(lower, upper, size=(n_mc_samples, state_dim))
    jacobian_backend = _select_dynamics_backend(dynamics, state_dim=state_dim, backend=backend)
    jacobians = np.stack(
        [
            _compute_dynamics_jacobian(
                dynamics,
                sample,
                state_dim=state_dim,
                backend=jacobian_backend,
                finite_difference_step=finite_difference_step,
                atol=atol,
            )[np.ix_(targets, list(range(state_dim)))]
            for sample in samples
        ],
        axis=0,
    )

    target_noise = covariance[np.ix_(targets, targets)]

    def effective_noise_for_subset(subset: Sequence[int]) -> np.ndarray:
        omitted = [index for index in range(state_dim) if index not in subset]
        effective = np.repeat(target_noise[None, :, :], n_mc_samples, axis=0)
        if omitted:
            omitted_variance_matrix = np.diag(intervention_variances[omitted])
            for index, jacobian in enumerate(jacobians):
                omitted_block = jacobian[:, omitted]
                effective[index] += omitted_block @ omitted_variance_matrix @ omitted_block.T
        return 0.5 * (effective + np.swapaxes(effective, 1, 2))

    def log_pdet_metric_for_subset(subset: Sequence[int], effective_noise: np.ndarray) -> np.ndarray:
        values = np.empty(n_mc_samples, dtype=float)
        for index, jacobian in enumerate(jacobians):
            block = jacobian[:, subset]
            noise_precision = np.linalg.pinv(effective_noise[index], rcond=atol)
            gram = block.T @ noise_precision @ block
            gram = 0.5 * (gram + gram.T)
            eigenvalues = np.linalg.eigvalsh(gram)
            positive = eigenvalues[eigenvalues > atol]
            if positive.size == 0:
                # The pseudo-determinant of a zero-rank positive semidefinite
                # matrix is the empty product, i.e. 1, so its log contribution is 0.
                values[index] = 0.0
            else:
                values[index] = float(np.log(np.clip(positive, atol, None)).sum())
        return values

    whole_effective_noise = effective_noise_for_subset(whole_sources)
    whole_log_pdet = log_pdet_metric_for_subset(whole_sources, whole_effective_noise)

    part_rows: list[dict[str, Any]] = []
    part_log_total = np.zeros(n_mc_samples, dtype=float)
    for block in normalized_partition:
        block_effective_noise = effective_noise_for_subset(block)
        block_log_pdet = log_pdet_metric_for_subset(block, block_effective_noise)
        part_log_total += block_log_pdet
        part_rows.append(
            {
                "sources": tuple(block),
                "effective_noise": block_effective_noise.mean(axis=0),
                "mean_log_pdet": float(block_log_pdet.mean()),
            }
        )

    syn_total = 0.5 * float(np.mean(whole_log_pdet - part_log_total)) / math.log(log_base)
    return {
        "syn_total": float(syn_total),
        "whole_sources": tuple(whole_sources),
        "target_indices": tuple(targets),
        "mean_log_pdet_whole": float(whole_log_pdet.mean()),
        "mean_log_pdet_parts_sum": float(part_log_total.mean()),
        "whole_effective_noise": whole_effective_noise.mean(axis=0),
        "part_terms": part_rows,
        "n_mc_samples": int(n_mc_samples),
    }


def linear_gaussian_effective_information(
    coupling: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int] | None = None,
    intervention_scale: float = 1.0,
    log_base: float = math.e,
    atol: float = 1e-12,
) -> float:
    """Compute EI for a linear-Gaussian mechanism under isotropic maximum-entropy proxy input."""

    matrix = np.asarray(coupling, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coupling must be a square matrix.")
    if noise.shape != matrix.shape:
        raise ValueError("noise_covariance must match coupling shape.")

    n_nodes = matrix.shape[0]
    sources = _normalize_indices(source_indices, n_nodes, "source_indices")
    targets = (
        list(range(n_nodes))
        if target_indices is None
        else _normalize_indices(target_indices, n_nodes, "target_indices")
    )
    if not sources:
        return 0.0

    complement = [idx for idx in range(n_nodes) if idx not in sources]
    intervention_variance = (float(intervention_scale) ** 2) / 12.0
    signal_map = matrix[np.ix_(targets, sources)]
    signal_cov = intervention_variance * signal_map @ signal_map.T

    effective_noise = noise[np.ix_(targets, targets)].copy()
    if complement:
        complement_map = matrix[np.ix_(targets, complement)]
        effective_noise += intervention_variance * complement_map @ complement_map.T

    target_dim = len(targets)
    information_matrix = np.eye(target_dim) + np.linalg.pinv(effective_noise, rcond=atol) @ signal_cov
    return 0.5 * _safe_logdet(information_matrix, atol=atol) / math.log(log_base)


def linear_gaussian_ei_decomposition(
    coupling: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    intervention_scale: float = 1.0,
    log_base: float = math.e,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compute joint EI, single-source EI budget, and highest-order synergy for a linear-Gaussian system."""

    matrix = np.asarray(coupling, dtype=float)
    n_nodes = matrix.shape[0]
    ei_full = linear_gaussian_effective_information(
        matrix,
        noise_covariance,
        source_indices=list(range(n_nodes)),
        target_indices=list(range(n_nodes)),
        intervention_scale=intervention_scale,
        log_base=log_base,
        atol=atol,
    )
    ei_singles = [
        linear_gaussian_effective_information(
            matrix,
            noise_covariance,
            source_indices=[node],
            target_indices=list(range(n_nodes)),
            intervention_scale=intervention_scale,
            log_base=log_base,
            atol=atol,
        )
        for node in range(n_nodes)
    ]
    un_budget = float(sum(ei_singles))
    syn_high = float(ei_full - un_budget)
    rho_high = 0.0 if abs(ei_full) <= atol else float(syn_high / ei_full)
    return {
        "ei_full": float(ei_full),
        "ei_singles": ei_singles,
        "un_budget": un_budget,
        "syn_high": syn_high,
        "rho_high": rho_high,
    }


def linear_gaussian_uniform_box_effective_information(
    coupling: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int] | None = None,
    box_size: float = math.sqrt(12.0),
    log_base: float = math.e,
    atol: float = 1e-12,
) -> float:
    """Approximate EI under a uniform-box intervention for a linear-Gaussian mechanism.

    This follows the high-SNR pseudodeterminant-style approximation discussed in the
    Gaussian iterative systems literature. The intervention is assumed uniform over
    `[-box_size/2, box_size/2]^k` on the selected source coordinates.
    """

    matrix = np.asarray(coupling, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coupling must be a square matrix.")
    if noise.shape != matrix.shape:
        raise ValueError("noise_covariance must match coupling shape.")
    if float(box_size) <= 0.0:
        raise ValueError("box_size must be positive.")

    n_nodes = matrix.shape[0]
    sources = _normalize_indices(source_indices, n_nodes, "source_indices")
    targets = (
        list(range(n_nodes))
        if target_indices is None
        else _normalize_indices(target_indices, n_nodes, "target_indices")
    )
    if not sources:
        return 0.0

    complement = [idx for idx in range(n_nodes) if idx not in sources]
    intervention_variance = (float(box_size) ** 2) / 12.0
    signal_map = matrix[np.ix_(targets, sources)]

    effective_noise = noise[np.ix_(targets, targets)].copy()
    if complement:
        complement_map = matrix[np.ix_(targets, complement)]
        effective_noise += intervention_variance * complement_map @ complement_map.T

    gram = signal_map.T @ np.linalg.pinv(effective_noise, rcond=atol) @ signal_map
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    positive = eigenvalues[eigenvalues > atol]
    if positive.size == 0:
        return 0.0

    source_dim = len(sources)
    ei = (
        source_dim * math.log(float(box_size))
        - 0.5 * source_dim * math.log(2.0 * math.pi * math.e)
        + 0.5 * float(np.log(positive).sum())
    ) / math.log(log_base)
    return float(ei)


def linear_gaussian_uniform_box_ei_decomposition(
    coupling: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    box_size: float = math.sqrt(12.0),
    log_base: float = math.e,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compute EI decomposition under a uniform-box intervention approximation."""

    matrix = np.asarray(coupling, dtype=float)
    n_nodes = matrix.shape[0]
    ei_full = linear_gaussian_uniform_box_effective_information(
        matrix,
        noise_covariance,
        source_indices=list(range(n_nodes)),
        target_indices=list(range(n_nodes)),
        box_size=box_size,
        log_base=log_base,
        atol=atol,
    )
    ei_singles = [
        linear_gaussian_uniform_box_effective_information(
            matrix,
            noise_covariance,
            source_indices=[node],
            target_indices=list(range(n_nodes)),
            box_size=box_size,
            log_base=log_base,
            atol=atol,
        )
        for node in range(n_nodes)
    ]
    un_budget = float(sum(ei_singles))
    syn_high = float(ei_full - un_budget)
    rho_high = 0.0 if abs(ei_full) <= atol else float(syn_high / ei_full)
    return {
        "ei_full": float(ei_full),
        "ei_singles": ei_singles,
        "un_budget": un_budget,
        "syn_high": syn_high,
        "rho_high": rho_high,
    }


def linear_gaussian_mediano_metrics(
    coupling: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    log_base: float = math.e,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compute a first-pass subset of Mediano-style linear-Gaussian benchmark metrics."""

    matrix = np.asarray(coupling, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    state_cov = solve_stationary_covariance(matrix, noise, atol=atol)
    n_nodes = matrix.shape[0]
    lag_cov = state_cov @ matrix.T
    joint_cov = np.block(
        [
            [state_cov, lag_cov],
            [lag_cov.T, state_cov],
        ]
    )

    current = list(range(n_nodes))
    future = [n_nodes + idx for idx in range(n_nodes)]
    tdmi = gaussian_mutual_information(
        joint_cov,
        source_indices=current,
        target_indices=future,
        log_base=log_base,
        atol=atol,
    )
    decomposition = linear_gaussian_ei_decomposition(
        matrix,
        noise,
        intervention_scale=1.0,
        log_base=log_base,
        atol=atol,
    )

    phi_values = []
    phi_tilde_values = []
    psi_values = []
    for left, right in _even_bipartitions(n_nodes):
        left_future = [n_nodes + idx for idx in left]
        right_future = [n_nodes + idx for idx in right]

        mi_left = gaussian_mutual_information(
            joint_cov,
            source_indices=left,
            target_indices=left_future,
            log_base=log_base,
            atol=atol,
        )
        mi_right = gaussian_mutual_information(
            joint_cov,
            source_indices=right,
            target_indices=right_future,
            log_base=log_base,
            atol=atol,
        )
        phi_values.append(tdmi - mi_left - mi_right)

        cond_full = gaussian_conditional_covariance(
            joint_cov,
            target_indices=current,
            given_indices=future,
            atol=atol,
        )
        cond_left = gaussian_conditional_covariance(
            joint_cov,
            target_indices=left,
            given_indices=left_future,
            atol=atol,
        )
        cond_right = gaussian_conditional_covariance(
            joint_cov,
            target_indices=right,
            given_indices=right_future,
            atol=atol,
        )
        phi_tilde_values.append(
            0.5
            * (
                _safe_logdet(cond_left, atol=atol)
                + _safe_logdet(cond_right, atol=atol)
                - _safe_logdet(cond_full, atol=atol)
            )
            / math.log(log_base)
        )

        mi_left_whole = gaussian_mutual_information(
            joint_cov,
            source_indices=left,
            target_indices=future,
            log_base=log_base,
            atol=atol,
        )
        mi_right_whole = gaussian_mutual_information(
            joint_cov,
            source_indices=right,
            target_indices=future,
            log_base=log_base,
            atol=atol,
        )
        psi_values.append(tdmi - max(mi_left_whole, mi_right_whole))

    correlation = _covariance_to_correlation(state_cov, atol=atol)
    off_diagonal = np.abs(correlation - np.eye(n_nodes))
    mean_abs_corr = 0.0
    if n_nodes > 1:
        mean_abs_corr = float(off_diagonal.sum() / (n_nodes * (n_nodes - 1)))

    causal_density = 0.0
    if n_nodes > 1:
        terms = []
        for source in range(n_nodes):
            for target in range(n_nodes):
                if source == target:
                    continue
                given = [idx for idx in current if idx != source]
                terms.append(
                    _gaussian_conditional_mutual_information(
                        joint_cov,
                        source_indices=[source],
                        target_indices=[n_nodes + target],
                        given_indices=given,
                        log_base=log_base,
                        atol=atol,
                    )
                )
        causal_density = float(sum(terms) / len(terms))

    return {
        "ei_full": decomposition["ei_full"],
        "ei_singles": decomposition["ei_singles"],
        "un_budget": decomposition["un_budget"],
        "syn_high": decomposition["syn_high"],
        "rho_high": decomposition["rho_high"],
        "phi": float(min(phi_values) if phi_values else 0.0),
        "phi_tilde": float(min(phi_tilde_values) if phi_tilde_values else 0.0),
        "psi": float(min(psi_values) if psi_values else 0.0),
        "causal_density": causal_density,
        "tdmi": float(tdmi),
        "mean_abs_corr": mean_abs_corr,
        "n_even_bipartitions": len(phi_values),
    }


def render_ground_truth_causal_graph_svg(
    title: str,
    *,
    n_nodes: int,
    directed_edges: Sequence[tuple[int, int, str]] | Sequence[tuple[int, int]],
    hyperedges: Sequence[Mapping[str, Any]] | None = None,
    node_labels: Sequence[str] | None = None,
    subtitle: str = "Ground-truth mechanism",
    width: int = 620,
    height: int = 280,
    pairwise_width_base: float = 1.0,
    pairwise_width_scale: float = 2.6,
    hyperedge_stroke_width: float = 1.8,
    hyperedge_target_stroke_width: float = 2.0,
    arrow_marker_width: float = 8.0,
    arrow_marker_height: float = 6.0,
    arrow_marker_ref_x: float = 7.0,
    arrow_marker_ref_y: float = 3.0,
    pairwise_edge_color: str | None = None,
    default_hyperedge_color: str | None = None,
    pairwise_curvature_scale: float = 0.06,
    pairwise_start_offset: float = 16.0,
    pairwise_end_offset: float = 16.0,
    hyperedge_source_offset: float = 14.0,
    hyperedge_target_offset: float = 16.0,
    hyperedge_junction_gap: float = 8.0,
    hyperedge_junction_x_frac: float = 0.47,
    hyperedge_junction_y_offset: float = 0.0,
    hyperedge_junction_vertical_gap: float = 0.0,
) -> str:
    """Render a ground-truth causal graph with motif labels."""

    pairwise = np.zeros((n_nodes, n_nodes), dtype=float)
    legend_items: list[tuple[str, str]] = []
    for edge in directed_edges:
        if len(edge) == 2:
            source, target = edge
            label = ""
        else:
            source, target, label = edge
        pairwise[int(source), int(target)] = 1.0
        if label:
            legend_items.append(
                (
                    str(label),
                    pairwise_edge_color or default_hyperedge_color or _motif_color(str(label)),
                )
            )
    for edge in hyperedges or ():
        label = str(edge.get("label", "")).strip()
        if label:
            legend_items.append((label, default_hyperedge_color or pairwise_edge_color or _motif_color(label)))

    deduped_legend_items: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for label, color in legend_items:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        deduped_legend_items.append((label, color))

    return render_causal_graph_svg(
        title,
        pairwise_matrix=pairwise,
        hyperedges=hyperedges or (),
        node_labels=node_labels,
        subtitle=subtitle,
        show_edge_values=False,
        show_text_labels=False,
        edge_threshold=0.5,
        hyperedge_threshold=0.0,
        width=width,
        height=height,
        legend_items=deduped_legend_items,
        pairwise_width_base=pairwise_width_base,
        pairwise_width_scale=pairwise_width_scale,
        hyperedge_stroke_width=hyperedge_stroke_width,
        hyperedge_target_stroke_width=hyperedge_target_stroke_width,
        arrow_marker_width=arrow_marker_width,
        arrow_marker_height=arrow_marker_height,
        arrow_marker_ref_x=arrow_marker_ref_x,
        arrow_marker_ref_y=arrow_marker_ref_y,
        pairwise_edge_color=pairwise_edge_color,
        default_hyperedge_color=default_hyperedge_color,
        pairwise_curvature_scale=pairwise_curvature_scale,
        pairwise_start_offset=pairwise_start_offset,
        pairwise_end_offset=pairwise_end_offset,
        hyperedge_source_offset=hyperedge_source_offset,
        hyperedge_target_offset=hyperedge_target_offset,
        hyperedge_junction_gap=hyperedge_junction_gap,
        hyperedge_junction_x_frac=hyperedge_junction_x_frac,
        hyperedge_junction_y_offset=hyperedge_junction_y_offset,
        hyperedge_junction_vertical_gap=hyperedge_junction_vertical_gap,
    )


def render_static_causal_graph_svg(
    title: str,
    *,
    node_labels: Sequence[str],
    directed_edges: Sequence[tuple[int, int, str]] | Sequence[tuple[int, int]] = (),
    hyperedges: Sequence[Mapping[str, Any]] | None = None,
    subtitle: str = "",
    show_hyperedge_labels: bool = True,
    width: int = 620,
    height: int = 340,
) -> str:
    """Render a paper-style single-slice causal graph."""

    n_nodes = len(node_labels)
    if n_nodes == 0:
        raise ValueError("node_labels must be non-empty.")

    cx = width * 0.42
    cy = height * 0.56
    radius = min(width * 0.22, height * 0.31)
    points = []
    for idx in range(n_nodes):
        angle = -math.pi / 2.0 + 2.0 * math.pi * idx / n_nodes
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    legend_items: list[tuple[str, str]] = []
    for edge in directed_edges:
        if len(edge) == 3 and str(edge[2]).strip():
            legend_items.append((str(edge[2]).strip(), _motif_color(str(edge[2]).strip())))
    for edge in hyperedges or ():
        label = str(edge.get("label", "")).strip()
        if label:
            legend_items.append((label, _motif_color(label)))

    deduped_legend_items: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for label, color in legend_items:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        deduped_legend_items.append((label, color))

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs><marker id='static-arrow' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto' markerUnits='strokeWidth'>"
        "<path d='M 0 0 L 8 3 L 0 6 z' fill='#6b7785'/></marker></defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
    ]
    if title:
        pieces.append(
            f"<text x='14' y='22' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>"
        )
    if subtitle:
        pieces.append(
            f"<text x='14' y='40' font-size='11' fill='#555'>{html.escape(subtitle)}</text>"
        )

    # ordinary directed edges
    for edge in directed_edges:
        if len(edge) == 2:
            source, target = edge
        else:
            source, target, _ = edge
        sx, sy = points[int(source)]
        tx, ty = points[int(target)]
        dx = tx - sx
        dy = ty - sy
        length = max(math.hypot(dx, dy), 1e-9)
        ux = dx / length
        uy = dy / length
        start_x = sx + 18.0 * ux
        start_y = sy + 18.0 * uy
        end_x = tx - 20.0 * ux
        end_y = ty - 20.0 * uy
        reverse_exists = any(
            (len(other) >= 2 and int(other[0]) == int(target) and int(other[1]) == int(source))
            for other in directed_edges
            if other is not edge
        )
        if reverse_exists:
            curvature = 0.16 if int(source) < int(target) else -0.16
            pieces.append(
                _svg_curved_path(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    stroke="#adb7c2",
                    stroke_width=1.8,
                    dash="",
                    curvature=curvature,
                    marker_end="url(#static-arrow)",
                )
            )
        else:
            pieces.append(
                f"<line x1='{start_x:.1f}' y1='{start_y:.1f}' x2='{end_x:.1f}' y2='{end_y:.1f}' "
                f"stroke='#adb7c2' stroke-width='1.8' marker-end='url(#static-arrow)'/>"
            )

    # hyperedges
    hyperedge_source_pieces: list[str] = []
    hyperedge_target_pieces: list[str] = []
    hyperedge_junction_pieces: list[str] = []
    for edge in hyperedges or ():
        sources = tuple(int(source) for source in edge.get("sources", ()))
        if len(sources) < 2:
            continue
        target = int(edge["target"])
        label = str(edge.get("label", "")).strip()
        color = _motif_color(label)
        source_points = [points[source] for source in sources]
        target_point = points[target]
        junction_x = sum(point[0] for point in source_points + [target_point]) / (len(source_points) + 1)
        junction_y = sum(point[1] for point in source_points + [target_point]) / (len(source_points) + 1)
        junction_radius = 6.5
        node_radius = 18.0

        for sx, sy in source_points:
            dx = junction_x - sx
            dy = junction_y - sy
            length = max(math.hypot(dx, dy), 1e-9)
            ux = dx / length
            uy = dy / length
            hyperedge_source_pieces.append(
                f"<line x1='{sx + node_radius * ux:.1f}' y1='{sy + node_radius * uy:.1f}' "
                f"x2='{junction_x - junction_radius * ux:.1f}' y2='{junction_y - junction_radius * uy:.1f}' "
                f"stroke='{color}' stroke-width='2.0' stroke-dasharray='5,3' opacity='0.95'/>"
            )
        dx = target_point[0] - junction_x
        dy = target_point[1] - junction_y
        length = max(math.hypot(dx, dy), 1e-9)
        ux = dx / length
        uy = dy / length
        hyperedge_target_pieces.append(
            f"<line x1='{junction_x + junction_radius * ux:.1f}' y1='{junction_y + junction_radius * uy:.1f}' "
            f"x2='{target_point[0] - node_radius * ux:.1f}' y2='{target_point[1] - node_radius * uy:.1f}' "
            f"stroke='{color}' stroke-width='2.2' stroke-dasharray='5,3' marker-end='url(#static-arrow)' opacity='0.95'/>"
        )
        hyperedge_junction_pieces.append(
            f"<circle cx='{junction_x:.1f}' cy='{junction_y:.1f}' r='{junction_radius:.1f}' fill='white' stroke='{color}' stroke-width='2.0'/>"
        )
        if label and show_hyperedge_labels:
            hyperedge_junction_pieces.append(
                f"<text x='{junction_x:.1f}' y='{junction_y - 11.0:.1f}' text-anchor='middle' font-size='10' fill='{color}'>{html.escape(label)}</text>"
            )

    pieces.extend(hyperedge_source_pieces)
    pieces.extend(hyperedge_junction_pieces)
    pieces.extend(hyperedge_target_pieces)

    for idx, (x, y) in enumerate(points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='18' fill='#eef3f7' stroke='#66727f' stroke-width='2.0'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' text-anchor='middle' font-size='12' font-weight='700' fill='#111'>{html.escape(str(node_labels[idx]))}</text>"
        )

    if deduped_legend_items:
        legend_x = width - 135.0
        legend_y = 74.0
        pieces.append(
            f"<rect x='{legend_x - 12:.1f}' y='{legend_y - 20:.1f}' width='118' height='{24 + 18 * len(deduped_legend_items):.1f}' "
            f"rx='8' ry='8' fill='#fbfcfd' stroke='#d7dee6' stroke-width='1'/>"
        )
        pieces.append(
            f"<text x='{legend_x:.1f}' y='{legend_y:.1f}' font-size='11' font-weight='700' fill='#333'>Mechanisms</text>"
        )
        for idx, (label, color) in enumerate(deduped_legend_items, start=1):
            y = legend_y + 16.0 * idx
            upper_label = str(label).upper()
            if "COPY" in upper_label:
                pieces.append(
                    f"<line x1='{legend_x:.1f}' y1='{y - 4:.1f}' x2='{legend_x + 18:.1f}' y2='{y - 4:.1f}' "
                    f"stroke='#adb7c2' stroke-width='2.0' marker-end='url(#static-arrow)'/>"
                )
            else:
                pieces.append(
                    f"<line x1='{legend_x:.1f}' y1='{y - 4:.1f}' x2='{legend_x + 18:.1f}' y2='{y - 4:.1f}' "
                    f"stroke='{color}' stroke-width='2.0' stroke-dasharray='5,3'/>"
                )
            pieces.append(
                f"<text x='{legend_x + 24:.1f}' y='{y:.1f}' font-size='10.5' fill='#333'>{html.escape(label)}</text>"
            )

    pieces.append("</svg>")
    return "".join(pieces)


def render_causal_graph_svg(
    title: str,
    *,
    pairwise_matrix: np.ndarray,
    hyperedges: Sequence[Mapping[str, Any]] | None = None,
    node_labels: Sequence[str] | None = None,
    subtitle: str = "",
    edge_labels: Mapping[tuple[int, int], str] | None = None,
    show_edge_values: bool = False,
    show_text_labels: bool = False,
    edge_threshold: float = 0.0,
    hyperedge_threshold: float = 0.0,
    width: int = 620,
    height: int = 280,
    value_decimals: int = 2,
    legend_items: Sequence[tuple[str, str]] | None = None,
    pairwise_width_base: float = 1.0,
    pairwise_width_scale: float = 2.6,
    hyperedge_stroke_width: float = 1.8,
    hyperedge_target_stroke_width: float = 2.0,
    arrow_marker_width: float = 8.0,
    arrow_marker_height: float = 6.0,
    arrow_marker_ref_x: float = 7.0,
    arrow_marker_ref_y: float = 3.0,
    pairwise_edge_color: str | None = None,
    default_hyperedge_color: str | None = None,
    pairwise_curvature_scale: float = 0.06,
    pairwise_start_offset: float = 16.0,
    pairwise_end_offset: float = 16.0,
    hyperedge_source_offset: float = 14.0,
    hyperedge_target_offset: float = 16.0,
    hyperedge_junction_gap: float = 8.0,
    hyperedge_junction_x_frac: float = 0.52,
    hyperedge_junction_y_offset: float = 0.0,
    hyperedge_junction_vertical_gap: float = 0.0,
) -> str:
    """Render a two-slice causal graph with optional hyperedges."""

    values = np.asarray(pairwise_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("pairwise_matrix must be square.")
    n_nodes = values.shape[0]
    if node_labels is None:
        node_labels = [f"x{idx}" for idx in range(n_nodes)]
    if len(node_labels) != n_nodes:
        raise ValueError("node_labels must match the matrix size.")

    legend_entries = list(legend_items or ())
    legend_reserved_width = 150.0 if legend_entries else 0.0
    left_x = 120.0
    right_x = width - 120.0 - legend_reserved_width
    top_y = 60.0
    bottom_y = height - 30.0
    spacing = (bottom_y - top_y) / max(n_nodes - 1, 1)
    left_points = [(left_x, top_y + idx * spacing) for idx in range(n_nodes)]
    right_points = [(right_x, top_y + idx * spacing) for idx in range(n_nodes)]
    bound = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    hyperedge_values = [abs(float(edge.get("value", edge.get("synergy", 0.0)))) for edge in hyperedges or ()]
    hyper_bound = max(hyperedge_values, default=1.0)
    deferred_labels: list[dict[str, Any]] = []

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid meet'>",
        "<defs>"
        + _svg_arrow_marker(
            "cg-arrow",
            width=arrow_marker_width,
            height=arrow_marker_height,
            ref_x=arrow_marker_ref_x,
            ref_y=arrow_marker_ref_y,
            fill="#5c6773",
        )
        + "</defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
    ]
    if title:
        pieces.append(
            f"<text x='12' y='20' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>"
        )
    if subtitle:
        pieces.append(
            f"<text x='12' y='38' font-size='10.5' fill='#555'>{html.escape(subtitle)}</text>"
        )

    pieces.append(
        f"<text x='{left_x:.1f}' y='38' font-size='11' text-anchor='middle' fill='#555'>sources at t</text>"
    )
    pieces.append(
        f"<text x='{right_x:.1f}' y='38' font-size='11' text-anchor='middle' fill='#555'>targets at t+1</text>"
    )

    for source in range(n_nodes):
        for target in range(n_nodes):
            value = float(values[source, target])
            if abs(value) <= edge_threshold:
                continue
            color = pairwise_edge_color or ("#2c7bb6" if value >= 0 else "#d73027")
            stroke_width = pairwise_width_base + pairwise_width_scale * abs(value) / bound
            start_x, start_y = left_points[source]
            end_x, end_y = right_points[target]
            curvature = pairwise_curvature_scale * (target - source)
            pieces.append(
                _svg_curved_path(
                    start_x + pairwise_start_offset,
                    start_y,
                    end_x - pairwise_end_offset,
                    end_y,
                    stroke=color,
                    stroke_width=stroke_width,
                    dash="",
                    curvature=curvature,
                    marker_end="url(#cg-arrow)",
                )
            )

            if show_text_labels:
                label_parts = []
                if edge_labels and (source, target) in edge_labels:
                    label_parts.append(str(edge_labels[(source, target)]))
                if show_edge_values:
                    label_parts.append(f"{value:.{value_decimals}f}")
            else:
                label_parts = []
            if label_parts:
                mid_x = (start_x + end_x) / 2.0
                mid_y = (start_y + end_y) / 2.0 - 4.0
                deferred_labels.append(
                    {
                        "x": mid_x,
                        "y": mid_y,
                        "text": " | ".join(label_parts),
                        "fill": color,
                    }
                )

    hyperedge_rows: list[dict[str, Any]] = []
    for edge in hyperedges or ():
        sources = tuple(int(source) for source in edge.get("sources", ()))
        if len(sources) < 2:
            continue
        if "targets" in edge:
            targets = tuple(int(target) for target in edge.get("targets", ()))
        else:
            targets = (int(edge["target"]),)
        if not targets:
            continue
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        label = str(edge.get("label", "")).strip()
        opacity = min(max(float(edge.get("opacity", 0.95)), 0.0), 1.0)
        if abs(value) <= hyperedge_threshold and not label:
            continue

        source_points = [left_points[source] for source in sources]
        target_points = [right_points[target] for target in targets]
        junction_x = width * hyperedge_junction_x_frac
        junction_y = (
            sum(point[1] for point in source_points + target_points)
            / (len(source_points) + len(target_points))
        )
        junction_y += hyperedge_junction_y_offset
        color = str(edge.get("color", "")).strip() or default_hyperedge_color or _motif_color(label)
        radius = 4.8 + 7.2 * abs(value) / max(hyper_bound, 1e-9)
        hyperedge_rows.append(
            {
                "source_points": source_points,
                "target_points": target_points,
                "junction_x": junction_x,
                "junction_y": junction_y,
                "color": color,
                "radius": radius,
                "label": label,
                "value": value,
                "opacity": opacity,
            }
        )

    if hyperedge_rows:
        max_radius = max(float(row["radius"]) for row in hyperedge_rows)
        staggered_ys = _stagger_axis_positions(
            [float(row["junction_y"]) for row in hyperedge_rows],
            min_gap=hyperedge_junction_vertical_gap,
            lower=top_y + max_radius,
            upper=bottom_y - max_radius,
        )
        for row, staggered_y in zip(hyperedge_rows, staggered_ys):
            row["junction_y"] = staggered_y

    for row in hyperedge_rows:
        source_points = row["source_points"]
        target_points = row["target_points"]
        junction_x = float(row["junction_x"])
        junction_y = float(row["junction_y"])
        color = str(row["color"])
        opacity = float(row["opacity"])
        for source_point in source_points:
            pieces.append(
                f"<line x1='{source_point[0] + hyperedge_source_offset:.1f}' y1='{source_point[1]:.1f}' "
                f"x2='{junction_x - hyperedge_junction_gap:.1f}' y2='{junction_y:.1f}' "
                f"stroke='{color}' stroke-width='{hyperedge_stroke_width:.1f}' stroke-dasharray='5,3' opacity='{opacity:.2f}'/>"
            )
        for target_point in target_points:
            pieces.append(
                f"<line x1='{junction_x + hyperedge_junction_gap:.1f}' y1='{junction_y:.1f}' "
                f"x2='{target_point[0] - hyperedge_target_offset:.1f}' y2='{target_point[1]:.1f}' "
                f"stroke='{color}' stroke-width='{hyperedge_target_stroke_width:.1f}' stroke-dasharray='5,3' marker-end='url(#cg-arrow)' opacity='{opacity:.2f}'/>"
            )
        radius = float(row["radius"])
        pieces.append(
            f"<circle cx='{junction_x:.1f}' cy='{junction_y:.1f}' r='{radius:.1f}' fill='white' stroke='{color}' stroke-width='1.0' opacity='{opacity:.2f}'/>"
        )
        label_bits = []
        if show_text_labels:
            label_bits = [
                bit
                for bit in [
                    str(row["label"]),
                    f"{float(row['value']):.{value_decimals}f}" if show_edge_values else "",
                ]
                if bit
            ]
        if label_bits:
            deferred_labels.append(
                {
                    "x": junction_x,
                    "y": junction_y - 10.0,
                    "text": " | ".join(label_bits),
                    "fill": color,
                }
            )

    for item in _stagger_svg_labels(
        deferred_labels,
        min_vertical_gap=13.0,
        x_collision_gap=42.0,
        top=48.0,
        bottom=height - 20.0,
    ):
        pieces.append(
            f"<text x='{item['x']:.1f}' y='{item['y']:.1f}' font-size='9.5' text-anchor='middle' fill='{item['fill']}'>"
            f"{html.escape(str(item['text']))}</text>"
        )

    if legend_entries:
        legend_x = right_x + 96.0
        legend_y = 66.0
        for index, (label, color) in enumerate(legend_entries):
            y = legend_y + index * 22.0
            upper_label = str(label).upper()
            if "COPY" in upper_label:
                pieces.append(
                    f"<line x1='{legend_x - 6.0:.1f}' y1='{y:.1f}' x2='{legend_x + 8.0:.1f}' y2='{y:.1f}' "
                    f"stroke='{color}' stroke-width='2.0' marker-end='url(#cg-arrow)'/>"
                )
            else:
                pieces.append(
                    f"<circle cx='{legend_x:.1f}' cy='{y:.1f}' r='5.5' fill='white' stroke='{color}' stroke-width='2'/>"
                )
            pieces.append(
                f"<text x='{legend_x + 12.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='start' fill='#333'>{html.escape(label)}</text>"
            )

    for idx, (x, y) in enumerate(left_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#edf3f8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{idx}</text>"
        )
        pieces.append(
            f"<text x='{x - 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='end' fill='#333'>{html.escape(str(node_labels[idx]))}(t)</text>"
        )

    for idx, (x, y) in enumerate(right_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#f8efe8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{idx}</text>"
        )
        pieces.append(
            f"<text x='{x + 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='start' fill='#333'>{html.escape(str(node_labels[idx]))}(t+1)</text>"
        )

    pieces.append("</svg>")
    return "".join(pieces)


def render_multi_target_hypergraph_svg(
    title: str,
    *,
    n_nodes: int,
    hyperedges: Sequence[Mapping[str, Any]],
    node_labels: Sequence[str] | None = None,
    subtitle: str = "",
    width: int = 700,
    height: int = 320,
    hyperedge_threshold: float = 0.0,
    show_text_labels: bool = False,
    show_edge_values: bool = False,
    value_decimals: int = 2,
) -> str:
    """Render source hyperedges whose targets are subsets with size at least two."""

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    if node_labels is None:
        node_labels = [f"x{idx}" for idx in range(n_nodes)]
    if len(node_labels) != n_nodes:
        raise ValueError("node_labels must match n_nodes.")

    filtered_edges = [
        edge
        for edge in hyperedges
        if len(tuple(int(target) for target in edge.get("targets", ()))) >= 2
        and abs(float(edge.get("value", edge.get("synergy", 0.0)))) > hyperedge_threshold
    ]
    filtered_edges.sort(
        key=lambda edge: -abs(float(edge.get("value", edge.get("synergy", 0.0))))
    )

    left_x = 120.0
    right_x = width - 150.0
    top_y = 72.0
    bottom_y = height - 34.0
    spacing = (bottom_y - top_y) / max(n_nodes - 1, 1)
    left_points = [(left_x, top_y + idx * spacing) for idx in range(n_nodes)]
    right_points = [(right_x, top_y + idx * spacing) for idx in range(n_nodes)]
    palette = ["#2a9d8f", "#e76f51", "#457b9d", "#8d5fd3", "#f4a261", "#6a994e"]
    bound = max(
        [abs(float(edge.get("value", edge.get("synergy", 0.0)))) for edge in filtered_edges],
        default=1.0,
    )

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid meet'>",
        "<defs><marker id='mt-arrow' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto' markerUnits='userSpaceOnUse'>"
        "<path d='M 0 0 L 8 3 L 0 6 z' fill='#5c6773'/></marker></defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='12' y='20' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>",
    ]
    if subtitle:
        pieces.append(
            f"<text x='12' y='38' font-size='10.5' fill='#555'>{html.escape(subtitle)}</text>"
        )
    pieces.append(
        f"<text x='{left_x:.1f}' y='48' font-size='11' text-anchor='middle' fill='#555'>sources at t</text>"
    )
    pieces.append(
        f"<text x='{right_x:.1f}' y='48' font-size='11' text-anchor='middle' fill='#555'>targets at t+1</text>"
    )

    deferred_labels: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(filtered_edges):
        sources = tuple(int(source) for source in edge.get("sources", ()))
        targets = tuple(int(target) for target in edge.get("targets", ()))
        if len(sources) < 2 or len(targets) < 2:
            continue
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        label = str(edge.get("label", "")).strip()
        color = _motif_color(label) if label else palette[edge_index % len(palette)]
        opacity = min(max(float(edge.get("opacity", 0.92)), 0.0), 1.0)
        source_points = [left_points[source] for source in sources]
        target_points = [right_points[target] for target in targets]
        radius = 4.6 + 6.8 * abs(value) / max(bound, 1e-9)

        source_anchor_x = width * (0.37 + 0.03 * edge_index)
        target_anchor_x = width * (0.55 + 0.02 * edge_index)
        center_y = (
            sum(point[1] for point in source_points) + sum(point[1] for point in target_points)
        ) / (len(source_points) + len(target_points))
        center_y += (edge_index - (len(filtered_edges) - 1) / 2.0) * 18.0

        for source_x, source_y in source_points:
            pieces.append(
                f"<line x1='{source_x + 14.0:.1f}' y1='{source_y:.1f}' x2='{source_anchor_x - 8.0:.1f}' y2='{center_y:.1f}' "
                f"stroke='{color}' stroke-width='1.8' stroke-dasharray='5,3' opacity='{opacity:.2f}'/>"
            )
        for target_x, target_y in target_points:
            pieces.append(
                f"<line x1='{source_anchor_x + radius + 2.0:.1f}' y1='{center_y:.1f}' "
                f"x2='{target_x - 24.0:.1f}' y2='{target_y:.1f}' "
                f"stroke='{color}' stroke-width='{1.6 + 2.2 * abs(value) / max(bound, 1e-9):.2f}' "
                f"stroke-dasharray='5,3' marker-end='url(#mt-arrow)' opacity='{opacity:.2f}'/>"
            )

        target_min_y = min(point[1] for point in target_points) - 18.0
        target_max_y = max(point[1] for point in target_points) + 18.0
        pieces.append(
            f"<rect x='{right_x - 30.0:.1f}' y='{target_min_y:.1f}' width='60.0' height='{target_max_y - target_min_y:.1f}' "
            f"rx='11' ry='11' fill='{color}' fill-opacity='0.08' stroke='{color}' stroke-width='1.2' stroke-dasharray='4,3' opacity='{opacity:.2f}'/>"
        )
        pieces.append(
            f"<circle cx='{source_anchor_x:.1f}' cy='{center_y:.1f}' r='{radius:.1f}' fill='white' stroke='{color}' stroke-width='1.1' opacity='{opacity:.2f}'/>"
        )

        if show_text_labels:
            target_text = "{" + ", ".join(str(node_labels[target]) for target in targets) + "}"
            text_bits = [target_text]
            if show_edge_values:
                text_bits.append(f"{value:.{value_decimals}f}")
            deferred_labels.append(
                {
                    "x": (source_anchor_x + target_anchor_x) / 2.0,
                    "y": center_y - 10.0,
                    "text": " | ".join(text_bits),
                    "fill": color,
                }
            )

    for item in _stagger_svg_labels(
        deferred_labels,
        min_vertical_gap=13.0,
        x_collision_gap=42.0,
        top=56.0,
        bottom=height - 18.0,
    ):
        pieces.append(
            f"<text x='{item['x']:.1f}' y='{item['y']:.1f}' font-size='9.5' text-anchor='middle' fill='{item['fill']}'>"
            f"{html.escape(str(item['text']))}</text>"
        )

    for idx, (x, y) in enumerate(left_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#edf3f8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{idx}</text>"
        )
        pieces.append(
            f"<text x='{x - 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='end' fill='#333'>{html.escape(str(node_labels[idx]))}(t)</text>"
        )

    for idx, (x, y) in enumerate(right_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#f8efe8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{idx}</text>"
        )
        pieces.append(
            f"<text x='{x + 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='start' fill='#333'>{html.escape(str(node_labels[idx]))}(t+1)</text>"
        )

    pieces.append("</svg>")
    return "".join(pieces)


def render_coarse_graining_bridge_svg(
    title: str,
    *,
    micro_labels: Sequence[str],
    macro_labels: Sequence[str],
    groups: Sequence[Sequence[int]],
    width: int = 360,
    height: int = 360,
) -> str:
    """Render a bridge diagram that shows which micro nodes merge into each macro node."""

    if len(macro_labels) != len(groups):
        raise ValueError("macro_labels and groups must have the same length.")
    if not micro_labels:
        raise ValueError("micro_labels must be non-empty.")

    micro_x = 86.0
    macro_x = width - 98.0
    top_y = 64.0
    bottom_y = height - 34.0
    micro_spacing = (bottom_y - top_y) / max(len(micro_labels) - 1, 1)
    macro_spacing = (bottom_y - top_y) / max(len(macro_labels) - 1, 1)
    micro_points = [(micro_x, top_y + index * micro_spacing) for index in range(len(micro_labels))]
    macro_points = [(macro_x, top_y + index * macro_spacing) for index in range(len(macro_labels))]

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='12' y='20' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>",
        f"<text x='12' y='38' font-size='10.5' fill='#555'>which micro nodes merge into each macro node</text>",
        f"<text x='{micro_x:.1f}' y='38' font-size='11' text-anchor='middle' fill='#555'>micro nodes</text>",
        f"<text x='{macro_x:.1f}' y='38' font-size='11' text-anchor='middle' fill='#555'>macro nodes</text>",
    ]

    for macro_index, group in enumerate(groups):
        macro_x_point, macro_y_point = macro_points[macro_index]
        for micro_index in group:
            micro_x_point, micro_y_point = micro_points[int(micro_index)]
            pieces.append(
                f"<line x1='{micro_x_point + 18.0:.1f}' y1='{micro_y_point:.1f}' "
                f"x2='{macro_x_point - 20.0:.1f}' y2='{macro_y_point:.1f}' "
                f"stroke='#a7b2bf' stroke-width='2.2' opacity='0.72'/>"
            )

    for index, (x, y) in enumerate(micro_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#edf3f8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{index}</text>"
        )
        pieces.append(
            f"<text x='{x - 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='end' fill='#333'>{html.escape(str(micro_labels[index]))}</text>"
        )

    for index, (x, y) in enumerate(macro_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='16' fill='#f4eefb' stroke='#6b57a5' stroke-width='1.6'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10.5' text-anchor='middle' fill='#111'>{html.escape(str(macro_labels[index]))}</text>"
        )

    pieces.append("</svg>")
    return "".join(pieces)


def render_coarse_graining_comparison_svg(
    title: str,
    *,
    micro_pairwise: np.ndarray,
    micro_labels: Sequence[str],
    macro_labels: Sequence[str],
    groups: Sequence[Sequence[int]],
    macro_pairwise: np.ndarray,
    macro_hyperedges: Sequence[Mapping[str, Any]] | None = None,
    micro_hyperedges: Sequence[Mapping[str, Any]] | None = None,
    width: int = 1180,
    height: int = 820,
) -> str:
    """Render aligned micro/macro causal graphs with coarse-graining on both time slices."""

    micro_values = np.asarray(micro_pairwise, dtype=float)
    macro_values = np.asarray(macro_pairwise, dtype=float)
    if micro_values.ndim != 2 or micro_values.shape[0] != micro_values.shape[1]:
        raise ValueError("micro_pairwise must be square.")
    if macro_values.ndim != 2 or macro_values.shape[0] != macro_values.shape[1]:
        raise ValueError("macro_pairwise must be square.")
    if micro_values.shape[0] != len(micro_labels):
        raise ValueError("micro_labels must match micro_pairwise.")
    if macro_values.shape[0] != len(macro_labels):
        raise ValueError("macro_labels must match macro_pairwise.")
    if len(groups) != len(macro_labels):
        raise ValueError("groups must match macro_labels.")

    top_margin = 36.0
    bottom_margin = 28.0
    macro_top = top_margin + 28.0
    macro_bottom = height * 0.36
    micro_top = height * 0.63
    micro_bottom = height - bottom_margin

    source_x = 116.0
    target_x = width - 150.0
    micro_source_x = source_x
    micro_target_x = target_x
    macro_source_x = source_x
    macro_target_x = target_x

    micro_spacing = (micro_bottom - micro_top) / max(len(micro_labels) - 1, 1)
    macro_spacing = (macro_bottom - macro_top) / max(len(macro_labels) - 1, 1)
    micro_left_points = [(micro_source_x, micro_top + idx * micro_spacing) for idx in range(len(micro_labels))]
    micro_right_points = [(micro_target_x, micro_top + idx * micro_spacing) for idx in range(len(micro_labels))]
    macro_left_points = [(macro_source_x, macro_top + idx * macro_spacing) for idx in range(len(macro_labels))]
    macro_right_points = [(macro_target_x, macro_top + idx * macro_spacing) for idx in range(len(macro_labels))]

    micro_bound = max(float(np.max(np.abs(micro_values))) if micro_values.size else 0.0, 1.0)
    macro_bound = max(float(np.max(np.abs(macro_values))) if macro_values.size else 0.0, 1.0)
    hyperedge_values = [
        abs(float(edge.get("value", edge.get("synergy", 0.0))))
        for edge in (micro_hyperedges or ())
    ] + [
        abs(float(edge.get("value", edge.get("synergy", 0.0))))
        for edge in (macro_hyperedges or ())
    ]
    hyper_bound = max(hyperedge_values, default=1.0)
    macro_palette = [
        ("#d9822b", "#f7e1cf"),
        ("#2f7d63", "#d9eee7"),
        ("#6b57a5", "#e5ddf5"),
        ("#b24c63", "#f6dbe3"),
        ("#3a78b3", "#dce9f6"),
    ]
    macro_colors = [macro_palette[index % len(macro_palette)] for index in range(len(macro_labels))]

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<defs><marker id='cg-arrow' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto' markerUnits='userSpaceOnUse'>"
        "<path d='M 0 0 L 8 3 L 0 6 z' fill='#5c6773'/></marker></defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='12' y='20' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>",
    ]

    for macro_index, group in enumerate(groups):
        stroke_color, fill_color = macro_colors[macro_index]
        left_points = [micro_left_points[int(node)] for node in group]
        right_points = [micro_right_points[int(node)] for node in group]
        left_y_min = min(point[1] for point in left_points) - 16.0
        left_y_max = max(point[1] for point in left_points) + 16.0
        right_y_min = min(point[1] for point in right_points) - 16.0
        right_y_max = max(point[1] for point in right_points) + 16.0
        left_box_x = micro_source_x - 58.0
        left_box_width = 76.0
        right_box_x = micro_target_x - 12.0
        right_box_width = 96.0
        pieces.append(
            f"<rect x='{left_box_x:.1f}' y='{left_y_min:.1f}' width='{left_box_width:.1f}' height='{left_y_max - left_y_min:.1f}' "
            f"rx='10' ry='10' fill='{fill_color}' stroke='{stroke_color}' stroke-width='1.4' opacity='0.36'/>"
        )
        pieces.append(
            f"<rect x='{right_box_x:.1f}' y='{right_y_min:.1f}' width='{right_box_width:.1f}' height='{right_y_max - right_y_min:.1f}' "
            f"rx='10' ry='10' fill='{fill_color}' stroke='{stroke_color}' stroke-width='1.4' opacity='0.36'/>"
        )

    for source in range(len(micro_labels)):
        for target in range(len(micro_labels)):
            value = float(micro_values[source, target])
            if abs(value) <= 0.0:
                continue
            start_x, start_y = micro_left_points[source]
            end_x, end_y = micro_right_points[target]
            color = "#2c7bb6" if value >= 0 else "#d73027"
            stroke_width = 1.0 + 2.6 * abs(value) / micro_bound
            curvature = 0.06 * (target - source)
            pieces.append(
                _svg_curved_path(
                    start_x + 16.0,
                    start_y,
                    end_x - 16.0,
                    end_y,
                    stroke=color,
                    stroke_width=stroke_width,
                    dash="",
                    curvature=curvature,
                    marker_end="url(#cg-arrow)",
                )
            )

    micro_junction_x = micro_source_x + 0.58 * (micro_target_x - micro_source_x)
    for edge in micro_hyperedges or ():
        sources = tuple(int(source) for source in edge.get("sources", ()))
        if len(sources) < 2:
            continue
        target = int(edge["target"])
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        opacity = min(max(float(edge.get("opacity", 0.9)), 0.0), 1.0)
        color = _motif_color(str(edge.get("label", "syn")))
        source_points = [micro_left_points[source] for source in sources]
        target_point = micro_right_points[target]
        junction_y = (sum(point[1] for point in source_points) + target_point[1]) / (len(source_points) + 1)
        for source_point in source_points:
            pieces.append(
                f"<line x1='{source_point[0] + 14.0:.1f}' y1='{source_point[1]:.1f}' "
                f"x2='{micro_junction_x - 8.0:.1f}' y2='{junction_y:.1f}' "
                f"stroke='{color}' stroke-width='1.8' stroke-dasharray='5,3' opacity='{opacity:.2f}'/>"
            )
        pieces.append(
            f"<line x1='{micro_junction_x + 8.0:.1f}' y1='{junction_y:.1f}' "
            f"x2='{target_point[0] - 16.0:.1f}' y2='{target_point[1]:.1f}' "
            f"stroke='{color}' stroke-width='2.0' stroke-dasharray='5,3' marker-end='url(#cg-arrow)' opacity='{opacity:.2f}'/>"
        )
        radius = 4.8 + 7.2 * abs(value) / max(hyper_bound, 1e-9)
        pieces.append(
            f"<circle cx='{micro_junction_x:.1f}' cy='{junction_y:.1f}' r='{radius:.1f}' fill='white' stroke='{color}' stroke-width='1.0' opacity='{opacity:.2f}'/>"
        )

    for macro_index, group in enumerate(groups):
        macro_source_point = macro_left_points[macro_index]
        macro_target_point = macro_right_points[macro_index]
        mapping_color, _ = macro_colors[macro_index]
        left_points = [micro_left_points[int(node)] for node in group]
        right_points = [micro_right_points[int(node)] for node in group]
        left_anchor_y = sum(point[1] for point in left_points) / len(left_points)
        right_anchor_y = sum(point[1] for point in right_points) / len(right_points)
        left_box_x = micro_source_x - 58.0
        left_box_right_x = micro_source_x - 58.0 + 76.0
        right_box_right_x = micro_target_x - 12.0 + 96.0
        pieces.append(
            _svg_curved_path(
                left_box_x,
                left_anchor_y + 2.0,
                macro_source_point[0],
                macro_source_point[1] - 16.0,
                stroke=mapping_color,
                stroke_width=2.4,
                dash="",
                curvature=-0.18,
            )
        )
        pieces.append(
            _svg_curved_path(
                right_box_right_x,
                right_anchor_y + 2.0,
                macro_target_point[0],
                macro_target_point[1] - 13.0,
                stroke=mapping_color,
                stroke_width=2.4,
                dash="",
                curvature=0.18,
            )
        )

    for source in range(len(macro_labels)):
        for target in range(len(macro_labels)):
            value = float(macro_values[source, target])
            if abs(value) <= 0.0:
                continue
            start_x, start_y = macro_left_points[source]
            end_x, end_y = macro_right_points[target]
            color = "#2c7bb6" if value >= 0 else "#d73027"
            stroke_width = 1.0 + 2.6 * abs(value) / macro_bound
            curvature = 0.09 * (target - source)
            pieces.append(
                _svg_curved_path(
                    start_x + 18.0,
                    start_y,
                    end_x - 16.0,
                    end_y,
                    stroke=color,
                    stroke_width=stroke_width,
                    dash="",
                    curvature=curvature,
                    marker_end="url(#cg-arrow)",
                )
            )

    macro_junction_x = macro_source_x + 0.58 * (macro_target_x - macro_source_x)
    for edge in macro_hyperedges or ():
        sources = tuple(int(source) for source in edge.get("sources", ()))
        if len(sources) < 2:
            continue
        target = int(edge["target"])
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        opacity = min(max(float(edge.get("opacity", 0.9)), 0.0), 1.0)
        color = _motif_color(str(edge.get("label", "syn")))
        source_points = [macro_left_points[source] for source in sources]
        target_point = macro_right_points[target]
        junction_y = (sum(point[1] for point in source_points) + target_point[1]) / (len(source_points) + 1)
        for source_point in source_points:
            pieces.append(
                f"<line x1='{source_point[0] + 18.0:.1f}' y1='{source_point[1]:.1f}' "
                f"x2='{macro_junction_x - 8.0:.1f}' y2='{junction_y:.1f}' "
                f"stroke='{color}' stroke-width='1.8' stroke-dasharray='5,3' opacity='{opacity:.2f}'/>"
            )
        pieces.append(
            f"<line x1='{macro_junction_x + 8.0:.1f}' y1='{junction_y:.1f}' "
            f"x2='{target_point[0] - 16.0:.1f}' y2='{target_point[1]:.1f}' "
            f"stroke='{color}' stroke-width='2.0' stroke-dasharray='5,3' marker-end='url(#cg-arrow)' opacity='{opacity:.2f}'/>"
        )
        radius = 4.8 + 7.2 * abs(value) / max(hyper_bound, 1e-9)
        pieces.append(
            f"<circle cx='{macro_junction_x:.1f}' cy='{junction_y:.1f}' r='{radius:.1f}' fill='white' stroke='{color}' stroke-width='1.0' opacity='{opacity:.2f}'/>"
        )

    for index, (x, y) in enumerate(micro_left_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#edf3f8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{index}</text>"
        )
        pieces.append(
            f"<text x='{x - 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='end' fill='#333'>{html.escape(str(micro_labels[index]))}(t)</text>"
        )

    for index, (x, y) in enumerate(micro_right_points):
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#f8efe8' stroke='#51606f' stroke-width='1.4'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{index}</text>"
        )
        pieces.append(
            f"<text x='{x + 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='start' fill='#333'>{html.escape(str(micro_labels[index]))}(t+1)</text>"
        )

    for index, (x, y) in enumerate(macro_left_points):
        stroke_color, fill_color = macro_colors[index]
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='16' fill='{fill_color}' stroke='{stroke_color}' stroke-width='1.8'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10.5' text-anchor='middle' fill='#111'>{index}</text>"
        )
        pieces.append(
            f"<text x='{x + 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='start' fill='#333'>{html.escape(str(macro_labels[index]))}(t)</text>"
        )

    for index, (x, y) in enumerate(macro_right_points):
        stroke_color, fill_color = macro_colors[index]
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='{fill_color}' stroke='{stroke_color}' stroke-width='1.6'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='middle' fill='#111'>{index}</text>"
        )
        pieces.append(
            f"<text x='{x + 22.0:.1f}' y='{y + 4.0:.1f}' font-size='10' text-anchor='start' fill='#333'>{html.escape(str(macro_labels[index]))}(t+1)</text>"
        )

    pieces.append("</svg>")
    return "".join(pieces)


def render_coarse_graining_comparison_png(
    output_path: str | Path,
    title: str,
    *,
    micro_pairwise: np.ndarray,
    micro_labels: Sequence[str],
    macro_labels: Sequence[str],
    groups: Sequence[Sequence[int]],
    macro_pairwise: np.ndarray,
    macro_hyperedges: Sequence[Mapping[str, Any]] | None = None,
    micro_hyperedges: Sequence[Mapping[str, Any]] | None = None,
    width: int = 1180,
    height: int = 820,
    dpi: int = 180,
) -> Path:
    """Render the coarse-graining comparison directly to PNG for PDF-friendly exports."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    micro_values = np.asarray(micro_pairwise, dtype=float)
    macro_values = np.asarray(macro_pairwise, dtype=float)
    if micro_values.ndim != 2 or micro_values.shape[0] != micro_values.shape[1]:
        raise ValueError("micro_pairwise must be square.")
    if macro_values.ndim != 2 or macro_values.shape[0] != macro_values.shape[1]:
        raise ValueError("macro_pairwise must be square.")
    if micro_values.shape[0] != len(micro_labels):
        raise ValueError("micro_labels must match micro_pairwise.")
    if macro_values.shape[0] != len(macro_labels):
        raise ValueError("macro_labels must match macro_pairwise.")
    if len(groups) != len(macro_labels):
        raise ValueError("groups must match macro_labels.")

    top_margin = 36.0
    bottom_margin = 28.0
    micro_top = top_margin + 28.0
    micro_bottom = height * 0.37
    macro_top = height * 0.64
    macro_bottom = height - bottom_margin

    source_x = 116.0
    target_x = width - 150.0
    micro_spacing = (micro_bottom - micro_top) / max(len(micro_labels) - 1, 1)
    macro_spacing = (macro_bottom - macro_top) / max(len(macro_labels) - 1, 1)
    micro_left_points = [(source_x, micro_top + idx * micro_spacing) for idx in range(len(micro_labels))]
    micro_right_points = [(target_x, micro_top + idx * micro_spacing) for idx in range(len(micro_labels))]
    macro_left_points = [(source_x, macro_top + idx * macro_spacing) for idx in range(len(macro_labels))]
    macro_right_points = [(target_x, macro_top + idx * macro_spacing) for idx in range(len(macro_labels))]

    micro_bound = max(float(np.max(np.abs(micro_values))) if micro_values.size else 0.0, 1.0)
    macro_bound = max(float(np.max(np.abs(macro_values))) if macro_values.size else 0.0, 1.0)
    hyperedge_values = [
        abs(float(edge.get("value", edge.get("synergy", 0.0))))
        for edge in (micro_hyperedges or ())
    ] + [
        abs(float(edge.get("value", edge.get("synergy", 0.0))))
        for edge in (macro_hyperedges or ())
    ]
    hyper_bound = max(hyperedge_values, default=1.0)
    macro_palette = [
        ("#d9822b", "#f7e1cf"),
        ("#2f7d63", "#d9eee7"),
        ("#6b57a5", "#e5ddf5"),
        ("#b24c63", "#f6dbe3"),
        ("#3a78b3", "#dce9f6"),
    ]
    macro_colors = [macro_palette[index % len(macro_palette)] for index in range(len(macro_labels))]

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, constrained_layout=True)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.text(12, 20, title, fontsize=16, fontweight="bold", color="#111", va="top")

    def add_curve(start, end, color, lw, rad, linestyle="-", alpha=0.95, arrow=True):
        patch = FancyArrowPatch(
            posA=start,
            posB=end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>" if arrow else "-",
            mutation_scale=10,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(patch)

    for macro_index, group in enumerate(groups):
        stroke_color, fill_color = macro_colors[macro_index]
        left_points = [micro_left_points[int(node)] for node in group]
        right_points = [micro_right_points[int(node)] for node in group]
        left_y_min = min(point[1] for point in left_points) - 16.0
        left_y_max = max(point[1] for point in left_points) + 16.0
        right_y_min = min(point[1] for point in right_points) - 16.0
        right_y_max = max(point[1] for point in right_points) + 16.0
        ax.add_patch(FancyBboxPatch((source_x - 58.0, left_y_min), 76.0, left_y_max - left_y_min,
                                    boxstyle="round,pad=0.0,rounding_size=10", linewidth=1.4,
                                    edgecolor=stroke_color, facecolor=fill_color, alpha=0.36))
        ax.add_patch(FancyBboxPatch((target_x - 12.0, right_y_min), 96.0, right_y_max - right_y_min,
                                    boxstyle="round,pad=0.0,rounding_size=10", linewidth=1.4,
                                    edgecolor=stroke_color, facecolor=fill_color, alpha=0.36))

    for source in range(len(micro_labels)):
        for target in range(len(micro_labels)):
            value = float(micro_values[source, target])
            if abs(value) <= 0.0:
                continue
            color = "#2c7bb6" if value >= 0 else "#d73027"
            stroke_width = 1.0 + 2.6 * abs(value) / micro_bound
            curvature = -0.12 * (target - source) / max(len(micro_labels), 1)
            add_curve((micro_left_points[source][0] + 16.0, micro_left_points[source][1]),
                      (micro_right_points[target][0] - 16.0, micro_right_points[target][1]),
                      color, stroke_width, curvature)

    micro_junction_x = source_x + 0.58 * (target_x - source_x)
    for edge in micro_hyperedges or ():
        sources = tuple(int(source) for source in edge.get("sources", ()))
        if len(sources) < 2:
            continue
        target = int(edge["target"])
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        opacity = min(max(float(edge.get("opacity", 0.9)), 0.0), 1.0)
        color = _motif_color(str(edge.get("label", "syn")))
        source_points = [micro_left_points[source] for source in sources]
        target_point = micro_right_points[target]
        junction_y = (sum(point[1] for point in source_points) + target_point[1]) / (len(source_points) + 1)
        for source_point in source_points:
            ax.plot([source_point[0] + 14.0, micro_junction_x - 8.0], [source_point[1], junction_y],
                    color=color, linewidth=1.8, linestyle=(0, (3, 2)), alpha=opacity)
        add_curve((micro_junction_x + 8.0, junction_y), (target_point[0] - 16.0, target_point[1]),
                  color, 2.0, 0.0, linestyle=(0, (3, 2)), alpha=opacity)
        radius = 4.8 + 7.2 * abs(value) / max(hyper_bound, 1e-9)
        ax.add_patch(Circle((micro_junction_x, junction_y), radius=radius, facecolor="white",
                            edgecolor=color, linewidth=1.0, alpha=opacity))

    for macro_index, group in enumerate(groups):
        mapping_color, _ = macro_colors[macro_index]
        left_points = [micro_left_points[int(node)] for node in group]
        right_points = [micro_right_points[int(node)] for node in group]
        left_anchor_y = sum(point[1] for point in left_points) / len(left_points)
        right_anchor_y = sum(point[1] for point in right_points) / len(right_points)
        add_curve((source_x - 58.0, left_anchor_y + 2.0), (macro_left_points[macro_index][0], macro_left_points[macro_index][1] - 16.0),
                  mapping_color, 2.4, 0.18, arrow=False)
        add_curve((target_x + 84.0, right_anchor_y + 2.0), (macro_right_points[macro_index][0], macro_right_points[macro_index][1] - 13.0),
                  mapping_color, 2.4, -0.18, arrow=False)

    for source in range(len(macro_labels)):
        for target in range(len(macro_labels)):
            value = float(macro_values[source, target])
            if abs(value) <= 0.0:
                continue
            color = "#2c7bb6" if value >= 0 else "#d73027"
            stroke_width = 1.0 + 2.6 * abs(value) / macro_bound
            curvature = -0.18 * (target - source) / max(len(macro_labels), 1)
            add_curve((macro_left_points[source][0] + 18.0, macro_left_points[source][1]),
                      (macro_right_points[target][0] - 16.0, macro_right_points[target][1]),
                      color, stroke_width, curvature)

    macro_junction_x = source_x + 0.58 * (target_x - source_x)
    for edge in macro_hyperedges or ():
        sources = tuple(int(source) for source in edge.get("sources", ()))
        if len(sources) < 2:
            continue
        target = int(edge["target"])
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        opacity = min(max(float(edge.get("opacity", 0.9)), 0.0), 1.0)
        color = _motif_color(str(edge.get("label", "syn")))
        source_points = [macro_left_points[source] for source in sources]
        target_point = macro_right_points[target]
        junction_y = (sum(point[1] for point in source_points) + target_point[1]) / (len(source_points) + 1)
        for source_point in source_points:
            ax.plot([source_point[0] + 18.0, macro_junction_x - 8.0], [source_point[1], junction_y],
                    color=color, linewidth=1.8, linestyle=(0, (3, 2)), alpha=opacity)
        add_curve((macro_junction_x + 8.0, junction_y), (target_point[0] - 16.0, target_point[1]),
                  color, 2.0, 0.0, linestyle=(0, (3, 2)), alpha=opacity)
        radius = 4.8 + 7.2 * abs(value) / max(hyper_bound, 1e-9)
        ax.add_patch(Circle((macro_junction_x, junction_y), radius=radius, facecolor="white",
                            edgecolor=color, linewidth=1.0, alpha=opacity))

    for index, (x, y) in enumerate(micro_left_points):
        ax.add_patch(Circle((x, y), radius=13, facecolor="#edf3f8", edgecolor="#51606f", linewidth=1.4))
        ax.text(x, y + 1.5, str(index), fontsize=10, color="#111", ha="center", va="center")
        ax.text(x - 22.0, y, f"{micro_labels[index]}(t)", fontsize=10, color="#333", ha="right", va="center")
    for index, (x, y) in enumerate(micro_right_points):
        ax.add_patch(Circle((x, y), radius=13, facecolor="#f8efe8", edgecolor="#51606f", linewidth=1.4))
        ax.text(x, y + 1.5, str(index), fontsize=10, color="#111", ha="center", va="center")
        ax.text(x + 22.0, y, f"{micro_labels[index]}(t+1)", fontsize=10, color="#333", ha="left", va="center")
    for index, (x, y) in enumerate(macro_left_points):
        stroke_color, fill_color = macro_colors[index]
        ax.add_patch(Circle((x, y), radius=16, facecolor=fill_color, edgecolor=stroke_color, linewidth=1.8))
        ax.text(x, y + 1.5, str(index), fontsize=10.5, color="#111", ha="center", va="center")
        ax.text(x + 22.0, y, f"{macro_labels[index]}(t)", fontsize=10, color="#333", ha="left", va="center")
    for index, (x, y) in enumerate(macro_right_points):
        stroke_color, fill_color = macro_colors[index]
        ax.add_patch(Circle((x, y), radius=13, facecolor=fill_color, edgecolor=stroke_color, linewidth=1.6))
        ax.text(x, y + 1.5, str(index), fontsize=10, color="#111", ha="center", va="center")
        ax.text(x + 22.0, y, f"{macro_labels[index]}(t+1)", fontsize=10, color="#333", ha="left", va="center")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output


def render_recovery_summary_svg(
    title: str,
    *,
    rows: Sequence[Mapping[str, Any]],
    width: int = 480,
    row_height: int = 26,
) -> str:
    """Render a compact SVG summary table for recovery metrics."""

    columns = list(rows[0].keys()) if rows else []
    height = 58 + row_height * (len(rows) + 1)
    left_margin = 12.0
    top_margin = 42.0
    col_width = max((width - 2 * left_margin) / max(len(columns), 1), 70.0)

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
    ]
    if title:
        pieces.append(
            f"<text x='12' y='20' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>"
        )

    header_y = top_margin
    pieces.append(
        f"<rect x='{left_margin:.1f}' y='{header_y - 16:.1f}' width='{width - 2 * left_margin:.1f}' height='{row_height:.1f}' fill='#eef3f7' stroke='#cdd6df' stroke-width='1'/>"
    )
    for column_index, column in enumerate(columns):
        x = left_margin + column_index * col_width + col_width / 2.0
        pieces.append(
            f"<text x='{x:.1f}' y='{header_y:.1f}' font-size='10' text-anchor='middle' fill='#223'>{html.escape(str(column))}</text>"
        )

    for row_index, row in enumerate(rows):
        y = header_y + (row_index + 1) * row_height
        fill = "#fbfcfd" if row_index % 2 == 0 else "#f3f6f8"
        pieces.append(
            f"<rect x='{left_margin:.1f}' y='{y - 16:.1f}' width='{width - 2 * left_margin:.1f}' height='{row_height:.1f}' fill='{fill}' stroke='#e0e6eb' stroke-width='1'/>"
        )
        for column_index, column in enumerate(columns):
            x = left_margin + column_index * col_width + col_width / 2.0
            value = row.get(column, "")
            if isinstance(value, float):
                text = f"{value:.2f}"
            else:
                text = str(value)
            pieces.append(
                f"<text x='{x:.1f}' y='{y:.1f}' font-size='10' text-anchor='middle' fill='#223'>{html.escape(text)}</text>"
            )

    pieces.append("</svg>")
    return "".join(pieces)


def _normalize_topology_connection_style(
    connection_style: Mapping[str, Mapping[str, Any]] | None,
    *,
    fallback_marker_width: float = 8.0,
    fallback_marker_height: float = 6.0,
    fallback_marker_ref_x: float = 7.0,
    fallback_marker_ref_y: float = 3.0,
    fallback_marker_fill: str = "#7d8a97",
) -> dict[str, dict[str, Any]]:
    defaults: dict[str, dict[str, Any]] = {
        "copy": {
            "stroke": "#a8b3bf",
            "stroke_width_base": 1.0,
            "stroke_width_scale": 0.9,
            "dash": "",
            "opacity": 0.9,
            "start_offset": 14.0,
            "end_offset": 16.0,
            "forward_curvature": 0.0,
            "bidirectional_curvature": 0.14,
            "marker_width": fallback_marker_width,
            "marker_height": fallback_marker_height,
            "marker_ref_x": fallback_marker_ref_x,
            "marker_ref_y": fallback_marker_ref_y,
            "marker_fill": fallback_marker_fill,
            "marker_stroke": "none",
            "marker_shape": "triangle",
        },
        "cooperation": {
            "stroke": "#e17c05",
            "stroke_width": 2.0,
            "dash": "5,4",
            "opacity": 0.95,
            "start_offset": 14.0,
            "end_offset": 17.0,
            "curvature": 0.34,
            "marker_width": fallback_marker_width,
            "marker_height": fallback_marker_height,
            "marker_ref_x": fallback_marker_ref_x,
            "marker_ref_y": fallback_marker_ref_y,
            "marker_fill": "#e17c05",
            "marker_stroke": "none",
            "marker_shape": "triangle",
            "linecap": "butt",
        },
        "parity": {
            "stroke": "#7b4ab5",
            "stroke_width": 2.0,
            "dash": "2,3",
            "opacity": 0.95,
            "start_offset": 14.0,
            "end_offset": 18.0,
            "curvature": 0.24,
            "marker_width": fallback_marker_width,
            "marker_height": fallback_marker_height,
            "marker_ref_x": fallback_marker_ref_x,
            "marker_ref_y": fallback_marker_ref_y,
            "marker_fill": "#7b4ab5",
            "marker_stroke": "none",
            "marker_shape": "triangle",
            "linecap": "round",
        },
    }
    provided = connection_style or {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, style in defaults.items():
        merged = dict(style)
        merged.update(dict(provided.get(key, {})))
        normalized[key] = merged
    return normalized


def _normalize_topology_connection_filter_style(
    connection_filter_style: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = {
        "drop_shared_parity_sources": False,
    }
    if connection_filter_style is not None:
        normalized.update(dict(connection_filter_style))
    return normalized


def _svg_marker_def(marker_id: str, style: Mapping[str, Any]) -> str:
    marker_width = float(style["marker_width"])
    marker_height = float(style["marker_height"])
    marker_ref_x = float(style["marker_ref_x"])
    marker_ref_y = float(style["marker_ref_y"])
    marker_shape = str(style.get("marker_shape", "triangle"))
    marker_fill = html.escape(str(style.get("marker_fill", style["stroke"])))
    marker_stroke = str(style.get("marker_stroke", "none"))
    stroke_attr = ""
    if marker_stroke != "none":
        stroke_attr = f" stroke='{html.escape(marker_stroke)}' stroke-width='0.8'"

    if marker_shape == "diamond":
        path = (
            f"M 0 {marker_ref_y:.1f} L {marker_width / 2.0:.1f} 0 "
            f"L {marker_width:.1f} {marker_ref_y:.1f} "
            f"L {marker_width / 2.0:.1f} {marker_height:.1f} z"
        )
    else:
        path = (
            f"M 0 0 L {marker_width:.1f} {marker_ref_y:.1f} "
            f"L 0 {marker_height:.1f} z"
        )

    return (
        f"<marker id='{marker_id}' markerWidth='{marker_width:.1f}' markerHeight='{marker_height:.1f}' "
        f"refX='{marker_ref_x:.1f}' refY='{marker_ref_y:.1f}' orient='auto' markerUnits='userSpaceOnUse'>"
        f"<path d='{path}' fill='{marker_fill}'{stroke_attr}/></marker>"
    )


def render_topology_mechanism_svg(
    name: str,
    label: str,
    adjacency: np.ndarray,
    node_specs: Sequence[Mapping[str, Any]],
    *,
    width: int = 440,
    height: int = 290,
    show_summary_box: bool = True,
    show_panel_title: bool = True,
    show_mechanism_hint: bool = True,
    show_mechanism_text_labels: bool = True,
    arrow_marker_width: float = 8.0,
    arrow_marker_height: float = 6.0,
    arrow_marker_ref_x: float = 7.0,
    arrow_marker_ref_y: float = 3.0,
    arrow_marker_fill: str = "#7d8a97",
    uniform_node_fill: str | None = None,
    uniform_node_stroke: str | None = None,
    uniform_node_stroke_width: float | None = None,
    connection_style: Mapping[str, Mapping[str, Any]] | None = None,
    connection_filter_style: Mapping[str, Any] | None = None,
) -> str:
    """Render a topology diagram with copy/cooperation/parity mechanism markers."""

    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")
    n_nodes = matrix.shape[0]
    if len(node_specs) != n_nodes:
        raise ValueError("Need one node specification per node.")

    styles = _normalize_topology_connection_style(
        connection_style,
        fallback_marker_width=arrow_marker_width,
        fallback_marker_height=arrow_marker_height,
        fallback_marker_ref_x=arrow_marker_ref_x,
        fallback_marker_ref_y=arrow_marker_ref_y,
        fallback_marker_fill=arrow_marker_fill,
    )
    filters = _normalize_topology_connection_filter_style(connection_filter_style)
    copy_style = styles["copy"]
    cooperation_style = styles["cooperation"]
    parity_style = styles["parity"]

    left_panel = 250.0
    cx = 110.0 if show_summary_box else width / 2.0
    cy = height / 2.0
    radius = min(height * 0.34, 82.0)
    points = []
    for idx in range(n_nodes):
        angle = -math.pi / 2.0 + 2.0 * math.pi * idx / n_nodes
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    def mechanism_curvature(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        magnitude: float,
        prefer_toward_center: bool,
    ) -> float:
        dx = x2 - x1
        dy = y2 - y1
        length = max(math.hypot(dx, dy), 1e-9)
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        nx = -dy / length
        ny = dx / length

        def score(curvature: float) -> float:
            control_x = mx + curvature * length * nx
            control_y = my + curvature * length * ny
            return (control_x - cx) ** 2 + (control_y - cy) ** 2

        positive_score = score(magnitude)
        negative_score = score(-magnitude)
        if prefer_toward_center:
            return magnitude if positive_score <= negative_score else -magnitude
        return magnitude if positive_score >= negative_score else -magnitude

    def offset_segment(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        start_offset: float,
        end_offset: float,
    ) -> tuple[float, float, float, float]:
        dx = x2 - x1
        dy = y2 - y1
        length = max(math.hypot(dx, dy), 1e-9)
        ux = dx / length
        uy = dy / length
        return (
            x1 + start_offset * ux,
            y1 + start_offset * uy,
            x2 - end_offset * ux,
            y2 - end_offset * uy,
        )

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<defs>"
        + _svg_marker_def("copy_arrowhead", copy_style)
        + _svg_marker_def("cooperation_arrowhead", cooperation_style)
        + _svg_marker_def("parity_arrowhead", parity_style)
        + "</defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
    ]
    if show_panel_title:
        pieces.append(
            f"<text x='12' y='20' font-size='16' font-weight='700'>{html.escape(name)}: {html.escape(label)}</text>"
        )
    if show_mechanism_hint:
        pieces.append(
            "<text x='12' y='38' font-size='10' fill='#555'>fill=parity strength, border=cooperation strength</text>"
        )

    mechanism_sources_by_target: list[set[int]] = [set() for _ in range(n_nodes)]
    for node, spec in enumerate(node_specs):
        coop_sources = tuple(int(i) for i in spec.get("coop_sources", ()))
        if len(coop_sources) >= 2:
            mechanism_sources_by_target[node].update(coop_sources)
        for entry in spec.get("coop_pairs", []):
            if len(entry) == 2:
                src_a, src_b = entry
            else:
                src_a, src_b, _ = entry
            mechanism_sources_by_target[node].update((int(src_a), int(src_b)))
        parity_sources = tuple(int(i) for i in spec.get("parity_sources", ()))
        if filters["drop_shared_parity_sources"]:
            parity_sources = tuple(source for source in parity_sources if source not in coop_sources)
        if len(parity_sources) >= 2:
            mechanism_sources_by_target[node].update(parity_sources)

    # Directed structural edges.
    for src in range(n_nodes):
        for dst in range(n_nodes):
            weight = matrix[src, dst]
            if weight <= 0:
                continue
            if src in mechanism_sources_by_target[dst]:
                continue
            x1, y1 = points[src]
            x2, y2 = points[dst]
            start_x, start_y, end_x, end_y = offset_segment(
                x1,
                y1,
                x2,
                y2,
                start_offset=float(copy_style["start_offset"]),
                end_offset=float(copy_style["end_offset"]),
            )
            stroke_width = float(copy_style["stroke_width_base"]) + float(copy_style["stroke_width_scale"]) * min(weight, 1.5)
            reverse_weight = matrix[dst, src]
            if reverse_weight > 0:
                curvature = float(copy_style["bidirectional_curvature"]) if src < dst else -float(copy_style["bidirectional_curvature"])
                pieces.append(
                    _svg_curved_path(
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        stroke=str(copy_style["stroke"]),
                        stroke_width=stroke_width,
                        dash=str(copy_style["dash"]),
                        curvature=curvature,
                        marker_end="url(#copy_arrowhead)",
                        opacity=float(copy_style["opacity"]),
                    )
                )
            else:
                pieces.append(
                    _svg_curved_path(
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        stroke=str(copy_style["stroke"]),
                        stroke_width=stroke_width,
                        dash=str(copy_style["dash"]),
                        curvature=float(copy_style["forward_curvature"]),
                        marker_end="url(#copy_arrowhead)",
                        opacity=float(copy_style["opacity"]),
                    )
                )

    # Cooperation and parity source links.
    for node, spec in enumerate(node_specs):
        target_x, target_y = points[node]
        coop_sources = tuple(int(i) for i in spec.get("coop_sources", ()))
        if len(coop_sources) < 2:
            coop_sources = ()
        for source in coop_sources:
            sx, sy = points[source]
            start_x, start_y, end_x, end_y = offset_segment(
                sx,
                sy,
                target_x,
                target_y,
                start_offset=float(cooperation_style["start_offset"]),
                end_offset=float(cooperation_style["end_offset"]),
            )
            pieces.append(
                _svg_curved_path(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    stroke=str(cooperation_style["stroke"]),
                    stroke_width=float(cooperation_style["stroke_width"]),
                    dash=str(cooperation_style["dash"]),
                    curvature=mechanism_curvature(
                        sx,
                        sy,
                        target_x,
                        target_y,
                        magnitude=float(cooperation_style["curvature"]),
                        prefer_toward_center=True,
                    ),
                    marker_end="url(#cooperation_arrowhead)",
                    opacity=float(cooperation_style["opacity"]),
                    linecap=str(cooperation_style.get("linecap", "butt")),
                )
            )
        if coop_sources and show_mechanism_text_labels:
            pieces.append(
                f"<text x='{target_x:.1f}' y='{target_y + 24:.1f}' font-size='9' text-anchor='middle' fill='#a05a00'>coop</text>"
            )
        for entry in spec.get("coop_pairs", []):
            if len(entry) == 2:
                src_a, src_b = entry
            else:
                src_a, src_b, _ = entry
            ax, ay = points[int(src_a)]
            bx, by = points[int(src_b)]
            start_x, start_y, end_x, end_y = offset_segment(
                ax,
                ay,
                bx,
                by,
                start_offset=float(cooperation_style["start_offset"]),
                end_offset=float(cooperation_style["end_offset"]),
            )
            pieces.append(
                _svg_curved_path(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    stroke=str(cooperation_style["stroke"]),
                    stroke_width=float(cooperation_style["stroke_width"]),
                    dash=str(cooperation_style["dash"]),
                    curvature=mechanism_curvature(
                        ax,
                        ay,
                        bx,
                        by,
                        magnitude=float(cooperation_style["curvature"]),
                        prefer_toward_center=True,
                    ),
                    marker_end="url(#cooperation_arrowhead)",
                    opacity=float(cooperation_style["opacity"]),
                    linecap=str(cooperation_style.get("linecap", "butt")),
                )
            )
            if show_mechanism_text_labels:
                mid_x = (ax + bx) / 2.0
                mid_y = (ay + by) / 2.0 - 10.0
                pieces.append(
                    f"<text x='{mid_x:.1f}' y='{mid_y:.1f}' font-size='9' text-anchor='middle' fill='#a05a00'>coop</text>"
                )
        parity_sources = tuple(int(i) for i in spec.get("parity_sources", ()))
        if filters["drop_shared_parity_sources"]:
            parity_sources = tuple(source for source in parity_sources if source not in coop_sources)
        if len(parity_sources) < 2:
            parity_sources = ()
        for source in parity_sources:
            sx, sy = points[source]
            start_x, start_y, end_x, end_y = offset_segment(
                sx,
                sy,
                target_x,
                target_y,
                start_offset=float(parity_style["start_offset"]),
                end_offset=float(parity_style["end_offset"]),
            )
            pieces.append(
                _svg_curved_path(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    stroke=str(parity_style["stroke"]),
                    stroke_width=float(parity_style["stroke_width"]),
                    dash=str(parity_style["dash"]),
                    curvature=mechanism_curvature(
                        sx,
                        sy,
                        target_x,
                        target_y,
                        magnitude=float(parity_style["curvature"]),
                        prefer_toward_center=False,
                    ),
                    marker_end="url(#parity_arrowhead)",
                    opacity=float(parity_style["opacity"]),
                    linecap=str(parity_style.get("linecap", "round")),
                )
            )
        if parity_sources and show_mechanism_text_labels:
            pieces.append(
                f"<text x='{target_x:.1f}' y='{target_y - 18:.1f}' font-size='9' text-anchor='middle' fill='#7b4ab5'>parity</text>"
            )

    # Nodes.
    for idx, (x, y) in enumerate(points):
        spec = node_specs[idx]
        beta = float(spec.get("beta", 0.0))
        gamma = float(spec.get("gamma", 0.0))
        fill = (
            uniform_node_fill
            if uniform_node_fill is not None
            else _blend_rgb((245, 245, 245), (196, 85, 39), min(max(gamma, 0.0), 1.2) / 1.2)
        )
        stroke = uniform_node_stroke if uniform_node_stroke is not None else ("#2f3e4e" if beta > 0 else "#9aa5b1")
        stroke_width = (
            uniform_node_stroke_width
            if uniform_node_stroke_width is not None
            else 1.4 + 1.8 * min(max(beta, 0.0), 1.2) / 1.2
        )
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='{fill}' stroke='{stroke}' stroke-width='{stroke_width:.2f}'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4:.1f}' font-size='11' text-anchor='middle' fill='#111'>{idx}</text>"
        )

    if show_summary_box:
        # Parameter summary box.
        summary_x = left_panel + 18
        summary_y = 58
        summary_lines = []
        for idx, spec in enumerate(node_specs):
            bits = []
            beta = float(spec.get("beta", 0.0))
            gamma = float(spec.get("gamma", 0.0))
            if beta > 0.15:
                bits.append(f"beta={beta:.2f}")
            if gamma > 0.05:
                bits.append(f"gamma={gamma:.2f}")
            coop_sources = tuple(int(i) for i in spec.get("coop_sources", ()))
            if coop_sources:
                bits.append("coop=" + ",".join(str(source) for source in coop_sources))
            coop_pairs = spec.get("coop_pairs", [])
            if coop_pairs and not coop_sources:
                if len(coop_pairs) == 1:
                    pair = coop_pairs[0]
                    pair_text = f"{int(pair[0])}-{int(pair[1])}"
                else:
                    pair_text = f"{len(coop_pairs)} pairs"
                bits.append(f"coop={pair_text}")
            parity_sources = spec.get("parity_sources", ())
            if parity_sources:
                bits.append("parity=" + ",".join(str(int(i)) for i in parity_sources))
            if bits:
                summary_lines.append(f"node {idx}: " + "; ".join(bits))

        pieces.append(
            f"<rect x='{summary_x - 8}' y='{summary_y - 18}' width='{width - summary_x - 18}' height='{height - summary_y + 6}' "
            f"fill='#fcfcfd' stroke='#d4dbe3' stroke-width='1' rx='8' ry='8'/>"
        )
        pieces.append(
            f"<text x='{summary_x:.1f}' y='{summary_y:.1f}' font-size='12' font-weight='700' fill='#222'>Mechanism summary</text>"
        )
        pieces.append(
            f"<text x='{summary_x:.1f}' y='{summary_y + 18:.1f}' font-size='10' fill='#555'>only non-default coordination signals are listed</text>"
        )
        text_y = summary_y + 40
        for line in summary_lines[:10]:
            pieces.append(
                f"<text x='{summary_x:.1f}' y='{text_y:.1f}' font-size='10.5' fill='#222'>{html.escape(line)}</text>"
            )
            text_y += 16

    pieces.append("</svg>")
    return "".join(pieces)


def render_downward_causation_comparison_svg(
    *,
    decoupling_parity_ei: float,
    decoupling_single_ei: float,
    decoupling_dc_values: Sequence[float],
    downward_target_ei: float,
    downward_single_ei: float,
    downward_dc_values: Sequence[float],
    width: int = 1160,
    height: int = 600,
) -> str:
    """Render a paper-style comparison between causal decoupling and downward causation."""

    if len(decoupling_dc_values) != 3 or len(downward_dc_values) != 3:
        raise ValueError("Both DC value sequences must have length 3.")

    bg = "#ffffff"
    navy = "#1F3B73"
    slate = "#5C677D"
    light_slate = "#9AA5B1"
    panel_fill = "#FCFCFD"
    panel_stroke = "#D8DEE6"
    node_fill = "#D9E6F2"
    node_stroke = "#6E7C87"
    xor_fill = "#111111"
    accent = "#C56B3C"
    font = "STIXGeneral, DejaVu Serif, Times New Roman, serif"

    left_x = 40.0
    top_y = 34.0
    panel_w = (width - 100.0) / 2.0
    gap = 20.0
    right_x = left_x + panel_w + gap
    panel_h = height - 60.0

    left_source_x = left_x + 68.0
    left_xor_x = left_x + 185.0
    left_group_x = left_x + 342.0
    right_source_x = right_x + 68.0
    right_xor_x = right_x + 185.0
    right_target_x = right_x + 342.0
    y_positions = [130.0, 240.0, 350.0]
    center_y = y_positions[1]
    card_y = 430.0

    def metric_card(x: float, y: float, lines: Sequence[str], *, card_w: float = 235.0) -> list[str]:
        line_height = 20.0
        card_h = 26.0 + line_height * len(lines)
        pieces_local = [
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{card_w:.1f}' height='{card_h:.1f}' "
            f"rx='12' ry='12' fill='{panel_fill}' stroke='{panel_stroke}' stroke-width='1.2'/>"
        ]
        text_y = y + 24.0
        for idx, line in enumerate(lines):
            fill = "#24313d"
            pieces_local.append(
                f"<text x='{x + 16.0:.1f}' y='{text_y + idx * line_height:.1f}' font-size='13' "
                f"font-family='{font}' fill='{fill}'>{html.escape(line)}</text>"
            )
        return pieces_local

    def arrow(x1: float, y1: float, x2: float, y2: float, *, width_value: float = 2.4) -> str:
        return (
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='{slate}' stroke-width='{width_value:.2f}' marker-end='url(#dc-arrow)'/>"
        )

    def node(x: float, y: float, *, radius: float = 22.0, label: str | None = None) -> list[str]:
        pieces_local = [
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{node_fill}' "
            f"stroke='{node_stroke}' stroke-width='2.2'/>"
        ]
        if label:
            pieces_local.append(
                f"<text x='{x:.1f}' y='{y + radius + 22.0:.1f}' text-anchor='middle' font-size='12.5' "
                f"font-family='{font}' fill='#4a5662'>{html.escape(label)}</text>"
            )
        return pieces_local

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs>"
        "<marker id='dc-arrow' markerWidth='10' markerHeight='8' refX='8.5' refY='4' orient='auto' markerUnits='strokeWidth'>"
        f"<path d='M 0 0 L 10 4 L 0 8 z' fill='{slate}'/></marker>"
        "</defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{bg}'/>",
        f"<rect x='{left_x:.1f}' y='{top_y:.1f}' width='{panel_w:.1f}' height='{panel_h:.1f}' rx='20' ry='20' fill='{bg}'/>",
        f"<rect x='{right_x:.1f}' y='{top_y:.1f}' width='{panel_w:.1f}' height='{panel_h:.1f}' rx='20' ry='20' fill='{bg}'/>",
        f"<text x='{left_x + 6.0:.1f}' y='56' font-size='18' font-weight='700' font-family='{font}' fill='{navy}'>A</text>",
        f"<text x='{right_x + 6.0:.1f}' y='56' font-size='18' font-weight='700' font-family='{font}' fill='{navy}'>B</text>",
        f"<text x='{left_x + 34.0:.1f}' y='56' font-size='30' font-weight='700' font-family='{font}' fill='#111'>Causal decoupling</text>",
        f"<text x='{right_x + 34.0:.1f}' y='56' font-size='30' font-weight='700' font-family='{font}' fill='#111'>Downward causation</text>",
    ]

    # Left panel: causal decoupling.
    for idx, y in enumerate(y_positions, start=1):
        pieces.extend(node(left_source_x, y, label=f"x{idx}(t)"))
        pieces.extend(node(left_group_x, y))
        pieces.append(arrow(left_source_x + 22.0, y, left_xor_x - 12.0, center_y, width_value=2.2))

    pieces.append(
        f"<rect x='{left_group_x - 42.0:.1f}' y='{y_positions[0] - 42.0:.1f}' width='84' height='264' "
        f"rx='34' ry='34' fill='none' stroke='{light_slate}' stroke-width='2.2' stroke-dasharray='9,9'/>"
    )
    pieces.append(
        f"<text x='{left_group_x:.1f}' y='{y_positions[2] + 46.0:.1f}' text-anchor='middle' font-size='12.5' "
        f"font-family='{font}' fill='#4a5662'>macro target group</text>"
    )
    pieces.append(
        f"<circle cx='{left_xor_x:.1f}' cy='{center_y:.1f}' r='11' fill='{xor_fill}'/>"
    )
    pieces.append(
        f"<text x='{left_xor_x:.1f}' y='{center_y - 20.0:.1f}' text-anchor='middle' font-size='19' font-weight='700' "
        f"font-family='{font}' fill='#111'>XOR</text>"
    )
    pieces.append(arrow(left_xor_x + 12.0, center_y, left_group_x - 54.0, center_y, width_value=2.6))
    pieces.extend(
        metric_card(
            left_x + (panel_w - 235.0) / 2.0,
            card_y,
            [
                f"EI(full -> parity_next) = {decoupling_parity_ei:.2f}",
                f"EI(x_i -> parity_next) = {decoupling_single_ei:.2f}",
                "DC_j = 0.00 for all j" if max(abs(float(v)) for v in decoupling_dc_values) < 1e-12 else
                "DC values: " + ", ".join(f"{float(v):.2f}" for v in decoupling_dc_values),
            ],
        )
    )

    # Right panel: downward causation.
    for idx, y in enumerate(y_positions, start=1):
        pieces.extend(node(right_source_x, y, label=f"x{idx}(t)"))
        pieces.extend(node(right_target_x, y, label=f"x{idx}(t+1)"))
        pieces.append(arrow(right_source_x + 22.0, y, right_xor_x - 12.0, center_y, width_value=2.2))

    pieces.append(
        f"<circle cx='{right_target_x:.1f}' cy='{y_positions[0]:.1f}' r='24' fill='none' stroke='{accent}' stroke-width='2.0' opacity='0.55'/>"
    )
    pieces.append(
        f"<circle cx='{right_xor_x:.1f}' cy='{center_y:.1f}' r='11' fill='{xor_fill}'/>"
    )
    pieces.append(
        f"<text x='{right_xor_x:.1f}' y='{center_y - 20.0:.1f}' text-anchor='middle' font-size='19' font-weight='700' "
        f"font-family='{font}' fill='#111'>XOR</text>"
    )
    pieces.append(arrow(right_xor_x + 12.0, center_y, right_target_x - 24.0, y_positions[0], width_value=2.8))
    pieces.extend(
        metric_card(
            right_x + (panel_w - 235.0) / 2.0,
            card_y,
            [
                f"EI(full -> x1_next) = {downward_target_ei:.2f}",
                f"EI(x_i -> x1_next) = {downward_single_ei:.2f}",
                f"DC_1 = {float(downward_dc_values[0]):.2f}",
                f"DC_2 = {float(downward_dc_values[1]):.2f}",
                f"DC_3 = {float(downward_dc_values[2]):.2f}",
            ],
        )
    )

    pieces.append("</svg>")
    return "".join(pieces)


def render_mixed_downward_causation_svg(
    *,
    full_ei: float,
    environment_ei: float,
    x3_ei: float,
    flexibility: float,
    environment_synergy: float,
    dc_value: float,
    width: int = 760,
    height: int = 440,
) -> str:
    """Render a paper-style SVG for the mixed downward-causation case."""

    bg = "#ffffff"
    slate = "#5C677D"
    node_fill = "#D9E6F2"
    node_stroke = "#6E7C87"
    gate_fill = "#111111"
    accent = "#C56B3C"
    font = "STIXGeneral, DejaVu Serif, Times New Roman, serif"

    source_x = 96.0
    and_x = 278.0
    xor_x = 432.0
    target_x = 624.0
    y_positions = [140.0, 250.0, 360.0]
    and_y = 195.0
    xor_y = 250.0
    def arrow(x1: float, y1: float, x2: float, y2: float, *, width_value: float = 2.4) -> str:
        return (
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='{slate}' stroke-width='{width_value:.2f}' marker-end='url(#dc-mixed-arrow)'/>"
        )

    def node(x: float, y: float, *, radius: float = 22.0, label: str | None = None) -> list[str]:
        pieces_local = [
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{node_fill}' "
            f"stroke='{node_stroke}' stroke-width='2.2'/>"
        ]
        if label:
            pieces_local.append(
                f"<text x='{x:.1f}' y='{y + radius + 22.0:.1f}' text-anchor='middle' font-size='12.5' "
                f"font-family='{font}' fill='#4a5662'>{html.escape(label)}</text>"
            )
        return pieces_local

    def gate(x: float, y: float, label: str) -> list[str]:
        return [
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='11' fill='{gate_fill}'/>",
            f"<text x='{x:.1f}' y='{y - 20.0:.1f}' text-anchor='middle' font-size='18' font-weight='700' "
            f"font-family='{font}' fill='#111'>{html.escape(label)}</text>",
        ]

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs>"
        "<marker id='dc-mixed-arrow' markerWidth='10' markerHeight='8' refX='8.5' refY='4' orient='auto' markerUnits='strokeWidth'>"
        f"<path d='M 0 0 L 10 4 L 0 8 z' fill='{slate}'/></marker>"
        "</defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{bg}'/>",
        f"<text x='54' y='56' font-size='30' font-weight='700' font-family='{font}' fill='#111'>Mixed downward causation</text>",
    ]

    for idx, y in enumerate(y_positions, start=1):
        pieces.extend(node(source_x, y, label=f"x{idx}(t)"))
        pieces.extend(node(target_x, y, label=f"x{idx}(t+1)"))

    pieces.append(
        f"<circle cx='{target_x:.1f}' cy='{y_positions[0]:.1f}' r='24' fill='none' stroke='{accent}' stroke-width='2.0' opacity='0.55'/>"
    )

    pieces.append(arrow(source_x + 22.0, y_positions[0], and_x - 12.0, and_y, width_value=2.2))
    pieces.append(arrow(source_x + 22.0, y_positions[1], and_x - 12.0, and_y, width_value=2.2))
    pieces.append(arrow(and_x + 12.0, and_y, xor_x - 12.0, xor_y, width_value=2.5))
    pieces.append(arrow(source_x + 22.0, y_positions[2], xor_x - 12.0, xor_y, width_value=2.2))
    pieces.append(arrow(xor_x + 12.0, xor_y, target_x - 24.0, y_positions[0], width_value=2.8))

    pieces.extend(gate(and_x, and_y, "AND"))
    pieces.extend(gate(xor_x, xor_y, "XOR"))

    pieces.append("</svg>")
    return "".join(pieces)


def render_matrix_heatmap_svg(
    title: str,
    matrix: np.ndarray,
    *,
    subtitle: str = "",
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    width: int = 270,
    height: int = 270,
    decimals: int = 2,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """Render a compact SVG heatmap for a small matrix."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D.")
    n_rows, n_cols = values.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("matrix must be non-empty.")

    if row_labels is None:
        row_labels = [str(i) for i in range(n_rows)]
    if col_labels is None:
        col_labels = [str(i) for i in range(n_cols)]
    if len(row_labels) != n_rows or len(col_labels) != n_cols:
        raise ValueError("row_labels and col_labels must match matrix shape.")

    bound = max(abs(vmin) if vmin is not None else 0.0, abs(vmax) if vmax is not None else 0.0)
    if vmin is None or vmax is None:
        bound = max(bound, float(np.max(np.abs(values))), 1e-12)
    if bound <= 0.0:
        bound = 1.0

    left_margin = 54.0
    top_margin = 52.0 if subtitle else 40.0
    right_margin = 12.0
    bottom_margin = 14.0
    grid_width = width - left_margin - right_margin
    grid_height = height - top_margin - bottom_margin
    cell_w = grid_width / n_cols
    cell_h = grid_height / n_rows

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='10' y='18' font-size='15' font-weight='700' fill='#111'>{html.escape(title)}</text>",
    ]
    if subtitle:
        pieces.append(
            f"<text x='10' y='35' font-size='10.5' fill='#555'>{html.escape(subtitle)}</text>"
        )

    for col, label in enumerate(col_labels):
        x = left_margin + (col + 0.5) * cell_w
        pieces.append(
            f"<text x='{x:.1f}' y='{top_margin - 8:.1f}' font-size='10' text-anchor='middle' fill='#444'>{html.escape(str(label))}</text>"
        )
    for row, label in enumerate(row_labels):
        y = top_margin + (row + 0.5) * cell_h + 3.0
        pieces.append(
            f"<text x='{left_margin - 8:.1f}' y='{y:.1f}' font-size='10' text-anchor='end' fill='#444'>{html.escape(str(label))}</text>"
        )

    for row in range(n_rows):
        for col in range(n_cols):
            value = float(values[row, col])
            x = left_margin + col * cell_w
            y = top_margin + row * cell_h
            fill = _matrix_heatmap_color(value, bound)
            text_fill = "#f8f8f8" if abs(value) > 0.62 * bound else "#222"
            pieces.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' fill='{fill}' stroke='#ffffff' stroke-width='1'/>"
            )
            pieces.append(
                f"<text x='{x + cell_w / 2.0:.1f}' y='{y + cell_h / 2.0 + 4.0:.1f}' font-size='10' text-anchor='middle' fill='{text_fill}'>"
                f"{value:.{decimals}f}</text>"
            )

    pieces.append(
        f"<rect x='{left_margin:.1f}' y='{top_margin:.1f}' width='{grid_width:.1f}' height='{grid_height:.1f}' fill='none' stroke='#c8d0d9' stroke-width='1'/>"
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _node_activation_probability(
    state: np.ndarray,
    adjacency: np.ndarray,
    node_specs: Sequence[Mapping[str, Any]],
    node: int,
) -> float:
    spec = node_specs[node]
    bias = float(spec.get("bias", 0.0))
    alpha = float(spec.get("alpha", 1.0))
    beta = float(spec.get("beta", 1.0))
    gamma = float(spec.get("gamma", 1.0))
    eta = float(spec.get("eta", 1.0))

    copy_weights = spec.get("copy_weights")
    if copy_weights is None:
        weights = adjacency[:, node]
    else:
        weights = np.zeros(adjacency.shape[0], dtype=float)
        if isinstance(copy_weights, Mapping):
            for source, weight in copy_weights.items():
                weights[int(source)] = float(weight)
        else:
            for entry in copy_weights:
                source, weight = entry
                weights[int(source)] = float(weight)

    centered_state = 2.0 * np.asarray(state, dtype=float) - 1.0
    copy_term = float(np.dot(weights, centered_state))

    coop_sources = tuple(int(i) for i in spec.get("coop_sources", ()))
    if coop_sources:
        coop_term = float(np.prod(np.asarray(state, dtype=float)[list(coop_sources)]) - 2.0 ** (-len(coop_sources)))
    else:
        coop_term = 0.0
        for entry in spec.get("coop_pairs", []):
            if len(entry) == 2:
                i, k = entry
                weight = 1.0
            else:
                i, k, weight = entry
            coop_term += float(weight) * (float(state[int(i)] * state[int(k)]) - 0.25)

    parity_sources = tuple(int(i) for i in spec.get("parity_sources", ()))
    if parity_sources:
        parity_bit = sum(int(state[i]) for i in parity_sources) % 2
        parity_term = eta * (2.0 * parity_bit - 1.0)
    else:
        parity_term = 0.0

    logit = bias + alpha * copy_term + beta * coop_term + gamma * parity_term
    return 1.0 / (1.0 + math.exp(-logit))


def _validate_system_tpm(system_tpm: np.ndarray, n_nodes: int, atol: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(system_tpm, dtype=float)
    expected_size = 2 ** n_nodes
    if matrix.shape != (expected_size, expected_size):
        raise ValueError("System TPM shape does not match n_nodes.")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Each system TPM row must sum to 1.")
    if np.any(matrix < -atol):
        raise ValueError("System TPM entries must be non-negative.")
    return np.clip(matrix, 0.0, None)


def _encode_binary_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.shape[1] == 0:
        return np.zeros(states.shape[0], dtype=int)
    weights = 2 ** np.arange(states.shape[1] - 1, -1, -1)
    return states @ weights


def _blend_rgb(start: tuple[int, int, int], end: tuple[int, int, int], factor: float) -> str:
    factor = min(max(factor, 0.0), 1.0)
    rgb = tuple(int(round(a + (b - a) * factor)) for a, b in zip(start, end))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _normalize_indices(indices: Sequence[int], size: int, name: str) -> list[int]:
    normalized = sorted(int(index) for index in indices)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique.")
    if any(index < 0 or index >= size for index in normalized):
        raise ValueError(f"{name} contain out-of-range values.")
    return normalized


def _safe_logdet(matrix: np.ndarray, *, atol: float) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign > 0:
        return float(value)
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    clipped = np.clip(eigenvalues, atol, None)
    return float(np.log(clipped).sum())


def _even_bipartitions(n_nodes: int) -> list[tuple[list[int], list[int]]]:
    if n_nodes % 2 != 0:
        raise ValueError("Even bipartitions require an even number of nodes.")
    half = n_nodes // 2
    partitions = []
    for combo in combinations(range(1, n_nodes), half - 1):
        left = [0, *combo]
        right = [index for index in range(n_nodes) if index not in left]
        partitions.append((left, right))
    return partitions


def _covariance_to_correlation(covariance: np.ndarray, *, atol: float) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(covariance), atol, None))
    return covariance / np.outer(scale, scale)


def _gaussian_conditional_mutual_information(
    covariance: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    given_indices: Sequence[int],
    log_base: float,
    atol: float,
) -> float:
    source = _normalize_indices(source_indices, covariance.shape[0], "source_indices")
    target = _normalize_indices(target_indices, covariance.shape[0], "target_indices")
    given = _normalize_indices(given_indices, covariance.shape[0], "given_indices")
    if not source or not target:
        return 0.0

    cond_source_given = gaussian_conditional_covariance(
        covariance,
        target_indices=source,
        given_indices=given,
        atol=atol,
    )
    cond_source_given_target = gaussian_conditional_covariance(
        covariance,
        target_indices=source,
        given_indices=[*given, *target],
        atol=atol,
    )
    return 0.5 * (
        _safe_logdet(cond_source_given, atol=atol)
        - _safe_logdet(cond_source_given_target, atol=atol)
    ) / math.log(log_base)


def _empirical_discrete_mutual_information(
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    log_base: float,
) -> float:
    source = np.asarray(source_values).reshape(-1)
    target = np.asarray(target_values).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source_values and target_values must have matching shape.")
    if source.size == 0:
        raise ValueError("Need at least one sample to estimate mutual information.")

    _, source_ids = np.unique(source, return_inverse=True)
    _, target_ids = np.unique(target, return_inverse=True)
    joint = np.zeros((source_ids.max() + 1, target_ids.max() + 1), dtype=float)
    np.add.at(joint, (source_ids, target_ids), 1.0)
    joint /= float(source.size)

    source_probs = joint.sum(axis=1, keepdims=True)
    target_probs = joint.sum(axis=0, keepdims=True)
    denominator = source_probs * target_probs
    mask = joint > 0.0
    ratios = np.divide(joint, denominator, out=np.zeros_like(joint), where=denominator > 0.0)
    return float(np.sum(joint[mask] * np.log(ratios[mask])) / math.log(log_base))


def _validate_explicit_output_covariance(
    covariance: np.ndarray | None,
    *,
    state_dim: int,
    atol: float,
) -> np.ndarray:
    if covariance is None:
        raise ValueError("output_covariance must be provided explicitly.")
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (state_dim, state_dim):
        raise ValueError("output_covariance must have shape (state_dim, state_dim).")
    if not np.allclose(matrix, matrix.T, atol=atol):
        raise ValueError("output_covariance must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues <= atol):
        raise ValueError("output_covariance must be positive definite.")
    return matrix


def _normalize_intervention_bounds(
    intervention_bound: float | Sequence[float] | np.ndarray,
    *,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isscalar(intervention_bound):
        width = float(intervention_bound)
        if width <= 0.0:
            raise ValueError("intervention_bound must be positive.")
        lower = np.full(state_dim, -width, dtype=float)
        upper = np.full(state_dim, width, dtype=float)
        return lower, upper

    bounds = np.asarray(intervention_bound, dtype=float)
    if bounds.shape == (2,):
        lower = np.full(state_dim, float(bounds[0]), dtype=float)
        upper = np.full(state_dim, float(bounds[1]), dtype=float)
        return lower, upper
    if bounds.shape == (state_dim, 2):
        return bounds[:, 0].astype(float), bounds[:, 1].astype(float)
    raise ValueError(
        "intervention_bound must be a positive scalar, a length-2 interval, or a (state_dim, 2) array."
    )


def _select_dynamics_backend(
    dynamics: Callable[[Any], Any],
    *,
    state_dim: int,
    backend: str,
) -> str:
    normalized = backend.lower()
    if normalized not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be one of {'auto', 'numpy', 'torch'}.")
    if normalized != "auto":
        return normalized

    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        if isinstance(dynamics, torch_module.nn.Module):
            return "torch"
        probe = torch_module.zeros(state_dim, dtype=torch_module.float64)
        try:
            result = dynamics(probe)
        except Exception:
            pass
        else:
            if isinstance(result, torch_module.Tensor):
                return "torch"
    return "numpy"


def _compute_dynamics_jacobian(
    dynamics: Callable[[Any], Any],
    sample: np.ndarray,
    *,
    state_dim: int,
    backend: str,
    finite_difference_step: float,
    atol: float,
) -> np.ndarray:
    if backend == "torch":
        return _torch_dynamics_jacobian(dynamics, sample, state_dim=state_dim)
    if backend == "numpy":
        return _finite_difference_dynamics_jacobian(
            dynamics,
            sample,
            state_dim=state_dim,
            step=max(float(finite_difference_step), atol),
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _finite_difference_dynamics_jacobian(
    dynamics: Callable[[Any], Any],
    sample: np.ndarray,
    *,
    state_dim: int,
    step: float,
) -> np.ndarray:
    center = _evaluate_numpy_dynamics(dynamics, sample, state_dim=state_dim)
    jacobian = np.zeros((state_dim, state_dim), dtype=float)
    for axis in range(state_dim):
        delta = np.zeros(state_dim, dtype=float)
        delta[axis] = step
        forward = _evaluate_numpy_dynamics(dynamics, sample + delta, state_dim=state_dim)
        backward = _evaluate_numpy_dynamics(dynamics, sample - delta, state_dim=state_dim)
        jacobian[:, axis] = (forward - backward) / (2.0 * step)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("Dynamics evaluation produced non-finite values.")
    return jacobian


def _evaluate_numpy_dynamics(
    dynamics: Callable[[Any], Any],
    sample: np.ndarray,
    *,
    state_dim: int,
) -> np.ndarray:
    result = dynamics(np.asarray(sample, dtype=float))
    array = np.asarray(result, dtype=float).reshape(-1)
    if array.shape != (state_dim,):
        raise ValueError("Dynamics output must be a flat vector of length state_dim.")
    return array


def _torch_dynamics_jacobian(
    dynamics: Callable[[Any], Any],
    sample: np.ndarray,
    *,
    state_dim: int,
) -> np.ndarray:
    if importlib.util.find_spec("torch") is None:
        raise ValueError("torch backend requested but torch is not installed.")
    import torch

    vector = torch.tensor(sample, dtype=torch.float64, requires_grad=True)

    def wrapped(x: Any) -> Any:
        result = dynamics(x)
        if not isinstance(result, torch.Tensor):
            raise ValueError("Torch backend requires the dynamics callable to return a torch.Tensor.")
        flat = result.reshape(-1)
        if flat.shape != (state_dim,):
            raise ValueError("Dynamics output must be a flat vector of length state_dim.")
        return flat

    jacobian = torch.autograd.functional.jacobian(wrapped, vector, create_graph=False)
    matrix = jacobian.detach().cpu().numpy().astype(float, copy=False)
    if matrix.shape != (state_dim, state_dim):
        raise ValueError("Jacobian must have shape (state_dim, state_dim).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Dynamics Jacobian contains non-finite values.")
    return matrix


def _motif_color(label: str) -> str:
    upper = label.upper()
    if "XOR" in upper or "PARITY" in upper:
        return "#7b4ab5"
    if "AND" in upper or "COOP" in upper:
        return "#e17c05"
    if "COPY" in upper:
        return "#2c7bb6"
    return "#2c7a5c"


def _stagger_svg_labels(
    labels: Sequence[Mapping[str, Any]],
    *,
    min_vertical_gap: float,
    x_collision_gap: float,
    top: float,
    bottom: float,
) -> list[dict[str, float | str]]:
    """Greedily separate nearby text labels so dense central bands stay readable."""

    placed: list[dict[str, float | str]] = []
    ordered = sorted(
        (
            {
                "order": index,
                "x": float(label["x"]),
                "y": float(label["y"]),
                "text": str(label["text"]),
                "fill": str(label["fill"]),
            }
            for index, label in enumerate(labels)
        ),
        key=lambda item: (item["y"], item["x"], item["order"]),
    )

    for item in ordered:
        x = float(item["x"])
        y = float(item["y"])
        while True:
            conflict = next(
                (
                    other
                    for other in placed
                    if abs(float(other["x"]) - x) <= x_collision_gap
                    and abs(float(other["y"]) - y) < min_vertical_gap
                ),
                None,
            )
            if conflict is None:
                break
            y = float(conflict["y"]) + min_vertical_gap
        y = min(max(y, top), bottom)
        placed.append(
            {
                "order": float(item["order"]),
                "x": x,
                "y": y,
                "text": item["text"],
                "fill": item["fill"],
            }
        )

    placed.sort(key=lambda item: item["order"])
    return [{"x": item["x"], "y": item["y"], "text": item["text"], "fill": item["fill"]} for item in placed]


def _stagger_axis_positions(
    values: Sequence[float],
    *,
    min_gap: float,
    lower: float,
    upper: float,
) -> list[float]:
    """Greedily separate 1D positions while preserving relative order."""

    positions = [float(value) for value in values]
    if len(positions) <= 1 or min_gap <= 0.0:
        return positions

    ordered = sorted(enumerate(positions), key=lambda item: item[1])
    adjusted = [value for _, value in ordered]

    for index in range(1, len(adjusted)):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + min_gap)

    overflow = adjusted[-1] - upper
    if overflow > 0.0:
        adjusted = [value - overflow for value in adjusted]

    underflow = lower - adjusted[0]
    if underflow > 0.0:
        adjusted = [value + underflow for value in adjusted]

    for index in range(1, len(adjusted)):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + min_gap)

    overflow = adjusted[-1] - upper
    if overflow > 0.0:
        adjusted = [value - overflow for value in adjusted]

    restored = [0.0] * len(positions)
    for (original_index, _), adjusted_value in zip(ordered, adjusted):
        restored[original_index] = adjusted_value
    return restored


def _svg_arrow_marker(marker_id: str, *, width: float, height: float, ref_x: float, ref_y: float, fill: str) -> str:
    path = f"M 0 0 L {width:.1f} {ref_y:.1f} L 0 {height:.1f} z"
    return (
        f"<marker id='{marker_id}' markerWidth='{width:.1f}' markerHeight='{height:.1f}' "
        f"refX='{ref_x:.1f}' refY='{ref_y:.1f}' orient='auto' markerUnits='userSpaceOnUse'>"
        f"<path d='{path}' fill='{html.escape(fill)}'/></marker>"
    )


def _svg_curved_path(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    stroke_width: float,
    dash: str,
    curvature: float = 0.16,
    marker_end: str | None = None,
    opacity: float = 0.95,
    linecap: str | None = None,
) -> str:
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    length = max(math.hypot(dx, dy), 1e-9)
    nx = -dy / length
    ny = dx / length
    cx = mx + curvature * length * nx
    cy = my + curvature * length * ny
    dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
    marker_attr = f" marker-end='{marker_end}'" if marker_end else ""
    linecap_attr = f" stroke-linecap='{linecap}'" if linecap else ""
    return (
        f"<path d='M {x1:.1f},{y1:.1f} Q {cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}' "
        f"fill='none' stroke='{stroke}' stroke-width='{stroke_width:.2f}'{dash_attr}{marker_attr}{linecap_attr} opacity='{opacity:.2f}'/>"
    )


def _matrix_heatmap_color(value: float, bound: float) -> str:
    ratio = min(max(abs(value) / max(bound, 1e-12), 0.0), 1.0)
    if value >= 0.0:
        return _blend_rgb((245, 247, 250), (44, 123, 182), ratio)
    return _blend_rgb((245, 247, 250), (215, 48, 39), ratio)
