from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rule_type: str
    inputs: tuple[str, ...]


def parse_rule(raw: object) -> Rule:
    if not isinstance(raw, dict):
        raise ValueError("Each update rule must be an object.")
    rule_type = raw.get("type")
    inputs = raw.get("inputs", ())
    if not isinstance(rule_type, str) or not rule_type:
        raise ValueError("Each update rule needs a nonempty type.")
    if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
        raise ValueError("Rule inputs must be a list of node names.")
    return Rule(rule_type=rule_type, inputs=tuple(inputs))
