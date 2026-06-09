export const booleanRuleTypes = ["xor", "and", "or", "copy", "majority"] as const;

export type BooleanRuleType = (typeof booleanRuleTypes)[number];

type RuleDefinition = {
  label: string;
  description: string;
  explanation: string;
  arity: number;
  evaluate: (inputs: number[]) => number;
};

const RULES: Record<BooleanRuleType, RuleDefinition> = {
  xor: {
    label: "XOR",
    description: "Pure interaction. Inputs are informative mainly together.",
    explanation: "The target becomes 1 when exactly one input is 1.",
    arity: 2,
    evaluate: ([a, b]) => a ^ b,
  },
  and: {
    label: "AND",
    description: "Threshold-like rule. Output is 1 only when all selected inputs are 1.",
    explanation: "The target becomes 1 only when both inputs are 1.",
    arity: 2,
    evaluate: ([a, b]) => Number(Boolean(a && b)),
  },
  or: {
    label: "OR",
    description: "Output is 1 when at least one selected input is 1.",
    explanation: "The target becomes 1 when at least one input is 1.",
    arity: 2,
    evaluate: ([a, b]) => Number(Boolean(a || b)),
  },
  copy: {
    label: "COPY",
    description: "One source directly determines the target.",
    explanation: "The target takes the same value as its input.",
    arity: 1,
    evaluate: ([a]) => a,
  },
  majority: {
    label: "MAJORITY",
    description: "Output follows the majority vote among inputs.",
    explanation: "The target becomes 1 when most of its three inputs are 1.",
    arity: 3,
    evaluate: (inputs) => Number(inputs.reduce((sum, value) => sum + value, 0) >= 2),
  },
};

export type RulePresentation = RuleDefinition & {
  type: BooleanRuleType;
  inputs: string[];
  target: string;
  formula: string;
};

export type TruthTableRow = {
  inputs: number[];
  output: number;
};

export function minimumNodeCountForRule(type: BooleanRuleType): number {
  return RULES[type].arity + 1;
}

export function getRulePresentation(type: BooleanRuleType, nodeCount: number): RulePresentation {
  const definition = RULES[type];
  const inputs = Array.from({ length: definition.arity }, (_, index) => `x${index}`);
  const target = `x${nodeCount - 1}`;
  return {
    ...definition,
    type,
    inputs,
    target,
    formula: `${target}_next = ${type}(${inputs.join(", ")})`,
  };
}

export function buildTruthTable(type: BooleanRuleType, nodeCount: number): TruthTableRow[] {
  const rule = getRulePresentation(type, nodeCount);
  return Array.from({ length: 2 ** rule.arity }, (_, rowIndex) => {
    const inputs = Array.from({ length: rule.arity }, (_, inputIndex) => (rowIndex >> (rule.arity - inputIndex - 1)) & 1);
    return { inputs, output: rule.evaluate(inputs) };
  });
}

export function interpretSynergy(value: number, threshold = 1e-6): string {
  if (Math.abs(value) < threshold) {
    return "This group does not add much information beyond the individual source effects.";
  }
  if (value > 0) {
    return "The joint intervention on these source nodes provides more effective information about the target than the sum of their individual effects. PEID therefore identifies this group as a synergistic causal mechanism.";
  }
  return "The joint effect is less than the sum of individual effects. This may indicate redundancy or overlapping information rather than synergy.";
}
