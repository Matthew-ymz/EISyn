import { describe, expect, it } from "vitest";
import { filterVisibleGraph } from "./filtering";

const result = {
  nodes: [{ id: "a", label: "a" }, { id: "b", label: "b" }],
  pairwise_edges: [
    { source: "a", target: "b", ei: 0.9 },
    { source: "b", target: "a", ei: 0.2 },
    { source: "a", target: "a", ei: 0.1 },
    { source: "b", target: "b", ei: 0.05 },
  ],
  hyperedges: [
    { sources: ["a", "b"], target: "a", source_order: 2, joint_ei: 1.0, single_ei_sum: 0.3, display_value: 0.7, synergy: 0.7 },
    { sources: ["a", "b"], target: "b", source_order: 2, joint_ei: 0.5, single_ei_sum: 0.2, display_value: 0.3, synergy: 0.3 },
  ],
};

describe("filterVisibleGraph", () => {
  it("keeps the strongest top percentage without mutating full results", () => {
    const visible = filterVisibleGraph(result, {
      pairwiseMode: "percent",
      pairwisePercent: 50,
      hyperedgeMode: "topK",
      hyperedgeTopK: 1,
    });

    expect(visible.pairwise_edges.map((edge) => edge.ei)).toEqual([0.9, 0.2]);
    expect(visible.hyperedges.map((edge) => edge.display_value)).toEqual([0.7]);
    expect(result.pairwise_edges).toHaveLength(4);
    expect(result.hyperedges).toHaveLength(2);
  });
});
