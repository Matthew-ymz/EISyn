export type NodeItem = {
  id: string;
  label?: string;
};

export type PairwiseEdge = {
  source: string;
  target: string;
  ei: number;
};

export type Hyperedge = {
  sources: string[];
  target: string;
  source_order: number;
  joint_ei: number;
  single_ei_sum: number;
  synergy: number;
  display_value: number;
  interaction_type?: string;
};

export type GraphResult = {
  nodes: NodeItem[];
  pairwise_edges: PairwiseEdge[];
  hyperedges: Hyperedge[];
  diagnostics?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

export type DisplayOptions = {
  pairwiseMode: "percent" | "topK";
  pairwisePercent: number;
  pairwiseTopK?: number;
  hyperedgeMode: "percent" | "topK";
  hyperedgePercent?: number;
  hyperedgeTopK?: number;
};

function topCount(total: number, mode: "percent" | "topK", percent: number, topK?: number): number {
  if (total <= 0) return 0;
  if (mode === "topK") return Math.max(0, Math.min(total, Math.floor(topK ?? total)));
  const bounded = Math.max(0, Math.min(100, percent));
  return Math.max(0, Math.min(total, Math.ceil((total * bounded) / 100)));
}

export function filterVisibleGraph(result: GraphResult, options: DisplayOptions): GraphResult {
  const pairwise = [...result.pairwise_edges].sort((a, b) => b.ei - a.ei);
  const hyperedges = [...result.hyperedges].sort((a, b) => b.display_value - a.display_value);
  const pairCount = topCount(pairwise.length, options.pairwiseMode, options.pairwisePercent, options.pairwiseTopK);
  const hyperCount = topCount(
    hyperedges.length,
    options.hyperedgeMode,
    options.hyperedgePercent ?? 100,
    options.hyperedgeTopK,
  );
  return {
    ...result,
    nodes: [...result.nodes],
    pairwise_edges: pairwise.slice(0, pairCount),
    hyperedges: hyperedges.slice(0, hyperCount),
  };
}

export function strongestPairwise(result: GraphResult): PairwiseEdge | undefined {
  return [...result.pairwise_edges].sort((a, b) => b.ei - a.ei)[0];
}
