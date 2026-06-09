import * as d3 from "d3";
import type { GraphResult, Hyperedge, NodeItem, PairwiseEdge } from "./filtering";

export type LayoutNode = NodeItem & {
  x: number;
  y: number;
};

export type GraphSelection =
  | { kind: "pairwise"; edge: PairwiseEdge }
  | { kind: "hyperedge"; edge: Hyperedge };

export function layoutNodes(nodes: NodeItem[], width: number, height: number): LayoutNode[] {
  const radius = Math.max(92, Math.min(width, height) * 0.32);
  const cx = width / 2;
  const cy = height / 2;
  return nodes.map((node, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(nodes.length, 1);
    return {
      ...node,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });
}

export function nodeMap(layout: LayoutNode[]): Map<string, LayoutNode> {
  return new Map(layout.map((node) => [node.id, node]));
}

export function pairwiseWidthScale(result: GraphResult) {
  const values = result.pairwise_edges.map((edge) => edge.ei);
  const maxValue = Math.max(...values, 1);
  return d3.scaleLinear().domain([0, maxValue]).range([1.4, 5.5]);
}

export function hyperedgeWidthScale(result: GraphResult) {
  const values = result.hyperedges.map((edge) => edge.display_value);
  const maxValue = Math.max(...values, 1);
  return d3.scaleLinear().domain([0, maxValue]).range([2, 6]);
}

export function hyperedgeHub(edge: Hyperedge, positions: Map<string, LayoutNode>, target: LayoutNode): { x: number; y: number } {
  const sources = edge.sources.map((source) => positions.get(source)).filter((node): node is LayoutNode => Boolean(node));
  if (!sources.length) return { x: target.x, y: target.y };
  const sourceX = d3.mean(sources, (node) => node.x) ?? target.x;
  const sourceY = d3.mean(sources, (node) => node.y) ?? target.y;
  return {
    x: sourceX * 0.56 + target.x * 0.44,
    y: sourceY * 0.56 + target.y * 0.44,
  };
}
