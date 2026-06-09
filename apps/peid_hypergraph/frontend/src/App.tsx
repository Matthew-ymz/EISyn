import { Activity, Download, FlaskConical, Network, Play, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";
import {
  type DisplayOptions,
  type GraphResult,
  type Hyperedge,
  type PairwiseEdge,
  filterVisibleGraph,
  strongestPairwise,
} from "./filtering";
import {
  type GraphSelection,
  hyperedgeHub,
  hyperedgeWidthScale,
  layoutNodes,
  nodeMap,
  pairwiseWidthScale,
} from "./graph";

type Mode = "boolean" | "continuous";

const functionChoices = ["xor", "and", "or", "copy", "majority"] as const;

const fallbackResult: GraphResult = {
  nodes: [{ id: "w", label: "w" }, { id: "x", label: "x" }, { id: "y", label: "y" }, { id: "z", label: "z" }],
  pairwise_edges: [
    { source: "w", target: "x", ei: 0.6188 },
    { source: "w", target: "y", ei: 0.6915 },
    { source: "x", target: "z", ei: 0.1071 },
    { source: "y", target: "z", ei: 0.1305 },
    { source: "w", target: "z", ei: 0.0071 },
  ],
  hyperedges: [
    {
      sources: ["x", "y"],
      target: "z",
      source_order: 2,
      joint_ei: 0.9957,
      single_ei_sum: 0.2376,
      synergy: 0.7581,
      display_value: 0.7581,
      interaction_type: "synergy",
    },
  ],
  diagnostics: { estimator: "fallback_sample", source_intervention: "maximum_entropy_independent" },
};

function buildBooleanPayload(nodeCount: number, ruleType: string) {
  const variables = Array.from({ length: nodeCount }, (_, index) => `x${index}`);
  const target = variables[variables.length - 1];
  const rules: Record<string, { type: string; inputs: string[] }> = {};
  for (const variable of variables) rules[variable] = { type: "copy", inputs: [variable] };
  const a = variables[0];
  const b = variables[Math.min(1, variables.length - 1)];
  const c = variables[Math.min(2, variables.length - 1)];
  if (ruleType === "copy") rules[target] = { type: "copy", inputs: [a] };
  else if (ruleType === "majority") rules[target] = { type: "majority", inputs: [a, b, c] };
  else rules[target] = { type: ruleType, inputs: [a, b] };
  return { variables, update_rules: rules, noise: 0, max_source_order: Math.min(3, nodeCount) };
}

async function computeGraph(mode: Mode, nodeCount: number, ruleType: string, alpha: number, beta: number): Promise<GraphResult> {
  const path = mode === "boolean" ? "/api/compute/boolean" : "/api/compute/continuous";
  const payload =
    mode === "boolean"
      ? buildBooleanPayload(nodeCount, ruleType)
      : {
          example: "sine_common_driver",
          alpha,
          beta,
          noise_std: 0.03,
          intervention_samples: 900,
          seed: 11,
          max_source_order: 2,
        };
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function formatNumber(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "-";
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

function GraphView({
  result,
  selected,
  onSelect,
}: {
  result: GraphResult;
  selected: GraphSelection | undefined;
  onSelect: (selection: GraphSelection) => void;
}) {
  const width = 720;
  const height = 540;
  const layout = layoutNodes(result.nodes, width, height);
  const positions = nodeMap(layout);
  const pairWidth = pairwiseWidthScale(result);
  const hyperWidth = hyperedgeWidthScale(result);

  return (
    <svg className="graph-surface" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Computed PEID causal hypergraph">
      <defs>
        <marker id="pair-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
          <path d="M0,0 L9,4.5 L0,9 Z" fill="#496274" />
        </marker>
        <marker id="hyper-arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#b9562b" />
        </marker>
      </defs>
      {result.pairwise_edges.map((edge) => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target || edge.ei <= 0) return null;
        const isSelected = selected?.kind === "pairwise" && selected.edge.source === edge.source && selected.edge.target === edge.target;
        return (
          <line
            key={`p-${edge.source}-${edge.target}`}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            className={isSelected ? "pair-edge selected" : "pair-edge"}
            strokeWidth={pairWidth(edge.ei)}
            markerEnd="url(#pair-arrow)"
            onClick={() => onSelect({ kind: "pairwise", edge })}
          />
        );
      })}
      {result.hyperedges.map((edge, index) => {
        const target = positions.get(edge.target);
        if (!target) return null;
        const hub = hyperedgeHub(edge, positions, target);
        const isSelected = selected?.kind === "hyperedge" && selected.edge.sources.join("+") === edge.sources.join("+") && selected.edge.target === edge.target;
        return (
          <g key={`h-${edge.sources.join("-")}-${edge.target}-${index}`} className={isSelected ? "hyper selected" : "hyper"}>
            {edge.sources.map((sourceId) => {
              const source = positions.get(sourceId);
              if (!source) return null;
              return (
                <path
                  key={sourceId}
                  d={`M ${source.x} ${source.y} Q ${(source.x + hub.x) / 2} ${hub.y - 24} ${hub.x} ${hub.y}`}
                  strokeWidth={hyperWidth(edge.display_value)}
                  onClick={() => onSelect({ kind: "hyperedge", edge })}
                />
              );
            })}
            <line
              x1={hub.x}
              y1={hub.y}
              x2={target.x}
              y2={target.y}
              strokeWidth={hyperWidth(edge.display_value)}
              markerEnd="url(#hyper-arrow)"
              onClick={() => onSelect({ kind: "hyperedge", edge })}
            />
            <circle cx={hub.x} cy={hub.y} r={15} onClick={() => onSelect({ kind: "hyperedge", edge })} />
            <text x={hub.x} y={hub.y + 4} textAnchor="middle">
              {edge.source_order}
            </text>
          </g>
        );
      })}
      {layout.map((node) => (
        <g key={node.id} className="node">
          <circle cx={node.x} cy={node.y} r={30} />
          <text x={node.x} y={node.y + 5} textAnchor="middle">
            {node.label ?? node.id}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Explanation({ selection }: { selection: GraphSelection | undefined }) {
  if (!selection) {
    return (
      <section className="side-section">
        <h2>Selected relation</h2>
        <p className="muted">No relation selected.</p>
      </section>
    );
  }
  if (selection.kind === "pairwise") {
    const edge = selection.edge;
    return (
      <section className="side-section">
        <h2>{edge.source} {"->"} {edge.target}</h2>
        <dl>
          <dt>Pairwise EI</dt>
          <dd>{formatNumber(edge.ei)} bits</dd>
        </dl>
      </section>
    );
  }
  const edge = selection.edge;
  return (
    <section className="side-section">
      <h2>{"{"}{edge.sources.join(", ")}{"}"} {"->"} {edge.target}</h2>
      <dl>
        <dt>Joint EI</dt>
        <dd>{formatNumber(edge.joint_ei)} bits</dd>
        <dt>Single EI sum</dt>
        <dd>{formatNumber(edge.single_ei_sum)} bits</dd>
        <dt>{edge.interaction_type === "signed_interaction" ? "Signed value" : "Synergy"}</dt>
        <dd>{formatNumber(edge.synergy)} bits</dd>
      </dl>
    </section>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("continuous");
  const [nodeCount, setNodeCount] = useState(3);
  const [ruleType, setRuleType] = useState<(typeof functionChoices)[number]>("xor");
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(0.75);
  const [result, setResult] = useState<GraphResult>(fallbackResult);
  const [selected, setSelected] = useState<GraphSelection | undefined>({ kind: "hyperedge", edge: fallbackResult.hyperedges[0] });
  const [status, setStatus] = useState("fallback sample loaded");
  const [display, setDisplay] = useState<DisplayOptions>({
    pairwiseMode: "percent",
    pairwisePercent: 40,
    hyperedgeMode: "percent",
    hyperedgePercent: 60,
  });

  const visible = useMemo(() => filterVisibleGraph(result, display), [result, display]);

  async function handleCompute() {
    setStatus("computing...");
    try {
      const next = await computeGraph(mode, nodeCount, ruleType, alpha, beta);
      setResult(next);
      const firstHyper = next.hyperedges[0];
      const firstPair = strongestPairwise(next);
      setSelected(firstHyper ? { kind: "hyperedge", edge: firstHyper } : firstPair ? { kind: "pairwise", edge: firstPair } : undefined);
      setStatus("computed");
    } catch (error) {
      setStatus(`service unavailable; showing sample`);
      setResult(fallbackResult);
      setSelected({ kind: "hyperedge", edge: fallbackResult.hyperedges[0] });
      console.error(error);
    }
  }

  const pairTopK = display.pairwiseTopK ?? Math.min(8, result.pairwise_edges.length);
  const hyperTopK = display.hyperedgeTopK ?? Math.min(6, result.hyperedges.length);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>PEID causal hypergraph</h1>
          <p>{status}</p>
        </div>
        <button className="icon-button" onClick={handleCompute} title="Compute PEID">
          <Play size={18} />
          <span>Compute</span>
        </button>
      </header>

      <section className="workspace">
        <aside className="builder-panel">
          <div className="panel-heading">
            <FlaskConical size={18} />
            <h2>Dynamics</h2>
          </div>

          <label className="field">
            <span>Mode</span>
            <select value={mode} onChange={(event) => setMode(event.target.value as Mode)}>
              <option value="boolean">Boolean exact</option>
              <option value="continuous">Continuous sine</option>
            </select>
          </label>

          {mode === "boolean" ? (
            <>
              <label className="field">
                <span>Node count</span>
                <input type="range" min="3" max="8" value={nodeCount} onChange={(event) => setNodeCount(Number(event.target.value))} />
                <strong>{nodeCount}</strong>
              </label>
              <label className="field">
                <span>Target function</span>
                <select value={ruleType} onChange={(event) => setRuleType(event.target.value as (typeof functionChoices)[number])}>
                  {functionChoices.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </label>
              <div className="rule-preview">
                <span>Target</span>
                <strong>x{nodeCount - 1}' = {ruleType}(x0, x1)</strong>
              </div>
            </>
          ) : (
            <>
              <label className="field">
                <span>alpha</span>
                <input type="range" min="0" max="1.5" step="0.05" value={alpha} onChange={(event) => setAlpha(Number(event.target.value))} />
                <strong>{alpha.toFixed(2)}</strong>
              </label>
              <label className="field">
                <span>common driver</span>
                <input type="range" min="0" max="1" step="0.05" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
                <strong>{beta.toFixed(2)}</strong>
              </label>
              <div className="rule-preview">
                <span>z'</span>
                <strong>0.22z + alpha sin(xy)</strong>
              </div>
            </>
          )}

          <div className="panel-heading compact">
            <SlidersHorizontal size={18} />
            <h2>Display</h2>
          </div>
          <label className="field">
            <span>Pairwise shown</span>
            <select value={display.pairwiseMode} onChange={(event) => setDisplay({ ...display, pairwiseMode: event.target.value as "percent" | "topK" })}>
              <option value="percent">Top percent</option>
              <option value="topK">Top K</option>
            </select>
          </label>
          {display.pairwiseMode === "percent" ? (
            <label className="field">
              <span>Pairwise %</span>
              <input type="range" min="5" max="100" step="5" value={display.pairwisePercent} onChange={(event) => setDisplay({ ...display, pairwisePercent: Number(event.target.value) })} />
              <strong>{display.pairwisePercent}%</strong>
            </label>
          ) : (
            <label className="field">
              <span>Pairwise K</span>
              <input type="number" min="0" max={result.pairwise_edges.length} value={pairTopK} onChange={(event) => setDisplay({ ...display, pairwiseTopK: Number(event.target.value) })} />
            </label>
          )}
          <label className="field">
            <span>Hyperedges K</span>
            <input type="number" min="0" max={result.hyperedges.length} value={hyperTopK} onChange={(event) => setDisplay({ ...display, hyperedgeMode: "topK", hyperedgeTopK: Number(event.target.value) })} />
          </label>
        </aside>

        <section className="graph-panel">
          <div className="panel-heading">
            <Network size={18} />
            <h2>Computed graph</h2>
            <span>{visible.pairwise_edges.length} pairwise / {visible.hyperedges.length} hyper</span>
          </div>
          <GraphView result={visible} selected={selected} onSelect={setSelected} />
        </section>

        <aside className="details-panel">
          <Explanation selection={selected} />
          <section className="side-section">
            <h2>Computed tables</h2>
            <div className="table-list">
              {result.hyperedges.slice(0, 6).map((edge) => (
                <button key={`${edge.sources.join("+")}-${edge.target}`} onClick={() => setSelected({ kind: "hyperedge", edge })}>
                  <span>{"{"}{edge.sources.join(",")}{"}"} {"->"} {edge.target}</span>
                  <strong>{formatNumber(edge.display_value)}</strong>
                </button>
              ))}
              {result.pairwise_edges.slice(0, 6).map((edge) => (
                <button key={`${edge.source}-${edge.target}`} onClick={() => setSelected({ kind: "pairwise", edge })}>
                  <span>{edge.source} {"->"} {edge.target}</span>
                  <strong>{formatNumber(edge.ei)}</strong>
                </button>
              ))}
            </div>
          </section>
          <button className="secondary-button" title="Export visible data" onClick={() => navigator.clipboard?.writeText(JSON.stringify(visible, null, 2))}>
            <Download size={16} />
            <span>Copy visible JSON</span>
          </button>
        </aside>
      </section>
    </main>
  );
}
