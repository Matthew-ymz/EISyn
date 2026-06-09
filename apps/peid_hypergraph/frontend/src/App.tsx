import { Download, FlaskConical, Network, Play, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";
import {
  type BooleanRuleType,
  booleanRuleTypes,
  buildTruthTable,
  getRulePresentation,
  interpretSynergy,
  minimumNodeCountForRule,
} from "./dynamics";
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

function NodeCountSummary({ nodeCount }: { nodeCount: number }) {
  const sources = Array.from({ length: nodeCount - 1 }, (_, index) => `x${index}`);
  return (
    <div className="node-summary">
      <strong>Total nodes: {nodeCount}</strong>
      <span>Source candidates: {sources.join(", ")}</span>
      <span>Default target: x{nodeCount - 1}</span>
      <small>The last node is used as the target by default.</small>
    </div>
  );
}

function MechanismPresets({
  selected,
  onSelect,
}: {
  selected: BooleanRuleType;
  onSelect: (rule: BooleanRuleType) => void;
}) {
  return (
    <section className="generator-section">
      <div className="section-label">Mechanism presets</div>
      <div className="preset-grid">
        {booleanRuleTypes.map((type) => {
          const rule = getRulePresentation(type, 3);
          return (
            <button
              key={type}
              className={selected === type ? "preset-button selected" : "preset-button"}
              title={rule.description}
              onClick={() => onSelect(type)}
            >
              {rule.label}
            </button>
          );
        })}
      </div>
      <p className="helper-text">{getRulePresentation(selected, 3).description}</p>
    </section>
  );
}

function RuleDiagram({ inputs, ruleLabel, target }: { inputs: string[]; ruleLabel: string; target: string }) {
  return (
    <div className="rule-diagram" aria-label={`${inputs.join(" and ")} flow through ${ruleLabel} to next ${target}`}>
      <div className="diagram-inputs">
        {inputs.map((input) => <span key={input}>{input}</span>)}
      </div>
      <span className="diagram-arrow">→</span>
      <strong className="diagram-rule">{ruleLabel}</strong>
      <span className="diagram-arrow">→</span>
      <span className="diagram-target">{target}_next</span>
    </div>
  );
}

function BooleanTruthTable({ ruleType, nodeCount }: { ruleType: BooleanRuleType; nodeCount: number }) {
  const rule = getRulePresentation(ruleType, nodeCount);
  const rows = buildTruthTable(ruleType, nodeCount);
  return (
    <div className="truth-table-wrap">
      <div className="section-label">Truth table preview</div>
      <table className="truth-table">
        <thead>
          <tr>
            {rule.inputs.map((input) => <th key={input}>{input}</th>)}
            <th>{rule.target}_next</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.inputs.join("")}>
              {row.inputs.map((value, index) => <td key={`${row.inputs.join("")}-${index}`}>{value}</td>)}
              <td>{row.output}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TargetRuleCard({ ruleType, nodeCount }: { ruleType: BooleanRuleType; nodeCount: number }) {
  const rule = getRulePresentation(ruleType, nodeCount);
  return (
    <section className="target-rule-card">
      <h3>Target update rule</h3>
      <div className="rule-facts">
        <span><strong>Target node:</strong> {rule.target}</span>
        <span><strong>Input nodes:</strong> {rule.inputs.join(", ")}</span>
        <span><strong>Rule:</strong> {rule.label}</span>
      </div>
      <code>{rule.formula}</code>
      <p>{rule.explanation}</p>
      <RuleDiagram inputs={rule.inputs} ruleLabel={rule.label} target={rule.target} />
      <BooleanTruthTable ruleType={ruleType} nodeCount={nodeCount} />
    </section>
  );
}

function GraphLegend() {
  return (
    <div className="graph-legend" aria-label="Graph legend">
      <span><i className="legend-line pair" />Pairwise edge</span>
      <span><i className="legend-line hyperedge" />Synergistic hyperedge</span>
      <span><i className="legend-line selected" />Selected mechanism</span>
    </div>
  );
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
            <text className="hyper-label" x={hub.x + 20} y={hub.y - 18}>
              {"{"}{edge.sources.join(", ")}{"}"} → {edge.target}
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
      <div className="interpretation">
        <h3>Interpretation</h3>
        <p>{interpretSynergy(edge.synergy)}</p>
      </div>
    </section>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("continuous");
  const [nodeCount, setNodeCount] = useState(5);
  const [ruleType, setRuleType] = useState<BooleanRuleType>("xor");
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

  function markSettingsChanged() {
    setStatus("settings changed; compute to update graph");
  }

  function selectRule(rule: BooleanRuleType) {
    setRuleType(rule);
    setNodeCount((current) => Math.max(current, minimumNodeCountForRule(rule)));
    markSettingsChanged();
  }

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
            <h2>System Generator</h2>
          </div>
          <p className="panel-subtitle">Define the dynamical rule used to generate the state transition table for PEID.</p>

          <label className="field">
            <span>Mode</span>
            <select value={mode} onChange={(event) => { setMode(event.target.value as Mode); markSettingsChanged(); }}>
              <option value="boolean">Boolean exact</option>
              <option value="continuous">Continuous sine</option>
            </select>
          </label>
          <p className="helper-text mode-help">
            {mode === "boolean"
              ? "Binary-state deterministic network. Each node is either 0 or 1. PEID is computed exactly from the full state transition table."
              : "Experimental continuous-valued dynamics with sine coupling. PEID is estimated from sampled trajectories, so results may depend on sampling settings."}
          </p>

          {mode === "boolean" ? (
            <>
              <label className="field">
                <span>Total nodes</span>
                <input type="range" min={minimumNodeCountForRule(ruleType)} max="8" value={nodeCount} onChange={(event) => { setNodeCount(Number(event.target.value)); markSettingsChanged(); }} />
              </label>
              <NodeCountSummary nodeCount={nodeCount} />
              <label className="field">
                <span>Target function</span>
                <select value={ruleType} onChange={(event) => selectRule(event.target.value as BooleanRuleType)}>
                  {booleanRuleTypes.map((choice) => (
                    <option key={choice} value={choice}>
                      {getRulePresentation(choice, nodeCount).label}
                    </option>
                  ))}
                </select>
              </label>
              <MechanismPresets selected={ruleType} onSelect={selectRule} />
              <TargetRuleCard ruleType={ruleType} nodeCount={nodeCount} />
            </>
          ) : (
            <>
              <label className="field">
                <span>Synergy strength (alpha)</span>
                <input type="range" min="0" max="1.5" step="0.05" value={alpha} onChange={(event) => { setAlpha(Number(event.target.value)); markSettingsChanged(); }} />
                <strong>{alpha.toFixed(2)}</strong>
              </label>
              <label className="field">
                <span>Common-driver strength</span>
                <input type="range" min="0" max="1" step="0.05" value={beta} onChange={(event) => { setBeta(Number(event.target.value)); markSettingsChanged(); }} />
                <strong>{beta.toFixed(2)}</strong>
              </label>
              <section className="target-rule-card">
                <h3>Target update rule</h3>
                <div className="rule-facts">
                  <span><strong>Target node:</strong> z</span>
                  <span><strong>Input nodes:</strong> x, y</span>
                  <span><strong>Rule:</strong> Sine interaction</span>
                </div>
                <code>z_next = 0.22z + alpha sin(xy)</code>
                <p>The product of x and y enters the target through a nonlinear sine coupling, which can create joint influence beyond either source alone.</p>
                <RuleDiagram inputs={["x", "y"]} ruleLabel="sin(xy)" target="z" />
              </section>
            </>
          )}

          <details className="education-note">
            <summary>How this affects PEID</summary>
            <p>The selected rule defines how the target node updates from its inputs. PEID compares the effective information carried by individual sources and source groups. If a group provides extra information beyond its members alone, the app draws a synergistic hyperedge.</p>
          </details>

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
          <GraphLegend />
          <GraphView result={visible} selected={selected} onSelect={setSelected} />
        </section>

        <aside className="details-panel">
          <Explanation selection={selected} />
          <section className="side-section">
            <h2>Computed tables</h2>
            <div className="table-list">
              {result.hyperedges.slice(0, 6).map((edge) => (
                <button key={`${edge.sources.join("+")}-${edge.target}`} onClick={() => setSelected({ kind: "hyperedge", edge })}>
                  <span><strong>{"{"}{edge.sources.join(", ")}{"}"} {"->"} {edge.target}</strong><small>Synergy: {formatNumber(edge.synergy)} bits</small></span>
                </button>
              ))}
              {result.pairwise_edges.slice(0, 6).map((edge) => (
                <button key={`${edge.source}-${edge.target}`} onClick={() => setSelected({ kind: "pairwise", edge })}>
                  <span><strong>{edge.source} {"->"} {edge.target}</strong><small>EI: {formatNumber(edge.ei)} bits</small></span>
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
