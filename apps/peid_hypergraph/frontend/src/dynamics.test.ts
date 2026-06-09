import { describe, expect, it } from "vitest";
import { buildTruthTable, getRulePresentation, interpretSynergy, minimumNodeCountForRule } from "./dynamics";

describe("getRulePresentation", () => {
  it("matches the simulator arity for Boolean presets", () => {
    expect(getRulePresentation("copy", 5).inputs).toEqual(["x0"]);
    expect(getRulePresentation("xor", 5).inputs).toEqual(["x0", "x1"]);
    expect(getRulePresentation("majority", 5).inputs).toEqual(["x0", "x1", "x2"]);
    expect(getRulePresentation("majority", 5).target).toBe("x4");
  });

  it("reserves a distinct target node for every preset", () => {
    expect(minimumNodeCountForRule("xor")).toBe(3);
    expect(minimumNodeCountForRule("majority")).toBe(4);
  });
});

describe("buildTruthTable", () => {
  it("builds XOR and COPY tables from their rule semantics", () => {
    expect(buildTruthTable("xor", 3).map((row) => row.output)).toEqual([0, 1, 1, 0]);
    expect(buildTruthTable("copy", 3)).toEqual([
      { inputs: [0], output: 0 },
      { inputs: [1], output: 1 },
    ]);
  });

  it("uses three inputs for majority", () => {
    const table = buildTruthTable("majority", 5);
    expect(table).toHaveLength(8);
    expect(table.find((row) => row.inputs.join("") === "011")?.output).toBe(1);
    expect(table.find((row) => row.inputs.join("") === "001")?.output).toBe(0);
  });
});

describe("interpretSynergy", () => {
  it("distinguishes synergy, negligible interaction, and redundancy", () => {
    expect(interpretSynergy(0.2)).toContain("more effective information");
    expect(interpretSynergy(1e-8)).toContain("does not add much information");
    expect(interpretSynergy(-0.2)).toContain("redundancy");
  });
});
