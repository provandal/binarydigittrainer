import { describe, it, expect } from "vitest";
import {
  flatToGrid,
  gridToFlat,
  parseLabel,
  getDecisionContribs,
  STEP_DESCRIPTIONS,
} from "../nn-helpers";

describe("flatToGrid / gridToFlat round-trip", () => {
  it("converts and back without loss", () => {
    const flat = Array(81)
      .fill(0)
      .map((_, i) => (i % 3 === 0 ? 1 : 0));
    const grid = flatToGrid(flat);
    expect(grid).toHaveLength(9);
    expect(grid[0]).toHaveLength(9);
    const back = gridToFlat(grid);
    expect(back).toEqual(flat);
  });

  it("produces 9x9 grid from 81-element array", () => {
    const grid = flatToGrid(Array(81).fill(0));
    expect(grid).toHaveLength(9);
    grid.forEach((row) => expect(row).toHaveLength(9));
  });
});

describe("parseLabel", () => {
  it("handles arrays directly", () => {
    expect(parseLabel([1, 0])).toEqual([1, 0]);
  });

  it("handles JSON strings", () => {
    expect(parseLabel("[0,1]")).toEqual([0, 1]);
  });

  it("handles double-quoted strings", () => {
    expect(parseLabel('"[1,0]"')).toEqual([1, 0]);
  });

  it("falls back to [1,0] for unknown types", () => {
    expect(parseLabel(42)).toEqual([1, 0]);
  });
});

describe("getDecisionContribs", () => {
  it("returns correct length", () => {
    const hiddenActivations = Array(24).fill(0.5);
    const outputWeights = [Array(24).fill(0.1), Array(24).fill(-0.1)];
    const contribs = getDecisionContribs(hiddenActivations, outputWeights);
    expect(contribs).toHaveLength(24);
  });

  it("computes correct sign for uniform weights", () => {
    const hiddenActivations = Array(24).fill(1);
    // w0 > w1 → positive contribution (favors digit 0)
    const outputWeights = [Array(24).fill(0.5), Array(24).fill(-0.5)];
    const contribs = getDecisionContribs(hiddenActivations, outputWeights);
    contribs.forEach((c) => {
      expect(c.contrib).toBeGreaterThan(0);
    });
  });
});

describe("STEP_DESCRIPTIONS", () => {
  it("has 6 steps", () => {
    expect(STEP_DESCRIPTIONS).toHaveLength(6);
  });

  it("each step has required fields", () => {
    STEP_DESCRIPTIONS.forEach((step) => {
      expect(step).toHaveProperty("name");
      expect(step).toHaveProperty("concept");
      expect(step).toHaveProperty("formula");
      expect(step).toHaveProperty("activeElements");
    });
  });
});
