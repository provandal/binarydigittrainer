import { describe, it, expect } from "vitest";
import {
  getBarColor,
  weightColor,
  getColorSchemeDescription,
  getPositiveColorName,
  vec81ToGrid9,
  type ColorScheme,
} from "../color-schemes";

const SCHEMES: ColorScheme[] = ["blue-red", "blue-orange", "green-purple", "high-contrast"];

describe("getBarColor", () => {
  it("returns valid CSS color strings for all schemes", () => {
    for (const scheme of SCHEMES) {
      const pos = getBarColor(0.5, scheme);
      const neg = getBarColor(-0.5, scheme);
      expect(pos).toMatch(/^#[0-9A-Fa-f]{6}$/);
      expect(neg).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });

  it("returns different colors for positive and negative", () => {
    for (const scheme of SCHEMES) {
      expect(getBarColor(1, scheme)).not.toBe(getBarColor(-1, scheme));
    }
  });
});

describe("weightColor", () => {
  it("returns valid rgb() strings", () => {
    for (const scheme of SCHEMES) {
      const color = weightColor(0.5, 1, scheme);
      expect(color).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
    }
  });

  it("returns white-ish for zero weight", () => {
    const color = weightColor(0, 1, "blue-red");
    expect(color).toBe("rgb(255, 255, 255)");
  });
});

describe("getColorSchemeDescription", () => {
  it("returns non-empty strings for all schemes", () => {
    for (const scheme of SCHEMES) {
      const desc = getColorSchemeDescription(scheme);
      expect(desc.length).toBeGreaterThan(10);
    }
  });
});

describe("getPositiveColorName", () => {
  it("returns a color name for all schemes", () => {
    for (const scheme of SCHEMES) {
      const name = getPositiveColorName(scheme);
      expect(name.length).toBeGreaterThan(0);
    }
  });
});

describe("vec81ToGrid9", () => {
  it("produces 9x9 grid", () => {
    const v = Array(81).fill(0);
    const grid = vec81ToGrid9(v);
    expect(grid).toHaveLength(9);
    grid.forEach((row) => expect(row).toHaveLength(9));
  });

  it("preserves values", () => {
    const v = Array(81)
      .fill(0)
      .map((_, i) => i);
    const grid = vec81ToGrid9(v);
    expect(grid[0][0]).toBe(0);
    expect(grid[0][8]).toBe(8);
    expect(grid[8][8]).toBe(80);
  });
});
