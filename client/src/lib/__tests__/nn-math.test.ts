import { describe, it, expect } from "vitest";
import { sigmoid, sigmoidDerivative, softmax, clip, initWeight, GRADIENT_CLIP } from "../nn-math";

describe("sigmoid", () => {
  it("returns 0.5 for input 0", () => {
    expect(sigmoid(0)).toBe(0.5);
  });

  it("approaches 1 for large positive input", () => {
    expect(sigmoid(100)).toBeCloseTo(1, 5);
  });

  it("approaches 0 for large negative input", () => {
    expect(sigmoid(-100)).toBeCloseTo(0, 5);
  });

  it("never returns NaN or Infinity", () => {
    for (const x of [-1000, -500, -100, 0, 100, 500, 1000]) {
      const result = sigmoid(x);
      expect(Number.isFinite(result)).toBe(true);
      expect(Number.isNaN(result)).toBe(false);
    }
  });

  it("is monotonically increasing", () => {
    const values = [-10, -5, -1, 0, 1, 5, 10].map(sigmoid);
    for (let i = 1; i < values.length; i++) {
      expect(values[i]).toBeGreaterThan(values[i - 1]);
    }
  });
});

describe("sigmoidDerivative", () => {
  it("returns 0.25 at z=0", () => {
    expect(sigmoidDerivative(0)).toBe(0.25);
  });

  it("is positive for all inputs", () => {
    for (const z of [-10, -1, 0, 1, 10]) {
      expect(sigmoidDerivative(z)).toBeGreaterThan(0);
    }
  });

  it("peaks at z=0", () => {
    const atZero = sigmoidDerivative(0);
    expect(sigmoidDerivative(-1)).toBeLessThan(atZero);
    expect(sigmoidDerivative(1)).toBeLessThan(atZero);
  });
});

describe("softmax", () => {
  it("returns [0.5, 0.5] for equal inputs", () => {
    const result = softmax([0, 0]);
    expect(result[0]).toBeCloseTo(0.5);
    expect(result[1]).toBeCloseTo(0.5);
  });

  it("sums to 1", () => {
    const result = softmax([1, 2, 3]);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it("is numerically stable with large inputs", () => {
    const result = softmax([1000, 1001]);
    expect(Number.isFinite(result[0])).toBe(true);
    expect(Number.isFinite(result[1])).toBe(true);
    expect(result[0] + result[1]).toBeCloseTo(1);
  });

  it("assigns highest probability to largest input", () => {
    const result = softmax([1, 3, 2]);
    expect(result[1]).toBeGreaterThan(result[0]);
    expect(result[1]).toBeGreaterThan(result[2]);
  });
});

describe("clip", () => {
  it("passes through values within bounds", () => {
    expect(clip(0.5)).toBe(0.5);
    expect(clip(-0.5)).toBe(-0.5);
  });

  it("clips values exceeding GRADIENT_CLIP", () => {
    expect(clip(5)).toBe(GRADIENT_CLIP);
    expect(clip(-5)).toBe(-GRADIENT_CLIP);
  });

  it("handles boundary values", () => {
    expect(clip(GRADIENT_CLIP)).toBe(GRADIENT_CLIP);
    expect(clip(-GRADIENT_CLIP)).toBe(-GRADIENT_CLIP);
  });
});

describe("initWeight", () => {
  it("returns a finite number", () => {
    const w = initWeight(81, 24);
    expect(Number.isFinite(w)).toBe(true);
  });

  it("stays within expected Xavier range", () => {
    const limit = Math.sqrt(2 / (81 + 24));
    for (let i = 0; i < 100; i++) {
      const w = initWeight(81, 24);
      expect(Math.abs(w)).toBeLessThanOrEqual(limit);
    }
  });
});
