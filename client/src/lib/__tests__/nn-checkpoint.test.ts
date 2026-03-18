import { describe, it, expect } from "vitest";
import { validateCheckpoint, nowStamp, type Checkpoint } from "../nn-checkpoint";
import { createInitialParams } from "../nn-engine";

function makeValidCheckpoint(): Checkpoint {
  const params = createInitialParams();
  return {
    format: "binary-digit-trainer-checkpoint@v1",
    createdAt: new Date().toISOString(),
    architecture: { input: 81, hidden: 24, output: 2 },
    normalize: { enabled: false, targetSize: 7 },
    optimizer: { learningRate: 0.01, lrDecayRate: 0.99, minLR: 0.0005, decayEnabled: false },
    stats: { epoch: 5, avgLoss: 0.1, examplesSeen: 100 },
    params,
  };
}

describe("validateCheckpoint", () => {
  it("accepts a valid checkpoint", () => {
    expect(validateCheckpoint(makeValidCheckpoint())).toBe(true);
  });

  it("rejects null/undefined", () => {
    expect(validateCheckpoint(null)).toBe(false);
    expect(validateCheckpoint(undefined)).toBe(false);
  });

  it("rejects wrong format string", () => {
    const cp = makeValidCheckpoint();
    cp.format = "wrong-format";
    expect(validateCheckpoint(cp)).toBe(false);
  });

  it("rejects wrong architecture", () => {
    const cp = makeValidCheckpoint();
    cp.architecture.hidden = 10;
    expect(validateCheckpoint(cp)).toBe(false);
  });

  it("rejects wrong weight shapes", () => {
    const cp = makeValidCheckpoint();
    cp.params.weights = cp.params.weights.slice(0, 5); // only 5 instead of 24
    expect(validateCheckpoint(cp)).toBe(false);
  });

  it("rejects wrong output weight shapes", () => {
    const cp = makeValidCheckpoint();
    cp.params.outputWeights = [[1, 2, 3]]; // wrong shape
    expect(validateCheckpoint(cp)).toBe(false);
  });

  it("round-trips: generate → validate", () => {
    const cp = makeValidCheckpoint();
    const json = JSON.parse(JSON.stringify(cp));
    expect(validateCheckpoint(json)).toBe(true);
  });
});

describe("nowStamp", () => {
  it("returns a string matching YYYYMMDD-HHMMSS pattern", () => {
    const stamp = nowStamp();
    expect(stamp).toMatch(/^\d{8}-\d{6}$/);
  });
});
