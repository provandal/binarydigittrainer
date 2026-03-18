import { describe, it, expect } from "vitest";
import {
  createInitialParams,
  forwardPassHidden,
  forwardPassOutput,
  calculateCrossEntropyLoss,
  backpropOutput,
  backpropHidden,
  runInference,
} from "../nn-engine";

describe("createInitialParams", () => {
  it("produces correct shapes", () => {
    const p = createInitialParams();
    expect(p.weights).toHaveLength(24);
    expect(p.weights[0]).toHaveLength(81);
    expect(p.biases).toHaveLength(24);
    expect(p.outputWeights).toHaveLength(2);
    expect(p.outputWeights[0]).toHaveLength(24);
    expect(p.outputBiases).toHaveLength(2);
  });
});

describe("forwardPassHidden", () => {
  it("produces 24 activations all in [0, 1]", () => {
    const params = createInitialParams();
    const inputs = Array(81)
      .fill(0)
      .map(() => Math.round(Math.random()));
    const { activations } = forwardPassHidden(inputs, params);
    expect(activations).toHaveLength(24);
    activations.forEach((a) => {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThanOrEqual(1);
    });
  });
});

describe("forwardPassOutput", () => {
  it("produces 2 activations that sum to 1 (softmax)", () => {
    const params = createInitialParams();
    const hiddenActivations = Array(24).fill(0.5);
    const { activations } = forwardPassOutput(hiddenActivations, params);
    expect(activations).toHaveLength(2);
    expect(activations[0] + activations[1]).toBeCloseTo(1);
  });
});

describe("calculateCrossEntropyLoss", () => {
  it("returns ~0 for perfect prediction", () => {
    const loss = calculateCrossEntropyLoss([0.9999, 0.0001], [1, 0]);
    expect(loss).toBeLessThan(0.01);
  });

  it("returns high loss for wrong prediction", () => {
    const loss = calculateCrossEntropyLoss([0.01, 0.99], [1, 0]);
    expect(loss).toBeGreaterThan(2);
  });

  it("never returns NaN", () => {
    const loss = calculateCrossEntropyLoss([0.5, 0.5], [1, 0]);
    expect(Number.isNaN(loss)).toBe(false);
  });
});

describe("backprop gradient direction", () => {
  it("loss decreases after one gradient update", () => {
    const params = createInitialParams();
    const inputs = Array(81).fill(0);
    inputs[0] = 1;
    inputs[1] = 1;
    const target = [1, 0];

    const hidden = forwardPassHidden(inputs, params);
    const output = forwardPassOutput(hidden.activations, params);
    const lossBefore = calculateCrossEntropyLoss(output.activations, target);

    const bpOut = backpropOutput(
      output.activations,
      hidden.activations,
      target,
      params.outputWeights,
      params.outputBiases,
      0.1,
    );
    const bpHid = backpropHidden(
      hidden.preActivations,
      bpOut.outputErrors,
      params.outputWeights,
      params.weights,
      params.biases,
      inputs,
      0.1,
    );

    const updatedParams = {
      weights: bpHid.weights,
      biases: bpHid.biases,
      outputWeights: bpOut.outputWeights,
      outputBiases: bpOut.outputBiases,
    };

    const hidden2 = forwardPassHidden(inputs, updatedParams);
    const output2 = forwardPassOutput(hidden2.activations, updatedParams);
    const lossAfter = calculateCrossEntropyLoss(output2.activations, target);

    expect(lossAfter).toBeLessThan(lossBefore);
  });
});

describe("convergence", () => {
  it("loss decreases over 100 iterations on a simple pattern", () => {
    let params = createInitialParams();
    const inputs = Array(81).fill(0);
    // Light up top-left corner
    for (let i = 0; i < 9; i++) inputs[i] = 1;
    const target = [1, 0];

    const losses: number[] = [];
    for (let iter = 0; iter < 100; iter++) {
      const hidden = forwardPassHidden(inputs, params);
      const output = forwardPassOutput(hidden.activations, params);
      const loss = calculateCrossEntropyLoss(output.activations, target);
      losses.push(loss);

      const bpOut = backpropOutput(
        output.activations,
        hidden.activations,
        target,
        params.outputWeights,
        params.outputBiases,
        0.1,
      );
      const bpHid = backpropHidden(
        hidden.preActivations,
        bpOut.outputErrors,
        params.outputWeights,
        params.weights,
        params.biases,
        inputs,
        0.1,
      );

      params = {
        weights: bpHid.weights,
        biases: bpHid.biases,
        outputWeights: bpOut.outputWeights,
        outputBiases: bpOut.outputBiases,
      };
    }

    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
    // Loss should drop significantly
    expect(losses[losses.length - 1]).toBeLessThan(0.5);
  });
});

describe("runInference", () => {
  it("returns digit in {0,1} and confidence in [0,1]", () => {
    const params = createInitialParams();
    const inputs = Array(81).fill(0);
    inputs[0] = 1;
    const result = runInference(inputs, params);
    expect([0, 1]).toContain(result.digit);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(result.hiddenActivations).toHaveLength(24);
    expect(result.outputActivations).toHaveLength(2);
  });
});
