// Neural network forward/backward pass — pure functions, no React

import { sigmoid, sigmoidDerivative, softmax, clip, initWeight } from "./nn-math";

export interface NetworkParams {
  weights: number[][]; // [H][I] — hidden weights
  biases: number[]; // [H]
  outputWeights: number[][]; // [O][H]
  outputBiases: number[]; // [O]
}

export interface NetworkState extends NetworkParams {
  hiddenActivations: number[];
  outputActivations: number[];
  hiddenPreActivations: number[];
  outputPreActivations: number[];
  loss: number;
  outputErrors: number[];
  currentTarget: number[];
  inputs: number[];
}

export interface ForwardResult {
  preActivations: number[];
  activations: number[];
}

export function createInitialParams(): NetworkParams {
  return {
    weights: Array.from({ length: 24 }, () =>
      Array(81)
        .fill(0)
        .map(() => initWeight(81, 24)),
    ),
    biases: Array(24).fill(0),
    outputWeights: Array.from({ length: 2 }, () =>
      Array(24)
        .fill(0)
        .map(() => initWeight(24, 2)),
    ),
    outputBiases: Array(2).fill(0),
  };
}

export function createInitialState(): NetworkState {
  const params = createInitialParams();
  return {
    ...params,
    hiddenActivations: Array(24).fill(0),
    outputActivations: Array(2).fill(0),
    hiddenPreActivations: Array(24).fill(0),
    outputPreActivations: Array(2).fill(0),
    loss: 0,
    outputErrors: Array(2).fill(0),
    currentTarget: [1, 0],
    inputs: Array(81).fill(0),
  };
}

export function forwardPassHidden(inputs: number[], params: NetworkParams): ForwardResult {
  const preActivations = params.weights.map((w, i) =>
    w.reduce((sum, weight, j) => sum + weight * inputs[j], params.biases[i]),
  );
  const activations = preActivations.map((z) => sigmoid(z));
  return { preActivations, activations };
}

export function forwardPassOutput(
  hiddenActivations: number[],
  params: NetworkParams,
): ForwardResult {
  const preActivations = params.outputWeights.map((w, i) =>
    w.reduce((sum, weight, j) => sum + weight * hiddenActivations[j], params.outputBiases[i]),
  );
  const activations = softmax(preActivations);
  return { preActivations, activations };
}

export function calculateCrossEntropyLoss(outputs: number[], target: number[]): number {
  const eps = 1e-7;
  return -target.reduce((sum, t, i) => {
    const p = Math.max(eps, Math.min(1 - eps, outputs[i]));
    return sum + t * Math.log(p);
  }, 0);
}

export interface BackpropOutputResult {
  outputWeights: number[][];
  outputBiases: number[];
  outputErrors: number[];
}

export function backpropOutput(
  outputActivations: number[],
  hiddenActivations: number[],
  target: number[],
  currentOutputWeights: number[][],
  currentOutputBiases: number[],
  lr: number,
): BackpropOutputResult {
  const outputErrors = outputActivations.map((output, i) => output - target[i]);
  const outputErrorsClipped = outputErrors.map(clip);

  const newOutputWeights = currentOutputWeights.map((weights, i) =>
    weights.map((weight, j) => weight - lr * outputErrorsClipped[i] * hiddenActivations[j]),
  );
  const newOutputBiases = currentOutputBiases.map((bias, i) => bias - lr * outputErrorsClipped[i]);

  return {
    outputWeights: newOutputWeights,
    outputBiases: newOutputBiases,
    outputErrors: outputErrorsClipped,
  };
}

export interface BackpropHiddenResult {
  weights: number[][];
  biases: number[];
}

export function backpropHidden(
  hiddenPreActivations: number[],
  outputErrors: number[],
  outputWeights: number[][],
  currentWeights: number[][],
  currentBiases: number[],
  inputs: number[],
  lr: number,
): BackpropHiddenResult {
  const hiddenErrors = hiddenPreActivations.map((preActivation, h) => {
    const errorSum = outputErrors.reduce(
      (sum, outputError, i) => sum + outputError * outputWeights[i][h],
      0,
    );
    return errorSum * sigmoidDerivative(preActivation);
  });
  const hiddenErrorsClipped = hiddenErrors.map(clip);

  const newWeights = currentWeights.map((weights, i) =>
    weights.map((weight, j) => weight - lr * hiddenErrorsClipped[i] * inputs[j]),
  );
  const newBiases = currentBiases.map((bias, i) => bias - lr * hiddenErrorsClipped[i]);

  return { weights: newWeights, biases: newBiases };
}

export function runInference(
  inputs: number[],
  params: NetworkParams,
): { digit: number; confidence: number; hiddenActivations: number[]; outputActivations: number[] } {
  const hidden = forwardPassHidden(inputs, params);
  const output = forwardPassOutput(hidden.activations, params);

  const predictedDigit = output.activations[0] > output.activations[1] ? 0 : 1;
  const confidence = output.activations[predictedDigit];

  return {
    digit: predictedDigit,
    confidence,
    hiddenActivations: hidden.activations,
    outputActivations: output.activations,
  };
}
