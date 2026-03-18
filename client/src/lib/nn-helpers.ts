// Array/data helpers and constants — pure functions

export const flatToGrid = (flatArray: number[]): number[][] => {
  const grid: number[][] = [];
  for (let i = 0; i < 9; i++) {
    grid.push(flatArray.slice(i * 9, (i + 1) * 9));
  }
  return grid;
};

export const gridToFlat = (grid: number[][]): number[] => {
  return grid.flat();
};

export const parseLabel = (label: any): number[] => {
  if (Array.isArray(label)) {
    return label;
  }
  if (typeof label === "string") {
    let labelStr = label;
    if (labelStr.startsWith('"') && labelStr.endsWith('"')) {
      labelStr = labelStr.slice(1, -1);
    }
    return JSON.parse(labelStr);
  }
  return [1, 0];
};

export const getCurrentTarget = (
  currentNetworkState: { current: { currentTarget?: number[] } },
  trainingMode: string,
  trainingExamples: any[],
  currentExampleIndex: number,
  selectedLabelRef: { current: number },
): number[] => {
  const cached = currentNetworkState.current.currentTarget;
  if (cached && cached.length === 2) {
    return cached;
  }

  if (trainingMode === "dataset" && trainingExamples[currentExampleIndex]) {
    const example = trainingExamples[currentExampleIndex];
    if (Array.isArray(example.label)) {
      return example.label;
    } else {
      let labelStr = example.label as string;
      if (labelStr.startsWith('"') && labelStr.endsWith('"')) {
        labelStr = labelStr.slice(1, -1);
      }
      return JSON.parse(labelStr);
    }
  }

  return selectedLabelRef.current === 0 ? [1, 0] : [0, 1];
};

export type DecisionContrib = {
  idx: number;
  contrib: number;
  w0: number;
  w1: number;
  h: number;
};

export const getDecisionContribs = (
  hiddenActivations: number[],
  outputWeights: number[][],
): DecisionContrib[] => {
  const w0 = outputWeights[0];
  const w1 = outputWeights[1];
  return w0.map((w0j, j) => ({
    idx: j,
    contrib: (w0j - w1[j]) * hiddenActivations[j],
    w0: w0j,
    w1: w1[j],
    h: hiddenActivations[j],
  }));
};

export const STEP_DESCRIPTIONS = [
  {
    name: "Ready - Draw your digit",
    concept:
      "The neural network is ready. In Training mode: draw and train step-by-step. In Predict mode: draw and get instant predictions.",
    formula: "Input preparation: x = [x₁, x₂, ..., x₈₁] where each xᵢ ∈ [0,1]",
    activeElements: [] as string[],
  },
  {
    name: "Forward Pass - Input to Hidden",
    concept:
      "Each input pixel is multiplied by connection weights and summed with bias to calculate hidden neuron activations.",
    formula: "hⱼ = σ(∑ᵢ wᵢⱼ·xᵢ + bⱼ) where σ(z) = 1/(1+e⁻ᶻ)",
    activeElements: ["input", "hidden", "inputWeights"],
  },
  {
    name: "Forward Pass - Hidden to Output",
    concept:
      "Hidden layer activations are combined using output weights and softmax to produce probability predictions for digits 0 and 1.",
    formula: "zₖ = ∑ⱼ wⱼₖ·hⱼ + bₖ, then pₖ = softmax(zₖ)",
    activeElements: ["hidden", "output", "outputWeights"],
  },
  {
    name: "Calculate Loss",
    concept:
      "The network's prediction is compared to the target label using Cross-Entropy Loss to measure accuracy.",
    formula: "Loss = -∑ₖ tₖ·log(pₖ) where tₖ is target and pₖ is predicted probability",
    activeElements: ["output", "loss"],
  },
  {
    name: "Backpropagation - Output Layer",
    concept:
      "Error signals flow backward to adjust output weights. The softmax + cross-entropy gradient is simplified.",
    formula: "δₖ = pₖ - tₖ, Δwⱼₖ = α·clip(δₖ)·hⱼ",
    activeElements: ["output", "outputWeights", "backprop"],
  },
  {
    name: "Backpropagation - Hidden Layer",
    concept:
      "Error signals propagate to hidden layer, adjusting input weights based on their contribution to the total error.",
    formula: "δⱼ = σ'(zⱼ)·∑ₖδₖ·wⱼₖ, Δwᵢⱼ = α·δⱼ·xᵢ",
    activeElements: ["hidden", "inputWeights", "backprop"],
  },
];
