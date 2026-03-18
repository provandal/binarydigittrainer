// Activation functions & numerical primitives

export const GRADIENT_CLIP = 1.0;

export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));

export const sigmoidDerivative = (z: number): number => {
  const s = sigmoid(z);
  return s * (1 - s);
};

export const softmax = (z: number[]): number[] => {
  const m = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
};

export const clip = (g: number): number => Math.max(-GRADIENT_CLIP, Math.min(GRADIENT_CLIP, g));

/** Xavier/Glorot weight initialization */
export const initWeight = (n_in: number, n_out: number): number =>
  (Math.random() - 0.5) * Math.sqrt(2 / (n_in + n_out));
