// Checkpoint serialization — pure functions

export type Checkpoint = {
  format: string;
  createdAt: string;
  architecture: { input: number; hidden: number; output: number };
  normalize: { enabled: boolean; targetSize: number };
  optimizer: {
    learningRate: number;
    lrDecayRate: number;
    minLR: number;
    decayEnabled: boolean;
  };
  stats: { epoch: number; avgLoss: number; examplesSeen: number };
  params: {
    weights: number[][]; // [H][I]
    biases: number[]; // [H]
    outputWeights: number[][]; // [O][H]
    outputBiases: number[]; // [O]
  };
};

export function validateCheckpoint(cp: any): cp is Checkpoint {
  if (!cp || typeof cp !== "object") return false;
  if (cp.format !== "binary-digit-trainer-checkpoint@v1") return false;
  const archOk =
    cp.architecture?.input === 81 &&
    cp.architecture?.hidden === 24 &&
    cp.architecture?.output === 2;
  const w = cp?.params?.weights,
    b = cp?.params?.biases,
    wo = cp?.params?.outputWeights,
    bo = cp?.params?.outputBiases;
  const shapesOk =
    Array.isArray(w) &&
    w.length === 24 &&
    w.every((row: any) => Array.isArray(row) && row.length === 81) &&
    Array.isArray(b) &&
    b.length === 24 &&
    Array.isArray(wo) &&
    wo.length === 2 &&
    wo.every((row: any) => Array.isArray(row) && row.length === 24) &&
    Array.isArray(bo) &&
    bo.length === 2;
  return archOk && shapesOk;
}

export function downloadBlobJSON(obj: any, filename: string): void {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function nowStamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}
