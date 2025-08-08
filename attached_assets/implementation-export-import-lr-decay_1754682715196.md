# Export / Import Checkpoints + LR Decay — Implementation Guide

This adds:
1) **Export Checkpoint** (download model weights/biases + metadata as JSON)  
2) **Import Checkpoint** (load a saved model)  
3) **Learning‑Rate Decay** (per‑epoch schedule, optional)  
4) *(Bonus)* Quick **Evaluate Model** on dataset (accuracy/Loss)

All snippets are designed to drop into your current React component that already has:
- `weights`, `biases`, `outputWeights`, `outputBiases` state
- `currentNetworkState.current` mirrors of the same
- softmax+CE or sigmoid+BCE (either works)
- `trainingExamples`, normalization (`preprocess`, `normalizeEnabled`, `targetSize`), and epoch runner

---

## 0) Checkpoint JSON schema

```jsonc
{
  "format": "binary-digit-trainer-checkpoint@v1",
  "createdAt": "2025-08-08T14:32:10.123Z",
  "architecture": {"input":81,"hidden":24,"output":2},
  "normalize": {"enabled": true, "targetSize": 7},
  "optimizer": {"learningRate": 0.01, "lrDecayRate": 0.99, "minLR": 0.0005, "decayEnabled": true},
  "stats": {"epoch": 3, "avgLoss": 0.0843, "examplesSeen": 200},
  "params": {
    "weights":        number[24][81],
    "biases":         number[24],
    "outputWeights":  number[2][24],
    "outputBiases":   number[2]
  }
}
```

---

## 1) Utilities (place near other helpers)

```ts
// ---------- Checkpoint helpers ----------
type Checkpoint = {
  format: string;
  createdAt: string;
  architecture: { input: number; hidden: number; output: number };
  normalize: { enabled: boolean; targetSize: number };
  optimizer: { learningRate: number; lrDecayRate: number; minLR: number; decayEnabled: boolean };
  stats: { epoch: number; avgLoss: number; examplesSeen: number };
  params: {
    weights: number[][];        // [H][I]
    biases: number[];           // [H]
    outputWeights: number[][];  // [O][H]
    outputBiases: number[];     // [O]
  };
};

function validateCheckpoint(cp: any): cp is Checkpoint {
  if (!cp || typeof cp !== "object") return false;
  if (cp.format !== "binary-digit-trainer-checkpoint@v1") return false;
  const archOk = cp.architecture?.input === 81 && cp.architecture?.hidden === 24 && cp.architecture?.output === 2;
  const w = cp?.params?.weights, b = cp?.params?.biases, wo = cp?.params?.outputWeights, bo = cp?.params?.outputBiases;
  const shapesOk =
    Array.isArray(w) && w.length === 24 && w.every((row: any) => Array.isArray(row) && row.length === 81) &&
    Array.isArray(b) && b.length === 24 &&
    Array.isArray(wo) && wo.length === 2 && wo.every((row: any) => Array.isArray(row) && row.length === 24) &&
    Array.isArray(bo) && bo.length === 2;
  return archOk && shapesOk;
}

function downloadBlobJSON(obj: any, filename: string) {
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

function nowStamp() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}
```

---

## 2) Component state for LR decay + stats

Add near your existing state:

```ts
// ----- LR decay -----
const [lrDecayEnabled, setLrDecayEnabled] = useState(false);
const [lrDecayRate, setLrDecayRate] = useState(0.99);   // per epoch multiply
const [minLR, setMinLR] = useState(0.0005);

// ----- Stats for checkpoint metadata -----
const [examplesSeen, setExamplesSeen] = useState(0);
const [lastEpochAvgLoss, setLastEpochAvgLoss] = useState<number | null>(null);
const [completedEpochs, setCompletedEpochs] = useState(0);
```

Increment `examplesSeen` inside your training loop (after each sample). At epoch end, set `lastEpochAvgLoss` and `completedEpochs++` (shown below).

---

## 3) Export Checkpoint button & handler

### Handler
```ts
function handleExportCheckpoint() {
  const cp: Checkpoint = {
    format: "binary-digit-trainer-checkpoint@v1",
    createdAt: new Date().toISOString(),
    architecture: { input: 81, hidden: 24, output: 2 },
    normalize: { enabled: normalizeEnabled, targetSize },
    optimizer: { learningRate, lrDecayRate, minLR, decayEnabled: lrDecayEnabled },
    stats: { epoch: completedEpochs, avgLoss: lastEpochAvgLoss ?? NaN, examplesSeen },
    params: {
      weights: currentNetworkState.current.weights.map(r => [...r]),
      biases:  [...currentNetworkState.current.biases],
      outputWeights: currentNetworkState.current.outputWeights.map(r => [...r]),
      outputBiases:  [...currentNetworkState.current.outputBiases]
    }
  };
  downloadBlobJSON(cp, `checkpoint-${nowStamp()}.json`);
}
```

### UI (Controls card, below other buttons)
```tsx
<div className="mt-4 grid grid-cols-2 gap-2">
  <Button
    onClick={handleExportCheckpoint}
    variant="outline"
    size="sm"
  >
    Export Checkpoint
  </Button>

  {/* Import reuses a hidden file input */}
  <div className="relative">
    <input
      type="file"
      accept=".json"
      onChange={handleImportCheckpointFile}
      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
    />
    <Button variant="outline" size="sm">Import Checkpoint</Button>
  </div>
</div>
```

---

## 4) Import Checkpoint handler

```ts
async function handleImportCheckpointFile(e: React.ChangeEvent<HTMLInputElement>) {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    const json = JSON.parse(text);
    if (!validateCheckpoint(json)) {
      alert("Invalid checkpoint format or shape mismatch (expected 81→24→2).");
      e.target.value = "";
      return;
    }

    const { params, normalize, optimizer, stats } = json as Checkpoint;

    // update refs first (source of truth for training)
    currentNetworkState.current.weights = params.weights.map(r => [...r]);
    currentNetworkState.current.biases  = [...params.biases];
    currentNetworkState.current.outputWeights = params.outputWeights.map(r => [...r]);
    currentNetworkState.current.outputBiases  = [...params.outputBiases];

    // reset caches
    currentNetworkState.current.hiddenActivations = Array(24).fill(0);
    currentNetworkState.current.outputActivations = Array(2).fill(0);
    currentNetworkState.current.hiddenPreActivations = Array(24).fill(0);
    currentNetworkState.current.outputPreActivations = Array(2).fill(0);
    currentNetworkState.current.loss = 0;
    currentNetworkState.current.outputErrors = Array(2).fill(0);

    // update React state to match (UI)
    setWeights(params.weights.map(r => [...r]));
    setBiases([...params.biases]);
    setOutputWeights(params.outputWeights.map(r => [...r]));
    setOutputBiases([...params.outputBiases]);

    // normalization + optimizer settings (optional but helpful)
    if (typeof normalize?.enabled === "boolean") setNormalizeEnabled(!!normalize.enabled);
    if (typeof normalize?.targetSize === "number") setTargetSize(normalize.targetSize);
    if (typeof optimizer?.learningRate === "number") setLearningRate(optimizer.learningRate);
    if (typeof optimizer?.lrDecayRate === "number") setLrDecayRate(optimizer.lrDecayRate);
    if (typeof optimizer?.minLR === "number") setMinLR(optimizer.minLR);
    if (typeof optimizer?.decayEnabled === "boolean") setLrDecayEnabled(optimizer.decayEnabled);

    // stats (for display only)
    if (typeof stats?.avgLoss === "number") setLastEpochAvgLoss(stats.avgLoss);
    if (typeof stats?.epoch === "number") setCompletedEpochs(stats.epoch);
    if (typeof stats?.examplesSeen === "number") setExamplesSeen(stats.examplesSeen);

    alert(`Loaded checkpoint: ${file.name}`);
  } catch (err) {
    console.error("Import error:", err);
    alert("Failed to import checkpoint.");
  } finally {
    e.target.value = ""; // allow reselecting same file
  }
}
```

---

## 5) Learning‑Rate Decay — state + UI + hook point

### UI controls (in “Network Info” or “Controls” card)
```tsx
<div className="mt-4 p-3 bg-gray-50 rounded">
  <div className="flex items-center gap-2 text-sm">
    <label className="flex items-center gap-2">
      <input
        type="checkbox"
        checked={lrDecayEnabled}
        onChange={(e) => setLrDecayEnabled(e.target.checked)}
      />
      LR Decay (per epoch)
    </label>
  </div>

  <div className="grid grid-cols-3 gap-2 mt-2 text-xs items-center">
    <div className="flex items-center justify-between">
      <span>Decay rate</span>
      <input
        type="number"
        step="0.001"
        min="0.90"
        max="0.999"
        value={lrDecayRate}
        onChange={e => setLrDecayRate(parseFloat(e.target.value) || 0.99)}
        className="w-20 border rounded px-1 py-0.5"
      />
    </div>
    <div className="flex items-center justify-between">
      <span>Min LR</span>
      <input
        type="number"
        step="0.0001"
        min="0.0001"
        max="0.1"
        value={minLR}
        onChange={e => setMinLR(parseFloat(e.target.value) || 0.0005)}
        className="w-20 border rounded px-1 py-0.5"
      />
    </div>
    <div className="flex items-center justify-between">
      <span>LR</span>
      <span className="font-mono">{learningRate.toFixed(5)}</span>
    </div>
  </div>
</div>
```

### Apply decay **at the end of each epoch**

Inside your multi‑epoch training function, you already compute average loss and push `epochLossHistory`. Right **after** storing the epoch’s average loss, add:

```ts
// after computing averageLoss at epoch end
setLastEpochAvgLoss(averageLoss);
setCompletedEpochs(prev => prev + 1);

// apply LR decay
if (lrDecayEnabled) {
  setLearningRate(prev => {
    const next = Math.max(minLR, prev * lrDecayRate);
    console.log(`[LR Decay] lr: ${prev.toFixed(6)} → ${next.toFixed(6)}`);
    return next;
  });
}
```

> Because your backprop uses the **`learningRate` state** each step, the newly decayed LR will take effect for the next epoch automatically.

Also, somewhere in the per‑sample runner:
```ts
setExamplesSeen(prev => prev + 1);
```

---

## 6) (Bonus) Evaluate current model on dataset

Add a button to quickly score accuracy/avg loss on the loaded dataset with current weights.

```ts
async function evaluateOnDataset() {
  if (!trainingExamples.length) { alert("No dataset loaded"); return; }
  let correct = 0, total = 0, lossSum = 0;
  for (const ex of trainingExamples) {
    let grid = Array.isArray(ex.pattern[0]) ? (ex.pattern as number[][]) : flatToGrid(ex.pattern as number[]);
    if (normalizeEnabled) grid = preprocess(grid, targetSize);
    const x = grid.flat();

    // forward (match your inference code)
    const z1 = currentNetworkState.current.weights.map((w,i) => w.reduce((s,wi,j)=>s+wi*x[j], currentNetworkState.current.biases[i]));
    const h = z1.map(sigmoid);
    const z2 = currentNetworkState.current.outputWeights.map((w,k) => w.reduce((s,wj,j)=>s+wj*h[j], currentNetworkState.current.outputBiases[k]));
    // softmax
    const maxZ = Math.max(...z2);
    const exps = z2.map(v => Math.exp(v - maxZ));
    const sumExp = exps.reduce((a,b)=>a+b,0) || 1;
    const p = exps.map(v => v / sumExp);

    const target = ex.label === 0 ? [1,0] : [0,1];
    const pred = p[0] > p[1] ? 0 : 1;
    if (pred === (target[0]===1?0:1)) correct++;
    total++;
    const ce = - (target[0]*Math.log(Math.max(1e-12,p[0])) + target[1]*Math.log(Math.max(1e-12,p[1])));
    lossSum += ce;
  }
  alert(`Eval — Acc: ${(100*correct/total).toFixed(1)}%  |  Avg CE: ${(lossSum/total).toFixed(4)}`);
}
```

UI button (Controls):
```tsx
<Button onClick={evaluateOnDataset} variant="outline" size="sm" className="w-full mt-2">
  Evaluate Model on Dataset
</Button>
```

---

## 7) Notes / Gotchas

- **Import replaces the model in memory.** Consider resetting any running training loops first (`setIsAutoTraining(false)`).
- **Shape validation** prevents accidental loading of wrong architectures.
- **Learning‑rate decay** below `minLR` is clamped to avoid stalling.
- **Normalization settings** are stored in the checkpoint so you replay experiments exactly.
- For comparison, export checkpoints after each epoch; import them back and use **Evaluate** to compare. You can also show the **Output Templates**/**Prototypes** to visually compare models.

---

## 8) Suggested insertion points

- Helpers (Section 1) → near other utils.
- State (Section 2) → alongside `learningRate` etc.
- Export/Import UI (Sections 3–4) → in your Controls card, below existing training buttons.
- LR Decay UI + hook (Section 5) → Network Info/Controls + end‑of‑epoch block in your epoch runner.
- Evaluate button (Section 6) → Controls card.

That’s it—this gives you reproducible **checkpoints** and an adjustable **LR schedule** so you can train, save, swap, and compare models quickly.