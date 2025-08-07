# Binary Digit Trainer — Async State Update Workaround

**Goal:** Eliminate training glitches caused by React’s async state updates (stale `pixelGrid`, `selectedLabel`, `currentExampleIndex`, etc.) during the automated training loop.

This plan avoids a full rewrite. It isolates the _training engine_ from React state and makes the loop deterministic.

---

## TL;DR (What to change)

1) **Stop reading React state for training logic.** Use refs/local snapshots for anything the loop depends on.
2) **Replace `setInterval`/recursive timers with an `async/await` loop** that you can cancel.
3) **Snapshot the current sample (grid + label) _before_ running steps** and run all 6 steps using the snapshot.
4) **Make `getPixelValues()` read from a ref, not state,** so step 0 never sees a stale grid.
5) **Use functional state updates for UI-only state.** Never rely on those values for training logic.
6) **Wire a single cancel flag (`shouldStopTraining.current`)** that the loop checks after each awaited delay.
7) **(Optional) Avoid `flushSync` unless you need to block on paint;** the ref approach makes it unnecessary.

---

## Why this fixes the bug

React batches and defers `setState`. In `runToNextSample`/multi-epoch code we’re setting `pixelGrid`, `selectedLabel`, `step`, and then immediately calling `nextStep()` inside a timer. The first `forwardPassHidden()` (step 0) can run **before** the state setters commit, so it reads **stale** inputs.

The fix is to **decouple** training logic from React’s async state by using **refs** and **local variables** for the _current sample_ and _inputs_.

---

## Concrete edits

### 1) Track the pixel grid in a ref and update it synchronously

Add a ref and a safe setter that updates both the ref and React state:

```ts
// NEW: at top-level state
const pixelGridRef = useRef<number[][]>(initialPixelGrid);

// REPLACE your setPixelGrid helper with this:
const setPixelGrid = (grid: number[][] | number[]) => {
  const normalized = Array.isArray(grid[0]) ? (grid as number[][]) : flatToGrid(grid as number[]);
  pixelGridRef.current = normalized;             // <- immediate, used by training logic
  setPixelGridState(normalized);                 // <- async, UI only
};
```

Then change `getPixelValues()` to read from the ref:

```ts
// REPLACE:
const getPixelValues = () => pixelGrid.flat();

// WITH:
const getPixelValues = () => pixelGridRef.current.flat();
```

### 2) Keep the selected label in a ref (for training logic)

```ts
const selectedLabelRef = useRef<number>(0);
useEffect(() => { selectedLabelRef.current = selectedLabel; }, [selectedLabel]);

const setSelectedLabelSafe = (label: number) => {
  selectedLabelRef.current = label;
  setSelectedLabel(label);
};
```

(If your dataset stores one-hot labels, you can also keep `currentTargetRef` (number[]) instead of the integer label.)

### 3) Snapshot the sample before running steps

Inside `runToNextSample` / the multi-epoch loop, take a **local snapshot** of everything needed:

```ts
const example = trainingExamples[idx];                 // do not read again inside timers
const grid = Array.isArray(example.pattern[0])
  ? (example.pattern as number[][])
  : flatToGrid(example.pattern as number[]);

const oneHot = example.label === 0 ? [1,0] : [0,1];
const uiDigit = oneHot[0] === 1 ? 0 : 1;

// Update refs & UI (UI can lag; refs are immediate)
setPixelGrid(grid);            // updates pixelGridRef immediately
setSelectedLabelSafe(uiDigit); // updates selectedLabelRef immediately

// Also cache target and inputs in a ref used by training logic:
currentNetworkState.current.currentTarget = oneHot;
currentNetworkState.current.inputs = grid.flat();
```

### 4) Replace timer recursion with a cancellable async loop

```ts
const sleep = (ms: number) => new Promise(res => setTimeout(res, ms));
const shouldStopTraining = useRef(false);

async function runStepsForCurrentSample() {
  for (const s of [0, 1, 2, 3, 4, 5]) {
    if (shouldStopTraining.current) return false;
    nextStep(s);                     // use the forced-step path; UI step is cosmetic
    await sleep(autoTrainingSpeed);
  }
  return true;
}

async function runEpochs() {
  setIsAutoTraining(true);
  setTrainingCompleted(false);

  for (let epoch = 1; epoch <= numberOfEpochs; epoch++) {
    currentEpochLoss.current = [];
    setCurrentEpoch(epoch);

    // Optional: shuffle order each epoch
    const order = Array.from({length: trainingExamples.length}, (_, i) => i).sort(() => Math.random() - 0.5);

    for (const idx of order) {
      if (shouldStopTraining.current) break;

      const ex = trainingExamples[idx];
      const grid = Array.isArray(ex.pattern[0]) ? ex.pattern as number[][] : flatToGrid(ex.pattern as number[]);
      const oneHot = ex.label === 0 ? [1,0] : [0,1];

      setPixelGrid(grid);
      setSelectedLabelSafe(oneHot[0] === 1 ? 0 : 1);
      currentNetworkState.current.currentTarget = oneHot;
      currentNetworkState.current.inputs = grid.flat();

      setCurrentTrainingIndex(idx); // UI only

      const completed = await runStepsForCurrentSample();
      if (!completed) break;

      currentEpochLoss.current.push(currentNetworkState.current.loss);
    }

    if (currentEpochLoss.current.length) {
      const avg = currentEpochLoss.current.reduce((a,b)=>a+b,0)/currentEpochLoss.current.length;
      setEpochLossHistory(prev => [...prev, { epoch, averageLoss: avg }]);
    }

    if (shouldStopTraining.current) break;
  }

  setIsAutoTraining(false);
  setTrainingCompleted(true);
}
```

Hook your “Process Training Set” button to `runEpochs()`. Add a “Stop” button that flips `shouldStopTraining.current = true`.

### 5) Training functions read from refs, not state

- `getPixelValues()` → returns `currentNetworkState.current.inputs ?? pixelGridRef.current.flat()`
- Loss/backprop targets: `const target = currentNetworkState.current.currentTarget;`

### 6) Prediction path unchanged

Inference mode can keep using React state since latency isn’t critical there.

---

## Testing checklist

- On automated training, step 0 uses the correct grid for the first forward pass every time.
- Stop cancels within one step delay.
- Epoch average loss decreases for a fixed dataset and seed.
- No off-by-one when wrapping dataset.
- Manual mode unchanged; inference still works.
