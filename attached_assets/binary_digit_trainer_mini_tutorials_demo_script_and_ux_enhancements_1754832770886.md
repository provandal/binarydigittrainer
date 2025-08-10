# Binary Digit Trainer — Mini‑Tutorials, Demo Script, and UX Enhancements

This package includes:

1. **Mini‑tutorials** for every key UI option and each of the **six training steps**. Each tutorial defines **every symbol** used in formulas and where each value comes from in the app.
2. A **5–10 minute demo script** that uses the tool to explain training and inference.
3. **Feature ideas** that make concepts more intuitive while you present.
4. Drop‑in **code snippets** (help icon + tutorial registry) to wire question‑mark popovers to these explanations.

---

## A. Drop‑in wiring for inline help (Popover + Registry)

> These snippets let you attach a small `?` icon next to any label. Clicking it opens the matching mini‑tutorial.

```tsx
// tutorials.ts — registry of mini‑tutorials (keys referenced by HelpIcon)
export type Tutorial = { title: string; html: string };

export const MINI_TUTORIALS: Record<string, Tutorial> = {
  // Filled in with content from Section B and C — copy the HTML strings there
};
```

```tsx
// HelpIcon.tsx — minimal popover using shadcn/ui
import { useState } from "react";
import { HelpCircle } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { MINI_TUTORIALS } from "./tutorials";

export function HelpIcon({ k }: { k: keyof typeof MINI_TUTORIALS }) {
  const t = MINI_TUTORIALS[k];
  if (!t) return null;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button aria-label={`Help: ${t.title}`} className="ml-1 text-blue-600 hover:text-blue-700 align-middle">
          <HelpCircle className="w-4 h-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="max-w-[34rem] prose prose-sm">
        <h4 className="font-semibold mb-2">{t.title}</h4>
        {/* We store HTML to allow subscript/superscript and italics in formulas */}
        <div dangerouslySetInnerHTML={{ __html: t.html }} />
      </PopoverContent>
    </Popover>
  );
}
```

```tsx
// Usage example in your UI label
<label className="text-sm font-medium">
  Learning Rate
  <HelpIcon k="learningRate" />
</label>
```

**How to populate the registry:** Copy the HTML blocks under Sections **B** (options) and **C** (training steps) into `MINI_TUTORIALS`.

---

## B. Mini‑tutorials — User Options

> Copy each block’s HTML into `MINI_TUTORIALS["key"].html`. Titles become `title`.

### 1) Learning Rate  (key: `learningRate`)

**What it is:** The step size for each weight/bias update.

**Update rule (per parameter θ):** \(\theta \leftarrow \theta - \eta \cdot \nabla\!L\)

- **\(\theta\)**: any trainable parameter (an input→hidden weight, a hidden→output weight, or a bias). In code, these live in `weights`, `outputWeights`, `biases`, `outputBiases`.
- **\(\eta\)** (eta): the **learning rate** (this UI field). Smaller = slower/safer; larger = faster/unstable.
- **\(\nabla\!L\)**: the gradient of the loss **L** with respect to \(\theta\), computed by backpropagation during Steps 3–4.

**Guidance:** Start around 0.05–0.2 for this tiny network. If loss jumps or oscillates, lower it. If learning crawls, raise it slightly.

---

### 2) Mode: Training vs Inference  (key: `mode`)

**Training:** Runs full cycle (Steps 0–4) and updates parameters.

**Inference (Predict):** Forward pass only (Steps 0–2 without updates). **No** parameter changes.

**Tip:** Inference should use the same preprocessing as training (e.g., centering/scaling if enabled).

---

### 3) Training Mode: Manual vs Dataset  (key: `trainingMode`)

- **Manual:** You draw a digit and select its label.
- **Dataset:** The app loads saved samples (`/api/training-examples`) and iterates through them.

**Important:** Dataset order should be shuffled each epoch to avoid learning order bias.

---

### 4) Epochs  (key: `epochs`)

**Definition:** One pass through the entire dataset.

**Average epoch loss:** mean of per-sample losses in that pass.

**Guidance:** For your 100-sample set, try 10–30 epochs with moderate learning rate.

---

### 5) Loss Function  (key: `loss`)

Two common choices (your app can show which is active):

**(a) Cross‑Entropy with Softmax** \(L = -\sum_{k \in \{0,1\}} y_k\,\log p_k\)

- **\(y_k\)**: one‑hot target for class **k** (\([1,0]\) for zero, \([0,1]\) for one). Comes from your label selection or dataset.
- **\(p_k\)**: predicted probability of class **k** after softmax (see below). Computed from logits in Step 1.

**(b) Mean Squared Error (MSE)** \(L = \tfrac12\sum_{k}(a_k - y_k)^2\)

- **\(a_k\)**: output activation of neuron **k** (with sigmoid). In code: `outputActivations[k]`.

**Tip:** For classification with softmax, cross‑entropy + softmax usually learns faster and gives simpler gradients: \(\delta_k = p_k - y_k\).

---

### 6) Activation (Sigmoid)  (key: `activationSigmoid`)

**Function:** \(\sigma(z) = 1/(1+e^{-z})\)

- **Input:** pre‑activation \(z\) (a weighted sum + bias). In code: `hiddenPreActivations[j]` or `outputPreActivations[k]`.
- **Output:** activation \(a=\sigma(z)\) in (0,1). In code: `hiddenActivations[j]`, `outputActivations[k]`.

**Derivative:** \(\sigma'(z) = \sigma(z)(1-\sigma(z))\). Used in backprop (Steps 3–4) unless using softmax+CE at the output.

---

### 7) Gradient Clipping  (key: `gradClip`)

**Purpose:** Prevents unstable jumps by capping gradient magnitude.

**Rule (elementwise):** \(g \leftarrow \mathrm{clip}(g, -c, c)\)

- **\(g\)**: a gradient component for some parameter.
- **\(c\)**: clip value (e.g., 1.0).

---

### 8) Learning‑Rate Decay  (key: `lrDecay`)

**Purpose:** Reduce \(\eta\) over epochs to fine‑tune near a minimum.

**Example:** \(\eta_{new} = \max(\eta_{min},\, \eta\cdot \text{decay})\) at epoch end.

---

### 9) Checkpoints: Export/Import  (key: `checkpoints`)

**Export:** Saves `weights`, `outputWeights`, `biases`, `outputBiases`, architecture, and optimizer metadata as JSON.

**Import:** Restores these arrays and updates the current network state; useful for A/B comparisons.

---

### 10) Decision vs Logit View  (key: `decisionVsLogit`)

**Logits:** raw scores before probability: \(z_k = \sum_j w_{j\to k}h_j + b_k\).

**Softmax probabilities:** \(p_k = e^{z_k}/(e^{z_0}+e^{z_1})\).

**Decision difference:** \(\Delta z = z_0 - z_1 = \sum_j (w_{j\to 0}-w_{j\to 1})h_j + (b_0-b_1)\).

- Positive contributions push toward class 0; negative toward class 1.

---

### 11) Activation Explorer  (key: `activationExplorer`)

Shows a 9×9 heatmap of a hidden neuron’s input weights.

- **Red cells:** positive weights (turning that pixel on increases the neuron’s pre‑activation).
- **Blue cells:** negative weights (turning that pixel on decreases the neuron’s pre‑activation).
- Scrub the training‑iteration slider to watch the learned template evolve.

---

## C. Mini‑tutorials — The Six Training Steps

> These match the Step cards in your UI. Use the keys below, e.g., `step0`, `step1`, …

### Step 0 — Forward Pass: Input → Hidden  (key: `step0`)

**Compute hidden pre‑activations:** \(z^{(h)}_j = \sum_{i=1}^{81} W_{i\to j}\,x_i + b^{(h)}_j\)

- **\(x_i\)**: input pixel **i** (0 or 1). From the 9×9 canvas; in code: `pixelGrid.flat()[i]`.
- **\(W_{i\to j}\)**: weight from input pixel **i** to hidden neuron **j**; in code: `weights[j][i]`.
- **\(b^{(h)}_j\)**: bias of hidden neuron **j**; in code: `biases[j]`.
- **\(z^{(h)}_j\)**: hidden pre‑activation stored in `hiddenPreActivations[j]`.

**Apply activation:** \(h_j = \sigma\big(z^{(h)}_j\big)\)

- **\(h_j\)**: hidden activation stored in `hiddenActivations[j]`.

---

### Step 1 — Forward Pass: Hidden → Output  (key: `step1`)

**Compute output logits:** \(z_k = \sum_{j=1}^{24} w_{j\to k}\,h_j + b_k\)

- **\(w_{j\to k}\)**: weight from hidden neuron **j** to output **k**; in code: `outputWeights[k][j]`.
- **\(b_k\)**: bias for output **k**; in code: `outputBiases[k]`.
- **\(z_k\)**: output pre‑activation (logit); in code: `outputPreActivations[k]`.

**Convert to probabilities (softmax):** \(p_k = \frac{e^{z_k}}{e^{z_0}+e^{z_1}}\)

- **\(p_k\)**: probability for class **k**; in code, you can derive from `outputPreActivations`.

---

### Step 2 — Compute Loss  (key: `step2`)

**Targets (one‑hot):** \(y=[1,0]\) for digit 0, \([0,1]\) for digit 1.

**Cross‑Entropy (recommended):** \(L = -\sum_{k} y_k\,\log p_k\)

- **Where values come from:** `selectedLabel` or dataset → target \(y\); Step 1 gives \(p_k\).

**(If MSE used):** \(L=\tfrac12\sum_k (a_k-y_k)^2\), where \(a_k=\sigma(z_k)\).

---

### Step 3 — Backprop: Output Layer  (key: `step3`)

**Error signal (with softmax + cross‑entropy):** \(\delta_k = p_k - y_k\)

- **\(\delta_k\)**: gradient of loss with respect to logit \(z_k\) (stored transiently; in code you keep `outputErrors`).

**Parameter updates:** \(w_{j\to k} \leftarrow w_{j\to k} - \eta\,\delta_k\,h_j,\qquad b_k \leftarrow b_k - \eta\,\delta_k\)

- Uses \(h_j\) from Step 0 and \(\eta\) from the Learning Rate field.

---

### Step 4 — Backprop: Hidden Layer  (key: `step4`)

**Hidden error signal:** \(\delta^{(h)}_j = \sigma'\!\big(z^{(h)}_j\big) \sum_{k} \delta_k\,w_{j\to k}\)

- **\(\delta^{(h)}_j\)**: gradient for hidden pre‑activation.
- **\(\sigma'\)**: derivative of sigmoid: \(\sigma(z)(1-\sigma(z))\).

**Parameter updates:** \(W_{i\to j} \leftarrow W_{i\to j} - \eta\,\delta^{(h)}_j\,x_i,\qquad b^{(h)}_j \leftarrow b^{(h)}_j - \eta\,\delta^{(h)}_j\)

---

### Step 5 — Next Sample / Reset  (key: `step5`)

Advance to the next example (or clear the canvas in manual mode). If using epochs, loop until all samples are processed.

**Tip:** When auto‑training, prefer a loop that passes the **index explicitly** rather than relying on asynchronous state updates.

---

## D. 5–10 Minute Demo Script (using the tool)

> Total time \~8 minutes. Bold items are actions to perform in the UI.

**1) Setup (0:00–1:00)**

- “We’ll train a tiny neural net (81→24→2) to distinguish **0** vs **1**.”
- **Reset Network**. **Show** learning rate. Click **?** to open its tutorial.

**2) Forward pass intuition (1:00–2:30)**

- **Draw** a simple “0”. **Step 0** to compute hidden activations.
- Open **Activation Explorer** for one hidden neuron. “This 9×9 map is its template. Red pixels push it up; blue push down.”
- **Step 1**. “Outputs compute raw scores called **logits**: \(z_k=\sum_j w_{j\to k}h_j+b_k\).” Open **Logit vs Decision** help.

**3) Probabilities & decision (2:30–3:30)**

- “Softmax turns logits into probabilities: \(p_k=e^{z_k}/(e^{z_0}+e^{z_1})\). We pick the larger.”
- Toggle **Decision (z₀−z₁)** view. “Each hidden unit votes via \((w_{j→0}-w_{j→1})h_j\).” Click a top contributor to jump to its heatmap.

**4) Loss & why gradients move weights (3:30–5:00)**

- **Step 2** to show loss. Open **Loss** help. “We use cross‑entropy.”
- **Step 3–4** once. “At the output, the error signal is \(\delta_k=p_k-y_k\). We nudge weights: \(w←w-\eta\,\delta_k\,h_j\). Hidden errors backprop using the chain rule.”
- **Scrub** the iteration slider in the dialog to show how a template changes.

**5) Train a full epoch (5:00–6:30)**

- Switch to **Training Set**. Set **Epochs=5**. **Start Training**. Show the loss trend.
- After training, switch to **Predict (Inference)**.

**6) Inference & interpretation (6:30–8:00)**

- **Draw** a “1”, then a “0”. Watch probabilities.
- Open **Output 0** dialog → “Top contributors.” Explain drivers vs suppressors in the **Decision** view.
- (Optional) Show **Checkpoints**: save, reload, compare behavior.

Close: “You’ve just seen forward pass → probabilities → loss → gradients → learned templates → interpretable decisions.”

---

## E. Extra features that help during the demo

1. **Drivers vs Suppressors lists** (Decision view)
   - Show top positive and top negative contributions \((w_{j→0}-w_{j→1})h_j\) with mini heatmaps.
2. **Input overlay on heatmaps**
   - Outline cells where the current input has 1s. This makes the dot‑product intuition visual: lit pixel × red weight ⇒ pushes up.
3. **Bias line item**
   - Display \(b_0-b_1\) (decision view) or \(b_k\) (logit view) as a separate contributor.
4. **Per‑neuron saturation indicator**
   - Flag hidden units with \(h_j\) near 0 or 1 often (sigmoid saturation) — teach why gradients vanish there.
5. **Misclassification replay**
   - Button to auto‑replay the last wrong prediction step‑by‑step with contributors highlighted.
6. **Noise/transform sliders**
   - Add “stroke thickness” or small translation/scale to show how robust the learned templates are.

---

## F. Ready‑to‑paste registry entries (HTML strings)

> Copy these into `MINI_TUTORIALS` in `tutorials.ts`.

```ts
export const MINI_TUTORIALS = {
  learningRate: {
    title: "Learning Rate",
    html: `
<p><strong>What it is</strong>: the step size used to update every weight and bias.</p>
<p><em>Update rule</em> (each parameter <code>θ</code>):</p>
<p>θ ← θ − η · ∇L</p>
<ul>
  <li><strong>θ</strong>: a trainable parameter (any entry in <code>weights</code>, <code>outputWeights</code>, <code>biases</code>, <code>outputBiases</code>).</li>
  <li><strong>η</strong> (eta): the learning rate (this field).</li>
  <li><strong>∇L</strong>: gradient of the loss with respect to θ, computed during backprop (Steps 3–4).</li>
</ul>
<p><strong>Tip</strong>: start moderate; if loss oscillates, lower η; if learning is too slow, raise slightly.</p>
`},
  mode: { title: "Mode: Training vs Inference", html: `
<p><strong>Training</strong>: runs forward + backprop and updates parameters.</p>
<p><strong>Inference (Predict)</strong>: forward pass only; no updates.</p>
<p>Use the same preprocessing in both modes for consistent behavior.</p>
`},
  trainingMode: { title: "Training Mode: Manual vs Dataset", html: `
<p><strong>Manual</strong>: draw a digit and select its label.</p>
<p><strong>Dataset</strong>: iterate through saved samples from the training set.</p>
<p><em>Tip</em>: shuffle order each epoch.</p>
`},
  epochs: { title: "Epochs", html: `
<p><strong>Epoch</strong>: one full pass through the dataset. The app reports average loss over that pass.</p>
`},
  loss: { title: "Loss Function", html: `
<p><strong>Cross-Entropy + Softmax</strong> (recommended): L = −∑ y_k log p_k</p>
<ul>
  <li><strong>y_k</strong>: one-hot target (e.g., [1,0] for zero).</li>
  <li><strong>p_k</strong>: predicted probability after softmax.</li>
</ul>
<p><strong>MSE</strong>: ½∑(a_k − y_k)^2 with a_k = σ(z_k).</p>
`},
  activationSigmoid: { title: "Activation (Sigmoid)", html: `
<p>σ(z) = 1/(1+e^{−z}). Input: pre-activation z (a weighted sum + bias). Output: activation a ∈ (0,1).</p>
<p>Derivative: σ'(z) = σ(z)(1−σ(z)) used in backprop.</p>
`},
  gradClip: { title: "Gradient Clipping", html: `
<p>Prevents unstable updates by limiting gradient magnitude: g ← clip(g, −c, c).</p>
`},
  lrDecay: { title: "Learning-Rate Decay", html: `
<p>Gradually reduce η each epoch: η_new = max(η_min, η · decay).</p>
`},
  checkpoints: { title: "Checkpoints (Export/Import)", html: `
<p>Save and restore network parameters (weights, biases, architecture, metadata) as JSON to compare models or resume training.</p>
`},
  decisionVsLogit: { title: "Decision vs Logit", html: `
<p><strong>Logit</strong> z_k = ∑ w_{j→k} h_j + b_k (raw score before probability).</p>
<p><strong>Softmax</strong> p_k = e^{z_k}/(e^{z_0}+e^{z_1}).</p>
<p><strong>Decision</strong> Δz = z_0 − z_1 = ∑(w_{j→0} − w_{j→1})h_j + (b_0 − b_1).</p>
`},
  activationExplorer: { title: "Activation Explorer", html: `
<p>9×9 heatmap of a hidden neuron’s input weights. Red: positive (pixel on pushes neuron up). Blue: negative (pixel on pulls down). Scrub training iterations to see the learned template form.</p>
`},
  step0: { title: "Step 0 — Input → Hidden", html: `
<p>Compute hidden pre-activations z_j^(h) = ∑ W_{i→j} x_i + b_j^(h), then activations h_j = σ(z_j^(h)).</p>
<ul>
  <li><strong>x_i</strong>: input pixels from the canvas (0 or 1).</li>
  <li><strong>W_{i→j}</strong>: input→hidden weights.</li>
  <li><strong>b_j^(h)</strong>: hidden biases.</li>
</ul>
`},
  step1: { title: "Step 1 — Hidden → Output", html: `
<p>Compute output logits z_k = ∑ w_{j→k} h_j + b_k; then probabilities p_k via softmax.</p>
`},
  step2: { title: "Step 2 — Loss", html: `
<p>Targets y are one-hot ([1,0] for zero; [0,1] for one). Cross-entropy: L = −∑ y_k log p_k.</p>
`},
  step3: { title: "Step 3 — Backprop (Output)", html: `
<p>With softmax+CE, error is δ_k = p_k − y_k. Update: w_{j→k} ← w_{j→k} − η δ_k h_j; b_k ← b_k − η δ_k.</p>
`},
  step4: { title: "Step 4 — Backprop (Hidden)", html: `
<p>Hidden error: δ_j^(h) = σ'(z_j^(h)) ∑ δ_k w_{j→k}. Update: W_{i→j} ← W_{i→j} − η δ_j^(h) x_i; b_j^(h) ← b_j^(h) − η δ_j^(h).</p>
`},
  step5: { title: "Step 5 — Next Sample", html: `
<p>Move to the next example (or clear the canvas) and repeat. For epochs, loop over the full set.</p>
`}
} satisfies Record<string, Tutorial>;
```

---

### Notes

- Formulas define each symbol and map directly to your state: `weights`, `biases`, `outputWeights`, `outputBiases`, `hiddenPreActivations`, `hiddenActivations`, `outputPreActivations`.
- If you add BatchNorm/Dropout later, add inference-mode notes where behavior differs.

