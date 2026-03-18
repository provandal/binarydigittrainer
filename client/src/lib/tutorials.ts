// tutorials.ts — registry of mini‑tutorials (keys referenced by HelpIcon)
export type Tutorial = { title: string; html: string };

export const MINI_TUTORIALS: Record<string, Tutorial> = {
  // User Options
  learningRate: {
    title: "Learning Rate",
    html: `
      <p><strong>What it is:</strong> The step size for each weight/bias update.</p>
      <p><strong>Update rule (per parameter θ):</strong> θ ← θ - η · ∇L</p>
      <ul>
        <li><strong>θ</strong>: any trainable parameter (an input→hidden weight, a hidden→output weight, or a bias). In code, these live in <code>weights</code>, <code>outputWeights</code>, <code>biases</code>, <code>outputBiases</code>.</li>
        <li><strong>η</strong> (eta): the <strong>learning rate</strong> (this UI field). Smaller = slower/safer; larger = faster/unstable.</li>
        <li><strong>∇L</strong>: the gradient of the loss <strong>L</strong> with respect to θ, computed by backpropagation during Steps 3–4.</li>
      </ul>
      <p><strong>Guidance:</strong> Start around 0.05–0.2 for this tiny network. If loss jumps or oscillates, lower it. If learning crawls, raise it slightly.</p>
    `,
  },

  mode: {
    title: "Mode: Training vs Inference",
    html: `
      <p><strong>Training:</strong> Runs full cycle (Steps 0–4) and updates parameters.</p>
      <p><strong>Inference (Predict):</strong> Forward pass only (Steps 0–2 without updates). <strong>No</strong> parameter changes.</p>
      <p><strong>Tip:</strong> Inference should use the same preprocessing as training (e.g., centering/scaling if enabled).</p>
    `,
  },

  trainingMode: {
    title: "Training Mode: Manual vs Dataset",
    html: `
      <ul>
        <li><strong>Manual:</strong> You draw a digit and select its label.</li>
        <li><strong>Dataset:</strong> The app loads saved samples from local storage and iterates through them.</li>
      </ul>
      <p><strong>Important:</strong> Dataset order should be shuffled each epoch to avoid learning order bias.</p>
    `,
  },

  epochs: {
    title: "Epochs",
    html: `
      <p><strong>Definition:</strong> One pass through the entire dataset.</p>
      <p><strong>Average epoch loss:</strong> mean of per-sample losses in that pass.</p>
      <p><strong>Guidance:</strong> For your 100-sample set, try 10–30 epochs with moderate learning rate.</p>
    `,
  },

  loss: {
    title: "Loss Function",
    html: `
      <p>Two common choices (your app can show which is active):</p>
      <p><strong>(a) Cross‑Entropy with Softmax</strong> L = -∑<sub>k ∈ {0,1}</sub> y<sub>k</sub> log p<sub>k</sub></p>
      <ul>
        <li><strong>y<sub>k</sub></strong>: one‑hot target for class <strong>k</strong> ([1,0] for zero, [0,1] for one). Comes from your label selection or dataset.</li>
        <li><strong>p<sub>k</sub></strong>: predicted probability of class <strong>k</strong> after softmax (see below). Computed from logits in Step 1.</li>
      </ul>
      <p><strong>(b) Mean Squared Error (MSE)</strong> L = ½∑<sub>k</sub>(a<sub>k</sub> - y<sub>k</sub>)²</p>
      <ul>
        <li><strong>a<sub>k</sub></strong>: output activation of neuron <strong>k</strong> (with sigmoid). In code: <code>outputActivations[k]</code>.</li>
      </ul>
      <p><strong>Tip:</strong> For classification with softmax, cross‑entropy + softmax usually learns faster and gives simpler gradients: δ<sub>k</sub> = p<sub>k</sub> - y<sub>k</sub>.</p>
    `,
  },

  activationSigmoid: {
    title: "Activation (Sigmoid)",
    html: `
      <p><strong>Function:</strong> σ(z) = 1/(1+e<sup>-z</sup>)</p>
      <ul>
        <li><strong>Input:</strong> pre‑activation <em>z</em> (a weighted sum + bias). In code: <code>hiddenPreActivations[j]</code> or <code>outputPreActivations[k]</code>.</li>
        <li><strong>Output:</strong> activation <em>a=σ(z)</em> in (0,1). In code: <code>hiddenActivations[j]</code>, <code>outputActivations[k]</code>.</li>
      </ul>
      <p><strong>Derivative:</strong> σ'(z) = σ(z)(1-σ(z)). Used in backprop (Steps 3–4) unless using softmax+CE at the output.</p>
    `,
  },

  gradClip: {
    title: "Gradient Clipping",
    html: `
      <p><strong>Purpose:</strong> Prevents unstable jumps by capping gradient magnitude.</p>
      <p><strong>Rule (elementwise):</strong> g ← clip(g, -c, c)</p>
      <ul>
        <li><strong>g</strong>: a gradient component for some parameter.</li>
        <li><strong>c</strong>: clip value (e.g., 1.0).</li>
      </ul>
    `,
  },

  lrDecay: {
    title: "Learning‑Rate Decay",
    html: `
      <p><strong>Purpose:</strong> Reduce η over epochs to fine‑tune near a minimum.</p>
      <p><strong>Example:</strong> η<sub>new</sub> = max(η<sub>min</sub>, η·decay) at epoch end.</p>
    `,
  },

  checkpoints: {
    title: "Checkpoints: Export/Import",
    html: `
      <p><strong>Export:</strong> Saves <code>weights</code>, <code>outputWeights</code>, <code>biases</code>, <code>outputBiases</code>, architecture, and optimizer metadata as JSON.</p>
      <p><strong>Import:</strong> Restores these arrays and updates the current network state; useful for A/B comparisons.</p>
    `,
  },

  decisionVsLogit: {
    title: "Decision vs Logit View",
    html: `
      <p><strong>Logits:</strong> raw scores before probability: z<sub>k</sub> = ∑<sub>j</sub> w<sub>j→k</sub>h<sub>j</sub> + b<sub>k</sub>.</p>
      <p><strong>Softmax probabilities:</strong> p<sub>k</sub> = e<sup>z<sub>k</sub></sup>/(e<sup>z<sub>0</sub></sup>+e<sup>z<sub>1</sub></sup>).</p>
      <p><strong>Decision difference:</strong> Δz = z<sub>0</sub> - z<sub>1</sub> = ∑<sub>j</sub> (w<sub>j→0</sub>-w<sub>j→1</sub>)h<sub>j</sub> + (b<sub>0</sub>-b<sub>1</sub>).</p>
      <ul>
        <li>Positive contributions push toward class 0; negative toward class 1.</li>
      </ul>
    `,
  },

  activationExplorer: {
    title: "Activation Explorer",
    html: `
      <p>Shows a 9×9 heatmap of a hidden neuron's input weights.</p>
      <ul>
        <li><strong>Red cells:</strong> positive weights (turning that pixel on increases the neuron's pre‑activation).</li>
        <li><strong>Blue cells:</strong> negative weights (turning that pixel on decreases the neuron's pre‑activation).</li>
        <li>Scrub the training‑iteration slider to watch the learned template evolve.</li>
      </ul>
    `,
  },

  // Training Steps
  step1: {
    title: "Step 1 — Forward Pass: Input → Hidden",
    html: `
      <p><strong>Compute hidden pre‑activations:</strong> z<sup>(h)</sup><sub>j</sub> = ∑<sub>i=1</sub><sup>81</sup> W<sub>i→j</sub> x<sub>i</sub> + b<sup>(h)</sup><sub>j</sub></p>
      <ul>
        <li><strong>x<sub>i</sub></strong>: input pixel <strong>i</strong> (0 or 1). From the 9×9 canvas; in code: <code>pixelGrid.flat()[i]</code>.</li>
        <li><strong>W<sub>i→j</sub></strong>: weight from input pixel <strong>i</strong> to hidden neuron <strong>j</strong>; in code: <code>weights[j][i]</code>.</li>
        <li><strong>b<sup>(h)</sup><sub>j</sub></strong>: bias of hidden neuron <strong>j</strong>; in code: <code>biases[j]</code>.</li>
        <li><strong>z<sup>(h)</sup><sub>j</sub></strong>: hidden pre‑activation stored in <code>hiddenPreActivations[j]</code>.</li>
      </ul>
      <p><strong>Apply activation:</strong> h<sub>j</sub> = σ(z<sup>(h)</sup><sub>j</sub>)</p>
      <ul>
        <li><strong>h<sub>j</sub></strong>: hidden activation stored in <code>hiddenActivations[j]</code>.</li>
      </ul>
    `,
  },

  step2: {
    title: "Step 2 — Forward Pass: Hidden → Output",
    html: `
      <p><strong>Compute output logits:</strong> z<sub>k</sub> = ∑<sub>j=1</sub><sup>24</sup> w<sub>j→k</sub> h<sub>j</sub> + b<sub>k</sub></p>
      <ul>
        <li><strong>w<sub>j→k</sub></strong>: weight from hidden neuron <strong>j</strong> to output <strong>k</strong>; in code: <code>outputWeights[k][j]</code>.</li>
        <li><strong>b<sub>k</sub></strong>: bias for output <strong>k</strong>; in code: <code>outputBiases[k]</code>.</li>
        <li><strong>z<sub>k</sub></strong>: output pre‑activation (logit); in code: <code>outputPreActivations[k]</code>.</li>
      </ul>
      <p><strong>Convert to probabilities (softmax):</strong> p<sub>k</sub> = e<sup>z<sub>k</sub></sup>/(e<sup>z<sub>0</sub></sup>+e<sup>z<sub>1</sub></sup>)</p>
      <ul>
        <li><strong>p<sub>k</sub></strong>: probability for class <strong>k</strong>; in code, you can derive from <code>outputPreActivations</code>.</li>
      </ul>
    `,
  },

  step3: {
    title: "Step 3 — Compute Loss",
    html: `
      <p><strong>Targets (one‑hot):</strong> y=[1,0] for digit 0, [0,1] for digit 1.</p>
      <p><strong>Cross‑Entropy (recommended):</strong> L = -∑<sub>k</sub> y<sub>k</sub> log p<sub>k</sub></p>
      <ul>
        <li><strong>Where values come from:</strong> <code>selectedLabel</code> or dataset → target <em>y</em>; Step 2 gives <em>p<sub>k</sub></em>.</li>
      </ul>
      <p><strong>(If MSE used):</strong> L=½∑<sub>k</sub> (a<sub>k</sub>-y<sub>k</sub>)², where a<sub>k</sub>=σ(z<sub>k</sub>).</p>
    `,
  },

  step4: {
    title: "Step 4 — Backprop: Output Layer",
    html: `
      <p><strong>Error signal (with softmax + cross‑entropy):</strong> δ<sub>k</sub> = p<sub>k</sub> - y<sub>k</sub></p>
      <ul>
        <li><strong>δ<sub>k</sub></strong>: gradient of loss with respect to logit <em>z<sub>k</sub></em> (stored transiently; in code you keep <code>outputErrors</code>).</li>
      </ul>
      <p><strong>Parameter updates:</strong> w<sub>j→k</sub> ← w<sub>j→k</sub> - η δ<sub>k</sub> h<sub>j</sub>, &nbsp;&nbsp;&nbsp;&nbsp; b<sub>k</sub> ← b<sub>k</sub> - η δ<sub>k</sub></p>
      <ul>
        <li>Uses <em>h<sub>j</sub></em> from Step 1 and <em>η</em> from the Learning Rate field.</li>
      </ul>
    `,
  },

  step5: {
    title: "Step 5 — Backprop: Hidden Layer",
    html: `
      <p><strong>Hidden error signal:</strong> δ<sup>(h)</sup><sub>j</sub> = σ'(z<sup>(h)</sup><sub>j</sub>) ∑<sub>k</sub> δ<sub>k</sub> w<sub>j→k</sub></p>
      <ul>
        <li><strong>δ<sup>(h)</sup><sub>j</sub></strong>: gradient for hidden pre‑activation.</li>
        <li><strong>σ'</strong>: derivative of sigmoid: σ(z)(1-σ(z)).</li>
      </ul>
      <p><strong>Parameter updates:</strong> W<sub>i→j</sub> ← W<sub>i→j</sub> - η δ<sup>(h)</sup><sub>j</sub> x<sub>i</sub>, &nbsp;&nbsp;&nbsp;&nbsp; b<sup>(h)</sup><sub>j</sub> ← b<sup>(h)</sup><sub>j</sub> - η δ<sup>(h)</sup><sub>j</sub></p>
    `,
  },

  step6: {
    title: "Step 6 — Next Sample / Reset",
    html: `
      <p>Advance to the next example (or clear the canvas in manual mode). If using epochs, loop until all samples are processed.</p>
      <p><strong>Tip:</strong> When auto‑training, prefer a loop that passes the <strong>index explicitly</strong> rather than relying on asynchronous state updates.</p>
    `,
  },
};
