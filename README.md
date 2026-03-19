<p align="center">
  <img src="assets/banner.png" alt="Binary Digit Trainer" width="100%">
</p>

# Binary Digit Trainer

An interactive neural network that learns to recognize binary digits (0 and 1) — entirely in your browser. Draw a digit on a 9x9 pixel grid, step through each phase of training, and watch the math happen in real time.

**[Try it live](https://provandal.github.io/binarydigittrainer/)**

## What is this?

Binary Digit Trainer is a hands-on learning tool that makes neural networks concrete. Instead of treating the network as a black box, it exposes every step:

1. **Forward Pass** — See how input pixels flow through weighted connections to hidden neurons (sigmoid) and output neurons (softmax)
2. **Loss Calculation** — Watch cross-entropy loss measure how wrong the prediction is
3. **Backpropagation** — Step through gradient computation as errors flow backward, updating weights layer by layer
4. **Convergence** — Train over multiple epochs and watch loss decrease as the network learns

The architecture is intentionally small (81 → 24 → 2) so every weight, activation, and gradient is visible and understandable.

## Features

- **Step-by-step training** — Walk through all 6 steps of a training cycle with formulas shown at each stage
- **14-step guided tour** — Interactive walkthrough from first drawing to trained inference
- **Multi-epoch training** — Automated training with real-time progress and loss tracking
- **Learning rate decay** — Configurable decay schedule with visual LR history
- **Weight visualization** — Heatmaps showing what each hidden neuron has learned
- **Checkpoint export/import** — Save and restore trained models as JSON
- **Inference mode** — Switch to inference, draw a digit, and get predictions with confidence scores
- **Mobile support** — Touch drawing and responsive layout for phones and tablets
- **No backend** — Everything runs client-side; training data persists in localStorage

## Getting started

### Try it online

Visit **[provandal.github.io/binarydigittrainer](https://provandal.github.io/binarydigittrainer/)** — no installation required.

### Run locally

```bash
git clone https://github.com/provandal/binarydigittrainer.git
cd binarydigittrainer
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Prerequisites

- [Node.js](https://nodejs.org/) 18 or later
- npm 9 or later

## Available scripts

| Command                | Description                              |
| ---------------------- | ---------------------------------------- |
| `npm run dev`          | Start development server with hot reload |
| `npm run build`        | Production build to `dist/`              |
| `npm run preview`      | Preview production build locally         |
| `npm run test`         | Run all tests                            |
| `npm run test:watch`   | Run tests in watch mode                  |
| `npm run typecheck`    | TypeScript type checking                 |
| `npm run lint`         | ESLint analysis                          |
| `npm run format`       | Format all files with Prettier           |
| `npm run format:check` | Check formatting without writing         |

## Architecture

```
81 inputs (9x9 pixel grid)
    ↓
24 hidden neurons (sigmoid activation)
    ↓
2 output neurons (softmax → probability distribution)
```

**Loss function:** Cross-entropy
**Weight initialization:** Xavier/Glorot
**Gradient clipping:** Prevents exploding gradients during training

## Project structure

```
client/src/
├── components/
│   ├── trainer/           # 10 UI sub-components
│   │   ├── AppHeader.tsx
│   │   ├── DrawingCanvas.tsx
│   │   ├── NetworkDiagram.tsx
│   │   ├── TrainingStepPanel.tsx
│   │   ├── WeightDetailView.tsx
│   │   ├── Heatmap9x9.tsx
│   │   ├── DatasetEditorDialog.tsx
│   │   ├── EpochSelectionDialog.tsx
│   │   ├── DebugHistoryPanel.tsx
│   │   └── AboutDialog.tsx
│   ├── binary-digit-trainer.tsx  # Orchestrator
│   └── GuidedTour.tsx
├── hooks/                 # 5 custom hooks
│   ├── useNetworkParams.ts
│   ├── useCanvasDrawing.ts
│   ├── useTrainingLoop.ts
│   ├── useTourState.ts
│   └── useModelManagement.ts
├── lib/                   # Pure logic modules
│   ├── nn-math.ts         # sigmoid, softmax, gradient clipping
│   ├── nn-engine.ts       # Forward/backward pass
│   ├── nn-checkpoint.ts   # Checkpoint serialization
│   ├── nn-helpers.ts      # Data utilities
│   ├── color-schemes.ts   # Visualization colors
│   └── __tests__/         # 52 unit tests
└── data/                  # Pre-trained model & sample dataset
```

## Tech stack

- **React 18** with TypeScript
- **Vite** for bundling and dev server
- **Tailwind CSS** with shadcn/ui components
- **Vitest** for testing (52 tests)
- **ESLint** + **Prettier** for code quality
- **GitHub Actions** for CI (typecheck → lint → format → test → build)
- **GitHub Pages** for deployment

## Who is this for?

- **Students** learning machine learning fundamentals for the first time
- **Educators** looking for an interactive demo that requires zero environment setup
- **Developers** who use ML libraries but want deeper intuition about what happens inside
- **Career switchers** building ML knowledge from the ground up

## Contributing

Found a bug or have an idea? Open a thread in [Discussions](https://github.com/provandal/binarydigittrainer/discussions) or submit an issue.

## License

MIT
