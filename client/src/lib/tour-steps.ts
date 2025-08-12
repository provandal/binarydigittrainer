import { TourStep } from '@/components/GuidedTour';

export const createTourSteps = (
  checkCanvasHasDrawing: () => boolean,
  checkTrainingStarted: () => boolean,
  validOneClick: () => boolean,
  validFullCycle: () => boolean,
  checkDatasetLoaded: () => boolean,
  checkNextSampleClicked: () => boolean,
  checkEpochTrainingStarted: () => boolean,
  checkTrainingCompletedAndCheckpointSaved: () => boolean,
  checkInferenceModeActive: () => boolean,
  checkWeightVisualizationOpened: () => boolean
): TourStep[] => [
  {
    id: 'welcome',
    title: 'Welcome to Binary Digit Trainer!',
    content: `
      <p>This interactive tour will guide you through all the key features of the Neural Network trainer.</p>
      <p><strong>You'll learn:</strong></p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>Manual training workflow</li>
        <li>Dataset-based training</li>
        <li>Multi-epoch training</li>
        <li>Checkpoints and inference</li>
        <li>Weight visualization</li>
      </ul>
      <p class="text-xs text-gray-500 mt-2">💡 <strong>Tip:</strong> Click ? icons for detailed mathematical explanations!</p>
      <div class="mt-3 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
        <strong>⚠️ Note:</strong> Starting this tour will reset the neural network to its default state to ensure a clean learning experience.
      </div>
    `
  },
  {
    id: 'reset',
    title: 'Starting Fresh',
    content: `
      <p>We've reset the network to start with clean weights and a blank canvas.</p>
      <p>The network architecture is <strong>81 → 24 → 2</strong>:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>81 input neurons (9×9 pixel grid)</li>
        <li>24 hidden neurons with sigmoid activation</li>
        <li>2 output neurons for digits 0 and 1</li>
      </ul>
    `
  },
  {
    id: 'draw-digit',
    title: 'Step 1: Draw a Binary Digit',
    content: `
      <p>Let's start with <strong>manual training</strong>. First, draw a binary digit (0 or 1) on the 9×9 canvas.</p>
      <p><strong>Drawing tips:</strong></p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>Click and drag to fill pixels</li>
        <li>Each pixel is binary: 0 (white) or 1 (black)</li>
        <li>Draw clear, simple shapes</li>
      </ul>
    `,
    target: '.grid.grid-cols-9',
    action: 'Draw a digit (0 or 1) on the canvas by clicking and dragging',
    waitForAction: true,
    validation: checkCanvasHasDrawing
  },
  {
    id: 'select-label',
    title: 'Step 2: Select the Correct Label',
    content: `
      <p>Great! Now select the correct label that matches what you drew.</p>
      <p>The network needs to know the correct answer to learn from mistakes during training.</p>
    `,
    target: '[data-tour-target="label-selector"]',
    action: 'Choose the correct label (0 or 1) that matches your drawing',
    waitForAction: false
  },
  {
    id: 'training-steps',
    title: 'Step 3: Start Training Process',
    content: `
      <p>Now let's begin the training process. Click the highlighted "Next Step" button once to start.</p>
      <p>Neural network training happens in 6 steps that you'll see step by step:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li><strong>Step 1-2:</strong> Forward pass (input → hidden → output)</li>
        <li><strong>Step 3:</strong> Calculate loss (error)</li>
        <li><strong>Step 4-5:</strong> Backpropagation (update weights)</li>
        <li><strong>Step 6:</strong> Complete the cycle</li>
      </ul>
    `,
    target: '[data-tour-target="next-step-button"]',
    action: 'Click "Next Step" one time to start',
    waitForAction: true,
    validation: validOneClick,
    autoAdvanceOnValid: true,
    pin: 'bottom-left'
  },
  {
    id: 'complete-training',
    title: 'Step 4: Complete the Training Cycle',
    content: `
      <p>Great! You've started training. Now keep clicking "Next Step" to complete the full training cycle:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>Watch activations flow through the network</li>
        <li>See the loss calculation</li>
        <li>Observe weight updates during backpropagation</li>
      </ul>
      <p class="text-xs text-gray-500 mt-2">Continue clicking "Next Step" to complete all 6 training steps.</p>
    `,
    target: '[data-tour-target="next-step-button"]',
    action: 'Finish the remaining training steps of this cycle',
    waitForAction: true,
    validation: validFullCycle,
    pin: 'bottom-left'
  },
  {
    id: 'dataset-training',
    title: 'Step 7: Training Set Mode',
    content: `
      <p>Now let's try <strong>training set mode</strong>. We'll load the pre-built training examples and step through them automatically.</p>
      <p>Switch to "Training Set" mode to access training examples with both 0s and 1s.</p>
    `,
    target: '[data-tour-target="dataset-button"]',
    action: 'Click the "Training Set" radio button to switch training modes',
    waitForAction: true,
    validation: checkDatasetLoaded,
    pin: 'bottom-left'
  },
  {
    id: 'next-sample',
    title: 'Step 8: Run to Next Sample',
    content: `
      <p>With training set mode active, you can now step through training examples automatically.</p>
      <p>The "Run to Next Sample" button will:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>Load the next training example</li>
        <li>Run all 6 training steps automatically</li>
        <li>Move to the next example in sequence</li>
      </ul>
    `,
    target: '[data-tour-target="run-next-sample-button"]',
    action: 'Click "Run to Next Sample" to process the next training example',
    waitForAction: true,
    validation: checkNextSampleClicked,
    pin: 'bottom-left'
  },
  {
    id: 'multi-epoch',
    title: 'Step 9: Multi-Epoch Training',
    content: `
      <p>For serious training, you can process the entire training set multiple times using <strong>epochs</strong>.</p>
      <p>An epoch means going through every training example once. Multiple epochs help the network learn better patterns.</p>
    `,
    target: '[data-tour-target="multi-epoch-button"]',
    action: 'Click "Process Training Set" to start automated training',
    waitForAction: true,
    validation: checkEpochTrainingStarted,
    pin: 'bottom-left'
  },
  {
    id: 'checkpoints',
    title: 'Step 10: Wait for Training or Save Progress',
    content: `
      <p>If training is still running, <strong>wait for it to complete</strong> or click <strong>"Stop Training"</strong> to stop it early.</p>
      <p>Once training stops, you can save your progress using checkpoints.</p>
      <p><strong>Checkpoint contains:</strong></p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>All network weights and biases</li>
        <li>Training history and loss data</li>
        <li>Learning rate and configuration</li>
      </ul>
    `,
    target: '[data-tour-target="save-checkpoint-button"], [data-tour-target="stop-training-button"]',
    action: 'Wait for training to complete, then click "Export" to save your model',
    waitForAction: true,
    validation: checkTrainingCompletedAndCheckpointSaved,
    pin: 'bottom-left'
  },
  {
    id: 'inference-mode',
    title: 'Step 11: Test Your Model',
    content: `
      <p>Switch to <strong>Inference Mode</strong> to test your trained network!</p>
      <p>In inference mode:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>Draw digits on the canvas</li>
        <li>Get instant predictions</li>
        <li>See confidence scores</li>
        <li>No training occurs - just testing</li>
      </ul>
    `,
    target: 'input[value="inference"]',
    action: 'Click the "Inference" radio button to switch to testing mode',
    waitForAction: true,
    validation: checkInferenceModeActive,
    pin: 'bottom-left'
  },
  {
    id: 'test-drawing',
    title: 'Step 12: Draw and Test',
    content: `
      <p>Perfect! Now you're in inference mode. Draw a digit and watch the network predict what it is.</p>
      <p>The prediction updates in real-time as you draw, showing both the predicted digit and confidence level.</p>
    `,
    target: '.grid.grid-cols-9',
    action: 'Draw a digit (0 or 1) to see the network make predictions',
    pin: 'bottom-left'
  },
  {
    id: 'weight-visualization',
    title: 'Step 13: Explore Network Internals',
    content: `
      <p>The final feature is <strong>weight visualization</strong>. Click on any output neuron in the network diagram to see:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>Which hidden neurons contribute most</li>
        <li>Excitatory vs inhibitory patterns</li>
        <li>9×9 weight templates for each neuron</li>
        <li>Decision contribution analysis</li>
      </ul>
    `,
    target: 'circle[fill="#EF4444"], circle[fill="#22C55E"]',
    action: 'Click on an output neuron (red or green circle) to open weight visualization',
    waitForAction: true,
    validation: checkWeightVisualizationOpened,
    pin: 'bottom-left'
  },
  {
    id: 'tour-complete',
    title: 'Tour Complete! 🎉',
    content: `
      <p><strong>Congratulations!</strong> You've learned all the key features:</p>
      <ul class="text-xs space-y-1 ml-4 list-disc">
        <li>✓ Manual and dataset training workflows</li>
        <li>✓ Multi-epoch batch processing</li>
        <li>✓ Checkpoint save/load system</li>
        <li>✓ Inference mode testing</li>
        <li>✓ Weight visualization and analysis</li>
      </ul>
      <p class="text-xs text-gray-500 mt-3">
        <strong>Remember:</strong> Click the ? icons throughout the interface for detailed mathematical explanations of each concept!
      </p>
    `
  }
];