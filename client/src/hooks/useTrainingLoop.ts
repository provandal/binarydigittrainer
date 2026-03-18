import { useState, useRef, useCallback } from "react";
import { sigmoid, sigmoidDerivative, softmax, clip } from "@/lib/nn-math";
import { flatToGrid, parseLabel, getCurrentTarget } from "@/lib/nn-helpers";
import type { NetworkStateRef } from "./useNetworkParams";
import type { TrainingExample } from "@shared/schema";

export interface UseTrainingLoopParams {
  // Refs from useNetworkParams
  currentNetworkState: React.MutableRefObject<NetworkStateRef>;
  selectedLabelRef: React.MutableRefObject<number>;
  trainingHistoryStore: React.MutableRefObject<any[]>;

  // Shared refs from parent (breaks circular deps)
  pixelGridRef: React.MutableRefObject<number[][]>;
  trainedSampleCountRef: React.MutableRefObject<number>;
  shouldStopTraining: React.MutableRefObject<boolean>;
  isAutoTrainingRef: React.MutableRefObject<boolean>;
  trainingCompletedRef: React.MutableRefObject<boolean>;

  // State from parent
  trainingExamples: TrainingExample[];

  // State/setters from useNetworkParams
  learningRate: number;
  setWeights: React.Dispatch<React.SetStateAction<number[][]>>;
  setBiases: React.Dispatch<React.SetStateAction<number[]>>;
  setOutputWeights: React.Dispatch<React.SetStateAction<number[][]>>;
  setOutputBiases: React.Dispatch<React.SetStateAction<number[]>>;
  setHiddenActivations: React.Dispatch<React.SetStateAction<number[]>>;
  setOutputActivations: React.Dispatch<React.SetStateAction<number[]>>;
  setLoss: React.Dispatch<React.SetStateAction<number>>;
  setStep: React.Dispatch<React.SetStateAction<number>>;
  setTrainingHistory: React.Dispatch<React.SetStateAction<any[]>>;
  setLearningRate: React.Dispatch<React.SetStateAction<number>>;

  // Pixel grid setter from useCanvasDrawing
  setPixelGrid: (grid: number[][] | number[]) => void;

  // Selected label setter from parent
  setSelectedLabelSafe: (label: number) => void;

  // Tour trigger ref from useTourState
  tourTriggerRef: React.MutableRefObject<(() => void) | null>;

  // Tour tracking setters
  setTourStepStarted: React.Dispatch<React.SetStateAction<boolean>>;
  setTourTrainingCycleCompleted: React.Dispatch<React.SetStateAction<boolean>>;
  setTourNextSampleClicked: React.Dispatch<React.SetStateAction<boolean>>;
  setTourMultiEpochStarted: React.Dispatch<React.SetStateAction<boolean>>;

  // Tour tracking refs
  clicksThisCycleRef: React.MutableRefObject<number>;
  cycleDoneRef: React.MutableRefObject<boolean>;
  multiEpochStartedRef: React.MutableRefObject<boolean>;
  nextSampleClickedRef: React.MutableRefObject<boolean>;

  // Model management state setters used by epoch completion
  setExamplesSeen: React.Dispatch<React.SetStateAction<number>>;
  setLastEpochAvgLoss: React.Dispatch<React.SetStateAction<number | null>>;
  setCompletedEpochs: React.Dispatch<React.SetStateAction<number>>;
  setLrHistory: React.Dispatch<React.SetStateAction<{ epoch: number; learningRate: number }[]>>;

  // LR decay config from useModelManagement
  lrDecayEnabled: boolean;
  lrDecayRate: number;
  minLR: number;

  // Mode from parent
  mode: "training" | "inference";
  pixelGrid: number[][];
  trainingMode: "manual" | "dataset";
}

export interface UseTrainingLoopReturn {
  // State
  isAutoTraining: boolean;
  autoTrainingSpeed: number;
  trainingCompleted: boolean;
  currentEpoch: number;
  numberOfEpochs: number;
  epochLossHistory: { epoch: number; averageLoss: number }[];
  trainedSampleCount: number;
  isEpochDialogOpen: boolean;
  currentExampleIndex: number;
  trainingMode: "manual" | "dataset";

  // Refs (only hook-internal ones; shared refs come from parent)
  currentEpochLoss: React.MutableRefObject<number[]>;
  trainingIntervalRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>;

  // Setters (exposed for parent/other hooks)
  setIsAutoTraining: React.Dispatch<React.SetStateAction<boolean>>;
  setAutoTrainingSpeed: React.Dispatch<React.SetStateAction<number>>;
  setTrainingCompleted: React.Dispatch<React.SetStateAction<boolean>>;
  setCurrentEpoch: React.Dispatch<React.SetStateAction<number>>;
  setNumberOfEpochs: React.Dispatch<React.SetStateAction<number>>;
  setEpochLossHistory: React.Dispatch<
    React.SetStateAction<{ epoch: number; averageLoss: number }[]>
  >;
  setTrainedSampleCount: React.Dispatch<React.SetStateAction<number>>;
  setIsEpochDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setCurrentExampleIndex: React.Dispatch<React.SetStateAction<number>>;
  setTrainingMode: React.Dispatch<React.SetStateAction<"manual" | "dataset">>;

  // Methods
  nextStep: (forceStep?: number) => void;
  processTrainingSet: () => void;
  stopTraining: () => void;
  startMultiEpochTraining: () => void;
  runToNextSample: () => Promise<void>;
  runEpochs: () => Promise<void>;

  // Computed
  totalPlannedSamples: number;
}

export function useTrainingLoop(params: UseTrainingLoopParams): UseTrainingLoopReturn {
  const {
    currentNetworkState,
    selectedLabelRef,
    trainingHistoryStore,
    pixelGridRef,
    trainedSampleCountRef,
    shouldStopTraining,
    isAutoTrainingRef,
    trainingCompletedRef,
    trainingExamples,
    learningRate,
    setWeights,
    setBiases,
    setOutputWeights,
    setOutputBiases,
    setHiddenActivations,
    setOutputActivations,
    setLoss,
    setStep,
    setTrainingHistory,
    setLearningRate,
    setPixelGrid,
    setSelectedLabelSafe,
    tourTriggerRef,
    setTourStepStarted,
    setTourTrainingCycleCompleted,
    setTourNextSampleClicked,
    setTourMultiEpochStarted,
    clicksThisCycleRef,
    cycleDoneRef,
    multiEpochStartedRef,
    nextSampleClickedRef,
    setExamplesSeen,
    setLastEpochAvgLoss,
    setCompletedEpochs,
    setLrHistory,
    lrDecayEnabled,
    lrDecayRate,
    minLR,
    mode,
    pixelGrid,
    trainingMode: _parentTrainingMode,
  } = params;

  // --- Local state ---
  const [isAutoTraining, setIsAutoTrainingState] = useState(false);
  const [autoTrainingSpeed, setAutoTrainingSpeed] = useState(50);
  const [trainingCompleted, setTrainingCompletedState] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [numberOfEpochs, setNumberOfEpochs] = useState(1);
  const [epochLossHistory, setEpochLossHistory] = useState<
    { epoch: number; averageLoss: number }[]
  >([]);
  const [trainedSampleCount, setTrainedSampleCount] = useState(0);
  const [isEpochDialogOpen, setIsEpochDialogOpen] = useState(false);
  const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
  const [trainingMode, setTrainingMode] = useState<"manual" | "dataset">("manual");

  // --- Refs (only hook-internal ones) ---
  const currentEpochLoss = useRef<number[]>([]);
  const trainingIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // --- Sync wrappers: update both React state and shared ref ---
  const setIsAutoTraining: React.Dispatch<React.SetStateAction<boolean>> = useCallback(
    (value: React.SetStateAction<boolean>) => {
      setIsAutoTrainingState((prev) => {
        const next = typeof value === "function" ? value(prev) : value;
        isAutoTrainingRef.current = next;
        return next;
      });
    },
    [isAutoTrainingRef],
  );

  const setTrainingCompleted: React.Dispatch<React.SetStateAction<boolean>> = useCallback(
    (value: React.SetStateAction<boolean>) => {
      setTrainingCompletedState((prev) => {
        const next = typeof value === "function" ? value(prev) : value;
        trainingCompletedRef.current = next;
        return next;
      });
    },
    [trainingCompletedRef],
  );

  // Computed
  const totalPlannedSamples = (trainingExamples.length || 0) * (numberOfEpochs || 0);

  // --- Helper: get pixel values for forward pass ---
  const getPixelValues = useCallback((): number[] => {
    // In inference mode, always use the current pixel grid (fresh drawing)
    if (mode === "inference") {
      return pixelGridRef.current?.flat() || pixelGrid.flat();
    }

    // During training, use cached inputs from network state if available
    if (
      currentNetworkState.current.inputs &&
      currentNetworkState.current.inputs.some((x) => x !== 0)
    ) {
      return currentNetworkState.current.inputs;
    }
    // Otherwise read from ref (immediate) or state (fallback)
    return pixelGridRef.current?.flat() || pixelGrid.flat();
  }, [mode, pixelGrid, pixelGridRef, currentNetworkState]);

  // --- Forward pass: Input -> Hidden ---
  const forwardPassHidden = useCallback(() => {
    const pixelValues = getPixelValues();
    // Calculate pre-activation values (z1)
    const newPreActivations = currentNetworkState.current.weights.map((w, i) =>
      w.reduce(
        (sum, weight, j) => sum + weight * pixelValues[j],
        currentNetworkState.current.biases[i],
      ),
    );
    // Apply sigmoid to get activations
    const newActivations = newPreActivations.map((z) => sigmoid(z));

    // Store both pre-activation and activation values
    currentNetworkState.current.hiddenPreActivations = newPreActivations;
    currentNetworkState.current.hiddenActivations = newActivations;
    setHiddenActivations(newActivations);
  }, [getPixelValues, currentNetworkState, setHiddenActivations]);

  // --- Forward pass: Hidden -> Output ---
  const forwardPassOutput = useCallback(() => {
    // Calculate pre-activation values (z2)
    const newPreActivations = currentNetworkState.current.outputWeights.map((w, i) =>
      w.reduce(
        (sum, weight, j) => sum + weight * currentNetworkState.current.hiddenActivations[j],
        currentNetworkState.current.outputBiases[i],
      ),
    );
    // Apply softmax to get probability distribution
    const newOutputActivations = softmax(newPreActivations);

    // Store both pre-activation and activation values
    currentNetworkState.current.outputPreActivations = newPreActivations;
    currentNetworkState.current.outputActivations = newOutputActivations;
    setOutputActivations(newOutputActivations);
  }, [currentNetworkState, setOutputActivations]);

  // --- Calculate loss ---
  const calculateLoss = useCallback(() => {
    // Use cached target to eliminate race conditions
    const target = getCurrentTarget(
      currentNetworkState,
      trainingMode,
      trainingExamples,
      currentExampleIndex,
      selectedLabelRef,
    );

    console.log(
      `🎯 Cross-Entropy Loss - Target: [${target}], Outputs: [${currentNetworkState.current.outputActivations.map((o) => o.toFixed(3))}]`,
    );

    // Cross-Entropy Loss with epsilon clamping
    const eps = 1e-7;
    const probs = currentNetworkState.current.outputActivations;
    const calculatedLoss = -target.reduce((sum: number, t: number, i: number) => {
      const p = Math.max(eps, Math.min(1 - eps, probs[i]));
      return sum + t * Math.log(p);
    }, 0);

    currentNetworkState.current.loss = calculatedLoss;
    setLoss(calculatedLoss);
  }, [
    currentNetworkState,
    trainingMode,
    trainingExamples,
    currentExampleIndex,
    selectedLabelRef,
    setLoss,
  ]);

  // --- Backpropagation: Output layer ---
  const backpropagationOutput = useCallback(() => {
    // Use cached target to eliminate race conditions
    const target = getCurrentTarget(
      currentNetworkState,
      trainingMode,
      trainingExamples,
      currentExampleIndex,
      selectedLabelRef,
    );

    // Cross-Entropy + Softmax gradient: delta_i = p_i - t_i
    const outputErrors = currentNetworkState.current.outputActivations.map(
      (output, i) => output - target[i],
    );

    // Apply gradient clipping for stability
    const outputErrorsClipped = outputErrors.map(clip);

    console.log(
      `🔄 Backprop Output (Cross-Entropy) - Target: [${target}], Errors: [${outputErrors.map((e) => e.toFixed(4))}], Clipped: [${outputErrorsClipped.map((e) => e.toFixed(4))}]`,
    );

    // Store clipped output errors for hidden layer backprop
    currentNetworkState.current.outputErrors = outputErrorsClipped;

    // Update output weights and biases using clipped errors
    const newOutputWeights = currentNetworkState.current.outputWeights.map((weights, i) =>
      weights.map(
        (weight, j) =>
          weight -
          learningRate * outputErrorsClipped[i] * currentNetworkState.current.hiddenActivations[j],
      ),
    );
    const newOutputBiases = currentNetworkState.current.outputBiases.map(
      (bias, i) => bias - learningRate * outputErrorsClipped[i],
    );

    // Update persistent store
    currentNetworkState.current.outputWeights = newOutputWeights;
    currentNetworkState.current.outputBiases = newOutputBiases;

    // Update React state for display
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
  }, [
    currentNetworkState,
    trainingMode,
    trainingExamples,
    currentExampleIndex,
    selectedLabelRef,
    learningRate,
    setOutputWeights,
    setOutputBiases,
  ]);

  // --- Backpropagation: Hidden layer ---
  const backpropagationHidden = useCallback(() => {
    // Use clipped output errors from output layer backpropagation
    const outputErrorsClipped = currentNetworkState.current.outputErrors;
    const pixelValues = getPixelValues();

    // Calculate hidden errors using PRE-ACTIVATION values and clipped output errors
    // delta_h = (Sum_i delta_i * w_ih) * sigmoid'(z_h) where z_h is PRE-activation
    const hiddenErrors = currentNetworkState.current.hiddenPreActivations.map(
      (preActivation, h) => {
        const errorSum = outputErrorsClipped.reduce(
          (sum, outputError, i) =>
            sum + outputError * currentNetworkState.current.outputWeights[i][h],
          0,
        );
        return errorSum * sigmoidDerivative(preActivation);
      },
    );

    // Apply gradient clipping to hidden errors as well
    const hiddenErrorsClipped = hiddenErrors.map(clip);

    // Update hidden weights and biases using clipped errors
    const newWeights = currentNetworkState.current.weights.map((weights, i) =>
      weights.map((weight, j) => weight - learningRate * hiddenErrorsClipped[i] * pixelValues[j]),
    );
    const newBiases = currentNetworkState.current.biases.map(
      (bias, i) => bias - learningRate * hiddenErrorsClipped[i],
    );

    // Update persistent store
    currentNetworkState.current.weights = newWeights;
    currentNetworkState.current.biases = newBiases;

    // Update React state for display
    setWeights(newWeights);
    setBiases(newBiases);
  }, [currentNetworkState, getPixelValues, learningRate, setWeights, setBiases]);

  // --- Sleep helper for async training ---
  const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

  // --- nextStep: execute a single training step ---
  const nextStep = useCallback(
    (forceStep?: number) => {
      const currentStep = forceStep !== undefined ? forceStep : undefined;
      // When forceStep is undefined, we read from the step state via the setter callback
      // For the logic below, we need the actual step value for forceStep cases
      console.log("nextStep executing step:", currentStep);

      // We use a helper that does the actual work, called either with forceStep or inside setStep
      const executeStep = (stepToExecute: number) => {
        // Mark tour step as started when first step is taken
        if (stepToExecute === 0) {
          setTourStepStarted(true);
        }

        switch (stepToExecute) {
          case 0:
            forwardPassHidden();
            break;
          case 1:
            forwardPassOutput();
            break;
          case 2:
            calculateLoss();
            break;
          case 3:
            backpropagationOutput();
            break;
          case 4: {
            console.log("Executing case 4 - backpropagation hidden (input->hidden weights)");
            backpropagationHidden();

            // Capture training history AFTER both weight updates are complete
            const historySnapshot = {
              iteration: trainingHistoryStore.current.length,
              weights: currentNetworkState.current.weights.map((w: number[]) => [...w]),
              outputWeights: currentNetworkState.current.outputWeights.map((w: number[]) => [...w]),
              biases: [...currentNetworkState.current.biases],
              outputBiases: [...currentNetworkState.current.outputBiases],
              loss: currentNetworkState.current.loss,
              hiddenActivations: [...currentNetworkState.current.hiddenActivations],
              outputActivations: [...currentNetworkState.current.outputActivations],
            };

            // Store in persistent history
            trainingHistoryStore.current.push(historySnapshot);

            // Update React state for display
            setTrainingHistory([...trainingHistoryStore.current]);

            console.log("Captured training history after both backprop steps:", {
              iteration: trainingHistoryStore.current.length,
              totalSnapshots: trainingHistoryStore.current.length,
              loss: currentNetworkState.current.loss,
              hiddenWeight: currentNetworkState.current.weights[0][0],
              outputWeight: currentNetworkState.current.outputWeights[0][0],
            });
            break;
          }
          case 5:
            // Complete cycle, start over - clear the canvas for next digit
            setPixelGrid(
              Array(9)
                .fill(0)
                .map(() => Array(9).fill(0)),
            );
            setStep(-1);
            setTourTrainingCycleCompleted(true); // Tour tracking - full 6-step cycle completed
            cycleDoneRef.current = true; // Set ref immediately for tour validation
            console.log("🎯 TOUR: Training cycle completed! Setting both state and ref to true");
            // Trigger tour validation check
            setTimeout(() => {
              if (tourTriggerRef.current) {
                console.log("🎯 TOUR: Triggering validation check...");
                tourTriggerRef.current();
              }
            }, 100);
            break;
        }
      };

      if (forceStep !== undefined) {
        executeStep(forceStep);
        setStep(forceStep);
        console.log("🎯 TOUR: Force step to", forceStep, "(no click tracking)");
      } else {
        // Read the current step value inside the setter to avoid stale closures
        setStep((prev) => {
          executeStep(prev);

          // Increment click counter for tour tracking (only on user clicks)
          clicksThisCycleRef.current += 1;
          console.log(
            "🎯 TOUR: User clicked Next Step, clicks this cycle:",
            clicksThisCycleRef.current,
          );

          const newStep = (prev + 1) % 6;
          console.log("🎯 TOUR: Step changed from", prev, "to", newStep);

          // Detect cycle completion when wrapping from step 5 back to 0
          if (prev === 5 && newStep === 0) {
            cycleDoneRef.current = true;
            console.log("🎯 TOUR: Training cycle completed! cycleDone set to true");
          }

          return newStep;
        });
      }

      // Always trigger tour validation after any step change
      setTimeout(() => {
        if (tourTriggerRef.current) {
          console.log("🎯 TOUR: Triggering validation after step change...");
          tourTriggerRef.current();
        }
      }, 100);
    },
    [
      forwardPassHidden,
      forwardPassOutput,
      calculateLoss,
      backpropagationOutput,
      backpropagationHidden,
      currentNetworkState,
      trainingHistoryStore,
      setStep,
      setTrainingHistory,
      setPixelGrid,
      setTourStepStarted,
      setTourTrainingCycleCompleted,
      tourTriggerRef,
      clicksThisCycleRef,
      cycleDoneRef,
    ],
  );

  // --- runStepsForCurrentSample: run all 6 steps for one sample ---
  const runStepsForCurrentSample = useCallback(async (): Promise<boolean> => {
    for (const s of [0, 1, 2, 3, 4, 5]) {
      if (shouldStopTraining.current) return false;
      console.log(`🔄 Running step ${s} for sample ${currentExampleIndex}`);
      nextStep(s); // use the forced-step path; UI step is cosmetic
      await sleep(autoTrainingSpeed);
    }
    return true;
  }, [nextStep, currentExampleIndex, autoTrainingSpeed]);

  // --- runToNextSample: train a single sample with visual feedback ---
  const runToNextSample = useCallback(async () => {
    if (trainingExamples.length === 0) return;

    console.log("🚀 Starting runToNextSample for example index:", currentExampleIndex);
    setIsAutoTraining(true);

    // Set tour tracking for step 8
    setTourNextSampleClicked(true);
    nextSampleClickedRef.current = true;

    // Load current training example with visual feedback
    const currentExample = trainingExamples[currentExampleIndex];
    console.log("📖 Loading example for runToNextSample:", currentExample.label);
    const pattern = currentExample.pattern as number[][] | number[];
    const grid = Array.isArray(pattern[0])
      ? (pattern as number[][])
      : flatToGrid(pattern as number[]);
    setPixelGrid(grid);
    const oneHotLabel = parseLabel(currentExample.label);
    setSelectedLabelSafe(oneHotLabel[0] === 1 ? 0 : 1);
    setStep(0);

    // Cache for training logic
    currentNetworkState.current.currentTarget = oneHotLabel;
    currentNetworkState.current.inputs = grid.flat();

    // Run all steps with good visual feedback
    const completed = await runStepsForCurrentSample();
    if (!completed) {
      setIsAutoTraining(false);
      return;
    }

    // Track progress and complete
    currentEpochLoss.current.push(currentNetworkState.current.loss);
    setExamplesSeen((prev) => prev + 1);

    // Move to next example
    const nextIndex = (currentExampleIndex + 1) % trainingExamples.length;
    setCurrentExampleIndex(nextIndex);

    setIsAutoTraining(false);
    console.log("✅ Single sample training completed");

    // Trigger tour validation
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log("🔔 TOUR: Triggering validation after single sample completion");
        tourTriggerRef.current();
      }
    }, 100);
  }, [
    trainingExamples,
    currentExampleIndex,
    setPixelGrid,
    setSelectedLabelSafe,
    setStep,
    currentNetworkState,
    runStepsForCurrentSample,
    setExamplesSeen,
    setTourNextSampleClicked,
    nextSampleClickedRef,
    tourTriggerRef,
  ]);

  // --- runEpochs: async multi-epoch training ---
  const runEpochs = useCallback(async () => {
    if (trainingExamples.length === 0) return;

    console.log(
      `🚀 NEW ASYNC runEpochs starting for ${numberOfEpochs} epoch(s) with ${trainingExamples.length} examples`,
    );

    setIsAutoTraining(true);
    setTrainingCompleted(false);
    setIsEpochDialogOpen(false);

    for (let epoch = 1; epoch <= numberOfEpochs; epoch++) {
      if (shouldStopTraining.current) break;

      currentEpochLoss.current = [];
      setCurrentEpoch(epoch);

      // Iterate through all examples in the training set
      for (let idx = 0; idx < trainingExamples.length; idx++) {
        if (shouldStopTraining.current) break;

        const example = trainingExamples[idx];
        const pattern = Array.isArray((example.pattern as any)[0])
          ? (example.pattern as number[][])
          : flatToGrid(example.pattern as number[]);
        const oneHot = parseLabel(example.label);
        const uiDigit = oneHot[0] === 1 ? 0 : 1;

        console.log(
          `📊 Epoch ${epoch}/${numberOfEpochs}, Example ${idx + 1}/${trainingExamples.length}, ExampleID: ${example.id}, Label: [${oneHot}]`,
        );

        // Snapshot the sample before running steps (eliminates async state issues)
        setPixelGrid(pattern);
        setSelectedLabelSafe(uiDigit);
        setCurrentExampleIndex(idx);

        // Cache target and inputs in network state for training logic
        currentNetworkState.current.currentTarget = oneHot;
        currentNetworkState.current.inputs = pattern.flat();
        console.log(
          `🔄 CACHE SET in runEpochs: currentTarget = [${oneHot}], ExampleID = ${example.id}`,
        );

        const completed = await runStepsForCurrentSample();
        if (!completed) break;

        currentEpochLoss.current.push(currentNetworkState.current.loss);
        setExamplesSeen((prev) => prev + 1);

        // Increment sample counter for tour validation
        trainedSampleCountRef.current += 1;
        setTrainedSampleCount((prev) => prev + 1);
      }

      // Calculate average loss for completed epoch
      if (currentEpochLoss.current.length > 0) {
        const avg =
          currentEpochLoss.current.reduce((a, b) => a + b, 0) / currentEpochLoss.current.length;
        setEpochLossHistory((prev) => [...prev, { epoch, averageLoss: avg }]);
        setLastEpochAvgLoss(avg);
        setCompletedEpochs(epoch);
        console.log(`✅ Epoch ${epoch} completed. Average loss: ${avg.toFixed(4)}`);

        // Apply learning rate decay
        if (lrDecayEnabled) {
          setLearningRate((prev) => {
            const next = Math.max(minLR, prev * lrDecayRate);
            console.log(`[LR Decay] lr: ${prev.toFixed(6)} → ${next.toFixed(6)}`);
            // Track learning rate history
            setLrHistory((prevHistory) => [...prevHistory, { epoch, learningRate: next }]);
            return next;
          });
        } else {
          // Track learning rate even without decay for consistency
          setLrHistory((prevHistory) => [...prevHistory, { epoch, learningRate }]);
        }
      }

      if (shouldStopTraining.current) break;
    }

    setIsAutoTraining(false);

    // Only set training completed if we've processed ALL planned samples for ALL epochs
    const allPlanned = totalPlannedSamples;
    const actuallyCompleted = trainedSampleCountRef.current >= allPlanned && allPlanned > 0;
    const allEpochsFinished = !shouldStopTraining.current; // Only true if we naturally finished all epochs
    if (actuallyCompleted && allEpochsFinished) {
      setTrainingCompleted(true);
      console.log(
        `🎉 Training completed! All ${trainedSampleCountRef.current}/${allPlanned} samples finished across all epochs.`,
      );
    }

    // Trigger tour validation check for training completion
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log("🔔 TOUR: Triggering validation check after training completion");
        tourTriggerRef.current();
      }
    }, 100);
  }, [
    trainingExamples,
    numberOfEpochs,
    setPixelGrid,
    setSelectedLabelSafe,
    currentNetworkState,
    runStepsForCurrentSample,
    setExamplesSeen,
    setLastEpochAvgLoss,
    setCompletedEpochs,
    setLearningRate,
    setLrHistory,
    lrDecayEnabled,
    lrDecayRate,
    minLR,
    learningRate,
    totalPlannedSamples,
    tourTriggerRef,
  ]);

  // --- processTrainingSet: open epoch dialog ---
  const processTrainingSet = useCallback(() => {
    // Check if training examples are loaded
    if (trainingExamples.length === 0) {
      console.warn("⚠️ TOUR: No training examples loaded. Cannot start multi-epoch training.");
      return;
    }

    setIsEpochDialogOpen(true);
    setTourMultiEpochStarted(true); // Tour tracking - React state
    multiEpochStartedRef.current = true; // Tour tracking - immediate ref
    console.log("🎯 TOUR: Process Training Set clicked! Setting both state and ref to true");

    // Trigger tour validation check
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log("🔔 TOUR: Triggering validation check after Process Training Set click");
        tourTriggerRef.current();
      }
    }, 100);
  }, [trainingExamples.length, setTourMultiEpochStarted, multiEpochStartedRef, tourTriggerRef]);

  // --- stopTraining ---
  const stopTraining = useCallback(() => {
    console.log("🛑 Training stopped by user");
    shouldStopTraining.current = true;
    setIsAutoTraining(false);

    // Only set training completed if we've processed all planned samples
    const allPlanned = totalPlannedSamples;
    const actuallyCompleted = trainedSampleCountRef.current >= allPlanned && allPlanned > 0;
    if (actuallyCompleted) {
      setTrainingCompleted(true);
      console.log(
        `🎉 Training completed via stop! All ${trainedSampleCountRef.current}/${allPlanned} samples finished.`,
      );
    }

    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }

    // Trigger tour validation check for manual training stop
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log("🔔 TOUR: Triggering validation check after manual training stop");
        tourTriggerRef.current();
      }
    }, 100);
  }, [totalPlannedSamples, tourTriggerRef]);

  // --- startMultiEpochTraining ---
  const startMultiEpochTraining = useCallback(() => {
    if (trainingExamples.length === 0) return;

    console.log(
      `Starting multi-epoch training for ${numberOfEpochs} epoch(s) with ${trainingExamples.length} examples`,
    );

    // Reset training progress counters
    trainedSampleCountRef.current = 0;
    setTrainedSampleCount(0);
    setTrainingCompleted(false);

    // Set multi-epoch started ref for tour tracking
    multiEpochStartedRef.current = true;

    shouldStopTraining.current = false;
    setIsAutoTraining(true);
    setCurrentExampleIndex(0);
    setIsEpochDialogOpen(false);

    // Reset epoch loss tracking
    setEpochLossHistory([]);
    currentEpochLoss.current = [];
    setCurrentEpoch(1);

    // Start the async training process
    runEpochs();
  }, [trainingExamples.length, numberOfEpochs, runEpochs, multiEpochStartedRef]);

  return {
    // State
    isAutoTraining,
    autoTrainingSpeed,
    trainingCompleted,
    currentEpoch,
    numberOfEpochs,
    epochLossHistory,
    trainedSampleCount,
    isEpochDialogOpen,
    currentExampleIndex,
    trainingMode,

    // Refs (only hook-internal ones; shared refs come from parent)
    currentEpochLoss,
    trainingIntervalRef,

    // Setters
    setIsAutoTraining,
    setAutoTrainingSpeed,
    setTrainingCompleted,
    setCurrentEpoch,
    setNumberOfEpochs,
    setEpochLossHistory,
    setTrainedSampleCount,
    setIsEpochDialogOpen,
    setCurrentExampleIndex,
    setTrainingMode,

    // Methods
    nextStep,
    processTrainingSet,
    stopTraining,
    startMultiEpochTraining,
    runToNextSample,
    runEpochs,

    // Computed
    totalPlannedSamples,
  };
}
