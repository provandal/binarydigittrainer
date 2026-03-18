import { useState, useCallback } from "react";
import {
  type Checkpoint,
  validateCheckpoint,
  downloadBlobJSON,
  nowStamp,
} from "@/lib/nn-checkpoint";
import { tourCheckpointData } from "@/data/tour-checkpoint";
import type { NetworkStateRef } from "./useNetworkParams";

export interface UseModelManagementParams {
  // From useNetworkParams
  currentNetworkState: React.MutableRefObject<NetworkStateRef>;
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

  // From useTrainingLoop
  setIsAutoTraining: React.Dispatch<React.SetStateAction<boolean>>;

  // Shared refs from parent (breaks circular deps)
  shouldStopTraining: React.MutableRefObject<boolean>;
  showModelManagementRef: React.MutableRefObject<boolean>;

  // From useTourState
  setTourCheckpointSaved: React.Dispatch<React.SetStateAction<boolean>>;
  setTourCheckpointLoaded: React.Dispatch<React.SetStateAction<boolean>>;
  tourTriggerRef: React.MutableRefObject<(() => void) | null>;
  checkpointSavedRef: React.MutableRefObject<boolean>;
}

export interface UseModelManagementReturn {
  // State
  showModelManagement: boolean;
  lrDecayEnabled: boolean;
  lrDecayRate: number;
  minLR: number;
  normalizeEnabled: boolean;
  targetSize: number;
  examplesSeen: number;
  lastEpochAvgLoss: number | null;
  completedEpochs: number;
  lrHistory: { epoch: number; learningRate: number }[];
  lastCheckpointLoaded: string | null;

  // Setters
  setShowModelManagement: React.Dispatch<React.SetStateAction<boolean>>;
  setLrDecayEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  setLrDecayRate: React.Dispatch<React.SetStateAction<number>>;
  setMinLR: React.Dispatch<React.SetStateAction<number>>;
  setNormalizeEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  setTargetSize: React.Dispatch<React.SetStateAction<number>>;
  setExamplesSeen: React.Dispatch<React.SetStateAction<number>>;
  setLastEpochAvgLoss: React.Dispatch<React.SetStateAction<number | null>>;
  setCompletedEpochs: React.Dispatch<React.SetStateAction<number>>;
  setLrHistory: React.Dispatch<React.SetStateAction<{ epoch: number; learningRate: number }[]>>;
  setLastCheckpointLoaded: React.Dispatch<React.SetStateAction<string | null>>;

  // Refs
  showModelManagementRef: React.MutableRefObject<boolean>;

  // Methods
  handleExportCheckpoint: () => void;
  handleImportCheckpointFile: (e: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
  loadCheckpointFromData: (checkpoint: Checkpoint) => void;
  loadTourCheckpoint: () => void;
  generateCurrentCheckpoint: () => Checkpoint;
}

export function useModelManagement(params: UseModelManagementParams): UseModelManagementReturn {
  const {
    currentNetworkState,
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
    setIsAutoTraining,
    shouldStopTraining,
    showModelManagementRef,
    setTourCheckpointSaved,
    setTourCheckpointLoaded,
    tourTriggerRef,
    checkpointSavedRef,
  } = params;

  // --- State ---
  const [showModelManagement, setShowModelManagement] = useState(false);
  const [lrDecayEnabled, setLrDecayEnabled] = useState(false);
  const [lrDecayRate, setLrDecayRate] = useState(0.99);
  const [minLR, setMinLR] = useState(0.0005);
  const [normalizeEnabled, setNormalizeEnabled] = useState(false);
  const [targetSize, setTargetSize] = useState(7);
  const [examplesSeen, setExamplesSeen] = useState(0);
  const [lastEpochAvgLoss, setLastEpochAvgLoss] = useState<number | null>(null);
  const [completedEpochs, setCompletedEpochs] = useState(0);
  const [lrHistory, setLrHistory] = useState<{ epoch: number; learningRate: number }[]>([]);
  const [lastCheckpointLoaded, setLastCheckpointLoaded] = useState<string | null>(null);

  // --- Generate current checkpoint ---
  const generateCurrentCheckpoint = useCallback((): Checkpoint => {
    return {
      format: "binary-digit-trainer-checkpoint@v1",
      createdAt: new Date().toISOString(),
      architecture: { input: 81, hidden: 24, output: 2 },
      normalize: { enabled: normalizeEnabled, targetSize },
      optimizer: {
        learningRate,
        lrDecayRate,
        minLR,
        decayEnabled: lrDecayEnabled,
      },
      stats: {
        epoch: completedEpochs,
        avgLoss: lastEpochAvgLoss ?? 0,
        examplesSeen,
      },
      params: {
        weights: currentNetworkState.current.weights.map((r) => [...r]),
        biases: [...currentNetworkState.current.biases],
        outputWeights: currentNetworkState.current.outputWeights.map((r) => [...r]),
        outputBiases: [...currentNetworkState.current.outputBiases],
      },
    };
  }, [
    normalizeEnabled,
    targetSize,
    learningRate,
    lrDecayRate,
    minLR,
    lrDecayEnabled,
    completedEpochs,
    lastEpochAvgLoss,
    examplesSeen,
    currentNetworkState,
  ]);

  // --- Export checkpoint ---
  const handleExportCheckpoint = useCallback(() => {
    const cp = generateCurrentCheckpoint();
    downloadBlobJSON(cp, `checkpoint-${nowStamp()}.json`);
    setTourCheckpointSaved(true); // Tour tracking - React state
    checkpointSavedRef.current = true; // Tour tracking - immediate ref
    console.log("🎯 TOUR: Export checkpoint clicked! Setting both state and ref to true");

    // Trigger tour validation check
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log("🔔 TOUR: Triggering validation check after Export click");
        tourTriggerRef.current();
      }
    }, 100);
  }, [generateCurrentCheckpoint, setTourCheckpointSaved, checkpointSavedRef, tourTriggerRef]);

  // --- Apply checkpoint data to network state ---
  const applyCheckpointToNetwork = useCallback(
    (checkpoint: Checkpoint) => {
      const { params: cpParams, normalize, optimizer, stats } = checkpoint;

      // update refs first (source of truth for training)
      currentNetworkState.current.weights = cpParams.weights.map((r) => [...r]);
      currentNetworkState.current.biases = [...cpParams.biases];
      currentNetworkState.current.outputWeights = cpParams.outputWeights.map((r) => [...r]);
      currentNetworkState.current.outputBiases = [...cpParams.outputBiases];

      // reset caches
      currentNetworkState.current.hiddenActivations = Array(24).fill(0);
      currentNetworkState.current.outputActivations = Array(2).fill(0);
      currentNetworkState.current.hiddenPreActivations = Array(24).fill(0);
      currentNetworkState.current.outputPreActivations = Array(2).fill(0);
      currentNetworkState.current.loss = 0;
      currentNetworkState.current.outputErrors = Array(2).fill(0);

      // update React state to match (UI)
      setWeights(cpParams.weights.map((r) => [...r]));
      setBiases([...cpParams.biases]);
      setOutputWeights(cpParams.outputWeights.map((r) => [...r]));
      setOutputBiases([...cpParams.outputBiases]);

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

      // Reset UI state
      setStep(0);
      setLoss(0);
      setHiddenActivations(Array(24).fill(0));
      setOutputActivations(Array(2).fill(0));
      setLrHistory([]); // Reset learning rate history when loading checkpoint
    },
    [
      currentNetworkState,
      setWeights,
      setBiases,
      setOutputWeights,
      setOutputBiases,
      setHiddenActivations,
      setOutputActivations,
      setLearningRate,
      setStep,
      setLoss,
    ],
  );

  // --- Import checkpoint from file ---
  const handleImportCheckpointFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
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

        // Stop any running training first
        setIsAutoTraining(false);
        shouldStopTraining.current = true;

        applyCheckpointToNetwork(json as Checkpoint);

        setLastCheckpointLoaded(file.name); // Track the loaded checkpoint
        setTourCheckpointLoaded(true); // Tour tracking

        alert(`Loaded checkpoint: ${file.name}`);
      } catch (err) {
        console.error("Import error:", err);
        alert("Failed to import checkpoint.");
      } finally {
        e.target.value = ""; // allow reselecting same file
      }
    },
    [applyCheckpointToNetwork, setIsAutoTraining, shouldStopTraining, setTourCheckpointLoaded],
  );

  // --- Load checkpoint from data (used by tour) ---
  const loadCheckpointFromData = useCallback(
    (checkpoint: Checkpoint) => {
      try {
        if (!validateCheckpoint(checkpoint)) {
          console.error("Invalid tour checkpoint format");
          return;
        }

        // Stop any running training first
        setIsAutoTraining(false);
        shouldStopTraining.current = true;

        applyCheckpointToNetwork(checkpoint);

        setIsAutoTraining(false);
        shouldStopTraining.current = true;
        setTrainingHistory([]);

        console.log("🎯 TOUR: Loaded tour checkpoint successfully");
      } catch (error) {
        console.error("Error loading tour checkpoint:", error);
      }
    },
    [applyCheckpointToNetwork, setIsAutoTraining, shouldStopTraining, setTrainingHistory],
  );

  // --- Load pre-trained tour checkpoint ---
  const loadTourCheckpoint = useCallback(() => {
    console.log("🎯 TOUR: Loading pre-trained checkpoint for inference testing");
    const checkpoint = tourCheckpointData;
    loadCheckpointFromData(checkpoint);
  }, [loadCheckpointFromData]);

  return {
    // State
    showModelManagement,
    lrDecayEnabled,
    lrDecayRate,
    minLR,
    normalizeEnabled,
    targetSize,
    examplesSeen,
    lastEpochAvgLoss,
    completedEpochs,
    lrHistory,
    lastCheckpointLoaded,

    // Setters
    setShowModelManagement,
    setLrDecayEnabled,
    setLrDecayRate,
    setMinLR,
    setNormalizeEnabled,
    setTargetSize,
    setExamplesSeen,
    setLastEpochAvgLoss,
    setCompletedEpochs,
    setLrHistory,
    setLastCheckpointLoaded,

    // Refs
    showModelManagementRef,

    // Methods
    handleExportCheckpoint,
    handleImportCheckpointFile,
    loadCheckpointFromData,
    loadTourCheckpoint,
    generateCurrentCheckpoint,
  };
}
