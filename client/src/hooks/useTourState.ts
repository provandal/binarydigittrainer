import { useState, useRef, useCallback } from "react";

export interface UseTourStateParams {
  // Shared refs from parent (breaks circular deps)
  pixelGridRef: React.MutableRefObject<number[][]>;
  isAutoTrainingRef: React.MutableRefObject<boolean>;
  trainingCompletedRef: React.MutableRefObject<boolean>;
  trainedSampleCountRef: React.MutableRefObject<number>;
  showModelManagementRef: React.MutableRefObject<boolean>;
}

export interface UseTourStateReturn {
  // Tour boolean state
  tourStepStarted: boolean;
  tourDrawnOnCanvas: boolean;
  tourStepExecuted: boolean;
  tourTrainingCycleCompleted: boolean;
  tourDatasetLoaded: boolean;
  tourNextSampleClicked: boolean;
  tourMultiEpochStarted: boolean;
  tourWeightVisualizationOpened: boolean;
  tourInferenceModeEnabled: boolean;
  tourCheckpointSaved: boolean;
  tourCheckpointLoaded: boolean;

  // Tour boolean setters
  setTourStepStarted: React.Dispatch<React.SetStateAction<boolean>>;
  setTourDrawnOnCanvas: React.Dispatch<React.SetStateAction<boolean>>;
  setTourStepExecuted: React.Dispatch<React.SetStateAction<boolean>>;
  setTourTrainingCycleCompleted: React.Dispatch<React.SetStateAction<boolean>>;
  setTourDatasetLoaded: React.Dispatch<React.SetStateAction<boolean>>;
  setTourNextSampleClicked: React.Dispatch<React.SetStateAction<boolean>>;
  setTourMultiEpochStarted: React.Dispatch<React.SetStateAction<boolean>>;
  setTourWeightVisualizationOpened: React.Dispatch<React.SetStateAction<boolean>>;
  setTourInferenceModeEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  setTourCheckpointSaved: React.Dispatch<React.SetStateAction<boolean>>;
  setTourCheckpointLoaded: React.Dispatch<React.SetStateAction<boolean>>;

  // Refs
  tourTriggerRef: React.MutableRefObject<(() => void) | null>;
  clicksThisCycleRef: React.MutableRefObject<number>;
  cycleDoneRef: React.MutableRefObject<boolean>;
  datasetLoadedRef: React.MutableRefObject<boolean>;
  nextSampleClickedRef: React.MutableRefObject<boolean>;
  multiEpochStartedRef: React.MutableRefObject<boolean>;
  checkpointSavedRef: React.MutableRefObject<boolean>;
  tourInferenceModeEnabledRef: React.MutableRefObject<boolean>;
  tourWeightVisualizationOpenedRef: React.MutableRefObject<boolean>;
  modeRef: React.MutableRefObject<"training" | "inference">;

  // Validation functions
  validationDrewSomething: () => boolean;
  checkTrainingStarted: () => boolean;
  validOneClick: () => boolean;
  validFullCycle: () => boolean;
  checkDatasetLoaded: () => boolean;
  checkNextSampleClicked: () => boolean;
  checkEpochTrainingStarted: () => boolean;
  checkTrainingCompleted: () => boolean;
  checkModelManagementExpanded: () => boolean;
  checkInferenceModeActive: () => boolean;
  checkWeightVisualizationOpened: () => boolean;
  checkCheckpointSaved: () => boolean;

  // Methods
  handleTourReset: () => void;
  resetCycleCounters: () => void;
}

export function useTourState(params: UseTourStateParams): UseTourStateReturn {
  const {
    pixelGridRef,
    isAutoTrainingRef,
    trainingCompletedRef,
    trainedSampleCountRef,
    showModelManagementRef,
  } = params;

  // --- Tour boolean state ---
  const [tourStepStarted, setTourStepStarted] = useState(false);
  const [tourDrawnOnCanvas, setTourDrawnOnCanvas] = useState(false);
  const [tourStepExecuted, setTourStepExecuted] = useState(false);
  const [tourTrainingCycleCompleted, setTourTrainingCycleCompleted] = useState(false);
  const [tourDatasetLoaded, setTourDatasetLoaded] = useState(false);
  const [tourNextSampleClicked, setTourNextSampleClicked] = useState(false);
  const [tourMultiEpochStarted, setTourMultiEpochStarted] = useState(false);
  const [tourWeightVisualizationOpened, setTourWeightVisualizationOpened] = useState(false);
  const [tourInferenceModeEnabled, setTourInferenceModeEnabled] = useState(false);
  const [tourCheckpointSaved, setTourCheckpointSaved] = useState(false);
  const [tourCheckpointLoaded, setTourCheckpointLoaded] = useState(false);

  // --- Refs ---
  const tourTriggerRef = useRef<(() => void) | null>(null);
  const clicksThisCycleRef = useRef(0);
  const cycleDoneRef = useRef(false);
  const datasetLoadedRef = useRef(false);
  const nextSampleClickedRef = useRef(false);
  const multiEpochStartedRef = useRef(false);
  const checkpointSavedRef = useRef(false);
  const tourInferenceModeEnabledRef = useRef(false);
  const tourWeightVisualizationOpenedRef = useRef(false);
  const modeRef = useRef<"training" | "inference">("training");

  // --- Validation functions ---

  const validationDrewSomething = useCallback(() => {
    const flat = pixelGridRef.current?.flat() || [];
    // Require at least 3 lit pixels so a single accidental click doesn't pass
    const lit = flat.reduce((s, v) => s + (v ? 1 : 0), 0);
    return lit >= 3;
  }, [pixelGridRef]);

  const checkTrainingStarted = useCallback(() => tourStepStarted, [tourStepStarted]);

  const validOneClick = useCallback(() => clicksThisCycleRef.current >= 1, []);

  const validFullCycle = useCallback(() => cycleDoneRef.current === true, []);

  const checkDatasetLoaded = useCallback(() => datasetLoadedRef.current === true, []);

  const checkNextSampleClicked = useCallback(() => nextSampleClickedRef.current === true, []);

  const checkEpochTrainingStarted = useCallback(() => multiEpochStartedRef.current === true, []);

  const checkTrainingCompleted = useCallback(() => {
    const multiEpochWasStarted = multiEpochStartedRef.current === true;
    const notCurrentlyTraining = !isAutoTrainingRef.current;
    const hasFinishedOrStopped =
      trainingCompletedRef.current || (notCurrentlyTraining && trainedSampleCountRef.current > 0);
    return multiEpochWasStarted && notCurrentlyTraining && hasFinishedOrStopped;
  }, [isAutoTrainingRef, trainingCompletedRef, trainedSampleCountRef]);

  const checkModelManagementExpanded = useCallback(
    () => showModelManagementRef.current,
    [showModelManagementRef],
  );

  const checkInferenceModeActive = useCallback(() => modeRef.current === "inference", []);

  const checkWeightVisualizationOpened = useCallback(
    () => tourWeightVisualizationOpenedRef.current,
    [],
  );

  const checkCheckpointSaved = useCallback(() => checkpointSavedRef.current === true, []);

  // --- Reset cycle counters ---
  const resetCycleCounters = useCallback(() => {
    clicksThisCycleRef.current = 0;
    cycleDoneRef.current = false;
  }, []);

  // --- handleTourReset: full reset for tour ---
  const handleTourReset = useCallback(() => {
    // NOTE: The actual network state reset (setWeights, setBiases, etc.) and
    // currentNetworkState.current reset must be handled by the parent
    // component when calling handleTourReset, since those belong to
    // useNetworkParams. This function resets only tour-owned state.

    // Reset tour tracking state
    setTourStepStarted(false);
    setTourDrawnOnCanvas(false);
    setTourStepExecuted(false);
    setTourTrainingCycleCompleted(false);
    setTourDatasetLoaded(false);
    setTourNextSampleClicked(false);
    setTourMultiEpochStarted(false);
    setTourWeightVisualizationOpened(false);
    setTourInferenceModeEnabled(false);
    setTourCheckpointSaved(false);
    setTourCheckpointLoaded(false);

    // Reset tour refs
    cycleDoneRef.current = false;
    datasetLoadedRef.current = false;
    nextSampleClickedRef.current = false;
    multiEpochStartedRef.current = false;
    checkpointSavedRef.current = false;
    tourInferenceModeEnabledRef.current = false;
    tourWeightVisualizationOpenedRef.current = false;
    showModelManagementRef.current = false;
    modeRef.current = "training";
  }, [showModelManagementRef]);

  return {
    // Tour boolean state
    tourStepStarted,
    tourDrawnOnCanvas,
    tourStepExecuted,
    tourTrainingCycleCompleted,
    tourDatasetLoaded,
    tourNextSampleClicked,
    tourMultiEpochStarted,
    tourWeightVisualizationOpened,
    tourInferenceModeEnabled,
    tourCheckpointSaved,
    tourCheckpointLoaded,

    // Tour boolean setters
    setTourStepStarted,
    setTourDrawnOnCanvas,
    setTourStepExecuted,
    setTourTrainingCycleCompleted,
    setTourDatasetLoaded,
    setTourNextSampleClicked,
    setTourMultiEpochStarted,
    setTourWeightVisualizationOpened,
    setTourInferenceModeEnabled,
    setTourCheckpointSaved,
    setTourCheckpointLoaded,

    // Refs
    tourTriggerRef,
    clicksThisCycleRef,
    cycleDoneRef,
    datasetLoadedRef,
    nextSampleClickedRef,
    multiEpochStartedRef,
    checkpointSavedRef,
    tourInferenceModeEnabledRef,
    tourWeightVisualizationOpenedRef,
    modeRef,

    // Validation functions
    validationDrewSomething,
    checkTrainingStarted,
    validOneClick,
    validFullCycle,
    checkDatasetLoaded,
    checkNextSampleClicked,
    checkEpochTrainingStarted,
    checkTrainingCompleted,
    checkModelManagementExpanded,
    checkInferenceModeActive,
    checkWeightVisualizationOpened,
    checkCheckpointSaved,

    // Methods
    handleTourReset,
    resetCycleCounters,
  };
}
