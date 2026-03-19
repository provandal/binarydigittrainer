import React, { useState, useEffect, useRef } from "react";
import type { TrainingExample } from "@shared/schema";
import {
  getTrainingExamples,
  createTrainingExample,
  updateTrainingExample,
  deleteTrainingExample,
  clearTrainingExamples,
  bulkUploadTrainingExamples,
} from "@/lib/local-storage";
import GuidedTour from "./GuidedTour";
import { createTourSteps } from "@/lib/tour-steps";
import { tourTrainingData } from "@/data/tour-training-data";

// Extracted pure modules
import { sigmoid, softmax } from "@/lib/nn-math";
import { flatToGrid, STEP_DESCRIPTIONS } from "@/lib/nn-helpers";
import { type ColorScheme } from "@/lib/color-schemes";

// Extracted hooks
import { useNetworkParams } from "@/hooks/useNetworkParams";
import { useTourState } from "@/hooks/useTourState";
import { useCanvasDrawing } from "@/hooks/useCanvasDrawing";
import { useModelManagement } from "@/hooks/useModelManagement";
import { useTrainingLoop } from "@/hooks/useTrainingLoop";

// Sub-components
import {
  AppHeader,
  DrawingCanvas,
  NetworkDiagram,
  TrainingStepPanel,
  WeightDetailView,
  DebugHistoryPanel,
  DatasetEditorDialog,
  EpochSelectionDialog,
  AboutDialog,
} from "@/components/trainer";

// 9x9 pixel grid (81 pixels total, each pixel is 0 or 1)
const initialPixelGrid = Array(9)
  .fill(0)
  .map(() => Array(9).fill(0));

export default function BinaryDigitTrainer() {
  // ========================
  // 1. Create shared refs at parent level (breaks circular dependencies)
  // ========================
  const pixelGridRef = useRef<number[][]>(initialPixelGrid);
  const trainedSampleCountRef = useRef(0);
  const showModelManagementRef = useRef(false);
  const shouldStopTrainingRef = useRef(false);
  const isAutoTrainingRef = useRef(false);
  const trainingCompletedRef = useRef(false);

  // ========================
  // 2. Training examples (localStorage) and local state needed before hooks
  // ========================
  const [trainingExamples, setTrainingExamples] = useState<TrainingExample[]>(() =>
    getTrainingExamples(),
  );
  const refreshExamples = () => setTrainingExamples(getTrainingExamples());

  // State that must be declared before hook calls (hooks depend on these)
  const [selectedLabel, setSelectedLabel] = useState(0);
  const [mode, setMode] = useState<"training" | "inference">("training");

  // Stable ref-based callback for setIsAutoTraining forwarding
  // (model management is created before training loop, but needs to forward to it)
  const setIsAutoTrainingRef_callback = useRef<React.Dispatch<React.SetStateAction<boolean>>>(
    (v) => {
      // Fallback: directly update the shared ref. Will be replaced once training hook is ready.
      if (typeof v === "function") {
        isAutoTrainingRef.current = v(isAutoTrainingRef.current);
      } else {
        isAutoTrainingRef.current = v;
      }
    },
  );

  // ========================
  // 3. Call hooks in dependency order
  // ========================

  // a. useNetworkParams — no deps
  const network = useNetworkParams();

  // b. useTourState — gets shared refs
  const tour = useTourState({
    pixelGridRef,
    isAutoTrainingRef,
    trainingCompletedRef,
    trainedSampleCountRef,
    showModelManagementRef,
  });

  // Safe setter for selected label
  const setSelectedLabelSafe = (label: number) => {
    network.selectedLabelRef.current = label;
    setSelectedLabel(label);
  };

  // c. useCanvasDrawing — gets tourTriggerRef from tour, shared pixelGridRef
  const canvas = useCanvasDrawing({
    tourTriggerRef: tour.tourTriggerRef,
    onPixelChange: () => {
      network.setStep(0); // Reset to first step when input changes
      tour.setTourDrawnOnCanvas(true); // Tour tracking
    },
    pixelGridRef,
  });

  // d. useModelManagement — gets setters from network, refs from tour, shared refs
  const model = useModelManagement({
    currentNetworkState: network.currentNetworkState,
    learningRate: network.learningRate,
    setWeights: network.setWeights,
    setBiases: network.setBiases,
    setOutputWeights: network.setOutputWeights,
    setOutputBiases: network.setOutputBiases,
    setHiddenActivations: network.setHiddenActivations,
    setOutputActivations: network.setOutputActivations,
    setLoss: network.setLoss,
    setStep: network.setStep,
    setTrainingHistory: network.setTrainingHistory,
    setLearningRate: network.setLearningRate,
    setIsAutoTraining: (v: React.SetStateAction<boolean>) => {
      setIsAutoTrainingRef_callback.current(v);
    },
    shouldStopTraining: shouldStopTrainingRef,
    showModelManagementRef,
    setTourCheckpointSaved: tour.setTourCheckpointSaved,
    setTourCheckpointLoaded: tour.setTourCheckpointLoaded,
    tourTriggerRef: tour.tourTriggerRef,
    checkpointSavedRef: tour.checkpointSavedRef,
  });

  // e. useTrainingLoop — gets everything it needs from above hooks + shared refs
  const training = useTrainingLoop({
    currentNetworkState: network.currentNetworkState,
    selectedLabelRef: network.selectedLabelRef,
    trainingHistoryStore: network.trainingHistoryStore,
    pixelGridRef,
    trainedSampleCountRef,
    shouldStopTraining: shouldStopTrainingRef,
    isAutoTrainingRef,
    trainingCompletedRef,
    trainingExamples,
    learningRate: network.learningRate,
    setWeights: network.setWeights,
    setBiases: network.setBiases,
    setOutputWeights: network.setOutputWeights,
    setOutputBiases: network.setOutputBiases,
    setHiddenActivations: network.setHiddenActivations,
    setOutputActivations: network.setOutputActivations,
    setLoss: network.setLoss,
    setStep: network.setStep,
    setTrainingHistory: network.setTrainingHistory,
    setLearningRate: network.setLearningRate,
    setPixelGrid: canvas.setPixelGrid,
    setSelectedLabelSafe,
    tourTriggerRef: tour.tourTriggerRef,
    setTourStepStarted: tour.setTourStepStarted,
    setTourTrainingCycleCompleted: tour.setTourTrainingCycleCompleted,
    setTourNextSampleClicked: tour.setTourNextSampleClicked,
    setTourMultiEpochStarted: tour.setTourMultiEpochStarted,
    clicksThisCycleRef: tour.clicksThisCycleRef,
    cycleDoneRef: tour.cycleDoneRef,
    multiEpochStartedRef: tour.multiEpochStartedRef,
    nextSampleClickedRef: tour.nextSampleClickedRef,
    setExamplesSeen: model.setExamplesSeen,
    setLastEpochAvgLoss: model.setLastEpochAvgLoss,
    setCompletedEpochs: model.setCompletedEpochs,
    setLrHistory: model.setLrHistory,
    lrDecayEnabled: model.lrDecayEnabled,
    lrDecayRate: model.lrDecayRate,
    minLR: model.minLR,
    mode,
    pixelGrid: canvas.pixelGrid,
    trainingMode: "manual",
  });

  // Now that training hook exists, wire up the stable callback ref
  setIsAutoTrainingRef_callback.current = training.setIsAutoTraining;

  // ========================
  // 4. Remaining local state (not in any hook)
  // ========================
  const [selectedWeightBox, setSelectedWeightBox] = useState<{
    type: "hidden" | "output";
    index: number;
  } | null>(null);
  const [weightDialogIteration, setWeightDialogIteration] = useState(0);
  const [activeElements, setActiveElements] = useState<string[]>([]);
  const [showDatasetEditor, setShowDatasetEditor] = useState(false);
  const [isDrawingInEditor, setIsDrawingInEditor] = useState(false);
  const [prediction, setPrediction] = useState<{ digit: number; confidence: number } | null>(null);
  const [showDebugDialog, setShowDebugDialog] = useState(false);
  const [debugHistory, setDebugHistory] = useState<
    {
      iteration: number;
      label: number[];
      outputActivations: number[];
      outputErrors: number[];
      outputBiases: number[];
      loss: number;
      step: number;
      timestamp: Date;
    }[]
  >([]);
  const [isGuidedTourOpen, setIsGuidedTourOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [showInputOverlay, setShowInputOverlay] = useState(false);
  const [useGlobalScale, setUseGlobalScale] = useState(false);
  const [colorScheme, setColorScheme] = useState<ColorScheme>("blue-red");
  const [viewMode, setViewMode] = useState<"decision" | "logit">("logit");

  // ========================
  // Convenience aliases (flatten hook returns for JSX readability)
  // ========================
  const {
    weights,
    biases,
    outputWeights,
    outputBiases,
    hiddenActivations,
    outputActivations,
    loss,
    step,
    trainingHistory,
    learningRate,
    currentNetworkState,
    selectedLabelRef,
    setHiddenActivations,
    setOutputActivations,
    setStep,
    setLearningRate,
    resetNetwork: resetNetworkBase,
  } = network;

  const {
    pixelGrid,
    canvasRef,
    handleMouseDown,
    handleMouseEnter,
    handleMouseUp,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
    clearCanvas: clearCanvasBase,
    setPixelGrid,
  } = canvas;

  const {
    tourStepExecuted,
    tourTriggerRef,
    datasetLoadedRef,
    nextSampleClickedRef,
    multiEpochStartedRef,
    checkpointSavedRef,
    tourInferenceModeEnabledRef,
    tourWeightVisualizationOpenedRef,
    modeRef,
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
    validationDrewSomething,
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
    resetCycleCounters,
  } = tour;

  const {
    showModelManagement,
    setShowModelManagement,
    lrDecayEnabled,
    setLrDecayEnabled,
    lrDecayRate,
    setLrDecayRate,
    minLR,
    setMinLR,
    normalizeEnabled,
    targetSize,
    examplesSeen,
    setExamplesSeen,
    lastEpochAvgLoss,
    setLastEpochAvgLoss,
    completedEpochs,
    setCompletedEpochs,
    lrHistory,
    setLrHistory,
    lastCheckpointLoaded,
    setLastCheckpointLoaded,
    handleExportCheckpoint,
    handleImportCheckpointFile,
    loadTourCheckpoint,
  } = model;

  const {
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
    trainingIntervalRef,
    setIsAutoTraining,
    setTrainingCompleted,
    setCurrentEpoch,
    setNumberOfEpochs,
    setEpochLossHistory,
    setTrainedSampleCount,
    setIsEpochDialogOpen,
    setCurrentExampleIndex,
    setTrainingMode,
    nextStep,
    processTrainingSet,
    stopTraining,
    startMultiEpochTraining,
    runToNextSample,
  } = training;

  // ========================
  // Clear canvas with prediction reset
  // ========================
  const clearCanvas = () => {
    clearCanvasBase();
    setPrediction(null);
  };

  // ========================
  // Reset network (extended from base)
  // ========================
  const resetNetwork = () => {
    resetNetworkBase();
    setPixelGrid(
      Array(9)
        .fill(0)
        .map(() => Array(9).fill(0)),
    );
    setSelectedWeightBox(null);
    setWeightDialogIteration(0);
    setTrainingCompleted(false);
    setEpochLossHistory([]);
    setLrHistory([]);

    // Reset training stats
    setExamplesSeen(0);
    setLastEpochAvgLoss(null);
    setCompletedEpochs(0);
    setLastCheckpointLoaded(null);

    // Reset training state
    shouldStopTrainingRef.current = false;
    setIsAutoTraining(false);
    setMode("training");

    // Clear debug history
    setDebugHistory([]);
    setShowDebugDialog(false);
    setTrainingMode("dataset");
    setCurrentExampleIndex(0);
    setCurrentEpoch(1);
  };

  // ========================
  // Calculate pixel values
  // ========================
  const getPixelValues = () => {
    if (mode === "inference") {
      return pixelGridRef.current?.flat() || pixelGrid.flat();
    }
    if (
      currentNetworkState.current.inputs &&
      currentNetworkState.current.inputs.some((x) => x !== 0)
    ) {
      return currentNetworkState.current.inputs;
    }
    return pixelGridRef.current?.flat() || pixelGrid.flat();
  };

  // ========================
  // Dataset editor functions
  // ========================
  const addDatasetExample = () => {
    const newExample = {
      pattern: Array(9)
        .fill(0)
        .map(() => Array(9).fill(0)),
      label: [1, 0],
    };
    createTrainingExample(newExample);
    refreshExamples();
  };

  const removeDatasetExample = (index: number) => {
    const example = trainingExamples[index];
    if (example?.id) {
      deleteTrainingExample(example.id);
      refreshExamples();
    }
  };

  const updateDatasetExample = (index: number, pattern: number[][] | number[], label: number) => {
    const example = trainingExamples[index];
    if (example?.id) {
      const oneHotLabel = label === 0 ? [1, 0] : [0, 1];
      updateTrainingExample(example.id, { pattern, label: oneHotLabel });
      refreshExamples();
    }
  };

  const handleEditorMouseDown = (exampleIndex: number, rowIndex: number, colIndex: number) => {
    setIsDrawingInEditor(true);
    toggleEditorPixel(exampleIndex, rowIndex, colIndex);
  };

  const handleEditorMouseEnter = (exampleIndex: number, rowIndex: number, colIndex: number) => {
    if (isDrawingInEditor) {
      toggleEditorPixel(exampleIndex, rowIndex, colIndex);
    }
  };

  const handleEditorMouseUp = () => {
    setIsDrawingInEditor(false);
  };

  const toggleEditorPixel = (exampleIndex: number, rowIndex: number, colIndex: number) => {
    const example = trainingExamples[exampleIndex];
    if (example?.id) {
      const pattern = example.pattern as number[][] | number[];
      let newPattern;

      if (Array.isArray(pattern[0])) {
        const grid = [...(pattern as number[][])];
        grid[rowIndex] = [...grid[rowIndex]];
        grid[rowIndex][colIndex] = grid[rowIndex][colIndex] ? 0 : 1;
        newPattern = grid;
      } else {
        const flatArray = [...(pattern as number[])];
        const flatIndex = rowIndex * 9 + colIndex;
        flatArray[flatIndex] = flatArray[flatIndex] ? 0 : 1;
        newPattern = flatArray;
      }

      updateTrainingExample(example.id, { pattern: newPattern, label: example.label as number[] });
      refreshExamples();
    } else {
      console.warn(`No valid example found at index ${exampleIndex}`, {
        example,
        trainingExamples,
      });
    }
  };

  const saveDataset = () => {
    setShowDatasetEditor(false);
    setCurrentExampleIndex(0);
  };

  const getPatternPreview = (pattern: number[][] | number[]) => {
    if (Array.isArray(pattern[0])) {
      return (pattern as number[][]).map((row) => row.reduce((sum, val) => sum + val, 0) / 9);
    } else {
      const flatPattern = pattern as number[];
      const preview = [];
      for (let i = 0; i < 9; i++) {
        const rowSum = flatPattern.slice(i * 9, (i + 1) * 9).reduce((sum, val) => sum + val, 0);
        preview.push(rowSum / 9);
      }
      return preview;
    }
  };

  // ========================
  // File upload function
  // ========================
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const jsonData = JSON.parse(content);

        if (Array.isArray(jsonData) && jsonData.length > 0) {
          const firstItem = jsonData[0];
          if (
            firstItem.input &&
            firstItem.target &&
            Array.isArray(firstItem.input) &&
            firstItem.input.length === 81 &&
            Array.isArray(firstItem.target) &&
            firstItem.target.length === 2
          ) {
            try {
              const count = bulkUploadTrainingExamples(jsonData);
              refreshExamples();
              alert(`Successfully uploaded ${count} training examples!`);
            } catch (err) {
              console.error("Bulk upload error:", err);
              alert("Failed to upload training data. Please check the file format.");
            }
          } else {
            alert(
              "Invalid file format. Expected array of objects with 'input' (81 numbers) and 'target' (2 numbers) fields.",
            );
          }
        } else {
          alert("Invalid file format. Expected a JSON array.");
        }
      } catch (error) {
        alert("Error reading file. Please make sure it's a valid JSON file.");
        console.error("File parsing error:", error);
      }
    };
    reader.readAsText(file);
    event.target.value = "";
  };

  // ========================
  // Inference
  // ========================
  const runInference = () => {
    if (mode !== "inference") return;

    const inputs = getPixelValues();

    const hiddenSums = currentNetworkState.current.weights.map(
      (neuronWeights, i) =>
        inputs.reduce((sum, input, j) => sum + input * neuronWeights[j], 0) +
        currentNetworkState.current.biases[i],
    );
    const hiddenOutputs = hiddenSums.map(sigmoid);

    const outputSums = currentNetworkState.current.outputWeights.map(
      (neuronWeights, i) =>
        hiddenOutputs.reduce((sum, hidden, j) => sum + hidden * neuronWeights[j], 0) +
        currentNetworkState.current.outputBiases[i],
    );
    const outputs = softmax(outputSums);

    setHiddenActivations(hiddenOutputs);
    setOutputActivations(outputs);

    const predictedDigit = outputs[0] > outputs[1] ? 0 : 1;
    const confidence = outputs[predictedDigit];

    setPrediction({ digit: predictedDigit, confidence });
  };

  // ========================
  // useEffects for syncing state
  // ========================

  // Load dataset example when in dataset mode
  useEffect(() => {
    if (trainingMode === "dataset" && trainingExamples[currentExampleIndex]) {
      const pattern = trainingExamples[currentExampleIndex].pattern as number[][] | number[];
      const grid = Array.isArray(pattern[0])
        ? (pattern as number[][])
        : flatToGrid(pattern as number[]);
      setPixelGrid(grid);
      let oneHotLabel;
      if (Array.isArray(trainingExamples[currentExampleIndex].label)) {
        oneHotLabel = trainingExamples[currentExampleIndex].label as number[];
      } else {
        let labelStr = trainingExamples[currentExampleIndex].label as string;
        if (labelStr.startsWith('"') && labelStr.endsWith('"')) {
          labelStr = labelStr.slice(1, -1);
        }
        oneHotLabel = JSON.parse(labelStr);
      }
      setSelectedLabel(oneHotLabel[0] === 1 ? 0 : 1);
    }
  }, [trainingMode, currentExampleIndex, trainingExamples]);

  // Update active elements based on current step
  useEffect(() => {
    if (STEP_DESCRIPTIONS[step]) {
      setActiveElements(STEP_DESCRIPTIONS[step].activeElements);
    } else {
      setActiveElements([]);
    }
  }, [step]);

  // Update weight dialog iteration to show latest when training history changes
  useEffect(() => {
    if (trainingHistory.length > 0) {
      setWeightDialogIteration(trainingHistory.length - 1);
    }
  }, [trainingHistory.length]);

  // Sync refs with state changes
  useEffect(() => {
    selectedLabelRef.current = selectedLabel;
  }, [selectedLabel]);
  useEffect(() => {
    pixelGridRef.current = pixelGrid;
  }, [pixelGrid]);

  // Trigger tour validation when training progress changes (for step 10 validation)
  useEffect(() => {
    if (tourTriggerRef.current) {
      console.log("🔔 TOUR: Training progress changed, triggering validation");
      tourTriggerRef.current();
    }
  }, [
    trainedSampleCount,
    trainingCompleted,
    isAutoTraining,
    currentEpoch,
    numberOfEpochs,
    trainingExamples.length,
  ]);

  // Auto-run inference when in inference mode and canvas changes
  useEffect(() => {
    if (mode === "inference") {
      runInference();
    }
  }, [mode, pixelGrid]);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    };
  }, []);

  // ========================
  // JSX — Composed from extracted sub-components
  // ========================
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-2 sm:p-4">
      <div className="mx-auto max-w-6xl">
        <AppHeader
          onOpenGuidedTour={() => setIsGuidedTourOpen(true)}
          onOpenAbout={() => setIsAboutOpen(true)}
        />

        <div className="grid grid-cols-1 gap-3 sm:gap-6 lg:grid-cols-4">
          {/* Drawing Canvas */}
          <DrawingCanvas
            pixelGrid={pixelGrid}
            canvasRef={canvasRef}
            handleMouseUp={handleMouseUp}
            handleTouchStart={handleTouchStart}
            handleTouchMove={handleTouchMove}
            handleTouchEnd={handleTouchEnd}
            handleMouseDown={handleMouseDown}
            handleMouseEnter={handleMouseEnter}
            clearCanvas={clearCanvas}
            mode={mode}
            selectedLabel={selectedLabel}
            setSelectedLabel={setSelectedLabel}
            setMode={setMode}
            modeRef={modeRef}
            setStep={setStep}
            setPixelGrid={setPixelGrid}
            setTourInferenceModeEnabled={setTourInferenceModeEnabled}
            tourInferenceModeEnabledRef={tourInferenceModeEnabledRef}
            prediction={prediction}
            setPrediction={setPrediction}
            learningRate={learningRate}
            setLearningRate={setLearningRate}
            trainingExamplesCount={trainingExamples.length}
          />

          {/* Neural Network Diagram */}
          <NetworkDiagram
            pixelValues={getPixelValues()}
            hiddenActivations={hiddenActivations}
            outputActivations={outputActivations}
            outputBiases={outputBiases}
            weights={weights}
            outputWeights={outputWeights}
            activeElements={activeElements}
            trainingHistoryLength={trainingHistory.length}
            mode={mode}
            loss={loss}
            showDebugDialog={showDebugDialog}
            setShowDebugDialog={setShowDebugDialog}
            setSelectedWeightBox={setSelectedWeightBox}
            setWeightDialogIteration={setWeightDialogIteration}
            setTourWeightVisualizationOpened={setTourWeightVisualizationOpened}
            tourWeightVisualizationOpenedRef={tourWeightVisualizationOpenedRef}
          />

          {/* Training Steps / Controls */}
          <TrainingStepPanel
            trainingMode={trainingMode}
            setTrainingMode={setTrainingMode}
            setTrainingCompleted={setTrainingCompleted}
            setTourDatasetLoaded={setTourDatasetLoaded}
            datasetLoadedRef={datasetLoadedRef}
            tourTriggerRef={tourTriggerRef}
            step={step}
            setStep={setStep}
            isAutoTraining={isAutoTraining}
            trainingCompleted={trainingCompleted}
            numberOfEpochs={numberOfEpochs}
            currentEpoch={currentEpoch}
            currentExampleIndex={currentExampleIndex}
            trainingExamples={trainingExamples}
            epochLossHistory={epochLossHistory}
            nextStep={nextStep}
            setTourStepExecuted={setTourStepExecuted}
            resetNetwork={resetNetwork}
            setShowDatasetEditor={setShowDatasetEditor}
            handleFileUpload={handleFileUpload}
            showModelManagement={showModelManagement}
            setShowModelManagement={setShowModelManagement}
            showModelManagementRef={showModelManagementRef}
            handleExportCheckpoint={handleExportCheckpoint}
            handleImportCheckpointFile={handleImportCheckpointFile}
            lrDecayEnabled={lrDecayEnabled}
            setLrDecayEnabled={setLrDecayEnabled}
            lrDecayRate={lrDecayRate}
            setLrDecayRate={setLrDecayRate}
            minLR={minLR}
            setMinLR={setMinLR}
            lrHistory={lrHistory}
            completedEpochs={completedEpochs}
            examplesSeen={examplesSeen}
            learningRate={learningRate}
            lastEpochAvgLoss={lastEpochAvgLoss}
            normalizeEnabled={normalizeEnabled}
            targetSize={targetSize}
            lastCheckpointLoaded={lastCheckpointLoaded}
            mode={mode}
            autoTrainingSpeed={autoTrainingSpeed}
            runToNextSample={runToNextSample}
            processTrainingSet={processTrainingSet}
            stopTraining={stopTraining}
            setCurrentExampleIndex={setCurrentExampleIndex}
          />
        </div>

        {/* Detailed Weight View - Below main grid */}
        {selectedWeightBox && (
          <WeightDetailView
            selectedWeightBox={selectedWeightBox}
            setSelectedWeightBox={setSelectedWeightBox}
            setTourWeightVisualizationOpened={setTourWeightVisualizationOpened}
            tourWeightVisualizationOpenedRef={tourWeightVisualizationOpenedRef}
            weightDialogIteration={weightDialogIteration}
            setWeightDialogIteration={setWeightDialogIteration}
            trainingHistory={trainingHistory}
            weights={weights}
            biases={biases}
            outputWeights={outputWeights}
            outputBiases={outputBiases}
            currentNetworkState={currentNetworkState}
            pixelGrid={pixelGrid}
            showInputOverlay={showInputOverlay}
            setShowInputOverlay={setShowInputOverlay}
            useGlobalScale={useGlobalScale}
            setUseGlobalScale={setUseGlobalScale}
            colorScheme={colorScheme}
            setColorScheme={setColorScheme}
            viewMode={viewMode}
            setViewMode={setViewMode}
          />
        )}

        {/* Debug History Dialog */}
        {showDebugDialog && (
          <DebugHistoryPanel debugHistory={debugHistory} setShowDebugDialog={setShowDebugDialog} />
        )}

        {/* Dataset Editor Dialog */}
        <DatasetEditorDialog
          showDatasetEditor={showDatasetEditor}
          setShowDatasetEditor={setShowDatasetEditor}
          trainingExamples={trainingExamples}
          addDatasetExample={addDatasetExample}
          removeDatasetExample={removeDatasetExample}
          updateDatasetExample={updateDatasetExample}
          handleEditorMouseDown={handleEditorMouseDown}
          handleEditorMouseEnter={handleEditorMouseEnter}
          handleEditorMouseUp={handleEditorMouseUp}
          saveDataset={saveDataset}
          getPatternPreview={getPatternPreview}
        />

        {/* Epoch Selection Dialog */}
        <EpochSelectionDialog
          isEpochDialogOpen={isEpochDialogOpen}
          setIsEpochDialogOpen={setIsEpochDialogOpen}
          trainingExamplesCount={trainingExamples.length}
          numberOfEpochs={numberOfEpochs}
          setNumberOfEpochs={setNumberOfEpochs}
          startMultiEpochTraining={startMultiEpochTraining}
          setTourMultiEpochStarted={setTourMultiEpochStarted}
        />

        {/* About Dialog */}
        <AboutDialog isAboutOpen={isAboutOpen} setIsAboutOpen={setIsAboutOpen} />

        {/* Guided Tour Component */}
        <GuidedTour
          isOpen={isGuidedTourOpen}
          onClose={() => setIsGuidedTourOpen(false)}
          onReset={() => {
            // Reset everything for clean tour start
            setPixelGrid(
              Array(9)
                .fill(0)
                .map(() => Array(9).fill(0)),
            );
            setStep(0);
            setMode("training");
            setTrainingMode("manual");
            setTrainingCompleted(false);
            setIsAutoTraining(false);
            setCurrentEpoch(1);
            setNumberOfEpochs(3);
            setHiddenActivations(Array(24).fill(0));
            setOutputActivations(Array(2).fill(0));
            // Reset tour tracking
            setTourDrawnOnCanvas(false);
            setTourStepExecuted(false);
            setTourDatasetLoaded(false);
            setTourNextSampleClicked(false);
            setTourMultiEpochStarted(false);
            setTourWeightVisualizationOpened(false);
            setTourInferenceModeEnabled(false);
            setTourCheckpointSaved(false);
            setTourCheckpointLoaded(false);
            setTourTrainingCycleCompleted(false);
            // Reset training completion state for tour
            setTrainingCompleted(false);
            trainedSampleCountRef.current = 0;
            setTrainedSampleCount(0);
            // Reset ref-based tracking
            multiEpochStartedRef.current = false;
            datasetLoadedRef.current = false;
            nextSampleClickedRef.current = false;
            checkpointSavedRef.current = false;
            tourInferenceModeEnabledRef.current = false;
            tourWeightVisualizationOpenedRef.current = false;
            showModelManagementRef.current = false;
            modeRef.current = "training";
            // Reset cycle counters for fresh tour start
            resetCycleCounters();
            // Load tour training dataset into localStorage
            clearTrainingExamples();
            bulkUploadTrainingExamples(tourTrainingData);
            refreshExamples();
          }}
          onValidationTrigger={(triggerFn) => {
            tourTriggerRef.current = triggerFn;
          }}
          tourSteps={createTourSteps(
            validationDrewSomething,
            () => tourStepExecuted,
            validOneClick,
            validFullCycle,
            checkDatasetLoaded,
            checkNextSampleClicked,
            checkEpochTrainingStarted,
            checkTrainingCompleted,
            checkModelManagementExpanded,
            checkCheckpointSaved,
            checkInferenceModeActive,
            checkWeightVisualizationOpened,
            stopTraining,
            loadTourCheckpoint,
          )}
        />
      </div>
    </div>
  );
}
