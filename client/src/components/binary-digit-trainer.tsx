import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Trash2, Plus, Edit3, Upload, ChevronDown, ChevronRight, Save, FolderOpen } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { TrainingExample, InsertTrainingExample } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";
import { HelpIcon } from "@/components/HelpIcon";
import GuidedTour from './GuidedTour';
import { createTourSteps } from '@/lib/tour-steps';


// Weight initialization helper using Xavier/Glorot initialization
const initWeight = (n_in: number, n_out: number) => 
  (Math.random() - 0.5) * Math.sqrt(2 / (n_in + n_out));

// 9x9 pixel grid (81 pixels total, each pixel is 0 or 1)
const initialPixelGrid = Array(9).fill(0).map(() => Array(9).fill(0)); // 9x9 grid of pixels
const initialWeights = Array.from({ length: 24 }, () => Array(81).fill(0).map(() => initWeight(81, 24)));
const initialBiases = Array(24).fill(0);
const initialOutputWeights = Array.from({ length: 2 }, () => Array(24).fill(0).map(() => initWeight(24, 2)));
const initialOutputBiases = Array(2).fill(0);

// Training dataset - 100+ examples each of 0 and 1
const generateTrainingDataset = () => {
  const dataset: { pattern: number[][], label: number }[] = [];
  
  // Empty dataset - user will create examples from scratch
  return dataset;
};

const sigmoid = (x: number) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const sigmoidDerivative = (z: number) => {
  const s = sigmoid(z);
  return s * (1 - s);
};

// Softmax function for output layer
const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const exps = z.map(v => Math.exp(v - m));
  const sum = exps.reduce((a,b)=>a+b, 0);
  return exps.map(e => e / sum);
};

// Gradient clipping constants
const GRADIENT_CLIP = 1.0;
const clip = (g: number) => Math.max(-GRADIENT_CLIP, Math.min(GRADIENT_CLIP, g));

// Helper functions to convert between flat array and 2D array formats
const flatToGrid = (flatArray: number[]): number[][] => {
  const grid: number[][] = [];
  for (let i = 0; i < 9; i++) {
    grid.push(flatArray.slice(i * 9, (i + 1) * 9));
  }
  return grid;
};

const gridToFlat = (grid: number[][]): number[] => {
  return grid.flat();
};

// Helper function to parse labels consistently
const parseLabel = (label: any): number[] => {
  if (Array.isArray(label)) {
    return label;
  }
  if (typeof label === 'string') {
    // Handle both regular JSON strings and double-quoted strings
    let labelStr = label;
    if (labelStr.startsWith('"') && labelStr.endsWith('"')) {
      labelStr = labelStr.slice(1, -1); // Remove outer quotes
    }
    return JSON.parse(labelStr);
  }
  // Fallback for any other format
  return [1, 0];
};

// Decision contribution types and helpers
type DecisionContrib = { 
  idx: number; 
  contrib: number; 
  w0: number; 
  w1: number; 
  h: number 
};

const getDecisionContribs = (hiddenActivations: number[], outputWeights: number[][]): DecisionContrib[] => {
  const w0 = outputWeights[0];
  const w1 = outputWeights[1];
  return w0.map((w0j, j) => ({ 
    idx: j, 
    contrib: (w0j - w1[j]) * hiddenActivations[j], 
    w0: w0j, 
    w1: w1[j], 
    h: hiddenActivations[j] 
  }));
};

// Helper function to get current target (eliminates race conditions)
const getCurrentTarget = (currentNetworkState: any, trainingMode: string, trainingExamples: any[], currentExampleIndex: number, selectedLabelRef: any): number[] => {
  // First check cached target from network state
  const cached = currentNetworkState.current.currentTarget;
  if (cached && cached.length === 2) {
    return cached;
  }
  
  // Fallback: rebuild target based on current mode
  if (trainingMode === 'dataset' && trainingExamples[currentExampleIndex]) {
    const example = trainingExamples[currentExampleIndex];
    if (Array.isArray(example.label)) {
      return example.label;
    } else {
      let labelStr = example.label as string;
      if (labelStr.startsWith('"') && labelStr.endsWith('"')) {
        labelStr = labelStr.slice(1, -1);
      }
      return JSON.parse(labelStr);
    }
  }
  
  // Last resort: manual mode
  return selectedLabelRef.current === 0 ? [1, 0] : [0, 1];
};

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

const STEP_DESCRIPTIONS = [
  {
    name: "Ready - Draw your digit",
    concept: "The neural network is ready. In Training mode: draw and train step-by-step. In Predict mode: draw and get instant predictions.",
    formula: "Input preparation: x = [x₁, x₂, ..., x₈₁] where each xᵢ ∈ [0,1]",
    activeElements: []
  },
  {
    name: "Forward Pass - Input to Hidden",
    concept: "Each input pixel is multiplied by connection weights and summed with bias to calculate hidden neuron activations.",
    formula: "hⱼ = σ(∑ᵢ wᵢⱼ·xᵢ + bⱼ) where σ(z) = 1/(1+e⁻ᶻ)",
    activeElements: ["input", "hidden", "inputWeights"]
  },
  {
    name: "Forward Pass - Hidden to Output", 
    concept: "Hidden layer activations are combined using output weights and softmax to produce probability predictions for digits 0 and 1.",
    formula: "zₖ = ∑ⱼ wⱼₖ·hⱼ + bₖ, then pₖ = softmax(zₖ)",
    activeElements: ["hidden", "output", "outputWeights"]
  },
  {
    name: "Calculate Loss",
    concept: "The network's prediction is compared to the target label using Cross-Entropy Loss to measure accuracy.",
    formula: "Loss = -∑ₖ tₖ·log(pₖ) where tₖ is target and pₖ is predicted probability",
    activeElements: ["output", "loss"]
  },
  {
    name: "Backpropagation - Output Layer",
    concept: "Error signals flow backward to adjust output weights. The softmax + cross-entropy gradient is simplified.",
    formula: "δₖ = pₖ - tₖ, Δwⱼₖ = α·clip(δₖ)·hⱼ",
    activeElements: ["output", "outputWeights", "backprop"]
  },
  {
    name: "Backpropagation - Hidden Layer",
    concept: "Error signals propagate to hidden layer, adjusting input weights based on their contribution to the total error.",
    formula: "δⱼ = hⱼ·(1-hⱼ)·∑ₖδₖ·wⱼₖ, Δwᵢⱼ = α·δⱼ·xᵢ",
    activeElements: ["hidden", "inputWeights", "backprop"]
  }
];

export default function BinaryDigitTrainer() {
  const queryClient = useQueryClient();
  
  // Fetch training examples from the API
  const { data: trainingExamples = [], isLoading: loadingExamples } = useQuery<TrainingExample[]>({
    queryKey: ["/api/training-examples"],
    queryFn: () => fetch("/api/training-examples").then(res => res.json()),
  });

  // Mutations for CRUD operations
  const createExampleMutation = useMutation({
    mutationFn: (example: InsertTrainingExample) => 
      apiRequest("POST", "/api/training-examples", example),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training-examples"] });
    },
  });

  const updateExampleMutation = useMutation({
    mutationFn: ({ id, example }: { id: number; example: InsertTrainingExample }) =>
      apiRequest("PUT", `/api/training-examples/${id}`, example),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training-examples"] });
    },
  });

  const deleteExampleMutation = useMutation({
    mutationFn: (id: number) =>
      apiRequest("DELETE", `/api/training-examples/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training-examples"] });
    },
  });

  const bulkUploadMutation = useMutation({
    mutationFn: (data: { input: number[]; target: number[] }[]) =>
      apiRequest("POST", "/api/training-examples/bulk-upload", data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ["/api/training-examples"] });
      alert(`Successfully uploaded ${data.count} training examples!`);
    },
    onError: (error) => {
      console.error("Bulk upload error:", error);
      alert("Failed to upload training data. Please check the file format.");
    }
  });

  const clearExamplesMutation = useMutation({
    mutationFn: () =>
      apiRequest("DELETE", "/api/training-examples"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training-examples"] });
      // Also clear any cached data to prevent stale ID issues
      queryClient.removeQueries({ queryKey: ["/api/training-examples"] });
    },
  });

  const [pixelGrid, setPixelGridState] = useState(initialPixelGrid);
  
  // Safe setter that updates both ref (immediate) and React state (async, UI only)
  const setPixelGrid = (grid: number[][] | number[]) => {
    const normalized = Array.isArray(grid[0]) ? (grid as number[][]) : flatToGrid(grid as number[]);
    pixelGridRef.current = normalized;             // <- immediate, used by training logic
    setPixelGridState(normalized);                 // <- async, UI only
  };
  
  // Safe setter for selected label
  const setSelectedLabelSafe = (label: number) => {
    selectedLabelRef.current = label;
    setSelectedLabel(label);
  };
  const [weights, setWeights] = useState(initialWeights);
  const [biases, setBiases] = useState(initialBiases);
  const [outputWeights, setOutputWeights] = useState(initialOutputWeights);
  const [outputBiases, setOutputBiases] = useState(initialOutputBiases);
  const [hiddenActivations, setHiddenActivations] = useState(Array(24).fill(0));
  const [outputActivations, setOutputActivations] = useState(Array(2).fill(0));
  const [selectedLabel, setSelectedLabel] = useState(0);
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(0);
  const [learningRate, setLearningRate] = useState(0.01);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hoveredPixel, setHoveredPixel] = useState<number | null>(null);
  const [selectedWeightBox, setSelectedWeightBox] = useState<{type: 'hidden' | 'output', index: number} | null>(null);
  const [weightDialogIteration, setWeightDialogIteration] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  
  // Guided tour and about dialog state
  const [isGuidedTourOpen, setIsGuidedTourOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  
  // Tour tracking state
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
  const tourTriggerRef = useRef<(() => void) | null>(null);
  const clicksThisCycleRef = useRef(0);
  const cycleDoneRef = useRef(false);
  const datasetLoadedRef = useRef(false);
  const nextSampleClickedRef = useRef(false);
  const multiEpochStartedRef = useRef(false);
  const checkpointSavedRef = useRef(false);
  const isDrawingRef = useRef(false);
  const changedCellsRef = useRef(0);

  // Tour validation functions
  const validationDrewSomething = () => {
    const flat = pixelGridRef.current?.flat() || pixelGrid.flat();
    // Require at least 3 lit pixels so a single accidental click doesn't pass
    const lit = flat.reduce((s, v) => s + (v ? 1 : 0), 0);
    return lit >= 3;
  };

  const checkTrainingStarted = () => tourStepStarted;
  // Reset cycle counters (call when starting fresh cycle)
  const resetCycleCounters = () => {
    console.log('🔄 TOUR: Resetting cycle counters');
    clicksThisCycleRef.current = 0;
    cycleDoneRef.current = false;
  };

  // Validation functions for tour steps
  const validOneClick = () => {
    const result = clicksThisCycleRef.current >= 1;
    console.log('🔍 TOUR: validOneClick - clicks:', clicksThisCycleRef.current, 'result:', result);
    return result;
  };
  const validFullCycle = () => {
    const result = cycleDoneRef.current === true;
    console.log('🔍 TOUR: validFullCycle - cycleDoneRef.current:', cycleDoneRef.current, 'result:', result);
    return result;
  };
  const checkTrainingStepsCompleted = () => {
    console.log('🔍 Tour validation - tourTrainingCycleCompleted:', tourTrainingCycleCompleted);
    return tourTrainingCycleCompleted;
  };
  const checkDatasetLoaded = () => {
    const result = datasetLoadedRef.current === true;
    console.log('🔍 TOUR: checkDatasetLoaded - datasetLoadedRef.current:', datasetLoadedRef.current, 'result:', result);
    return result;
  };
  const checkNextSampleClicked = () => {
    const result = nextSampleClickedRef.current === true;
    console.log('🔍 TOUR: checkNextSampleClicked - nextSampleClickedRef.current:', nextSampleClickedRef.current, 'result:', result);
    return result;
  };
  const checkEpochTrainingStarted = () => {
    const result = multiEpochStartedRef.current === true;
    console.log('🔍 TOUR: checkEpochTrainingStarted - multiEpochStartedRef.current:', multiEpochStartedRef.current, 'result:', result);
    return result;
  };
  const checkCheckpointSaved = () => {
    const result = checkpointSavedRef.current === true;
    console.log('🔍 TOUR: checkCheckpointSaved - checkpointSavedRef.current:', checkpointSavedRef.current, 'result:', result);
    return result;
  };
  const checkTrainingCompleted = () => {
    // Training is considered complete when:
    // 1. Multi-epoch training was started (multiEpochStartedRef.current is true), AND
    // 2. trainingCompleted is true (either finished all epochs or manually stopped), AND
    // 3. isAutoTraining is false (no active training process)
    const multiEpochWasStarted = multiEpochStartedRef.current === true;
    const result = multiEpochWasStarted && trainingCompleted && !isAutoTraining;
    console.log('🔍 TOUR: checkTrainingCompleted - multiEpochStarted:', multiEpochWasStarted, 'trainingCompleted:', trainingCompleted, 'isAutoTraining:', isAutoTraining, 'result:', result);
    return result;
  };
  const checkModelManagementExpanded = () => {
    const result = showModelManagement;
    console.log('🔍 TOUR: checkModelManagementExpanded - showModelManagement:', showModelManagement, 'result:', result);
    return result;
  };
  const checkInferenceModeActive = () => mode === 'inference';
  const checkWeightVisualizationOpened = () => tourWeightVisualizationOpened;

  // Reset function for tour
  const handleTourReset = () => {
    // Reset network
    const newWeights = Array.from({ length: 24 }, () => Array(81).fill(0).map(() => initWeight(81, 24)));
    const newBiases = Array(24).fill(0);
    const newOutputWeights = Array.from({ length: 2 }, () => Array(24).fill(0).map(() => initWeight(24, 2)));
    const newOutputBiases = Array(2).fill(0);
    
    setWeights(newWeights);
    setBiases(newBiases);
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
    setHiddenActivations(Array(24).fill(0));
    setOutputActivations(Array(2).fill(0));
    setStep(0);
    setLoss(0);
    
    // Update persistent state
    currentNetworkState.current = {
      weights: newWeights.map(w => [...w]),
      biases: [...newBiases],
      outputWeights: newOutputWeights.map(w => [...w]),
      outputBiases: [...newOutputBiases],
      hiddenActivations: Array(24).fill(0),
      outputActivations: Array(2).fill(0),
      loss: 0,
      outputErrors: Array(2).fill(0),
      currentTarget: [0, 0]
    };
    
    // Clear canvas
    const clearGrid = Array(9).fill(0).map(() => Array(9).fill(0));
    setPixelGrid(clearGrid);
    
    // Reset training mode and other states
    setMode('training');
    setTrainingMode('manual');
    setSelectedLabel(0);
    setPrediction(null);
    setIsAutoTraining(false);
    setTrainingCompleted(false);
    
    // Reset tour tracking
    setTourStepStarted(false);
    setTourDatasetLoaded(false);
    setTourNextSampleClicked(false);
    setTourMultiEpochStarted(false);
    setTourCheckpointSaved(false);
    setTourWeightVisualizationOpened(false);
    
    // Reset tour refs
    cycleDoneRef.current = false;
    datasetLoadedRef.current = false;
    nextSampleClickedRef.current = false;
    multiEpochStartedRef.current = false;
    checkpointSavedRef.current = false;
  };
  
  // New state for enhanced features
  const [trainingMode, setTrainingMode] = useState<'manual' | 'dataset'>('manual');
  const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
  const [stepHistory, setStepHistory] = useState<any[]>([]);
  const [currentStepInHistory, setCurrentStepInHistory] = useState(0);
  const [activeElements, setActiveElements] = useState<string[]>([]);
  const [showDatasetEditor, setShowDatasetEditor] = useState(false);
  const [isDrawingInEditor, setIsDrawingInEditor] = useState(false);
  
  // New state for automated training and inference mode
  const [mode, setMode] = useState<'training' | 'inference'>('training');
  const [isAutoTraining, setIsAutoTraining] = useState(false);
  const [autoTrainingSpeed, setAutoTrainingSpeed] = useState(50); // ms between steps - much faster for automated training
  const [prediction, setPrediction] = useState<{digit: number, confidence: number} | null>(null);
  const [isEpochDialogOpen, setIsEpochDialogOpen] = useState(false);
  const [numberOfEpochs, setNumberOfEpochs] = useState(1);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  
  // Using Cross-Entropy Loss (fixed)
  const lossFunction = 'crossentropy';
  
  // Epoch loss tracking
  const [epochLossHistory, setEpochLossHistory] = useState<{epoch: number, averageLoss: number}[]>([]);
  const currentEpochLoss = useRef<number[]>([]);
  const [trainingCompleted, setTrainingCompleted] = useState(false);
  
  // Debug info display
  const [showDebugDialog, setShowDebugDialog] = useState(false);
  const [debugHistory, setDebugHistory] = useState<{
    iteration: number;
    label: number[]; // Changed to number[] for one-hot format
    outputActivations: number[];
    outputErrors: number[];
    outputBiases: number[];
    loss: number;
    step: number;
    timestamp: Date;
  }[]>([]);

  // ----- LR decay and checkpoint stats -----
  const [lrDecayEnabled, setLrDecayEnabled] = useState(false);
  const [lrDecayRate, setLrDecayRate] = useState(0.99);   // per epoch multiply
  const [minLR, setMinLR] = useState(0.0005);
  const [normalizeEnabled, setNormalizeEnabled] = useState(false); // Placeholder for normalization
  const [targetSize, setTargetSize] = useState(7); // Placeholder for normalization

  // ----- Stats for checkpoint metadata -----
  const [examplesSeen, setExamplesSeen] = useState(0);
  const [lastEpochAvgLoss, setLastEpochAvgLoss] = useState<number | null>(null);
  const [completedEpochs, setCompletedEpochs] = useState(0);
  
  // ----- Learning rate history for visualization -----
  const [lrHistory, setLrHistory] = useState<{epoch: number, learningRate: number}[]>([]);

  // ----- Model Management UI -----
  const [showModelManagement, setShowModelManagement] = useState(false);
  const [lastCheckpointLoaded, setLastCheckpointLoaded] = useState<string | null>(null);
  
  // ----- Activation Explorer UI -----
  const [showInputOverlay, setShowInputOverlay] = useState(false);
  const [useGlobalScale, setUseGlobalScale] = useState(false);
  const [colorScheme, setColorScheme] = useState<'blue-red' | 'blue-orange' | 'green-purple' | 'high-contrast'>('blue-red');
  const [viewMode, setViewMode] = useState<'decision' | 'logit'>('logit'); // Default to logit view

  // ----- Canvas utility functions -----
  const clearCanvas = () => {
    setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
    setPrediction(null);
  };

  // ----- Color scheme helper for bar graphs -----
  const getBarColor = (weight: number, scheme: string = colorScheme) => {
    const isPositive = weight > 0;
    
    switch (scheme) {
      case 'blue-red':
        return isPositive ? "#3B82F6" : "#EF4444"; // Blue : Red
      
      case 'blue-orange':
        return isPositive ? "#3B82F6" : "#F97316"; // Blue : Orange
      
      case 'green-purple':
        return isPositive ? "#22C55E" : "#A855F7"; // Green : Purple
      
      case 'high-contrast':
        return isPositive ? "#6B7280" : "#1F2937"; // Light gray : Dark gray
      
      default:
        return isPositive ? "#3B82F6" : "#EF4444";
    }
  };

  // ----- Color scheme descriptions -----
  const getColorSchemeDescription = (scheme: string) => {
    switch (scheme) {
      case 'blue-red':
        return "Blue indicates positive values, the darker the color the more positive the value. Red indicates negative values, the darker the color the more negative the value.";
      
      case 'blue-orange':
        return "Blue indicates positive values, the darker the color the more positive the value. Orange indicates negative values, the darker the color the more negative the value.";
      
      case 'green-purple':
        return "Green indicates positive values, the darker the color the more positive the value. Purple indicates negative values, the darker the color the more negative the value.";
      
      case 'high-contrast':
        return "Light gray indicates positive values, the darker the color the more positive the value. Dark gray indicates negative values, the darker the color the more negative the value.";
      
      default:
        return "Blue indicates positive values, the darker the color the more positive the value. Red indicates negative values, the darker the color the more negative the value.";
    }
  };

  // Helper to get the positive color name for current scheme
  const getPositiveColorName = (scheme: string = colorScheme) => {
    switch (scheme) {
      case 'blue-red':
        return "blue";
      case 'blue-orange':
        return "blue";
      case 'green-purple':
        return "green";
      case 'high-contrast':
        return "light gray";
      default:
        return "blue";
    }
  };

  // Persistent training history store - independent of React state
  const trainingHistoryStore = useRef<any[]>([]);
  
  // Training control refs
  const trainingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const shouldStopTraining = useRef(false);
  
  // Refs for synchronous access during training (eliminates async state issues)
  const pixelGridRef = useRef<number[][]>(initialPixelGrid);
  const selectedLabelRef = useRef<number>(0);
  
  // Current network state that gets updated during training
  const currentNetworkState = useRef({
    weights: initialWeights.map(w => [...w]),
    biases: [...initialBiases],
    outputWeights: initialOutputWeights.map(w => [...w]),
    outputBiases: [...initialOutputBiases],
    hiddenActivations: Array(24).fill(0),
    outputActivations: Array(2).fill(0),
    hiddenPreActivations: Array(24).fill(0), // z1 values (pre-activation)
    outputPreActivations: Array(2).fill(0),   // z2 values (pre-activation)
    loss: 0,
    outputErrors: Array(2).fill(0),
    // Training-specific state (isolated from React)
    currentTarget: [1, 0] as number[],
    inputs: Array(81).fill(0) as number[]
  });

  // Load dataset example when in dataset mode
  useEffect(() => {
    if (trainingMode === 'dataset' && trainingExamples[currentExampleIndex]) {
      const pattern = trainingExamples[currentExampleIndex].pattern as number[][] | number[];
      // Convert flat array to 2D grid if needed
      const grid = Array.isArray(pattern[0]) ? pattern as number[][] : flatToGrid(pattern as number[]);
      setPixelGrid(grid);
      // Convert one-hot label back to integer for UI display
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
  useEffect(() => { selectedLabelRef.current = selectedLabel; }, [selectedLabel]);
  useEffect(() => { pixelGridRef.current = pixelGrid; }, [pixelGrid]);

  // Calculate pixel values - read from ref for training logic, fallback to state for UI
  const getPixelValues = () => {
    // In inference mode, always use the current pixel grid (fresh drawing)
    if (mode === 'inference') {
      return pixelGridRef.current?.flat() || pixelGrid.flat();
    }
    
    // During training, use cached inputs from network state if available
    if (currentNetworkState.current.inputs && currentNetworkState.current.inputs.some(x => x !== 0)) {
      return currentNetworkState.current.inputs;
    }
    // Otherwise read from ref (immediate) or state (fallback)
    return pixelGridRef.current?.flat() || pixelGrid.flat();
  };

  const togglePixel = (rowIndex: number, colIndex: number) => {
    setPixelGridState((prev) => {
      const next = prev.map((row, r) =>
        r === rowIndex ? row.map((v, c) => (c === colIndex ? (v ? 0 : 1) : v)) : row
      );
      // Update ref immediately for training logic
      pixelGridRef.current = next;
      changedCellsRef.current += 1;
      return next;
    });
    setStep(0); // Reset to first step when input changes
    setTourDrawnOnCanvas(true); // Tour tracking
  };

  const handleMouseDown = (rowIndex: number, colIndex: number) => {
    setIsDrawing(true);
    isDrawingRef.current = true;
    changedCellsRef.current = 0;
    togglePixel(rowIndex, colIndex);
  };

  const handleMouseEnter = (rowIndex: number, colIndex: number) => {
    if (isDrawing) {
      togglePixel(rowIndex, colIndex);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      // Only signal the tour if something actually changed
      if (changedCellsRef.current > 0) {
        tourTriggerRef.current?.();
      }
    }
  };

  const handlePixelHover = (pixelIndex: number) => {
    setHoveredPixel(pixelIndex);
  };

  const handlePixelLeave = () => {
    setHoveredPixel(null);
  };

  // Turn 81 weights into a 9x9 matrix
  const vec81ToGrid9 = (v: number[]) => {
    const g: number[][] = [];
    for (let r = 0; r < 9; r++) g.push(v.slice(r * 9, (r + 1) * 9));
    return g;
  };

  // Diverging color map with multiple scheme support
  const weightColor = (x: number, maxAbs: number, scheme: string = colorScheme) => {
    const a = maxAbs || 1e-6;
    const t = Math.max(-1, Math.min(1, x / a)); // normalize to [-1,1]
    const intensity = Math.abs(t);
    const c = Math.round(255 * intensity);
    
    switch (scheme) {
      case 'blue-red':
        if (t >= 0) {
          return `rgb(${255 - c}, ${255 - c}, 255)`; // Blue: positive
        } else {
          return `rgb(255, ${255 - c}, ${255 - c})`; // Red: negative
        }
      
      case 'blue-orange':
        if (t >= 0) {
          return `rgb(${255 - c}, ${255 - c}, 255)`; // Blue: positive
        } else {
          return `rgb(255, ${255 - Math.round(c * 0.6)}, ${255 - c})`; // Orange: negative
        }
      
      case 'green-purple':
        if (t >= 0) {
          return `rgb(${255 - c}, 255, ${255 - c})`; // Green: positive
        } else {
          return `rgb(${255 - Math.round(c * 0.3)}, ${255 - c}, 255)`; // Purple: negative
        }
      
      case 'high-contrast':
        if (t >= 0) {
          return `rgb(${255 - c}, ${255 - c}, ${255 - c})`; // Light gray to white: positive
        } else {
          return `rgb(${c}, ${c}, ${c})`; // Dark gray to black: negative
        }
      
      default:
        if (t >= 0) {
          return `rgb(${255 - c}, ${255 - c}, 255)`;
        } else {
          return `rgb(255, ${255 - c}, ${255 - c})`;
        }
    }
  };

  // Heatmap component (tiny, fast, no dependencies)
  function Heatmap9x9({ grid, cell = 18, showInputOverlay = false, inputGrid = null, globalMaxAbs = null }: { 
    grid: number[][]; 
    cell?: number; 
    showInputOverlay?: boolean;
    inputGrid?: number[][] | null;
    globalMaxAbs?: number | null;
  }) {
    const flat = grid.flat();
    const localMaxAbs = flat.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    const maxAbs = globalMaxAbs !== null ? globalMaxAbs : localMaxAbs;
    return (
      <div className="inline-grid" style={{ gridTemplateColumns: `repeat(9, ${cell}px)` }}>
        {grid.map((row, r) =>
          row.map((v, c) => {
            const isInputActive = inputGrid?.[r]?.[c] === 1;
            const baseStyle = {
              width: cell, 
              height: cell, 
              background: weightColor(v, maxAbs, colorScheme),
              opacity: showInputOverlay && !isInputActive ? 0.3 : 1,
              border: showInputOverlay && isInputActive ? '2px solid #000' : '1px solid rgba(255,255,255,0.4)'
            };
            return (
              <div
                key={`${r}-${c}`}
                title={`Weight: ${v.toFixed(3)}${showInputOverlay ? `, Input: ${isInputActive ? '1' : '0'}` : ''}`}
                style={baseStyle}
                className="transition-opacity duration-200"
              />
            );
          })
        )}
      </div>
    );
  }

  const forwardPassHidden = () => {
    const pixelValues = getPixelValues();
    // Calculate pre-activation values (z1)
    const newPreActivations = currentNetworkState.current.weights.map((w, i) => 
      w.reduce((sum, weight, j) => sum + weight * pixelValues[j], currentNetworkState.current.biases[i])
    );
    // Apply sigmoid to get activations
    const newActivations = newPreActivations.map(z => sigmoid(z));
    
    // Store both pre-activation and activation values
    currentNetworkState.current.hiddenPreActivations = newPreActivations;
    currentNetworkState.current.hiddenActivations = newActivations;
    setHiddenActivations(newActivations);
  };

  const forwardPassOutput = () => {
    // Calculate pre-activation values (z2)
    const newPreActivations = currentNetworkState.current.outputWeights.map((w, i) => 
      w.reduce((sum, weight, j) => sum + weight * currentNetworkState.current.hiddenActivations[j], currentNetworkState.current.outputBiases[i])
    );
    // Apply softmax to get probability distribution
    const newOutputActivations = softmax(newPreActivations);
    
    // Store both pre-activation and activation values
    currentNetworkState.current.outputPreActivations = newPreActivations;
    currentNetworkState.current.outputActivations = newOutputActivations;
    setOutputActivations(newOutputActivations);
    
    // Debug info will be captured in backpropagationOutput after errors are calculated
  };

  const calculateLoss = () => {
    // Use cached target to eliminate race conditions
    const target = getCurrentTarget(currentNetworkState, trainingMode, trainingExamples, currentExampleIndex, selectedLabelRef);
    
    console.log(`🎯 Cross-Entropy Loss - Target: [${target}], Outputs: [${currentNetworkState.current.outputActivations.map(o => o.toFixed(3))}]`);
    
    // Cross-Entropy Loss with epsilon clamping
    const eps = 1e-7;
    const probs = currentNetworkState.current.outputActivations;
    const calculatedLoss = -target.reduce((sum: number, t: number, i: number) => {
      const p = Math.max(eps, Math.min(1 - eps, probs[i]));
      return sum + t * Math.log(p);
    }, 0);
    
    currentNetworkState.current.loss = calculatedLoss;
    setLoss(calculatedLoss);
  };

  const backpropagationOutput = () => {
    // Use cached target to eliminate race conditions
    const target = getCurrentTarget(currentNetworkState, trainingMode, trainingExamples, currentExampleIndex, selectedLabelRef);
    
    // Cross-Entropy + Softmax gradient: δᵢ = pᵢ - tᵢ (clean and simple)
    const outputErrors = currentNetworkState.current.outputActivations.map((output, i) => 
      output - target[i]);
    
    // Apply gradient clipping for stability
    const outputErrorsClipped = outputErrors.map(clip);
    
    console.log(`🔄 Backprop Output (Cross-Entropy) - Target: [${target}], Errors: [${outputErrors.map(e => e.toFixed(4))}], Clipped: [${outputErrorsClipped.map(e => e.toFixed(4))}]`);
    
    // Store clipped output errors for hidden layer backprop
    currentNetworkState.current.outputErrors = outputErrorsClipped;
    
    // Update output weights and biases using clipped errors
    const newOutputWeights = currentNetworkState.current.outputWeights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * outputErrorsClipped[i] * currentNetworkState.current.hiddenActivations[j]));
    const newOutputBiases = currentNetworkState.current.outputBiases.map((bias, i) => 
      bias - learningRate * outputErrorsClipped[i]);
    
    // Update persistent store
    currentNetworkState.current.outputWeights = newOutputWeights;
    currentNetworkState.current.outputBiases = newOutputBiases;
    
    // Update React state for display
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
    
    // Capture debug info right after outputErrors are computed
    captureDebugInfo('backpropagationOutput');
  };

  const backpropagationHidden = () => {
    // Use clipped output errors from output layer backpropagation
    const outputErrorsClipped = currentNetworkState.current.outputErrors;
    const pixelValues = getPixelValues();
    
    // Calculate hidden errors using PRE-ACTIVATION values and clipped output errors
    // δₕ = (Σᵢ δᵢ · wᵢₕ) · σ'(zₕ) where zₕ is PRE-activation
    const hiddenErrors = currentNetworkState.current.hiddenPreActivations.map((preActivation, h) => {
      const errorSum = outputErrorsClipped.reduce((sum, outputError, i) => 
        sum + outputError * currentNetworkState.current.outputWeights[i][h], 0);
      return errorSum * sigmoidDerivative(preActivation);
    });
    
    // Apply gradient clipping to hidden errors as well
    const hiddenErrorsClipped = hiddenErrors.map(clip);
    
    // Update hidden weights and biases using clipped errors
    const newWeights = currentNetworkState.current.weights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * hiddenErrorsClipped[i] * pixelValues[j]));
    const newBiases = currentNetworkState.current.biases.map((bias, i) => 
      bias - learningRate * hiddenErrorsClipped[i]);
    
    // Update persistent store
    currentNetworkState.current.weights = newWeights;
    currentNetworkState.current.biases = newBiases;
    
    // Update React state for display
    setWeights(newWeights);
    setBiases(newBiases);
  };



  // Function to capture debug information
  const captureDebugInfo = (stage: string) => {
    let currentLabel;
    let currentOneHotTarget;
    
    // FIRST: Check if we have a cached target from async training (eliminates async state issues)
    if (currentNetworkState.current.currentTarget) {
      currentOneHotTarget = currentNetworkState.current.currentTarget;
      console.log(`🔍 Debug: Using cached target from async training - Target: [${currentOneHotTarget}]`);
    } else {
      console.log(`🔍 Debug: NOT using cached target - cachedTarget=${currentNetworkState.current.currentTarget}`);
      // FALLBACK: Read from React state (for manual training or when cache is unavailable)
      if (trainingMode === 'dataset') {
        const example = trainingExamples[currentExampleIndex];
        if (example) {
          // Use the current example (parse to ensure it's an array)
          currentOneHotTarget = parseLabel(example.label);
          console.log(`🔍 Debug: Dataset mode - currentExampleIndex: ${currentExampleIndex}, ExampleID: ${example.id}, rawLabel: ${JSON.stringify(example.label)}, parsedLabel: [${currentOneHotTarget}]`);
        } else {
          // Convert selectedLabel to one-hot for consistency
          currentOneHotTarget = selectedLabel === 0 ? [1, 0] : [0, 1];
          console.log(`🔍 Debug: Fallback to selectedLabel: ${selectedLabel} -> [${currentOneHotTarget}]`);
        }
      } else {
        // Manual drawing mode, convert selected label to one-hot
        currentOneHotTarget = selectedLabel === 0 ? [1, 0] : [0, 1];
        console.log(`🔍 Debug: Manual drawing - selectedLabel: ${selectedLabel} -> [${currentOneHotTarget}]`);
      }
    }
    
    const debugEntry = {
      iteration: trainingHistoryStore.current.length,
      label: currentOneHotTarget as number[], // Now showing one-hot format instead of integer
      outputActivations: [...currentNetworkState.current.outputActivations],
      outputErrors: [...currentNetworkState.current.outputErrors],
      outputBiases: [...currentNetworkState.current.outputBiases],
      loss: currentNetworkState.current.loss,
      step: step,
      timestamp: new Date()
    };
    
    setDebugHistory(prev => [...prev, debugEntry]);
  };

  const nextStep = (forceStep?: number) => {
    let currentLoss = loss;
    let currentHiddenActivations = hiddenActivations;
    let currentOutputActivations = outputActivations;
    
    const currentStep = forceStep !== undefined ? forceStep : step;
    console.log('nextStep executing step:', currentStep);
    
    // Mark tour step as started when first step is taken
    if (currentStep === 0 && !tourStepStarted) {
      setTourStepStarted(true);
    }
    
    switch (currentStep) {
      case 0:
        forwardPassHidden();
        break;
      case 1:
        forwardPassOutput();
        break;
      case 2:
        // Calculate loss using persistent store
        calculateLoss();
        break;
      case 3:
        backpropagationOutput();
        break;
      case 4:
        console.log('Executing case 4 - backpropagation hidden (input→hidden weights)');
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
          outputActivations: [...currentNetworkState.current.outputActivations]
        };
        
        // Store in persistent history
        trainingHistoryStore.current.push(historySnapshot);
        
        // Update React state for display
        setTrainingHistory([...trainingHistoryStore.current]);
        
        console.log('Captured training history after both backprop steps:', {
          iteration: trainingHistoryStore.current.length,
          totalSnapshots: trainingHistoryStore.current.length,
          loss: currentNetworkState.current.loss,
          hiddenWeight: currentNetworkState.current.weights[0][0],
          outputWeight: currentNetworkState.current.outputWeights[0][0]
        });
        break;
      case 5:
        // Complete cycle, start over - clear the canvas for next digit
        setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
        setStep(-1);
        setTourTrainingCycleCompleted(true); // Tour tracking - full 6-step cycle completed
        cycleDoneRef.current = true; // Set ref immediately for tour validation
        console.log('🎯 TOUR: Training cycle completed! Setting both state and ref to true');
        // Trigger tour validation check
        setTimeout(() => {
          if (tourTriggerRef.current) {
            console.log('🎯 TOUR: Triggering validation check...');
            tourTriggerRef.current();
          }
        }, 100);
        break;
    }
    
    // Track clicks and update step state
    if (forceStep === undefined) {
      // Increment click counter for tour tracking (only on user clicks)
      clicksThisCycleRef.current += 1;
      console.log('🎯 TOUR: User clicked Next Step, clicks this cycle:', clicksThisCycleRef.current);
      
      setStep((prev) => {
        const newStep = (prev + 1) % 6;
        console.log('🎯 TOUR: Step changed from', prev, 'to', newStep);
        
        // Detect cycle completion when wrapping from step 5 back to 0
        if (prev === 5 && newStep === 0) {
          cycleDoneRef.current = true;
          console.log('🎯 TOUR: Training cycle completed! cycleDone set to true');
        }
        
        return newStep;
      });
    } else {
      // Force step without incrementing click counter
      setStep(forceStep);
      console.log('🎯 TOUR: Force step to', forceStep, '(no click tracking)');
    }
    
    // Always trigger tour validation after any step change
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log('🎯 TOUR: Triggering validation after step change...');
        tourTriggerRef.current();
      }
    }, 100);
  };

  const resetNetwork = () => {
    // 81→24→2 architecture: 81 inputs (9x9 grid), 24 hidden neurons, 2 output neurons
    const newWeights = Array.from({ length: 24 }, () => Array(81).fill(0).map(() => initWeight(81, 24)));
    const newBiases = Array(24).fill(0);
    const newOutputWeights = Array.from({ length: 2 }, () => Array(24).fill(0).map(() => initWeight(24, 2)));
    const newOutputBiases = Array(2).fill(0);
    
    // Update persistent state
    currentNetworkState.current = {
      weights: newWeights,
      biases: newBiases,
      outputWeights: newOutputWeights,
      outputBiases: newOutputBiases,
      hiddenActivations: Array(24).fill(0),
      outputActivations: Array(2).fill(0),
      hiddenPreActivations: Array(24).fill(0),
      outputPreActivations: Array(2).fill(0),
      loss: 0,
      outputErrors: Array(2).fill(0),
      currentTarget: [1, 0] as number[],
      inputs: Array(81).fill(0) as number[]
    };
    
    // Clear training history
    trainingHistoryStore.current = [];
    
    // Update React state
    setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
    setWeights(newWeights);
    setBiases(newBiases);
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
    setHiddenActivations(Array(24).fill(0));
    setOutputActivations(Array(2).fill(0));
    setLoss(0);
    setStep(0);
    setTrainingHistory([]);
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
    
    // Reset training state to ensure Automated Training section remains visible
    shouldStopTraining.current = false;
    setIsAutoTraining(false);
    setMode('training');
    
    // Clear debug history
    setDebugHistory([]);
    setShowDebugDialog(false);
    setTrainingMode('dataset'); // Keep dataset mode to show automated training controls
    setCurrentExampleIndex(0);
    setCurrentEpoch(1);
  };

  // Dataset editor functions
  const addDatasetExample = () => {
    const newExample = {
      pattern: Array(9).fill(0).map(() => Array(9).fill(0)),
      label: [1, 0] // Default to digit 0 in one-hot format
    };
    createExampleMutation.mutate(newExample);
  };

  const removeDatasetExample = (index: number) => {
    const example = trainingExamples[index];
    if (example?.id) {
      deleteExampleMutation.mutate(example.id);
    }
  };

  const updateDatasetExample = (index: number, pattern: number[][] | number[], label: number) => {
    const example = trainingExamples[index];
    if (example?.id) {
      // Convert integer label to one-hot format
      const oneHotLabel = label === 0 ? [1, 0] : [0, 1];
      updateExampleMutation.mutate({ 
        id: example.id, 
        example: { pattern, label: oneHotLabel } 
      });
    }
  };

  // Editor drawing functions
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
        // 2D array format
        const grid = [...(pattern as number[][])];
        grid[rowIndex] = [...grid[rowIndex]];
        grid[rowIndex][colIndex] = grid[rowIndex][colIndex] ? 0 : 1;
        newPattern = grid;
      } else {
        // Flat array format
        const flatArray = [...(pattern as number[])];
        const flatIndex = rowIndex * 9 + colIndex;
        flatArray[flatIndex] = flatArray[flatIndex] ? 0 : 1;
        newPattern = flatArray;
      }
      
      updateExampleMutation.mutate({ 
        id: example.id, 
        example: { pattern: newPattern, label: example.label as number[] } 
      });
    } else {
      console.warn(`No valid example found at index ${exampleIndex}`, { example, trainingExamples });
    }
  };

  const saveDataset = () => {
    setShowDatasetEditor(false);
    setCurrentExampleIndex(0);
  };

  const getPatternPreview = (pattern: number[][] | number[]) => {
    // Handle both 2D array (9x9) and flat array (81 elements) formats
    if (Array.isArray(pattern[0])) {
      // 2D array format
      return (pattern as number[][]).map(row => row.reduce((sum, val) => sum + val, 0) / 9);
    } else {
      // Flat array format - convert to 2D preview
      const flatPattern = pattern as number[];
      const preview = [];
      for (let i = 0; i < 9; i++) {
        const rowSum = flatPattern.slice(i * 9, (i + 1) * 9).reduce((sum, val) => sum + val, 0);
        preview.push(rowSum / 9);
      }
      return preview;
    }
  };

  // File upload function
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const jsonData = JSON.parse(content);
        
        // Validate that it's an array of objects with input and target fields
        if (Array.isArray(jsonData) && jsonData.length > 0) {
          const firstItem = jsonData[0];
          if (firstItem.input && firstItem.target && 
              Array.isArray(firstItem.input) && firstItem.input.length === 81 &&
              Array.isArray(firstItem.target) && firstItem.target.length === 2) {
            bulkUploadMutation.mutate(jsonData);
          } else {
            alert("Invalid file format. Expected array of objects with 'input' (81 numbers) and 'target' (2 numbers) fields.");
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
    
    // Reset the input so the same file can be selected again
    event.target.value = '';
  };

  // ---------- Checkpoint functions ----------
  const handleExportCheckpoint = () => {
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
    setTourCheckpointSaved(true); // Tour tracking - React state
    checkpointSavedRef.current = true; // Tour tracking - immediate ref
    console.log('🎯 TOUR: Export checkpoint clicked! Setting both state and ref to true');
    
    // Trigger tour validation check
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log('🔔 TOUR: Triggering validation check after Export click');
        tourTriggerRef.current();
      }
    }, 100);
  };

  const handleImportCheckpointFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
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

      // Stop any running training first
      setIsAutoTraining(false);
      shouldStopTraining.current = true;

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

      // Reset UI state
      setStep(0);
      setLoss(0);
      setHiddenActivations(Array(24).fill(0));
      setOutputActivations(Array(2).fill(0));
      setLrHistory([]); // Reset learning rate history when loading checkpoint
      setLastCheckpointLoaded(file.name); // Track the loaded checkpoint
      setTourCheckpointLoaded(true); // Tour tracking
      
      alert(`Loaded checkpoint: ${file.name}`);
    } catch (err) {
      console.error("Import error:", err);
      alert("Failed to import checkpoint.");
    } finally {
      e.target.value = ""; // allow reselecting same file
    }
  };

  // Helper functions for async training
  const sleep = (ms: number) => new Promise(res => setTimeout(res, ms));
  
  const runStepsForCurrentSample = async (): Promise<boolean> => {
    for (const s of [0, 1, 2, 3, 4, 5]) {
      if (shouldStopTraining.current) return false;
      console.log(`🔄 Running step ${s} for sample ${currentExampleIndex}`);
      nextStep(s);                     // use the forced-step path; UI step is cosmetic
      await sleep(autoTrainingSpeed);
    }
    return true;
  };
  
  // Automated training functions - LEGACY (will be replaced)
  const runToNextSampleLegacy = () => {
    if (trainingExamples.length === 0) return;
    
    console.log('Starting runToNextSampleLegacy for example index:', currentExampleIndex);
    setIsAutoTraining(true);
    
    // Load current training example
    const currentExample = trainingExamples[currentExampleIndex];
    console.log('Loading example for runToNextSampleLegacy:', currentExample.label);
    const pattern = currentExample.pattern as number[][] | number[];
    // Convert flat array to 2D grid if needed
    const grid = Array.isArray(pattern[0]) ? pattern as number[][] : flatToGrid(pattern as number[]);
    setPixelGrid(grid);
    // Convert one-hot label back to integer for UI display
    const oneHotLabel = parseLabel(currentExample.label);
    console.log('Parsed label for runToNextSampleLegacy:', oneHotLabel);
    setSelectedLabel(oneHotLabel[0] === 1 ? 0 : 1);
    setStep(0); // Start at step 0
    
    // Run through all 6 steps automatically using nextStep() with forced step numbers
    let stepCount = 0;
    const interval = setInterval(() => {
      if (stepCount < 6) {
        console.log('runToNextSampleLegacy - calling nextStep(), step:', stepCount);
        nextStep(stepCount); // Force the step number to avoid React state timing issues
        stepCount++;
      } else {
        clearInterval(interval);
        console.log('runToNextSampleLegacy completed all 6 steps. Training history length:', trainingHistoryStore.current.length);
        // Update React step state to final step and then complete
        setStep(0);
        
        // Add current loss to epoch tracking
        currentEpochLoss.current.push(currentNetworkState.current.loss);
        
        // Move to next example
        setTimeout(() => {
          if (shouldStopTraining.current) return;
          
          const nextIndex = (currentExampleIndex + 1) % trainingExamples.length;
          console.log(`🔄 Moving from example ${currentExampleIndex} to ${nextIndex} (total: ${trainingExamples.length})`);
          
          if (nextIndex === 0) {
            // Completed one epoch
            const newEpoch = currentEpoch + 1;
            console.log(`✅ Completed epoch ${currentEpoch}/${numberOfEpochs}`);
            
            // Calculate average loss for completed epoch
            if (currentEpochLoss.current.length > 0) {
              const avgLoss = currentEpochLoss.current.reduce((a, b) => a + b, 0) / currentEpochLoss.current.length;
              setEpochLossHistory(prev => [...prev, { epoch: currentEpoch, averageLoss: avgLoss }]);
              currentEpochLoss.current = []; // Reset for next epoch
            }
            
            if (newEpoch <= numberOfEpochs) {
              // Continue to next epoch
              setCurrentEpoch(newEpoch);
              setCurrentExampleIndex(0);
              console.log(`🚀 Starting epoch ${newEpoch}/${numberOfEpochs}`);
              runToNextSampleLegacy();
            } else {
              // All epochs completed
              console.log(`🎉 Training completed! All ${numberOfEpochs} epochs finished.`);
              setIsAutoTraining(false);
              setTrainingCompleted(true);
            }
          } else {
            // Continue with next example in current epoch
            setCurrentExampleIndex(nextIndex);
            runToNextSampleLegacy();
          }
        }, autoTrainingSpeed / 2);
      }
    }, autoTrainingSpeed);
  };

  // New async multi-epoch training function
  const runEpochs = async () => {
    if (trainingExamples.length === 0) return;
    
    console.log(`🚀 NEW ASYNC runEpochs starting for ${numberOfEpochs} epoch(s) with ${trainingExamples.length} examples`);
    
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
          ? example.pattern as number[][] 
          : flatToGrid(example.pattern as number[]);
        const oneHot = parseLabel(example.label);
        const uiDigit = oneHot[0] === 1 ? 0 : 1;
        
        console.log(`📊 Epoch ${epoch}/${numberOfEpochs}, Example ${idx + 1}/${trainingExamples.length}, ExampleID: ${example.id}, Label: [${oneHot}]`);
        
        // Snapshot the sample before running steps (eliminates async state issues)
        setPixelGrid(pattern);
        setSelectedLabelSafe(uiDigit);
        setCurrentExampleIndex(idx);
        
        // Cache target and inputs in network state for training logic
        currentNetworkState.current.currentTarget = oneHot;
        currentNetworkState.current.inputs = pattern.flat();
        console.log(`🔄 CACHE SET in runEpochs: currentTarget = [${oneHot}], ExampleID = ${example.id}`);
        
        const completed = await runStepsForCurrentSample();
        if (!completed) break;
        
        currentEpochLoss.current.push(currentNetworkState.current.loss);
        setExamplesSeen(prev => prev + 1);
      }
      
      // Calculate average loss for completed epoch
      if (currentEpochLoss.current.length > 0) {
        const avg = currentEpochLoss.current.reduce((a,b) => a+b, 0) / currentEpochLoss.current.length;
        setEpochLossHistory(prev => [...prev, { epoch, averageLoss: avg }]);
        setLastEpochAvgLoss(avg);
        setCompletedEpochs(epoch);
        console.log(`✅ Epoch ${epoch} completed. Average loss: ${avg.toFixed(4)}`);
        
        // Apply learning rate decay
        if (lrDecayEnabled) {
          setLearningRate(prev => {
            const next = Math.max(minLR, prev * lrDecayRate);
            console.log(`[LR Decay] lr: ${prev.toFixed(6)} → ${next.toFixed(6)}`);
            // Track learning rate history
            setLrHistory(prevHistory => [...prevHistory, {epoch, learningRate: next}]);
            return next;
          });
        } else {
          // Track learning rate even without decay for consistency
          setLrHistory(prevHistory => [...prevHistory, {epoch, learningRate: learningRate}]);
        }
      }
      
      if (shouldStopTraining.current) break;
    }
    
    setIsAutoTraining(false);
    setTrainingCompleted(true);
    console.log(`🎉 Training completed! All ${numberOfEpochs} epochs finished.`);
    
    // Trigger tour validation check for training completion
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log('🔔 TOUR: Triggering validation check after training completion');
        tourTriggerRef.current();
      }
    }, 100);
  };
  
  // Process entire training set by calling runEpochs
  const processTrainingSet = () => {
    // Check if training examples are loaded
    if (trainingExamples.length === 0) {
      console.warn('⚠️ TOUR: No training examples loaded. Cannot start multi-epoch training.');
      return;
    }
    
    setIsEpochDialogOpen(true);
    setTourMultiEpochStarted(true); // Tour tracking - React state
    multiEpochStartedRef.current = true; // Tour tracking - immediate ref
    console.log('🎯 TOUR: Process Training Set clicked! Setting both state and ref to true');
    
    // Trigger tour validation check
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log('🔔 TOUR: Triggering validation check after Process Training Set click');
        tourTriggerRef.current();
      }
    }, 100);
  };

  const stopTraining = () => {
    console.log('🛑 Training stopped by user');
    shouldStopTraining.current = true;
    setIsAutoTraining(false);
    setTrainingCompleted(true);
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
    
    // Trigger tour validation check for manual training stop
    setTimeout(() => {
      if (tourTriggerRef.current) {
        console.log('🔔 TOUR: Triggering validation check after manual training stop');
        tourTriggerRef.current();
      }
    }, 100);
  };

  const startMultiEpochTraining = () => {
    if (trainingExamples.length === 0) return;
    
    console.log(`Starting multi-epoch training for ${numberOfEpochs} epoch(s) with ${trainingExamples.length} examples`);
    
    shouldStopTraining.current = false;
    setIsAutoTraining(true);
    setCurrentExampleIndex(0);
    setIsEpochDialogOpen(false);
    
    // Reset epoch loss tracking
    setEpochLossHistory([]);
    currentEpochLoss.current = [];
    setTrainingCompleted(false);
    setCurrentEpoch(1);
    
    // Start the async training process
    runEpochs();
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    };
  }, []);

  // Inference mode function
  const runInference = () => {
    if (mode !== 'inference') return;
    
    const inputs = getPixelValues();
    
    // Forward pass only (no training) - use current network state
    const hiddenSums = currentNetworkState.current.weights.map((neuronWeights, i) => 
      inputs.reduce((sum, input, j) => sum + input * neuronWeights[j], 0) + currentNetworkState.current.biases[i]
    );
    const hiddenOutputs = hiddenSums.map(sigmoid);
    
    const outputSums = currentNetworkState.current.outputWeights.map((neuronWeights, i) => 
      hiddenOutputs.reduce((sum, hidden, j) => sum + hidden * neuronWeights[j], 0) + currentNetworkState.current.outputBiases[i]
    );
    const outputs = softmax(outputSums); // Use softmax for proper probabilities
    
    // Update activations for visualization
    setHiddenActivations(hiddenOutputs);
    setOutputActivations(outputs);
    
    // Determine prediction using softmax probabilities
    const predictedDigit = outputs[0] > outputs[1] ? 0 : 1;
    const confidence = outputs[predictedDigit]; // Use the actual probability for the predicted digit
    
    setPrediction({ digit: predictedDigit, confidence });
  };

  // Auto-run inference when in inference mode and canvas changes
  useEffect(() => {
    if (mode === 'inference') {
      runInference();
    }
  }, [mode, pixelGrid]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8 relative">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">🧠 Binary Digit Trainer</h1>
          <p className="text-gray-600">Step-by-step Neural Network Learning Simulator</p>
          
          {/* Action buttons in top right */}
          <div className="absolute top-0 right-0 flex gap-2">
            <button
              onClick={() => setIsGuidedTourOpen(true)}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
            >
              Take Guided Tour
            </button>
            <button
              onClick={() => setIsAboutOpen(true)}
              className="px-3 py-1 bg-gray-600 text-white text-sm rounded-md hover:bg-gray-700 transition-colors"
            >
              About
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Drawing Canvas */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Drawing Canvas (9×9 pixels)</h2>
              <div 
                className="grid grid-cols-9 gap-0 mb-4 w-48 h-48 mx-auto border-2 border-gray-400 bg-gray-100"
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              >
                {(Array.isArray(pixelGrid[0]) ? pixelGrid : flatToGrid(pixelGrid as any)).map((row, rowIndex) => 
                  row.map((pixel, colIndex) => (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                      onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                      className={`w-full h-full border border-gray-200 cursor-crosshair select-none transition-colors duration-100 ${
                        pixel ? "bg-gray-800" : "bg-white hover:bg-gray-100"
                      }`}
                    />
                  ))
                )}
              </div>
              
              <div className="text-center">
                <p className="text-xs text-gray-600 mb-2">Click and drag to draw. Hover over pixels to see values.</p>
                <Button 
                  onClick={clearCanvas}
                  variant="outline" 
                  size="sm"
                  className="text-xs"
                >
                  Clear Canvas
                </Button>
              </div>
              
              <div className="space-y-3">
                {/* Target Label - Only show in training mode */}
                {mode === 'training' && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Target Label</h3>
                    <div className="flex gap-2 justify-center" data-tour-target="label-selector">
                      {[0, 1].map((label) => (
                        <label key={label} className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="radio"
                            name="label"
                            value={label}
                            checked={selectedLabel === label}
                            onChange={() => setSelectedLabel(label)}
                            className="text-blue-600"
                          />
                          <span>Digit {label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}

                {/* Mode Selection */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    Mode
                    <HelpIcon k="mode" />
                  </h3>
                  <div className="flex gap-2 justify-center">
                    {[
                      { value: 'training', label: 'Training' },
                      { value: 'inference', label: 'Inferencing' }
                    ].map((modeOption) => (
                      <label key={modeOption.value} className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="mode"
                          value={modeOption.value}
                          checked={mode === modeOption.value}
                          onChange={() => {
                            setMode(modeOption.value as 'training' | 'inference');
                            if (modeOption.value === 'inference') {
                              setStep(0);
                              setPrediction(null);
                              // Clear canvas for fresh drawing in inference mode
                              setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
                              setTourInferenceModeEnabled(true); // Tour tracking
                            }
                          }}
                          className="text-blue-600"
                        />
                        <span>{modeOption.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Prediction Display (Inference Mode) */}
                {mode === 'inference' && prediction && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-green-800 mb-1">Prediction</h4>
                    <div className="text-2xl font-bold text-green-700">
                      Digit: {prediction.digit}
                    </div>
                    <div className="text-xs text-green-600">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>

              {/* Network Info */}
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Network Info</h3>
                <div className="text-xs text-gray-600 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center">
                      Learning Rate:
                      <HelpIcon k="learningRate" />
                    </span>
                    <input
                      type="number"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.1)}
                      step="0.01"
                      min="0.001"
                      max="1.0"
                      className="w-16 px-1 py-0.5 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>Architecture: 81 → 24 → 2</div>
                  <div>Hidden: Sigmoid, Output: Softmax</div>
                  <div>Loss: Cross-Entropy</div>
                  <div>Dataset: {trainingExamples.length} examples</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Neural Network Diagram */}
          <Card className="col-span-2">
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Neural Network Diagram</h2>
              
              <div className="relative h-[550px] bg-gray-50 rounded-lg overflow-auto flex items-start">
                <svg 
                  className="block w-full self-start" 
                  viewBox="0 -20 750 1340" 
                  preserveAspectRatio="xMinYMin meet"
                  style={{ minHeight: 1340 }}
                >
                  {/* Input Layer */}
                  <g className="input-layer">
                    <text x="38" y="5" fontSize="20" fill="#666" fontWeight="bold">Input (81)</text>
                    {getPixelValues().map((value, i) => (
                      <g key={`input-${i}`}>
                        <circle
                          cx="75"
                          cy={25 + i * 20}
                          r="12"
                          fill={value > 0.5 ? "#3B82F6" : "#E5E7EB"}
                          stroke={activeElements.includes('input') ? "#F59E0B" : "#9CA3AF"}
                          strokeWidth={activeElements.includes('input') ? "2" : "1"}
                          className={activeElements.includes('input') ? "animate-pulse" : ""}
                        />
                        <text x="75" y={28 + i * 20} fontSize="7" fill="#000" textAnchor="middle" fontWeight="bold">
                          {value}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Hidden Layer */}
                  <g className="hidden-layer">
                    <text x="250" y="5" fontSize="20" fill="#666" fontWeight="bold">Hidden (24)</text>
                    {hiddenActivations.map((activation, i) => (
                      <g key={`hidden-${i}`}>
                        <circle
                          cx="313"
                          cy={25 + i * 22}
                          r="12"
                          fill={activation > 0.5 ? "#8B5CF6" : "#E5E7EB"}
                          stroke={activeElements.includes('hidden') ? "#F59E0B" : "#9CA3AF"}
                          strokeWidth={activeElements.includes('hidden') ? "2" : "1"}
                          className={activeElements.includes('hidden') ? "animate-pulse" : ""}
                        />
                        <text x="313" y={29 + i * 22} fontSize="8" fill="#000" textAnchor="middle" fontWeight="bold">
                          {activation.toFixed(2)}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Output Layer */}
                  <g className="output-layer">
                    <text x="525" y="5" fontSize="20" fill="#666" fontWeight="bold">Output (2)</text>
                    {outputActivations.map((activation, i) => (
                      <g key={`output-${i}`}>
                        <circle
                          cx="600"
                          cy={70 + i * 120}
                          r="25"
                          fill={activation === Math.max(...outputActivations) ? "#10B981" : "#E5E7EB"}
                          stroke={activeElements.includes('output') ? "#F59E0B" : "#9CA3AF"}
                          strokeWidth={activeElements.includes('output') ? "5" : "3.75"}
                          className={activeElements.includes('output') ? "animate-pulse" : ""}
                        />
                        <text x="600" y={77 + i * 120} fontSize="15" fill="#000" textAnchor="middle" fontWeight="bold">
                          {activation.toFixed(2)}
                        </text>
                        <text x="638" y={73 + i * 120} fontSize="15" fill="#666" fontWeight="bold">
                          {i}: {(activation * 100).toFixed(0)}%
                        </text>
                        <text x="638" y={95 + i * 120} fontSize="12" fill="#555" fontWeight="bold">
                          bias: {outputBiases[i].toFixed(3)}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Input to Hidden connections */}
                  {weights.map((hiddenWeights, hiddenIdx) =>
                    hiddenWeights.map((weight, inputIdx) => (
                      <line
                        key={`line-ih-${hiddenIdx}-${inputIdx}`}
                        x1="87"
                        y1={25 + inputIdx * 20}
                        x2="301"
                        y2={25 + hiddenIdx * 22}
                        stroke={activeElements.includes('connections') ? "#F59E0B" : "#9CA3AF"}
                        strokeWidth={activeElements.includes('connections') ? "1" : "0.3"}
                        opacity={activeElements.includes('connections') ? "0.8" : "0.2"}
                        className={activeElements.includes('connections') ? "animate-pulse" : ""}
                      />
                    ))
                  )}

                  {/* Hidden to Output connections */}
                  {outputWeights.map((outputWeightArray, outputIdx) =>
                    outputWeightArray.map((weight, hiddenIdx) => (
                      <line
                        key={`line-ho-${outputIdx}-${hiddenIdx}`}
                        x1="325"
                        y1={25 + hiddenIdx * 22}
                        x2="572"
                        y2={70 + outputIdx * 120}
                        stroke={activeElements.includes('connections') ? "#F59E0B" : "#9CA3AF"}
                        strokeWidth={activeElements.includes('connections') ? "2.5" : "1.25"}
                        opacity={activeElements.includes('connections') ? "0.8" : "0.4"}
                        className={activeElements.includes('connections') ? "animate-pulse" : ""}
                      />
                    ))
                  )}

                  {/* Weight detail buttons - rendered on top */}
                  {/* Hidden layer plus buttons */}
                  {hiddenActivations.map((activation, i) => (
                    <g key={`hidden-plus-${i}`} className="cursor-pointer" onClick={() => {
                      setSelectedWeightBox({type: 'hidden', index: i});
                      setWeightDialogIteration(trainingHistory.length === 0 ? 0 : Math.max(0, trainingHistory.length - 1));
                      setTourWeightVisualizationOpened(true); // Tour tracking
                    }}>
                      {/* Green circle */}
                      <circle
                        cx="280"
                        cy={25 + i * 22}
                        r="10"
                        fill="#10B981"
                        stroke="#059669"
                        strokeWidth="2"
                        opacity="0.9"
                      />
                      {/* Plus symbol */}
                      <line
                        x1="275"
                        y1={25 + i * 22}
                        x2="285"
                        y2={25 + i * 22}
                        stroke="white"
                        strokeWidth="2"
                        strokeLinecap="round"
                      />
                      <line
                        x1="280"
                        y1={20 + i * 22}
                        x2="280"
                        y2={30 + i * 22}
                        stroke="white"
                        strokeWidth="2"
                        strokeLinecap="round"
                      />
                    </g>
                  ))}

                  {/* Output layer plus buttons */}
                  {outputActivations.map((activation, i) => (
                    <g key={`output-plus-${i}`} className="cursor-pointer" onClick={() => {
                      setSelectedWeightBox({type: 'output', index: i});
                      setWeightDialogIteration(trainingHistory.length === 0 ? 0 : Math.max(0, trainingHistory.length - 1));
                      setTourWeightVisualizationOpened(true); // Tour tracking
                    }}>
                      {/* Green circle */}
                      <circle
                        cx="540"
                        cy={70 + i * 120}
                        r="12"
                        fill="#10B981"
                        stroke="#059669"
                        strokeWidth="2"
                        opacity="0.9"
                      />
                      {/* Plus symbol */}
                      <line
                        x1="534"
                        y1={70 + i * 120}
                        x2="546"
                        y2={70 + i * 120}
                        stroke="white"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                      />
                      <line
                        x1="540"
                        y1={64 + i * 120}
                        x2="540"
                        y2={76 + i * 120}
                        stroke="white"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                      />
                    </g>
                  ))}

                  {/* Debug Info Icon */}
                  <g className="debug-icon cursor-pointer" onClick={() => {
                    setShowDebugDialog(true);
                  }}>
                    {/* Debug icon background */}
                    <circle
                      cx="680"
                      cy="25"
                      r="15"
                      fill={showDebugDialog ? "#3B82F6" : "#6B7280"}
                      stroke={showDebugDialog ? "#1D4ED8" : "#4B5563"}
                      strokeWidth="2"
                      opacity="0.9"
                    />
                    {/* Info icon (i) */}
                    <text x="680" y="31" fontSize="16" fill="white" textAnchor="middle" fontWeight="bold">
                      i
                    </text>
                  </g>


                </svg>
              </div>

              {/* Weight Details Legend */}
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="text-sm">
                  <div className="font-medium text-gray-700 mb-2">Weight Details:</div>
                  <div className="space-y-2 text-xs text-gray-600">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                        <span className="text-white text-xs font-bold">+</span>
                      </div>
                      <span>Click green plus button to view detailed weights for each neuron</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-gray-500 rounded-full flex items-center justify-center">
                        <span className="text-white text-xs font-bold">i</span>
                      </div>
                      <span>Click info button (top right) to view network debug information</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Network Summary */}
              {mode === 'training' && (
                <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                  <div className="text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Current Loss:</span>
                      <span className="font-mono">{loss.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              )}
              
            </CardContent>
          </Card>

          {/* Controls - Made 50% wider */}
          <Card className="lg:col-span-1 lg:min-w-[400px]">
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Training Steps</h2>
              
              {/* Training Mode Toggle */}
              <div className="mb-2">
                <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                  Training Mode
                  <HelpIcon k="trainingMode" />
                </h3>
              </div>
              <div className="mb-4 flex gap-2">
                <Button 
                  onClick={() => { setTrainingMode('manual'); setTrainingCompleted(false); }}
                  variant={trainingMode === 'manual' ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                >
                  Manual Draw
                </Button>
                <Button 
                  onClick={() => { 
                    setTrainingMode('dataset'); 
                    setTrainingCompleted(false); 
                    setTourDatasetLoaded(true); // Tour tracking - React state
                    datasetLoadedRef.current = true; // Tour tracking - immediate ref
                    console.log('🎯 TOUR: Training Set clicked! Setting both state and ref to true');
                    // Trigger tour validation check
                    setTimeout(() => {
                      if (tourTriggerRef.current) {
                        console.log('🔔 TOUR: Triggering validation check after Training Set click');
                        tourTriggerRef.current();
                      }
                    }, 100);
                  }}
                  variant={trainingMode === 'dataset' ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                  data-tour-target="dataset-button"
                >
                  Training Set
                </Button>
              </div>

              {/* Current Step Info - Show detailed info only when not auto-training and not completed */}
              {!isAutoTraining && !trainingCompleted ? (
                <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                  <div className="text-sm font-medium text-blue-900 mb-2 flex items-center">
                    Step {step + 1} of 6: {STEP_DESCRIPTIONS[step] ? STEP_DESCRIPTIONS[step].name : 'Ready'}
                    {step < 6 && <HelpIcon k={`step${step + 1}` as keyof typeof MINI_TUTORIALS} />}
                  </div>
                  
                  {/* Concept Explanation */}
                  <div className="text-sm text-blue-800 mb-3">
                    <strong>Concept:</strong> {STEP_DESCRIPTIONS[step] ? STEP_DESCRIPTIONS[step].concept : 'Ready to begin training'}
                  </div>
                  
                  {/* Mathematical Formula */}
                  <div className="text-xs text-blue-700 font-mono bg-blue-100 p-2 rounded">
                    <strong>Formula:</strong> {STEP_DESCRIPTIONS[step] ? STEP_DESCRIPTIONS[step].formula : 'Click Next Step to begin'}
                  </div>
                </div>
              ) : (isAutoTraining || trainingCompleted) ? (
                <div className="mb-4 p-4 bg-purple-50 rounded-lg">
                  <div className="text-sm font-medium text-purple-900 mb-2">
                    {isAutoTraining ? 'Automated Training in Progress' : 'Training Complete'}
                  </div>
                  <div className="text-sm text-purple-800 mb-2">
                    {isAutoTraining 
                      ? (numberOfEpochs > 1 ? `Epoch ${currentEpoch} of ${numberOfEpochs}` : 'Processing training examples automatically')
                      : `Completed ${numberOfEpochs} epoch(s) with ${trainingExamples.length} samples`
                    }
                  </div>
                  
                  {/* Epoch Progress Bar (only show if multiple epochs) */}
                  {numberOfEpochs > 1 && (
                    <div className="mb-3">
                      <div className="w-full bg-purple-300 rounded-full h-1.5">
                        <div 
                          className="bg-purple-700 h-1.5 rounded-full transition-all duration-300" 
                          style={{ width: `${(currentEpoch / numberOfEpochs) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  {/* Sample Progress Bar */}
                  <div className="w-full bg-purple-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300" 
                      style={{ 
                        width: `${trainingExamples.length > 0 ? ((currentExampleIndex + 1) / trainingExamples.length) * 100 : 0}%` 
                      }}
                    ></div>
                  </div>
                  <div className="text-xs text-purple-700 mt-2 text-center">
                    Sample {currentExampleIndex + 1} of {trainingExamples.length}
                    {numberOfEpochs > 1 && ` • Epoch ${currentEpoch}/${numberOfEpochs}`}
                  </div>
                  
                  {/* Epoch Loss History */}
                  {epochLossHistory.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-purple-200">
                      <div className="text-xs font-medium text-purple-900 mb-2">Learning Progress</div>
                      <div className="space-y-1 max-h-20 overflow-y-auto">
                        {epochLossHistory.slice(-5).map((epochData, index) => (
                          <div key={epochData.epoch} className="flex justify-between text-xs text-purple-800">
                            <span>Epoch {epochData.epoch}:</span>
                            <span className="font-mono">{epochData.averageLoss.toFixed(6)}</span>
                          </div>
                        ))}
                      </div>
                      {epochLossHistory.length > 1 && (
                        <div className="text-xs text-purple-700 mt-2 text-center">
                          {epochLossHistory[epochLossHistory.length - 1].averageLoss < epochLossHistory[0].averageLoss ? '📉 Loss decreasing!' : '📈 Loss trend varies'}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : null}

              {/* Navigation Controls */}
              <div className="space-y-2 mb-4">
                <div className="flex gap-2">
                  <Button 
                    onClick={() => setStep(Math.max(0, step - 1))}
                    disabled={step === 0}
                    variant="outline"
                    size="sm"
                    className="flex-1"
                  >
                    ← Previous
                  </Button>
                  <Button 
                    onClick={() => {
                      nextStep();
                      setTourStepExecuted(true); // Tour tracking
                    }}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                    size="sm"
                    data-tour-target="next-step-button"
                  >
                    Next Step →
                  </Button>
                </div>
                
                <Button 
                  onClick={resetNetwork}
                  variant="outline"
                  className="w-full"
                  size="sm"
                >
                  Reset Network
                </Button>
                
                <Button 
                  onClick={() => setShowDatasetEditor(true)}
                  variant="outline"
                  className="w-full"
                  size="sm"
                >
                  <Edit3 className="w-4 h-4 mr-2" />
                  Edit Training Set
                </Button>
                
                <div className="relative">
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleFileUpload}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    id="bulk-upload-input"
                  />
                  <Button 
                    variant="outline"
                    className="w-full"
                    size="sm"
                    disabled={bulkUploadMutation.isPending}
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    {bulkUploadMutation.isPending ? 'Uploading...' : 'Upload Training Set'}
                  </Button>
                </div>
              </div>

              {/* Model Management Section */}
              <div className="mt-4">
                <Button
                  onClick={() => setShowModelManagement(!showModelManagement)}
                  variant="ghost"
                  size="sm"
                  className="w-full justify-between p-2 h-auto"
                  data-tour-target="model-management-toggle"
                >
                  <span className="text-sm font-medium text-gray-700">Model Management</span>
                  {showModelManagement ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </Button>
                
                {showModelManagement && (
                  <div className="mt-2 p-3 bg-gray-50 rounded-lg space-y-3">
                    {/* Checkpoint Export/Import */}
                    <div className="space-y-2">
                      <div className="text-xs font-medium text-gray-600 uppercase tracking-wide flex items-center">
                        Checkpoints
                        <HelpIcon k="checkpoints" />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          onClick={handleExportCheckpoint}
                          variant="outline"
                          size="sm"
                          className="text-xs"
                          data-tour-target="save-checkpoint-button"
                        >
                          <Save className="w-3 h-3 mr-1" />
                          Export
                        </Button>

                        <div className="relative">
                          <input
                            type="file"
                            accept=".json"
                            onChange={handleImportCheckpointFile}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                          />
                          <Button variant="outline" size="sm" className="w-full text-xs">
                            <FolderOpen className="w-3 h-3 mr-1" />
                            Import
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Learning Rate Decay Controls */}
                    <div className="space-y-2">
                      <div className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                        Learning Rate Decay
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="lr-decay-enabled"
                            checked={lrDecayEnabled}
                            onChange={(e) => setLrDecayEnabled(e.target.checked)}
                            className="rounded"
                          />
                          <label htmlFor="lr-decay-enabled" className="text-xs text-gray-700">
                            Enable decay (per epoch)
                          </label>
                        </div>
                        
                        {lrDecayEnabled && (
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="space-y-1">
                              <label className="text-gray-600">Decay rate</label>
                              <input
                                type="number"
                                step="0.001"
                                min="0.90"
                                max="0.999"
                                value={lrDecayRate}
                                onChange={e => setLrDecayRate(parseFloat(e.target.value) || 0.99)}
                                className="w-full text-xs border rounded px-2 py-1"
                              />
                            </div>
                            <div className="space-y-1">
                              <label className="text-gray-600">Min LR</label>
                              <input
                                type="number"
                                step="0.0001"
                                min="0.0001"
                                max="0.1"
                                value={minLR}
                                onChange={e => setMinLR(parseFloat(e.target.value) || 0.0005)}
                                className="w-full text-xs border rounded px-2 py-1"
                              />
                            </div>
                          </div>
                        )}
                        
                        {/* Learning Rate History Chart */}
                        {lrHistory.length > 1 && (
                          <div className="mt-2 p-2 bg-white rounded border">
                            <div className="text-xs text-gray-600 mb-1">LR over epochs</div>
                            <div className="h-12 flex items-end gap-0.5">
                              {lrHistory.slice(-10).map((point, i) => {
                                const maxLR = Math.max(...lrHistory.map(p => p.learningRate));
                                const height = (point.learningRate / maxLR) * 100;
                                return (
                                  <div key={i} className="flex-1 bg-blue-200 min-w-1" 
                                       style={{height: `${height}%`}}
                                       title={`Epoch ${point.epoch}: ${point.learningRate.toFixed(5)}`}>
                                  </div>
                                );
                              })}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              Last 10 epochs (hover for values)
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Model Stats */}
                    <div className="space-y-2">
                      <div className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                        Model Stats
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Architecture:</span>
                          <span className="font-mono">81→24→2</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Epochs:</span>
                          <span className="font-mono">{completedEpochs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Examples:</span>
                          <span className="font-mono">{examplesSeen}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Current LR:</span>
                          <span className="font-mono">{learningRate.toFixed(5)}</span>
                        </div>
                        <div className="flex justify-between col-span-2">
                          <span className="text-gray-600">Last Loss:</span>
                          <span className="font-mono">
                            {lastEpochAvgLoss !== null ? lastEpochAvgLoss.toFixed(4) : 'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between col-span-2">
                          <span className="text-gray-600">Normalization:</span>
                          <span className="font-mono">
                            {normalizeEnabled ? `Enabled (${targetSize}x${targetSize})` : 'Disabled'}
                          </span>
                        </div>
                        {lrDecayEnabled && (
                          <>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Decay Rate:</span>
                              <span className="font-mono">{lrDecayRate}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Min LR:</span>
                              <span className="font-mono">{minLR}</span>
                            </div>
                          </>
                        )}
                        {lastCheckpointLoaded && (
                          <div className="flex justify-between col-span-2 pt-1 border-t">
                            <span className="text-gray-600">Last Loaded:</span>
                            <span className="font-mono text-xs truncate max-w-24" title={lastCheckpointLoaded}>
                              {lastCheckpointLoaded}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Automated Training Controls - Only in Training Mode */}
              {mode === 'training' && trainingMode === 'dataset' && trainingExamples.length > 0 && (
                <div className="mt-4 space-y-2">
                  <div className="text-sm font-medium text-gray-700 mb-2">Automated Training</div>
                  <Button 
                    onClick={() => {
                      // Single sample training using new async approach
                      if (trainingExamples.length === 0) return;
                      const example = trainingExamples[currentExampleIndex];
                      const pattern = Array.isArray((example.pattern as any)[0]) 
                        ? example.pattern as number[][] 
                        : flatToGrid(example.pattern as number[]);
                      const oneHot = parseLabel(example.label);
                      const uiDigit = oneHot[0] === 1 ? 0 : 1;
                      
                      console.log(`🚀 Single sample training - Example ${currentExampleIndex}, Label: [${oneHot}]`);
                      
                      // Snapshot the sample before running steps (eliminates async state issues)
                      setPixelGrid(pattern);
                      setSelectedLabelSafe(uiDigit);
                      currentNetworkState.current.currentTarget = oneHot;
                      currentNetworkState.current.inputs = pattern.flat();
                      console.log(`📌 CACHE SET: currentTarget = [${oneHot}], example index = ${currentExampleIndex}`);
                      
                      setIsAutoTraining(true);
                      setTourNextSampleClicked(true); // Tour tracking - React state
                      nextSampleClickedRef.current = true; // Tour tracking - immediate ref
                      console.log('🎯 TOUR: Run to Next Sample clicked! Setting both state and ref to true');
                      // Trigger tour validation check
                      setTimeout(() => {
                        if (tourTriggerRef.current) {
                          console.log('🔔 TOUR: Triggering validation check after Run to Next Sample click');
                          tourTriggerRef.current();
                        }
                      }, 100);
                      runStepsForCurrentSample().then((completed) => {
                        setIsAutoTraining(false);
                        if (completed) {
                          // Track the training stats
                          setExamplesSeen(prev => prev + 1);
                          // Move to next example
                          const nextIndex = (currentExampleIndex + 1) % trainingExamples.length;
                          setCurrentExampleIndex(nextIndex);
                        }
                      });
                    }}
                    disabled={isAutoTraining}
                    size="sm"
                    className="w-full bg-green-600 hover:bg-green-700 text-white"
                    data-tour-target="run-next-sample-button"
                  >
                    {isAutoTraining ? 'Training...' : 'Run to Next Sample'}
                  </Button>
                  <Button 
                    onClick={processTrainingSet}
                    disabled={isAutoTraining}
                    size="sm"
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white"
                    data-tour-target="multi-epoch-button"
                  >
                    {isAutoTraining ? 'Processing Set...' : 'Process Training Set'}
                  </Button>
                  
                  {/* Stop Training Button - Only show when training is active */}
                  {isAutoTraining && (
                    <Button 
                      onClick={stopTraining}
                      size="sm"
                      variant="outline"
                      className="w-full border-red-500 text-red-600 hover:bg-red-50 hover:text-red-700 hover:border-red-600"
                      data-tour-target="stop-training-button"
                    >
                      🛑 Stop Training
                    </Button>
                  )}
                  <div className="text-xs text-gray-600 text-center">
                    Sample {currentExampleIndex + 1} of {trainingExamples.length} • Speed: {autoTrainingSpeed}ms
                  </div>
                </div>
              )}

              {/* Dataset Info and Navigation */}
              {trainingMode === 'dataset' && (
                <div className="mt-4 space-y-3">
                  <div className="p-3 bg-green-50 rounded-lg">
                    <div className="text-sm font-medium text-green-900 mb-1">
                      Training Dataset
                    </div>
                    <div className="text-xs text-green-700">
                      Example {currentExampleIndex + 1} of {trainingExamples.length} • Target: {Array.isArray(trainingExamples[currentExampleIndex]?.label) 
                        ? ((trainingExamples[currentExampleIndex]?.label as number[])?.[0] === 1 ? '0' : '1')
                        : String(trainingExamples[currentExampleIndex]?.label || 0)}
                      <br />
                      One-hot: [{Array.isArray(trainingExamples[currentExampleIndex]?.label) 
                        ? (trainingExamples[currentExampleIndex]?.label as number[]).join(',')
                        : trainingExamples[currentExampleIndex]?.label === 0 ? '1,0' : '0,1'}] (Neuron0: digit0, Neuron1: digit1)
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button 
                      onClick={() => setCurrentExampleIndex(Math.max(0, currentExampleIndex - 1))}
                      disabled={currentExampleIndex === 0}
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      ← Prev Example
                    </Button>
                    <Button 
                      onClick={() => setCurrentExampleIndex(Math.min(trainingExamples.length - 1, currentExampleIndex + 1))}
                      disabled={currentExampleIndex === trainingExamples.length - 1}
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      Next Example →
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Detailed Weight View - Below main grid */}
        {selectedWeightBox && (
          <div className="mt-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">
                    {selectedWeightBox.type === 'hidden' 
                      ? `Hidden Neuron ${selectedWeightBox.index + 1} Weights (81 input connections)`
                      : `Output Neuron ${selectedWeightBox.index} Weights (24 hidden connections)`}
                  </h2>
                  <Button 
                    onClick={() => {
                      setSelectedWeightBox(null);
                      setTourWeightVisualizationOpened(true); // Tour tracking
                    }}
                    variant="outline"
                    size="sm"
                  >
                    ×
                  </Button>
                </div>
                
                <div className="mb-4">
                  <label className="text-sm font-medium">Training Iteration: </label>
                  {trainingHistory.length > 0 ? (
                    <>
                      <input
                        type="range"
                        min="0"
                        max={trainingHistory.length - 1}
                        value={weightDialogIteration}
                        onChange={(e) => setWeightDialogIteration(parseInt(e.target.value))}
                        className="ml-2 w-32"
                      />
                      <span className="ml-2 text-sm">{weightDialogIteration + 1} / {trainingHistory.length}</span>
                    </>
                  ) : (
                    <span className="ml-2 text-sm text-gray-500">Training iteration information is not available for models loaded by checkpoint</span>
                  )}
                </div>
                
                {/* Weight Visualization */}
                <div className="bg-gray-50 p-4 rounded-lg">
                  {selectedWeightBox.type === 'hidden' && (
                    <div className="flex gap-6">
                      {/* Left side: Activation Explorer with 9x9 heatmap */}
                      <div className="flex-shrink-0">
                        <div className="mb-4">
                          <div className="flex items-baseline justify-between mb-2">
                            <h3 className="text-sm font-semibold flex items-center">
                              Activation Explorer
                              <HelpIcon k="activationExplorer" />
                            </h3>
                            <div className="text-xs text-gray-600 font-mono">
                              {(() => {
                                const z = currentNetworkState.current.hiddenPreActivations?.[selectedWeightBox.index] ?? 0;
                                const a = currentNetworkState.current.hiddenActivations?.[selectedWeightBox.index] ?? 0;
                                return `z: ${z.toFixed(3)} \u00A0\u00A0 a: ${a.toFixed(3)}`;
                              })()}
                            </div>
                          </div>

                          {/* Input Overlay Toggle */}
                          <div className="flex items-center gap-2 mb-2">
                            <input
                              type="checkbox"
                              id="input-overlay"
                              checked={showInputOverlay}
                              onChange={(e) => setShowInputOverlay(e.target.checked)}
                              className="rounded"
                            />
                            <label htmlFor="input-overlay" className="text-xs text-gray-600 cursor-pointer">
                              Show input overlay
                            </label>
                          </div>

                          {/* Global Scale Toggle */}
                          <div className="flex items-center gap-2 mb-2">
                            <input
                              type="checkbox"
                              id="global-scale"
                              checked={useGlobalScale}
                              onChange={(e) => setUseGlobalScale(e.target.checked)}
                              className="rounded"
                            />
                            <label htmlFor="global-scale" className="text-xs text-gray-600 cursor-pointer">
                              Use global scale
                            </label>
                          </div>

                          {/* Color Scheme Selector */}
                          <div className="mb-3">
                            <label className="text-xs text-gray-600 block mb-1">Color scheme:</label>
                            <select
                              value={colorScheme}
                              onChange={(e) => setColorScheme(e.target.value as any)}
                              className="text-xs border border-gray-300 rounded px-2 py-1 bg-white"
                            >
                              <option value="blue-red">Blue/Red (default)</option>
                              <option value="blue-orange">Blue/Orange</option>
                              <option value="green-purple">Green/Purple</option>
                              <option value="high-contrast">High contrast</option>
                            </select>
                            <div className="text-xs text-gray-600 mt-2 p-2 bg-gray-50 rounded">
                              {getColorSchemeDescription(colorScheme)}
                            </div>
                          </div>

                          {/* Weight template as heatmap */}
                          <div className="mb-3">
                            {(() => {
                              const w81 = (trainingHistory[weightDialogIteration]?.weights?.[selectedWeightBox.index]) ?? weights[selectedWeightBox.index];
                              const grid = vec81ToGrid9(w81);
                              const inputGrid = showInputOverlay ? pixelGrid : null;
                              
                              // Calculate global max if needed
                              let globalMaxAbs = null;
                              if (useGlobalScale) {
                                const allWeights = (trainingHistory[weightDialogIteration]?.weights) ?? weights;
                                globalMaxAbs = allWeights.reduce((max, neuronWeights) => {
                                  const neuronMax = neuronWeights.reduce((m, w) => Math.max(m, Math.abs(w)), 0);
                                  return Math.max(max, neuronMax);
                                }, 0);
                              }
                              
                              return <Heatmap9x9 
                                grid={grid} 
                                showInputOverlay={showInputOverlay} 
                                inputGrid={inputGrid}
                                globalMaxAbs={globalMaxAbs}
                              />;
                            })()}
                          </div>

                          <p className="text-xs text-gray-600 max-w-[200px]">
                            The colors visualize this neuron's input weights (blue=positive, red=negative). Blue pixels 
                            push activation up when the corresponding input pixel is on; red pulls it down.
                            Scrub the "Training Iteration" slider above to watch this template evolve.
                          </p>
                        </div>
                      </div>

                      {/* Right side: Weight bars (fixed spacing and bias visibility) */}
                      <div className="flex-grow">
                        <h3 className="text-sm font-semibold mb-2">Weight Details</h3>
                        <div className="h-[400px] overflow-auto">
                          <svg viewBox="0 0 600 1350" className="w-full" style={{ minHeight: '1350px' }}>
                              {/* Background */}
                              <rect x="50" y="30" width="500" height="1300" fill="white" stroke="#9CA3AF" strokeWidth="2"/>
                              <line x1="300" y1="30" x2="300" y2="1330" stroke="#666" strokeWidth="2" opacity="0.5"/>
                              
                              {/* Weight bars - reduced spacing from 22px to 15px */}
                              {(trainingHistory[weightDialogIteration]?.weights[selectedWeightBox.index] || weights[selectedWeightBox.index]).map((weight: number, i: number) => {
                                const barY = 45 + i * 15;
                                const barWidth = Math.abs(weight) * 250;
                                const barX = weight >= 0 ? 300 : 300 - barWidth;
                                return (
                                  <g key={i}>
                                    <rect
                                      x={barX}
                                      y={barY}
                                      width={barWidth}
                                      height="10"
                                      fill={getBarColor(weight)}
                                      opacity="0.8"
                                    />
                                    <text x="20" y={barY + 8} fontSize="8" fill="#666">
                                      Input {i + 1}:
                                    </text>
                                    <text x={weight >= 0 ? barX + barWidth + 5 : barX - 5} y={barY + 8} 
                                          fontSize="8" fill="#333" textAnchor={weight >= 0 ? "start" : "end"}>
                                      {weight.toFixed(3)}
                                    </text>
                                  </g>
                                );
                              })}
                              
                              {/* Bias visualization - fixed positioning */}
                              {(() => {
                                const bias = (trainingHistory[weightDialogIteration]?.biases && trainingHistory[weightDialogIteration]?.biases[selectedWeightBox.index]) || biases[selectedWeightBox.index];
                                const biasY = 45 + 81 * 15 + 10; // Add extra space for bias
                                const biasWidth = Math.abs(bias) * 250;
                                const biasX = bias >= 0 ? 300 : 300 - biasWidth;
                                return (
                                  <g>
                                    <rect
                                      x={biasX}
                                      y={biasY}
                                      width={biasWidth}
                                      height="12"
                                      fill={getBarColor(bias)}
                                      opacity="0.8"
                                    />
                                    <text x="20" y={biasY + 9} fontSize="10" fill="#666" fontWeight="bold">
                                      Bias:
                                    </text>
                                    <text x={bias >= 0 ? biasX + biasWidth + 5 : biasX - 5} y={biasY + 9} 
                                          fontSize="10" fill="#333" textAnchor={bias >= 0 ? "start" : "end"} fontWeight="bold">
                                      {bias.toFixed(3)}
                                    </text>
                                  </g>
                                );
                              })()}
                              
                              {/* Labels */}
                              <text x="55" y="1325" fontSize="12" fill="#666">-1</text>
                              <text x="295" y="1325" fontSize="12" fill="#666">0</text>
                              <text x="535" y="1325" fontSize="12" fill="#666">+1</text>
                          </svg>
                        </div>
                      </div>
                    </div>
                  )}

                  {selectedWeightBox.type === 'output' && (
                    <div>
                      <div className="flex gap-6">
                        {/* Left side: Top Contributors with mini thumbnails */}
                        <div className="w-1/3 flex-shrink-0">
                          <div className="flex items-baseline justify-between mb-4">
                            <h3 className="text-sm font-semibold">
                              {viewMode === 'decision' 
                                ? 'Decision Contributors (0 vs 1)' 
                                : `Output ${selectedWeightBox.index} — Logit (z₀${selectedWeightBox.index === 0 ? '₀' : '₁'})`}
                            </h3>
                          </div>


                          
                          {(() => {
                            // Calculate global max if needed
                            let globalMaxAbs = null;
                            if (useGlobalScale) {
                              const allWeights = (trainingHistory[weightDialogIteration]?.weights) ?? weights;
                              globalMaxAbs = allWeights.reduce((max, neuronWeights) => {
                                const neuronMax = neuronWeights.reduce((m, w) => Math.max(m, Math.abs(w)), 0);
                                return Math.max(max, neuronMax);
                              }, 0);
                            }

                            if (viewMode === 'decision') {
                              // Decision mode: Calculate decision contributions
                              const hiddenActivs = (trainingHistory[weightDialogIteration]?.hiddenActivations) ?? currentNetworkState.current.hiddenActivations;
                              const outputWeightsData = [
                                (trainingHistory[weightDialogIteration]?.outputWeights?.[0]) ?? outputWeights[0],
                                (trainingHistory[weightDialogIteration]?.outputWeights?.[1]) ?? outputWeights[1]
                              ];
                              
                              const decisionContribs = getDecisionContribs(hiddenActivs, outputWeightsData);
                              
                              const helpsZero = decisionContribs
                                .filter(c => c.contrib > 0)
                                .sort((a, b) => b.contrib - a.contrib)
                                .slice(0, 6);
                              
                              const helpsOne = decisionContribs
                                .filter(c => c.contrib < 0)
                                .sort((a, b) => a.contrib - b.contrib)
                                .slice(0, 6);

                              const renderDecisionGrid = (contributors: DecisionContrib[], title: string, description: string) => (
                                <div className="mb-6">
                                  <div className="mb-3">
                                    <h4 className="text-sm font-medium text-gray-800 mb-1">{title}</h4>
                                    <p className="text-xs text-gray-600">{description}</p>
                                  </div>
                                  <div className="grid grid-cols-3 gap-2">
                                    {contributors.map(({ idx, contrib, w0, w1, h }) => {
                                      const w81 = (trainingHistory[weightDialogIteration]?.weights?.[idx]) ?? weights[idx];
                                      const grid = vec81ToGrid9(w81);
                                      return (
                                        <div 
                                          key={idx} 
                                          className="p-2 bg-white rounded border cursor-pointer hover:bg-gray-50 transition-colors w-fit"
                                          onClick={() => setSelectedWeightBox({ type: 'hidden', index: idx })}
                                          title="Click to view detailed analysis of this hidden neuron"
                                        >
                                          <div className="text-xs mb-1">
                                            <span className="font-medium">Hidden {idx + 1}</span>
                                          </div>
                                          <Heatmap9x9 
                                            grid={grid} 
                                            cell={12} 
                                            showInputOverlay={showInputOverlay} 
                                            inputGrid={showInputOverlay ? pixelGrid : null}
                                            globalMaxAbs={globalMaxAbs}
                                          />
                                          <div className="text-xs text-center mt-1">
                                            <div className="font-mono text-blue-600 font-medium">
                                              Contrib: {contrib.toFixed(3)}
                                            </div>
                                            <div className="text-gray-500 text-xs">
                                              h={h.toFixed(2)}, w₀-w₁={(w0-w1).toFixed(3)}
                                            </div>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              );

                              return (
                                <div>
                                  {/* Helps classify as 0 */}
                                  {helpsZero.length > 0 && renderDecisionGrid(
                                    helpsZero,
                                    "Helps classify as 0",
                                    "These patterns push the decision toward digit 0 (positive contributions)"
                                  )}

                                  {/* Helps classify as 1 */}
                                  {helpsOne.length > 0 && renderDecisionGrid(
                                    helpsOne,
                                    "Helps classify as 1", 
                                    "These patterns push the decision toward digit 1 (negative contributions)"
                                  )}

                                  {/* Decision Bias Contribution */}
                                  {(() => {
                                    const b0 = (trainingHistory[weightDialogIteration]?.outputBiases?.[0]) ?? outputBiases[0];
                                    const b1 = (trainingHistory[weightDialogIteration]?.outputBiases?.[1]) ?? outputBiases[1];
                                    const biasDelta = b0 - b1;
                                    return (
                                      <div className="mb-6 border-t pt-4">
                                        <div className="mb-3">
                                          <h4 className="text-sm font-medium text-gray-800 mb-1">Bias Contribution</h4>
                                          <p className="text-xs text-gray-600">How much the bias terms contribute to the decision</p>
                                        </div>
                                        <div className="p-3 bg-gray-50 rounded">
                                          <div className="text-sm">
                                            <span className="font-medium">Decision bias (b₀ - b₁): </span>
                                            <span className={`font-mono ${biasDelta >= 0 ? 'text-blue-600' : 'text-red-600'}`}>
                                              {biasDelta.toFixed(3)}
                                            </span>
                                          </div>
                                          <div className="text-xs text-gray-600 mt-1">
                                            b₀={b0.toFixed(3)}, b₁={b1.toFixed(3)}
                                          </div>
                                        </div>
                                      </div>
                                    );
                                  })()}
                                </div>
                              );
                            } else {
                              // Logit mode: Original excitatory/inhibitory view
                              const k = selectedWeightBox.index;
                              const ow = (trainingHistory[weightDialogIteration]?.outputWeights?.[k]) ?? outputWeights[k];

                              const positiveWeights = ow
                                .map((w, i) => ({ i, w }))
                                .filter(({ w }) => w > 0)
                                .sort((a, b) => b.w - a.w)
                                .slice(0, 6);

                              const negativeWeights = ow
                                .map((w, i) => ({ i, w }))
                                .filter(({ w }) => w < 0)
                                .sort((a, b) => a.w - b.w)
                                .slice(0, 6);

                              const renderContributorGrid = (contributors: Array<{i: number, w: number}>, title: string, description: string) => (
                                <div className="mb-6">
                                  <div className="mb-3">
                                    <h4 className="text-sm font-medium text-gray-800 mb-1">{title}</h4>
                                    <p className="text-xs text-gray-600">{description}</p>
                                  </div>
                                  <div className="grid grid-cols-3 gap-2">
                                    {contributors.map(({ i, w }) => {
                                      const w81 = (trainingHistory[weightDialogIteration]?.weights?.[i]) ?? weights[i];
                                      const grid = vec81ToGrid9(w81);
                                      return (
                                        <div 
                                          key={i} 
                                          className="p-2 bg-white rounded border cursor-pointer hover:bg-gray-50 transition-colors w-fit"
                                          onClick={() => setSelectedWeightBox({ type: 'hidden', index: i })}
                                          title="Click to view detailed analysis of this hidden neuron"
                                        >
                                          <div className="text-xs mb-1">
                                            <span className="font-medium">Hidden {i + 1}</span>
                                          </div>
                                          <Heatmap9x9 
                                            grid={grid} 
                                            cell={12} 
                                            showInputOverlay={showInputOverlay} 
                                            inputGrid={showInputOverlay ? pixelGrid : null}
                                            globalMaxAbs={globalMaxAbs}
                                          />
                                          <div className="text-xs text-center mt-1 text-gray-600">
                                            <span className="font-mono">Weight: {w.toFixed(3)}</span>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              );

                              return (
                                <div>
                                  {/* Excitatory Contributors */}
                                  {positiveWeights.length > 0 && renderContributorGrid(
                                    positiveWeights,
                                    "Excitatory Contributors",
                                    `Patterns consisting of strongly positive values (${getPositiveColorName()}) make the neuron more likely to fire`
                                  )}

                                  {/* Inhibitory Contributors */}
                                  {negativeWeights.length > 0 && renderContributorGrid(
                                    negativeWeights,
                                    "Inhibitory Contributors", 
                                    `Patterns consisting of strongly positive values (${getPositiveColorName()}) make the neuron less likely to fire`
                                  )}
                                </div>
                              );
                            }
                          })()}

                          {/* Controls moved below both classes */}
                          <div className="border-t pt-4 mt-4">
                            {/* View Mode Toggle */}
                            <div className="mb-2">
                              <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                                View Mode
                                <HelpIcon k="decisionVsLogit" />
                              </h4>
                            </div>
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                              <div className="flex items-center gap-4 mb-2">
                                <label className="flex items-center gap-2">
                                  <input
                                    type="radio"
                                    name="viewMode"
                                    value="logit"
                                    checked={viewMode === 'logit'}
                                    onChange={(e) => setViewMode('logit')}
                                    className="text-blue-600"
                                  />
                                  <span className="text-sm font-medium">Output score (before probability)</span>
                                </label>
                                <label className="flex items-center gap-2">
                                  <input
                                    type="radio"
                                    name="viewMode"
                                    value="decision"
                                    checked={viewMode === 'decision'}
                                    onChange={(e) => setViewMode('decision')}
                                    className="text-blue-600"
                                  />
                                  <span className="text-sm font-medium">Decision (which class wins)</span>
                                </label>
                              </div>
                              <div className="text-xs text-gray-600">
                                {viewMode === 'decision' 
                                  ? 'Shows contributions to z₀−z₁, which controls the 0-vs-1 choice'
                                  : 'Shows contributions to z_k = Σ w_{j→k}h_j + b_k before probabilities'}
                              </div>
                            </div>

                            {/* Input Overlay Toggle for Output View */}
                            <div className="flex items-center gap-2 mb-2">
                              <input
                                type="checkbox"
                                id="output-input-overlay"
                                checked={showInputOverlay}
                                onChange={(e) => setShowInputOverlay(e.target.checked)}
                                className="rounded"
                              />
                              <label htmlFor="output-input-overlay" className="text-xs text-gray-600 cursor-pointer">
                                Show input overlay
                              </label>
                            </div>

                            {/* Global Scale Toggle for Output View */}
                            <div className="flex items-center gap-2 mb-2">
                              <input
                                type="checkbox"
                                id="output-global-scale"
                                checked={useGlobalScale}
                                onChange={(e) => setUseGlobalScale(e.target.checked)}
                                className="rounded"
                              />
                              <label htmlFor="output-global-scale" className="text-xs text-gray-600 cursor-pointer">
                                Use global scale
                              </label>
                            </div>

                            {/* Color Scheme Selector for Output View */}
                            <div className="mb-3">
                              <label className="text-xs text-gray-600 block mb-1">Color scheme:</label>
                              <select
                                value={colorScheme}
                                onChange={(e) => setColorScheme(e.target.value as any)}
                                className="text-xs border border-gray-300 rounded px-2 py-1 bg-white"
                              >
                                <option value="blue-red">Blue/Red (default)</option>
                                <option value="blue-orange">Blue/Orange</option>
                                <option value="green-purple">Green/Purple</option>
                                <option value="high-contrast">High contrast</option>
                              </select>
                              <div className="text-xs text-gray-600 mt-2 p-2 bg-gray-50 rounded">
                                {getColorSchemeDescription(colorScheme)}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Right side: Weight Details */}
                        <div className="flex-grow">
                          <h3 className="text-sm font-semibold mb-4">Weight Details</h3>
                          <svg width="100%" height="500" viewBox="0 0 600 500">
                            <g>
                              {/* Large weight box */}
                              <rect x="50" y="30" width="500" height="460" fill="white" stroke="#9CA3AF" strokeWidth="2"/>
                              <line x1="300" y1="30" x2="300" y2="490" stroke="#666" strokeWidth="2" opacity="0.5"/>
                              
                              {/* Weight bars - reduced spacing from 22px to 16px */}
                              {(trainingHistory[weightDialogIteration]?.outputWeights[selectedWeightBox.index] || outputWeights[selectedWeightBox.index]).map((weight: number, i: number) => {
                                const barY = 50 + i * 16;
                                const barWidth = Math.abs(weight) * 250;
                                const barX = weight >= 0 ? 300 : 300 - barWidth;
                                return (
                                  <g key={i}>
                                    <rect
                                      x={barX}
                                      y={barY}
                                      width={barWidth}
                                      height="12"
                                      fill={getBarColor(weight)}
                                      opacity="0.8"
                                    />
                                    <text x="20" y={barY + 9} fontSize="10" fill="#666">
                                      Hidden {i + 1}:
                                    </text>
                                    <text x={weight >= 0 ? barX + barWidth + 5 : barX - 5} y={barY + 9} 
                                          fontSize="10" fill="#333" textAnchor={weight >= 0 ? "start" : "end"}>
                                      {weight.toFixed(3)}
                                    </text>
                                  </g>
                                );
                              })}
                              
                              {/* Bias visualization - fixed positioning */}
                              {(() => {
                                const bias = (trainingHistory[weightDialogIteration]?.outputBiases && trainingHistory[weightDialogIteration]?.outputBiases[selectedWeightBox.index]) || outputBiases[selectedWeightBox.index];
                                const biasY = 50 + 24 * 16 + 10; // Add extra space for bias
                                const biasWidth = Math.abs(bias) * 250;
                                const biasX = bias >= 0 ? 300 : 300 - biasWidth;
                                return (
                                  <g>
                                    <rect
                                      x={biasX}
                                      y={biasY}
                                      width={biasWidth}
                                      height="14"
                                      fill={getBarColor(bias)}
                                      opacity="0.8"
                                    />
                                    <text x="20" y={biasY + 10} fontSize="11" fill="#666" fontWeight="bold">
                                      Bias:
                                    </text>
                                    <text x={bias >= 0 ? biasX + biasWidth + 5 : biasX - 5} y={biasY + 10} 
                                          fontSize="11" fill="#333" textAnchor={bias >= 0 ? "start" : "end"} fontWeight="bold">
                                      {bias.toFixed(3)}
                                    </text>
                                  </g>
                                );
                              })()}
                              
                              {/* Labels */}
                              <text x="55" y="485" fontSize="12" fill="#666">-1</text>
                              <text x="295" y="485" fontSize="12" fill="#666">0</text>
                              <text x="535" y="485" fontSize="12" fill="#666">+1</text>
                            </g>
                          </svg>
                          
                          <p className="text-xs text-gray-600 mt-4 max-w-[500px] ml-20">
                            Bars show connection strength from each hidden unit to this output. Thumbnails show each 
                            hidden unit's input template. Together they explain how mid-level patterns combine to vote 
                            for "0" or "1". Click a thumbnail to view that hidden neuron's detailed analysis.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Debug History Dialog */}
        {showDebugDialog && (
          <div className="mt-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">
                    Debug History ({debugHistory.length} entries)
                  </h2>
                  <Button 
                    onClick={() => setShowDebugDialog(false)}
                    variant="outline"
                    size="sm"
                  >
                    ×
                  </Button>
                </div>
                
                {debugHistory.length === 0 ? (
                  <div className="text-sm text-gray-600 italic p-4 text-center">
                    No debug data captured yet. Run some training steps to see debug information.
                  </div>
                ) : (
                  <div className="border border-gray-200 rounded-lg overflow-hidden">
                    {/* Fixed Table Headers */}
                    <div className="bg-gray-50 border-b border-gray-200">
                      <div className="grid grid-cols-8 gap-2 p-3 text-xs font-medium text-gray-700">
                        <div className="text-center">Iteration #</div>
                        <div className="text-center">Time</div>
                        <div className="text-center">Loss</div>
                        <div className="text-center">Output Activations</div>
                        <div className="text-center">Output Errors</div>
                        <div className="text-center">Output Biases</div>
                        <div className="text-center">Step #</div>
                        <div className="text-center">Label</div>
                      </div>
                    </div>
                    
                    {/* Scrollable Data Rows */}
                    <div className="max-h-96 overflow-y-auto">
                      {debugHistory.map((entry, index) => (
                        <div 
                          key={index} 
                          className={`grid grid-cols-8 gap-2 p-3 text-xs border-b border-gray-100 ${
                            index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                          }`}
                        >
                          <div className="text-center font-mono">{entry.iteration}</div>
                          <div className="text-center font-mono text-xs">
                            {entry.timestamp.toLocaleTimeString()}
                          </div>
                          <div className="text-center font-mono">
                            {entry.loss.toFixed(6)}
                          </div>
                          <div className="text-center font-mono text-xs">
                            [{entry.outputActivations.map((a: number) => a.toFixed(3)).join(', ')}]
                          </div>
                          <div className="text-center font-mono text-xs">
                            [{entry.outputErrors.map((e: number) => e.toFixed(3)).join(', ')}]
                          </div>
                          <div className="text-center font-mono text-xs">
                            [{entry.outputBiases.map((b: number) => b.toFixed(3)).join(', ')}]
                          </div>
                          <div className="text-center font-mono">{entry.step}</div>
                          <div className="text-center font-mono">[{Array.isArray(entry.label) ? entry.label.join(',') : entry.label}]</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Dataset Editor Dialog */}
        <Dialog open={showDatasetEditor} onOpenChange={setShowDatasetEditor}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto" onMouseUp={handleEditorMouseUp}>
            <DialogHeader>
              <DialogTitle>Edit Training Dataset</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <p className="text-sm text-gray-600">
                  {trainingExamples.length} examples total • 
                  {trainingExamples.filter((ex: TrainingExample) => {
                    const label = ex.label as number[];
                    return Array.isArray(label) && label[0] === 1; // [1,0] = digit 0
                  }).length} zeros, {trainingExamples.filter((ex: TrainingExample) => {
                    const label = ex.label as number[];
                    return Array.isArray(label) && label[1] === 1; // [0,1] = digit 1
                  }).length} ones
                </p>
                <Button onClick={addDatasetExample} size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  Add Example
                </Button>
              </div>

              <div className="grid gap-4 max-h-96 overflow-y-auto">
                {trainingExamples.map((example: TrainingExample, index: number) => {
                  const pixelValues = getPatternPreview(example.pattern as number[][]);
                  return (
                    <div key={example.id} className="border rounded-lg p-4 bg-gray-50">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium">Example {index + 1}</span>
                          <div className="flex items-center gap-2">
                            <Label htmlFor={`label-${index}`} className="text-sm">Label:</Label>
                            <select
                              id={`label-${index}`}
                              value={Array.isArray(example.label) 
                                ? ((example.label as number[])?.[0] === 1 ? '0' : '1')
                                : String(example.label || 0)}
                              onChange={(e) => updateDatasetExample(index, example.pattern as number[][] | number[], parseInt(e.target.value))}
                              className="px-2 py-1 border rounded text-sm"
                            >
                              <option value={0}>0</option>
                              <option value={1}>1</option>
                            </select>
                          </div>
                        </div>
                        <Button 
                          onClick={() => removeDatasetExample(index)}
                          variant="outline"
                          size="sm"
                          className="text-red-600 hover:text-red-700"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                      
                      <div className="flex items-center gap-4">
                        {/* 9x9 pixel grid */}
                        <div className="grid grid-cols-9 gap-0 w-32 h-32 border-2 border-gray-400 bg-gray-100">
                          {(() => {
                            const pattern = example.pattern as number[][] | number[];
                            const grid = Array.isArray(pattern[0]) ? pattern as number[][] : flatToGrid(pattern as number[]);
                            return grid.map((row: number[], rowIndex: number) => 
                              row.map((pixel: number, colIndex: number) => (
                                <div
                                  key={`${rowIndex}-${colIndex}`}
                                  className={`w-full h-full border border-gray-200 cursor-crosshair select-none transition-colors duration-100 ${
                                    pixel ? "bg-gray-800" : "bg-white hover:bg-gray-100"
                                  }`}
                                  onMouseDown={() => handleEditorMouseDown(index, rowIndex, colIndex)}
                                  onMouseEnter={() => handleEditorMouseEnter(index, rowIndex, colIndex)}
                                />
                              ))
                            );
                          })()}
                        </div>
                        
                        <div className="text-xs text-gray-600">
                          <div>Pattern (81 pixels): [{(() => {
                            const pattern = example.pattern as number[][] | number[];
                            const flatPattern = Array.isArray(pattern[0]) ? (pattern as number[][]).flat() : pattern as number[];
                            return flatPattern.slice(0, 12).map(v => v.toString()).join(',') + (flatPattern.length > 12 ? '...' : '');
                          })()}]</div>
                          <div className="mt-1">Click pixels to toggle. Target: {Array.isArray(example.label) 
                            ? ((example.label as number[])?.[0] === 1 ? '0' : '1')
                            : String(example.label)}</div>
                          <div className="mt-1">Each pixel is 0 (white) or 1 (black)</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex gap-3 pt-4 border-t">
                <Button onClick={saveDataset} className="flex-1">
                  Save Changes
                </Button>
                <Button 
                  onClick={() => {
                    setShowDatasetEditor(false);
                  }}
                  variant="outline"
                  className="flex-1"
                >
                  Cancel
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Epoch Selection Dialog */}
        <Dialog open={isEpochDialogOpen} onOpenChange={setIsEpochDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Select Number of Epochs</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="text-sm text-gray-600">
                An epoch is one complete pass through all {trainingExamples.length} training examples.
                Multiple epochs help the neural network learn patterns better.
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="epochs" className="flex items-center">
                  Number of Epochs:
                  <HelpIcon k="epochs" />
                </Label>
                <Input
                  id="epochs"
                  type="number"
                  min="1"
                  max="100"
                  value={numberOfEpochs}
                  onChange={(e) => setNumberOfEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full"
                />
              </div>
              
              <div className="text-xs text-gray-500">
                Total training steps: {numberOfEpochs} × {trainingExamples.length} = {numberOfEpochs * trainingExamples.length}
              </div>
              
              <div className="flex gap-3 pt-4">
                <Button 
                  onClick={() => {
                    startMultiEpochTraining();
                    setTourMultiEpochStarted(true); // Tour tracking
                  }}
                  className="flex-1 bg-purple-600 hover:bg-purple-700"
                >
                  Start Training
                </Button>
                <Button 
                  onClick={() => setIsEpochDialogOpen(false)}
                  variant="outline"
                  className="flex-1"
                >
                  Cancel
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* About Dialog */}
        <Dialog open={isAboutOpen} onOpenChange={setIsAboutOpen}>
          <DialogContent className="max-w-md">
            <div className="text-center space-y-4">
              <h2 className="text-xl font-bold">About Binary Digit Trainer</h2>
              
              <div className="text-sm text-gray-600 space-y-3">
                <p>
                  An educational neural network training platform for binary digit recognition,
                  featuring comprehensive step-by-step visualization and interactive learning tools.
                </p>
                
                <div className="border-t pt-3">
                  <h3 className="font-medium text-gray-800 mb-2">Created by</h3>
                  <div className="space-y-1">
                    <div>
                      <strong>Erik Smith</strong><br />
                      <a href="mailto:erik.smith@dell.com" className="text-blue-600 hover:underline">
                        erik.smith@dell.com
                      </a>
                    </div>
                  </div>
                </div>
                
                <div className="border-t pt-3">
                  <h3 className="font-medium text-gray-800 mb-2">Co-created with</h3>
                  <div>
                    <strong>Replit Agent</strong><br />
                    <span className="text-xs">
                      AI-powered development assistant providing implementation, 
                      architecture design, and educational content creation
                    </span>
                  </div>
                </div>
              </div>
              
              <Button 
                onClick={() => setIsAboutOpen(false)}
                className="w-full mt-4"
              >
                Close
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Guided Tour Component */}
        <GuidedTour
          isOpen={isGuidedTourOpen}
          onClose={() => setIsGuidedTourOpen(false)}
          onReset={() => {
            // Reset everything for clean tour start
            setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0))); // Clear canvas for fresh start
            setStep(0);
            setMode('training');
            setTrainingMode('manual');
            setTrainingCompleted(false);
            setIsAutoTraining(false);
            setCurrentEpoch(1);
            setNumberOfEpochs(3);
            // Clear activations
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
            // Reset cycle counters for fresh tour start
            resetCycleCounters();
          }}
          onValidationTrigger={(triggerFn) => {
            tourTriggerRef.current = triggerFn;
          }}
          tourSteps={createTourSteps(
            validationDrewSomething, // Use the ref-based validation function
            () => tourStepExecuted,
            validOneClick,
            validFullCycle,
            checkDatasetLoaded, // Use the ref-based validation function
            checkNextSampleClicked, // Use the ref-based validation function
            checkEpochTrainingStarted, // Use the ref-based validation function
            checkTrainingCompleted, // Use the ref-based validation function
            checkModelManagementExpanded, // Use the ref-based validation function
            checkCheckpointSaved, // Use the ref-based validation function
            () => tourInferenceModeEnabled,
            () => tourWeightVisualizationOpened
          )}
        />
      </div>
    </div>
  );
}
