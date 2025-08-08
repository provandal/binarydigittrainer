import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Trash2, Plus, Edit3, Upload, Download, BarChart3, Info, Settings } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { TrainingExample, InsertTrainingExample } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";

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
  if (typeof label === 'number') {
    return label === 0 ? [1, 0] : [0, 1];
  }
  throw new Error(`Invalid label format: ${label}`);
};

// Checkpoint interface for export/import
interface Checkpoint {
  format: string;
  createdAt: string;
  architecture: { input: number; hidden: number; output: number };
  normalize: { enabled: boolean; targetSize: number };
  optimizer: { learningRate: number; lrDecayRate: number; minLR: number; decayEnabled: boolean };
  stats: { epoch: number; avgLoss: number; examplesSeen: number };
  params: {
    weights: number[][];
    biases: number[];
    outputWeights: number[][];
    outputBiases: number[];
  };
}

// Validation function for checkpoint
const validateCheckpoint = (json: any): boolean => {
  if (!json.format || json.format !== "binary-digit-trainer-checkpoint@v1") return false;
  if (!json.params) return false;
  const { weights, biases, outputWeights, outputBiases } = json.params;
  return weights?.length === 24 && 
         weights[0]?.length === 81 && 
         biases?.length === 24 &&
         outputWeights?.length === 2 && 
         outputWeights[0]?.length === 24 && 
         outputBiases?.length === 2;
};

// Helper functions for checkpoints
const nowStamp = () => new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const downloadBlobJSON = (obj: any, filename: string) => {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

export default function BinaryDigitTrainer() {
  // UI State
  const [pixelGrid, setPixelGrid] = useState<number[][]>(initialPixelGrid);
  const [selectedLabel, setSelectedLabel] = useState<number>(0);
  const [learningRate, setLearningRate] = useState(0.3);
  const [mode, setMode] = useState<'train' | 'inference'>('train');
  const [trainingMode, setTrainingMode] = useState<'manual' | 'dataset'>('manual'); 
  const [currentStep, setCurrentStep] = useState(0);
  
  // Neural Network State
  const [weights, setWeights] = useState<number[][]>(initialWeights);
  const [biases, setBiases] = useState<number[]>(initialBiases);
  const [outputWeights, setOutputWeights] = useState<number[][]>(initialOutputWeights);
  const [outputBiases, setOutputBiases] = useState<number[]>(initialOutputBiases);
  
  // Training State  
  const [hiddenActivations, setHiddenActivations] = useState<number[]>(Array(24).fill(0));
  const [outputActivations, setOutputActivations] = useState<number[]>(Array(2).fill(0));
  const [hiddenPreActivations, setHiddenPreActivations] = useState<number[]>(Array(24).fill(0));
  const [outputPreActivations, setOutputPreActivations] = useState<number[]>(Array(2).fill(0));
  const [loss, setLoss] = useState<number>(0);
  const [outputErrors, setOutputErrors] = useState<number[]>(Array(2).fill(0));
  
  // Additional Training Context
  const [currentInputs, setCurrentInputs] = useState<number[]>(Array(81).fill(0));
  const [currentTarget, setCurrentTarget] = useState<number[]>([1, 0]);
  
  // Multi-example training state
  const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
  const [completedEpochs, setCompletedEpochs] = useState(0);
  const [examplesSeen, setExamplesSeen] = useState(0);
  const [lastEpochAvgLoss, setLastEpochAvgLoss] = useState<number | null>(null);
  
  // Async training state
  const [isAutoTraining, setIsAutoTraining] = useState(false);
  const [trainingCompleted, setTrainingCompleted] = useState(false);
  const [autoTrainingSpeed, setAutoTrainingSpeed] = useState(300);
  const [numberOfEpochs, setNumberOfEpochs] = useState(1);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const shouldStopTraining = useRef(false);
  const currentEpochLoss = useRef<number[]>([]);
  
  // Training states UI
  const [isEpochDialogOpen, setIsEpochDialogOpen] = useState(false);
  
  // Advanced features state
  const [lrDecayEnabled, setLrDecayEnabled] = useState(false);
  const [lrDecayRate, setLrDecayRate] = useState(0.95);
  const [minLR, setMinLR] = useState(0.01);
  const [isLRDialogOpen, setIsLRDialogOpen] = useState(false);
  
  // Dialogs
  const [weightDialogOpen, setWeightDialogOpen] = useState(false);
  const [selectedInputNeuron, setSelectedInputNeuron] = useState<number | null>(null);
  const [selectedHiddenNeuron, setSelectedHiddenNeuron] = useState<number | null>(null);
  const [debugInfoOpen, setDebugInfoOpen] = useState(false);
  
  // Use a ref to maintain training state consistency during async operations
  const currentNetworkState = useRef({
    weights: initialWeights.map(r => [...r]),
    biases: [...initialBiases],
    outputWeights: initialOutputWeights.map(r => [...r]),
    outputBiases: [...initialOutputBiases],
    hiddenActivations: Array(24).fill(0),
    outputActivations: Array(2).fill(0),
    hiddenPreActivations: Array(24).fill(0),
    outputPreActivations: Array(2).fill(0),
    loss: 0,
    outputErrors: Array(2).fill(0),
    currentTarget: [1, 0] as number[],
    inputs: Array(81).fill(0) as number[]
  });

  // Advanced features: checkpoint export/import and model evaluation
  const handleExportCheckpoint = () => {
    const cp: Checkpoint = {
      format: "binary-digit-trainer-checkpoint@v1",
      createdAt: new Date().toISOString(),
      architecture: { input: 81, hidden: 24, output: 2 },
      normalize: { enabled: false, targetSize: 7 },
      optimizer: { learningRate, lrDecayRate, minLR, decayEnabled: lrDecayEnabled },
      stats: { epoch: completedEpochs, avgLoss: lastEpochAvgLoss ?? NaN, examplesSeen },
      params: {
        weights: currentNetworkState.current.weights.map(r => [...r]),
        biases: [...currentNetworkState.current.biases],
        outputWeights: currentNetworkState.current.outputWeights.map(r => [...r]),
        outputBiases: [...currentNetworkState.current.outputBiases]
      }
    };
    downloadBlobJSON(cp, `checkpoint-${nowStamp()}.json`);
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
      const { params, optimizer, stats } = json as Checkpoint;

      // Update refs (training source of truth)
      currentNetworkState.current.weights = params.weights.map((r: number[]) => [...r]);
      currentNetworkState.current.biases = [...params.biases];
      currentNetworkState.current.outputWeights = params.outputWeights.map((r: number[]) => [...r]);
      currentNetworkState.current.outputBiases = [...params.outputBiases];

      // Reset activations
      currentNetworkState.current.hiddenActivations = Array(24).fill(0);
      currentNetworkState.current.outputActivations = Array(2).fill(0);
      currentNetworkState.current.hiddenPreActivations = Array(24).fill(0);
      currentNetworkState.current.outputPreActivations = Array(2).fill(0);
      currentNetworkState.current.loss = 0;
      currentNetworkState.current.outputErrors = Array(2).fill(0);

      // Update UI state
      setWeights(params.weights.map((r: number[]) => [...r]));
      setBiases([...params.biases]);
      setOutputWeights(params.outputWeights.map((r: number[]) => [...r]));
      setOutputBiases([...params.outputBiases]);

      // Update optimizer settings if available
      if (optimizer?.learningRate) setLearningRate(optimizer.learningRate);
      if (optimizer?.lrDecayRate) setLrDecayRate(optimizer.lrDecayRate);
      if (optimizer?.minLR) setMinLR(optimizer.minLR);
      if (typeof optimizer?.decayEnabled === "boolean") setLrDecayEnabled(optimizer.decayEnabled);

      // Update stats if available
      if (typeof stats?.avgLoss === "number") setLastEpochAvgLoss(stats.avgLoss);
      if (typeof stats?.epoch === "number") setCompletedEpochs(stats.epoch);
      if (typeof stats?.examplesSeen === "number") setExamplesSeen(stats.examplesSeen);

      alert(`Loaded checkpoint: ${file.name}`);
    } catch (err) {
      console.error("Import error:", err);
      alert("Failed to import checkpoint.");
    } finally {
      e.target.value = "";
    }
  };

  const evaluateOnDataset = async () => {
    if (!trainingExamples.length) { 
      alert("No dataset loaded"); 
      return; 
    }
    
    let correct = 0, total = 0, lossSum = 0;
    
    for (const ex of trainingExamples) {
      // Handle pattern type correctly
      let grid: number[][];
      if (Array.isArray((ex.pattern as any)[0])) {
        grid = ex.pattern as number[][];
      } else {
        grid = flatToGrid(ex.pattern as number[]);
      }
      const x = grid.flat();

      // Forward pass
      const z1 = currentNetworkState.current.weights.map((w,i) => 
        w.reduce((s,wi,j)=>s+wi*x[j], currentNetworkState.current.biases[i])
      );
      const h = z1.map(sigmoid);
      const z2 = currentNetworkState.current.outputWeights.map((w,k) => 
        w.reduce((s,wj,j)=>s+wj*h[j], currentNetworkState.current.outputBiases[k])
      );
      
      // Softmax
      const p = softmax(z2);

      const target = ex.label === 0 ? [1,0] : [0,1];
      const pred = p[0] > p[1] ? 0 : 1;
      if (pred === (target[0]===1?0:1)) correct++;
      total++;
      
      // Cross-entropy loss
      const ce = - (target[0]*Math.log(Math.max(1e-12,p[0])) + target[1]*Math.log(Math.max(1e-12,p[1])));
      lossSum += ce;
    }
    
    alert(`Evaluation Results:\nAccuracy: ${(100*correct/total).toFixed(1)}%\nAverage Loss: ${(lossSum/total).toFixed(4)}\nDataset Size: ${total} examples`);
  };

  // Data fetching with React Query
  const queryClient = useQueryClient();
  
  const { data: trainingExamples = [], isLoading: examplesLoading } = useQuery({
    queryKey: ['/api/training-examples'],
    enabled: trainingMode === 'dataset'
  });

  // Create training example mutation
  const createExampleMutation = useMutation({
    mutationFn: (newExample: InsertTrainingExample) => 
      apiRequest('/api/training-examples', {
        method: 'POST',
        body: JSON.stringify(newExample)
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training-examples'] });
    }
  });

  // Update training example mutation
  const updateExampleMutation = useMutation({
    mutationFn: ({ id, ...updates }: { id: number } & Partial<InsertTrainingExample>) =>
      apiRequest(`/api/training-examples/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(updates)
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training-examples'] });
    }
  });

  // Delete training example mutation
  const deleteExampleMutation = useMutation({
    mutationFn: (id: number) => 
      apiRequest(`/api/training-examples/${id}`, {
        method: 'DELETE'
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training-examples'] });
    }
  });

  // Bulk upload mutation for JSON files
  const bulkUploadMutation = useMutation({
    mutationFn: (examples: Array<{ input: number[], target: number[] }>) => {
      const transformedExamples: InsertTrainingExample[] = examples.map(ex => ({
        pattern: ex.input,
        label: JSON.stringify(ex.target)
      }));
      
      return Promise.all(
        transformedExamples.map(example => 
          apiRequest('/api/training-examples', {
            method: 'POST',
            body: JSON.stringify(example)
          })
        )
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/training-examples'] });
    }
  });

  // Initialize network weights
  const initializeWeights = () => {
    const newWeights = Array.from({ length: 24 }, () => 
      Array(81).fill(0).map(() => initWeight(81, 24))
    );
    const newBiases = Array(24).fill(0);
    const newOutputWeights = Array.from({ length: 2 }, () => 
      Array(24).fill(0).map(() => initWeight(24, 2))
    );
    const newOutputBiases = Array(2).fill(0);
    
    setWeights(newWeights);
    setBiases(newBiases);
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
    
    // Also update refs
    currentNetworkState.current.weights = newWeights.map(r => [...r]);
    currentNetworkState.current.biases = [...newBiases];
    currentNetworkState.current.outputWeights = newOutputWeights.map(r => [...r]);
    currentNetworkState.current.outputBiases = [...newOutputBiases];
    
    // Reset activations
    setHiddenActivations(Array(24).fill(0));
    setOutputActivations(Array(2).fill(0));
    setHiddenPreActivations(Array(24).fill(0));
    setOutputPreActivations(Array(2).fill(0));
    setLoss(0);
    setOutputErrors(Array(2).fill(0));
    setCurrentStep(0);
    
    // Reset training stats
    setCompletedEpochs(0);
    setExamplesSeen(0);
    setLastEpochAvgLoss(null);
  };

  // Safety wrapper for setSelectedLabel to prevent out-of-bounds errors
  const setSelectedLabelSafe = (label: number) => {
    if (label >= 0 && label <= 1) {
      setSelectedLabel(label);
    }
  };

  // Helper to create training examples
  const createTrainingExample = () => {
    if (trainingMode === 'manual') {
      const pattern = pixelGrid;
      const oneHotLabel = selectedLabel === 0 ? [1, 0] : [0, 1];
      
      const newExample: InsertTrainingExample = {
        pattern: gridToFlat(pattern),
        label: JSON.stringify(oneHotLabel)
      };
      
      createExampleMutation.mutate(newExample);
    }
  };

  // Helper to get pattern display for grid visualization
  const getPatternDisplay = (pattern: number[][] | number[]) => {
    const flatPattern = Array.isArray(pattern[0]) ? gridToFlat(pattern as number[][]) : pattern as number[];
    
    // Check if it's a simple pattern (mostly zeros or ones)
    const nonZeroCount = flatPattern.filter(x => x > 0.1).length;
    const mostlyZero = nonZeroCount < 10;
    const mostlyOne = nonZeroCount > 70;
    
    if (mostlyZero) {
      return '◦'; // Light circle for sparse patterns
    } else if (mostlyOne) {
      return '●'; // Filled circle for dense patterns  
    } else {
      // For intermediate patterns, create a small visual preview
      const preview: number[] = [];
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

  // Multi-epoch training with LR decay
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
      
      // Apply learning rate decay
      if (lrDecayEnabled && epoch > 1) {
        const newLR = Math.max(minLR, learningRate * Math.pow(lrDecayRate, epoch - 1));
        setLearningRate(newLR);
        console.log(`📉 LR Decay: Epoch ${epoch}, LR: ${newLR.toFixed(4)}`);
      }
      
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
        currentNetworkState.current.currentTarget = [...oneHot];
        currentNetworkState.current.inputs = pattern.flat();
        
        const completed = await runStepsForCurrentSample();
        if (!completed) break;
        
        // Track loss for this epoch
        currentEpochLoss.current.push(currentNetworkState.current.loss);
        setExamplesSeen(prev => prev + 1);
        
        await sleep(50); // Brief pause between examples
      }
      
      // Calculate and store epoch statistics
      if (currentEpochLoss.current.length > 0) {
        const avgLoss = currentEpochLoss.current.reduce((a, b) => a + b, 0) / currentEpochLoss.current.length;
        setLastEpochAvgLoss(avgLoss);
        setCompletedEpochs(epoch);
        console.log(`✅ Epoch ${epoch} completed. Avg Loss: ${avgLoss.toFixed(4)}`);
      }
      
      if (shouldStopTraining.current) break;
    }
    
    setIsAutoTraining(false);
    setTrainingCompleted(true);
    console.log(`🎯 Training completed! Total epochs: ${numberOfEpochs}`);
  };

  // Stop training function
  const stopTraining = () => {
    shouldStopTraining.current = true;
    setIsAutoTraining(false);
    console.log("🛑 Training stopped by user");
  };

  // Start epoch training
  const startEpochTraining = () => {
    if (trainingExamples.length === 0) {
      alert("No training examples available. Please create some examples first.");
      return;
    }
    
    shouldStopTraining.current = false;
    runEpochs();
  };

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

  // Handle pixel click
  const handlePixelClick = (row: number, col: number) => {
    if (mode === 'inference') return; // Don't allow editing in inference mode
    
    const newGrid = [...pixelGrid];
    newGrid[row][col] = newGrid[row][col] === 0 ? 1 : 0;
    setPixelGrid(newGrid);
  };

  // Neural network forward pass
  const forwardPass = (inputGrid: number[][]) => {
    const inputs = inputGrid.flat();
    
    // Input to hidden layer
    const hiddenPre = weights.map((neuronWeights, i) =>
      neuronWeights.reduce((sum, weight, j) => sum + weight * inputs[j], biases[i])
    );
    const hiddenAct = hiddenPre.map(sigmoid);
    
    // Hidden to output layer
    const outputPre = outputWeights.map((neuronWeights, i) =>
      neuronWeights.reduce((sum, weight, j) => sum + weight * hiddenAct[j], outputBiases[i])
    );
    
    // Apply softmax to output layer for binary classification
    const outputAct = softmax(outputPre);
    
    return {
      inputs,
      hiddenPreActivations: hiddenPre,
      hiddenActivations: hiddenAct,
      outputPreActivations: outputPre,
      outputActivations: outputAct
    };
  };

  // Training step implementation
  const nextStep = (forceStep?: number) => {
    const stepToRun = forceStep !== undefined ? forceStep : currentStep;
    
    // Get current target (from dataset or manual selection)
    const target = trainingMode === 'dataset' && trainingExamples[currentExampleIndex] 
      ? parseLabel(trainingExamples[currentExampleIndex].label)
      : (selectedLabel === 0 ? [1, 0] : [0, 1]);

    switch (stepToRun) {
      case 0: // Forward pass: Input to Hidden
        const inputs = pixelGrid.flat();
        const hiddenPre = currentNetworkState.current.weights.map((neuronWeights, i) =>
          neuronWeights.reduce((sum, weight, j) => sum + weight * inputs[j], currentNetworkState.current.biases[i])
        );
        
        currentNetworkState.current.inputs = inputs;
        currentNetworkState.current.hiddenPreActivations = hiddenPre;
        currentNetworkState.current.currentTarget = target;
        
        setCurrentInputs(inputs);
        setHiddenPreActivations(hiddenPre);
        setCurrentTarget(target);
        break;

      case 1: // Forward pass: Hidden Activation
        const hiddenAct = currentNetworkState.current.hiddenPreActivations.map(sigmoid);
        currentNetworkState.current.hiddenActivations = hiddenAct;
        setHiddenActivations(hiddenAct);
        break;

      case 2: // Forward pass: Hidden to Output
        const outputPre = currentNetworkState.current.outputWeights.map((neuronWeights, i) =>
          neuronWeights.reduce((sum, weight, j) => sum + weight * currentNetworkState.current.hiddenActivations[j], currentNetworkState.current.outputBiases[i])
        );
        currentNetworkState.current.outputPreActivations = outputPre;
        setOutputPreActivations(outputPre);
        break;

      case 3: // Forward pass: Output Activation (Softmax)
        const outputAct = softmax(currentNetworkState.current.outputPreActivations);
        currentNetworkState.current.outputActivations = outputAct;
        setOutputActivations(outputAct);
        
        // Calculate loss (cross-entropy)
        const lossValue = -currentNetworkState.current.currentTarget.reduce((sum, target, i) => 
          sum + target * Math.log(Math.max(1e-12, outputAct[i])), 0
        );
        currentNetworkState.current.loss = lossValue;
        setLoss(lossValue);
        break;

      case 4: // Backpropagation: Output layer
        const outputErr = currentNetworkState.current.outputActivations.map((output, i) => 
          output - currentNetworkState.current.currentTarget[i]
        );
        currentNetworkState.current.outputErrors = outputErr;
        setOutputErrors(outputErr);
        break;

      case 5: // Backpropagation: Update weights
        // Update output layer weights and biases
        const newOutputWeights = currentNetworkState.current.outputWeights.map((neuronWeights, i) =>
          neuronWeights.map((weight, j) => {
            const gradient = currentNetworkState.current.outputErrors[i] * currentNetworkState.current.hiddenActivations[j];
            return weight - learningRate * clip(gradient);
          })
        );
        
        const newOutputBiases = currentNetworkState.current.outputBiases.map((bias, i) => {
          const gradient = currentNetworkState.current.outputErrors[i];
          return bias - learningRate * clip(gradient);
        });
        
        // Calculate hidden layer errors
        const hiddenErrors = currentNetworkState.current.hiddenActivations.map((_, j) => {
          const error = currentNetworkState.current.outputErrors.reduce((sum, outErr, i) => 
            sum + outErr * currentNetworkState.current.outputWeights[i][j], 0
          );
          return error * sigmoidDerivative(currentNetworkState.current.hiddenPreActivations[j]);
        });
        
        // Update hidden layer weights and biases
        const newWeights = currentNetworkState.current.weights.map((neuronWeights, i) =>
          neuronWeights.map((weight, j) => {
            const gradient = hiddenErrors[i] * currentNetworkState.current.inputs[j];
            return weight - learningRate * clip(gradient);
          })
        );
        
        const newBiases = currentNetworkState.current.biases.map((bias, i) => {
          const gradient = hiddenErrors[i];
          return bias - learningRate * clip(gradient);
        });
        
        // Update ref state
        currentNetworkState.current.weights = newWeights;
        currentNetworkState.current.biases = newBiases;
        currentNetworkState.current.outputWeights = newOutputWeights;
        currentNetworkState.current.outputBiases = newOutputBiases;
        
        // Update UI state
        setWeights(newWeights);
        setBiases(newBiases);
        setOutputWeights(newOutputWeights);
        setOutputBiases(newOutputBiases);
        break;
    }
    
    if (forceStep === undefined) {
      setCurrentStep((currentStep + 1) % 6);
    }
  };

  // Run inference
  useEffect(() => {
    if (mode === 'inference') {
      const result = forwardPass(pixelGrid);
      setCurrentInputs(result.inputs);
      setHiddenPreActivations(result.hiddenPreActivations);
      setHiddenActivations(result.hiddenActivations);
      setOutputPreActivations(result.outputPreActivations);
      setOutputActivations(result.outputActivations);
      
      // Calculate loss for display
      const target = selectedLabel === 0 ? [1, 0] : [0, 1];
      const lossValue = -target.reduce((sum, t, i) => 
        sum + t * Math.log(Math.max(1e-12, result.outputActivations[i])), 0
      );
      setLoss(lossValue);
    }
  }, [mode, pixelGrid]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Binary Digit Trainer</h1>
          <p className="text-lg text-gray-600">
            Neural Network Architecture: 81 → 24 → 2 (Input → Hidden → Output)
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
              Epochs: {completedEpochs}
            </span>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
              Examples: {examplesSeen}
            </span>
            {lastEpochAvgLoss !== null && (
              <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                Avg Loss: {lastEpochAvgLoss.toFixed(4)}
              </span>
            )}
            <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm">
              LR: {learningRate.toFixed(4)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
          {/* Drawing Canvas */}
          <div className="lg:col-span-3">
            <Card>
              <CardContent className="p-4">
                <div className="mb-4">
                  <h3 className="font-semibold mb-2">Draw Binary Digit</h3>
                  <div className="grid grid-cols-9 gap-1 w-64 h-64 mx-auto">
                    {pixelGrid.map((row, rowIndex) =>
                      row.map((pixel, colIndex) => (
                        <button
                          key={`${rowIndex}-${colIndex}`}
                          className={`w-6 h-6 border border-gray-300 ${
                            pixel === 1 ? 'bg-black' : 'bg-white'
                          } ${mode === 'inference' ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                          onClick={() => handlePixelClick(rowIndex, colIndex)}
                          disabled={mode === 'inference'}
                        />
                      ))
                    )}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Expected Output:</Label>
                  <div className="flex gap-2">
                    <Button
                      variant={selectedLabel === 0 ? "default" : "outline"}
                      onClick={() => setSelectedLabel(0)}
                      disabled={mode === 'inference'}
                      className="flex-1"
                    >
                      0
                    </Button>
                    <Button
                      variant={selectedLabel === 1 ? "default" : "outline"}
                      onClick={() => setSelectedLabel(1)}
                      disabled={mode === 'inference'}
                      className="flex-1"
                    >
                      1
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Neural Network Visualization */}
          <div className="lg:col-span-6">
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold">Neural Network</h3>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setDebugInfoOpen(true)}
                    >
                      <Info className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="flex justify-between items-center h-96 overflow-y-auto">
                  {/* Input Layer */}
                  <div className="flex flex-col items-center">
                    <h4 className="text-sm font-medium mb-2">Input (81)</h4>
                    <div className="grid grid-cols-9 gap-1">
                      {currentInputs.slice(0, 81).map((activation, i) => (
                        <div
                          key={i}
                          className={`w-3 h-3 border cursor-pointer ${
                            activation > 0.5 ? 'bg-blue-500' : 'bg-gray-200'
                          }`}
                          onClick={() => setSelectedInputNeuron(selectedInputNeuron === i ? null : i)}
                          title={`Input ${i}: ${activation.toFixed(3)}`}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Hidden Layer */}
                  <div className="flex flex-col items-center">
                    <h4 className="text-sm font-medium mb-2">Hidden (24)</h4>
                    <div className="grid grid-cols-6 gap-1">
                      {hiddenActivations.map((activation, i) => (
                        <div
                          key={i}
                          className={`w-4 h-4 rounded-full border-2 cursor-pointer ${
                            activation > 0.5 ? 'bg-green-500 border-green-600' : 'bg-gray-200 border-gray-300'
                          }`}
                          onClick={() => setSelectedHiddenNeuron(selectedHiddenNeuron === i ? null : i)}
                          title={`Hidden ${i}: ${activation.toFixed(3)}`}
                          style={{
                            opacity: Math.max(0.3, activation)
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Output Layer */}
                  <div className="flex flex-col items-center">
                    <h4 className="text-sm font-medium mb-2">Output (2)</h4>
                    <div className="space-y-2">
                      {outputActivations.map((activation, i) => (
                        <div
                          key={i}
                          className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs font-bold ${
                            activation > 0.5 ? 'bg-red-500 border-red-600 text-white' : 'bg-gray-200 border-gray-300'
                          }`}
                          title={`Output ${i}: ${activation.toFixed(3)}`}
                          style={{
                            opacity: Math.max(0.3, activation)
                          }}
                        >
                          {i}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Weight Inspector */}
                {(selectedInputNeuron !== null || selectedHiddenNeuron !== null) && (
                  <div className="mt-4">
                    <Button
                      onClick={() => setWeightDialogOpen(true)}
                      variant="outline"
                      size="sm"
                    >
                      Inspect Weights
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Control Panel */}
          <div className="lg:col-span-3">
            <Card>
              <CardContent className="p-4 space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Mode</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant={mode === 'train' ? "default" : "outline"}
                      onClick={() => setMode('train')}
                      size="sm"
                    >
                      Train
                    </Button>
                    <Button
                      variant={mode === 'inference' ? "default" : "outline"}
                      onClick={() => setMode('inference')}
                      size="sm"
                    >
                      Inference
                    </Button>
                  </div>
                </div>

                {mode === 'train' && (
                  <>
                    <div>
                      <h3 className="font-semibold mb-2">Training Mode</h3>
                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          variant={trainingMode === 'manual' ? "default" : "outline"}
                          onClick={() => setTrainingMode('manual')}
                          size="sm"
                        >
                          Manual
                        </Button>
                        <Button
                          variant={trainingMode === 'dataset' ? "default" : "outline"}
                          onClick={() => setTrainingMode('dataset')}
                          size="sm"
                        >
                          Dataset
                        </Button>
                      </div>
                    </div>

                    <div>
                      <Label htmlFor="learning-rate">Learning Rate: {learningRate.toFixed(4)}</Label>
                      <Input
                        id="learning-rate"
                        type="range"
                        min="0.001"
                        max="1"
                        step="0.001"
                        value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        className="mt-1"
                      />
                    </div>

                    {/* Advanced Features */}
                    <div className="space-y-2">
                      <Button
                        onClick={() => setIsLRDialogOpen(true)}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        <Settings className="w-4 h-4 mr-2" />
                        LR Decay Settings
                      </Button>
                      
                      <Button
                        onClick={handleExportCheckpoint}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Export Checkpoint
                      </Button>
                      
                      <label htmlFor="import-checkpoint" className="w-full">
                        <Button variant="outline" size="sm" className="w-full" asChild>
                          <span>
                            <Upload className="w-4 h-4 mr-2" />
                            Import Checkpoint
                          </span>
                        </Button>
                        <input
                          id="import-checkpoint"
                          type="file"
                          accept=".json"
                          onChange={handleImportCheckpointFile}
                          className="hidden"
                        />
                      </label>
                      
                      <Button
                        onClick={evaluateOnDataset}
                        variant="outline"
                        size="sm"
                        className="w-full"
                        disabled={!trainingExamples.length}
                      >
                        <BarChart3 className="w-4 h-4 mr-2" />
                        Evaluate Model
                      </Button>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Training Steps</h3>
                      <div className="text-sm text-gray-600 mb-2">
                        Current Step: {currentStep + 1}/6
                      </div>
                      <div className="space-y-2">
                        <Button
                          onClick={() => nextStep()}
                          className="w-full"
                          disabled={isAutoTraining}
                        >
                          Next Step
                        </Button>
                        <Button
                          onClick={initializeWeights}
                          variant="outline"
                          className="w-full"
                          disabled={isAutoTraining}
                        >
                          Reset Network
                        </Button>
                      </div>
                    </div>

                    {trainingMode === 'dataset' && (
                      <div>
                        <h3 className="font-semibold mb-2">Batch Training</h3>
                        <div className="space-y-2">
                          <Button
                            onClick={() => setIsEpochDialogOpen(true)}
                            className="w-full"
                            disabled={isAutoTraining || trainingExamples.length === 0}
                          >
                            Run Epochs
                          </Button>
                          {isAutoTraining && (
                            <Button
                              onClick={stopTraining}
                              variant="destructive"
                              className="w-full"
                            >
                              Stop Training
                            </Button>
                          )}
                        </div>
                        
                        {isAutoTraining && (
                          <div className="text-sm text-center mt-2">
                            <div>Epoch: {currentEpoch}/{numberOfEpochs}</div>
                            <div>Sample: {currentExampleIndex + 1}/{trainingExamples.length}</div>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}

                {mode === 'inference' && (
                  <div>
                    <h3 className="font-semibold mb-2">Prediction</h3>
                    <div className="space-y-2">
                      <div className="text-sm">
                        <div>Digit 0: {(outputActivations[0] * 100).toFixed(1)}%</div>
                        <div>Digit 1: {(outputActivations[1] * 100).toFixed(1)}%</div>
                      </div>
                      <div className="text-lg font-bold">
                        Predicted: {outputActivations[0] > outputActivations[1] ? '0' : '1'}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Training Status */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <h3 className="font-semibold mb-2">Training Progress</h3>
              <div className="space-y-1 text-sm">
                <div>Loss: {loss.toFixed(4)}</div>
                <div>Step: {currentStep + 1}/6</div>
                <div>LR: {learningRate.toFixed(4)}</div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <h3 className="font-semibold mb-2">Network Stats</h3>
              <div className="space-y-1 text-sm">
                <div>Hidden Avg: {(hiddenActivations.reduce((a, b) => a + b, 0) / 24).toFixed(3)}</div>
                <div>Output Sum: {outputActivations.reduce((a, b) => a + b, 0).toFixed(3)}</div>
                <div>Weights: {weights.flat().length + outputWeights.flat().length}</div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <h3 className="font-semibold mb-2">Dataset Info</h3>
              <div className="space-y-1 text-sm">
                <div>Examples: {trainingExamples.length}</div>
                <div>Current: {currentExampleIndex + 1}</div>
                <div>Mode: {trainingMode}</div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Dataset Management */}
        {trainingMode === 'dataset' && (
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold">Dataset Management</h3>
                <div className="flex gap-2">
                  <Button
                    onClick={createTrainingExample}
                    disabled={createExampleMutation.isPending}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Current Pattern
                  </Button>
                  <label htmlFor="file-upload">
                    <Button variant="outline" asChild>
                      <span>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload JSON
                      </span>
                    </Button>
                    <input
                      id="file-upload"
                      type="file"
                      accept=".json"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>

              {examplesLoading ? (
                <div>Loading examples...</div>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {trainingExamples.map((example, index) => (
                    <div
                      key={example.id}
                      className={`flex items-center justify-between p-2 border rounded ${
                        index === currentExampleIndex ? 'bg-blue-50 border-blue-200' : ''
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-mono w-8">#{example.id}</span>
                        <div className="text-lg">
                          {typeof getPatternDisplay(example.pattern) === 'string' 
                            ? getPatternDisplay(example.pattern)
                            : '▦'
                          }
                        </div>
                        <span className="text-sm">
                          Label: {Array.isArray(example.label) ? 
                            `[${(example.label as number[]).join(',')}]` : 
                            example.label
                          }
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setCurrentExampleIndex(index)}
                        >
                          Select
                        </Button>
                      </div>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            const newPattern = gridToFlat(pixelGrid);
                            const newLabel = JSON.stringify(selectedLabel === 0 ? [1, 0] : [0, 1]);
                            updateExampleMutation.mutate({
                              id: example.id,
                              pattern: newPattern,
                              label: newLabel
                            });
                          }}
                          disabled={updateExampleMutation.isPending}
                        >
                          <Edit3 className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteExampleMutation.mutate(example.id)}
                          disabled={deleteExampleMutation.isPending}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Learning Rate Decay Dialog */}
        <Dialog open={isLRDialogOpen} onOpenChange={setIsLRDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Learning Rate Decay Settings</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={lrDecayEnabled}
                  onChange={(e) => setLrDecayEnabled(e.target.checked)}
                  id="lr-decay-enabled"
                />
                <Label htmlFor="lr-decay-enabled">Enable Learning Rate Decay</Label>
              </div>
              
              {lrDecayEnabled && (
                <>
                  <div>
                    <Label htmlFor="lr-decay-rate">Decay Rate: {lrDecayRate}</Label>
                    <Input
                      id="lr-decay-rate"
                      type="range"
                      min="0.8"
                      max="0.99"
                      step="0.01"
                      value={lrDecayRate}
                      onChange={(e) => setLrDecayRate(parseFloat(e.target.value))}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="min-lr">Minimum LR: {minLR}</Label>
                    <Input
                      id="min-lr"
                      type="range"
                      min="0.001"
                      max="0.1"
                      step="0.001"
                      value={minLR}
                      onChange={(e) => setMinLR(parseFloat(e.target.value))}
                    />
                  </div>
                  
                  <div className="text-sm text-gray-600">
                    New LR = Current LR × {lrDecayRate}^(epoch-1), minimum: {minLR}
                  </div>
                </>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* Epoch Training Dialog */}
        <Dialog open={isEpochDialogOpen} onOpenChange={setIsEpochDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Multi-Epoch Training</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label htmlFor="num-epochs">Number of Epochs</Label>
                <Input
                  id="num-epochs"
                  type="number"
                  min="1"
                  max="100"
                  value={numberOfEpochs}
                  onChange={(e) => setNumberOfEpochs(parseInt(e.target.value) || 1)}
                />
              </div>
              <div>
                <Label htmlFor="training-speed">Training Speed (ms between steps)</Label>
                <Input
                  id="training-speed"
                  type="number"
                  min="50"
                  max="2000"
                  step="50"
                  value={autoTrainingSpeed}
                  onChange={(e) => setAutoTrainingSpeed(parseInt(e.target.value) || 300)}
                />
              </div>
              <div className="text-sm text-gray-600">
                This will train on all {trainingExamples.length} examples for {numberOfEpochs} epoch(s).
              </div>
              <Button onClick={startEpochTraining} className="w-full">
                Start Training
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Weight Dialog */}
        <Dialog open={weightDialogOpen} onOpenChange={setWeightDialogOpen}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle>Weight Inspector</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              {selectedInputNeuron !== null && (
                <div>
                  <h4 className="font-semibold mb-2">Input Neuron {selectedInputNeuron} → Hidden Layer</h4>
                  
                  {/* 9x9 Grid Visualization */}
                  <div className="mb-4">
                    <h5 className="text-sm font-medium mb-2">Weight Grid (9×9)</h5>
                    <div className="grid grid-cols-9 gap-1 w-64 h-64 mx-auto">
                      {Array.from({ length: 81 }, (_, i) => {
                        const weightSum = weights.reduce((sum, neuronWeights) => 
                          sum + neuronWeights[selectedInputNeuron], 0
                        );
                        const avgWeight = weightSum / 24;
                        const intensity = Math.abs(avgWeight);
                        const isPositive = avgWeight >= 0;
                        
                        return (
                          <div
                            key={i}
                            className={`w-6 h-6 border border-gray-300 ${
                              i === selectedInputNeuron ? 'border-red-500 border-2' : ''
                            }`}
                            style={{
                              backgroundColor: isPositive 
                                ? `rgba(34, 197, 94, ${Math.min(1, intensity * 2)})` 
                                : `rgba(239, 68, 68, ${Math.min(1, intensity * 2)})`
                            }}
                            title={`Position ${i}: ${avgWeight.toFixed(4)}`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Individual Weight Bars */}
                  <div className="grid grid-cols-12 gap-1">
                    {weights.map((neuronWeights, hiddenIndex) => {
                      const weight = neuronWeights[selectedInputNeuron];
                      const maxWeight = Math.max(...weights.flat().map(Math.abs));
                      const intensity = Math.abs(weight) / maxWeight;
                      const isPositive = weight >= 0;
                      
                      return (
                        <div key={hiddenIndex} className="text-center">
                          <div className="text-xs mb-1">H{hiddenIndex}</div>
                          <div
                            className={`h-16 w-4 mx-auto border ${
                              isPositive ? 'bg-green-500' : 'bg-red-500'
                            }`}
                            style={{ opacity: intensity }}
                            title={`Weight: ${weight.toFixed(4)}`}
                          />
                          <div className="text-xs mt-1">{weight.toFixed(2)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {selectedHiddenNeuron !== null && (
                <div>
                  <h4 className="font-semibold mb-2">Hidden Neuron {selectedHiddenNeuron} → Output Layer</h4>
                  <div className="grid grid-cols-2 gap-4">
                    {outputWeights.map((neuronWeights, outputIndex) => {
                      const weight = neuronWeights[selectedHiddenNeuron];
                      const maxWeight = Math.max(...outputWeights.flat().map(Math.abs));
                      const intensity = Math.abs(weight) / maxWeight;
                      const isPositive = weight >= 0;
                      
                      return (
                        <div key={outputIndex} className="text-center">
                          <div className="text-sm mb-1">Output {outputIndex}</div>
                          <div
                            className={`h-24 w-8 mx-auto border ${
                              isPositive ? 'bg-green-500' : 'bg-red-500'
                            }`}
                            style={{ opacity: intensity }}
                            title={`Weight: ${weight.toFixed(4)}`}
                          />
                          <div className="text-sm mt-1">{weight.toFixed(4)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* Debug Info Dialog */}
        <Dialog open={debugInfoOpen} onOpenChange={setDebugInfoOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Debug Information</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              <div>
                <h4 className="font-semibold">Current Inputs</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  [{currentInputs.slice(0, 10).map(x => x.toFixed(3)).join(', ')}...]
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold">Hidden Pre-activations</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  [{hiddenPreActivations.slice(0, 8).map(x => x.toFixed(3)).join(', ')}...]
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold">Hidden Activations</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  [{hiddenActivations.slice(0, 8).map(x => x.toFixed(3)).join(', ')}...]
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold">Output Pre-activations</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  [{outputPreActivations.map(x => x.toFixed(3)).join(', ')}]
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold">Output Activations (Softmax)</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  [{outputActivations.map(x => x.toFixed(3)).join(', ')}]
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold">Target & Loss</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  Target: [{currentTarget.map(x => x.toFixed(3)).join(', ')}]<br/>
                  Loss: {loss.toFixed(6)}
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold">Output Errors</h4>
                <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                  [{outputErrors.map(x => x.toFixed(3)).join(', ')}]
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}