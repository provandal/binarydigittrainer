import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Trash2, Plus, Edit3, Upload } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { TrainingExample, InsertTrainingExample } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";


// 9x9 pixel grid (81 pixels total, each pixel is 0 or 1)
const initialPixelGrid = Array(9).fill(0).map(() => Array(9).fill(0)); // 9x9 grid of pixels
const initialWeights = Array.from({ length: 24 }, () => Array(81).fill(0).map(() => (Math.random() - 0.5) * 0.4));
const initialBiases = Array(24).fill(0).map(() => (Math.random() - 0.5) * 0.2);
const initialOutputWeights = Array.from({ length: 2 }, () => Array(24).fill(0).map(() => (Math.random() - 0.5) * 0.4));
const initialOutputBiases = Array(2).fill(0).map(() => (Math.random() - 0.5) * 0.2);

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
    concept: "Hidden layer activations are combined using output weights to produce final predictions for digits 0 and 1.",
    formula: "oₖ = σ(∑ⱼ wⱼₖ·hⱼ + bₖ) for output neurons k ∈ {0,1}",
    activeElements: ["hidden", "output", "outputWeights"]
  },
  {
    name: "Calculate Loss",
    concept: "The network's prediction is compared to the target label using Mean Squared Error to measure accuracy.",
    formula: "Loss = ½∑ₖ(tₖ - oₖ)² where tₖ is target and oₖ is output",
    activeElements: ["output", "loss"]
  },
  {
    name: "Backpropagation - Output Layer",
    concept: "Error signals flow backward to adjust output weights. Larger errors cause bigger weight changes.",
    formula: "δₖ = (tₖ - oₖ)·oₖ·(1-oₖ), Δwⱼₖ = α·δₖ·hⱼ",
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
  
  // Safe setter that ensures pixelGrid is always a 2D array
  const setPixelGrid = (grid: number[][] | number[]) => {
    if (Array.isArray(grid[0])) {
      setPixelGridState(grid as number[][]);
    } else {
      setPixelGridState(flatToGrid(grid as number[]));
    }
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
  const [learningRate] = useState(0.5);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hoveredPixel, setHoveredPixel] = useState<number | null>(null);
  const [selectedWeightBox, setSelectedWeightBox] = useState<{type: 'hidden' | 'output', index: number} | null>(null);
  const [weightDialogIteration, setWeightDialogIteration] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  
  // New state for enhanced features
  const [trainingMode, setTrainingMode] = useState<'manual' | 'dataset'>('manual');
  const [datasetIndex, setDatasetIndex] = useState(0);
  const [stepHistory, setStepHistory] = useState<any[]>([]);
  const [currentStepInHistory, setCurrentStepInHistory] = useState(0);
  const [activeElements, setActiveElements] = useState<string[]>([]);
  const [showDatasetEditor, setShowDatasetEditor] = useState(false);
  const [isDrawingInEditor, setIsDrawingInEditor] = useState(false);
  
  // New state for automated training and inference mode
  const [mode, setMode] = useState<'training' | 'inference'>('training');
  const [isAutoTraining, setIsAutoTraining] = useState(false);
  const [currentTrainingIndex, setCurrentTrainingIndex] = useState(0);
  const [autoTrainingSpeed, setAutoTrainingSpeed] = useState(50); // ms between steps - much faster for automated training
  const [prediction, setPrediction] = useState<{digit: number, confidence: number} | null>(null);
  const [isEpochDialogOpen, setIsEpochDialogOpen] = useState(false);
  const [numberOfEpochs, setNumberOfEpochs] = useState(1);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  
  // Epoch loss tracking
  const [epochLossHistory, setEpochLossHistory] = useState<{epoch: number, averageLoss: number}[]>([]);
  const currentEpochLoss = useRef<number[]>([]);

  // Persistent training history store - independent of React state
  const trainingHistoryStore = useRef<any[]>([]);
  
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
    outputErrors: Array(2).fill(0)
  });

  // Load dataset example when in dataset mode
  useEffect(() => {
    if (trainingMode === 'dataset' && trainingExamples[datasetIndex]) {
      const pattern = trainingExamples[datasetIndex].pattern as number[][] | number[];
      // Convert flat array to 2D grid if needed
      const grid = Array.isArray(pattern[0]) ? pattern as number[][] : flatToGrid(pattern as number[]);
      setPixelGrid(grid);
      setSelectedLabel(trainingExamples[datasetIndex].label);
    }
  }, [trainingMode, datasetIndex, trainingExamples]);

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

  // Calculate pixel values - flatten 9x9 grid to 81 inputs (each pixel is 0 or 1)
  const getPixelValues = () => {
    return pixelGrid.flat(); // Flatten 9x9 grid to 81 inputs
  };

  const togglePixel = (rowIndex: number, colIndex: number) => {
    const newPixelGrid = [...pixelGrid];
    newPixelGrid[rowIndex] = [...newPixelGrid[rowIndex]];
    newPixelGrid[rowIndex][colIndex] = newPixelGrid[rowIndex][colIndex] === 0 ? 1 : 0;
    setPixelGrid(newPixelGrid);
    setStep(0); // Reset to first step when input changes
  };

  const handleMouseDown = (rowIndex: number, colIndex: number) => {
    setIsDrawing(true);
    togglePixel(rowIndex, colIndex);
  };

  const handleMouseEnter = (rowIndex: number, colIndex: number) => {
    if (isDrawing) {
      togglePixel(rowIndex, colIndex);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handlePixelHover = (pixelIndex: number) => {
    setHoveredPixel(pixelIndex);
  };

  const handlePixelLeave = () => {
    setHoveredPixel(null);
  };

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
    // Apply sigmoid to get activations
    const newOutputActivations = newPreActivations.map(z => sigmoid(z));
    
    // Store both pre-activation and activation values
    currentNetworkState.current.outputPreActivations = newPreActivations;
    currentNetworkState.current.outputActivations = newOutputActivations;
    setOutputActivations(newOutputActivations);
  };

  const calculateLoss = () => {
    let target;
    if (trainingMode === 'dataset' && trainingExamples[datasetIndex]) {
      // Use one-hot targets from training data: [digit_0, digit_1]
      const example = trainingExamples[datasetIndex];
      target = example.label === 0 ? [1, 0] : [0, 1];
      console.log(`🎯 Dataset Loss - Label: ${example.label}, Target: [${target}], Outputs: [${currentNetworkState.current.outputActivations.map(o => o.toFixed(3))}]`);
    } else {
      // Manual mode: convert selectedLabel to one-hot
      target = [selectedLabel === 0 ? 1 : 0, selectedLabel === 1 ? 1 : 0]; // [digit_0, digit_1]
      console.log(`🎯 Manual Loss - Label: ${selectedLabel}, Target: [${target}], Outputs: [${currentNetworkState.current.outputActivations.map(o => o.toFixed(3))}]`);
    }
    const mse = currentNetworkState.current.outputActivations.reduce((sum, output, i) => 
      sum + Math.pow(output - target[i], 2), 0) / 2;
    currentNetworkState.current.loss = mse;
    setLoss(mse);
  };

  const backpropagationOutput = () => {
    let target;
    if (trainingMode === 'dataset' && trainingExamples[datasetIndex]) {
      // Use one-hot targets from training data: [digit_0, digit_1]
      const example = trainingExamples[datasetIndex];
      target = example.label === 0 ? [1, 0] : [0, 1];
    } else {
      // Manual mode: convert selectedLabel to one-hot
      target = [selectedLabel === 0 ? 1 : 0, selectedLabel === 1 ? 1 : 0]; // [digit_0, digit_1]
    }
    
    // Calculate individual output deltas using PRE-ACTIVATION values (matching Python)
    // δᵢ = (aᵢ - yᵢ) · σ'(zᵢ) where zᵢ is PRE-activation
    const outputErrors = currentNetworkState.current.outputActivations.map((output, i) => 
      (output - target[i]) * sigmoidDerivative(currentNetworkState.current.outputPreActivations[i]));
    
    console.log(`🔄 Backprop Output - Target: [${target}], Errors: [${outputErrors.map(e => e.toFixed(4))}], PreActivations: [${currentNetworkState.current.outputPreActivations.map(z => z.toFixed(3))}]`);
    
    // Update output weights and biases for each output neuron
    const newOutputWeights = currentNetworkState.current.outputWeights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * outputErrors[i] * currentNetworkState.current.hiddenActivations[j]));
    const newOutputBiases = currentNetworkState.current.outputBiases.map((bias, i) => 
      bias - learningRate * outputErrors[i]);
    
    // Store output errors for hidden layer backpropagation
    currentNetworkState.current.outputErrors = outputErrors;
    
    // Update persistent store
    currentNetworkState.current.outputWeights = newOutputWeights;
    currentNetworkState.current.outputBiases = newOutputBiases;
    
    // Update React state for display
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
  };

  const backpropagationHidden = () => {
    // Use stored output errors from output layer backpropagation
    const outputErrors = currentNetworkState.current.outputErrors;
    const pixelValues = getPixelValues();
    
    // Calculate hidden errors using PRE-ACTIVATION values (matching Python)
    // δₕ = (Σᵢ δᵢ · wᵢₕ) · σ'(zₕ) where zₕ is PRE-activation
    const hiddenErrors = currentNetworkState.current.hiddenPreActivations.map((preActivation, h) => {
      const errorSum = outputErrors.reduce((sum, outputError, i) => 
        sum + outputError * currentNetworkState.current.outputWeights[i][h], 0);
      return errorSum * sigmoidDerivative(preActivation);
    });
    
    // Update hidden weights and biases
    const newWeights = currentNetworkState.current.weights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * hiddenErrors[i] * pixelValues[j]));
    const newBiases = currentNetworkState.current.biases.map((bias, i) => 
      bias - learningRate * hiddenErrors[i]);
    
    // Update persistent store
    currentNetworkState.current.weights = newWeights;
    currentNetworkState.current.biases = newBiases;
    
    // Update React state for display
    setWeights(newWeights);
    setBiases(newBiases);
  };



  const nextStep = (forceStep?: number) => {
    let currentLoss = loss;
    let currentHiddenActivations = hiddenActivations;
    let currentOutputActivations = outputActivations;
    
    const currentStep = forceStep !== undefined ? forceStep : step;
    console.log('nextStep executing step:', currentStep);
    
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
        break;
    }
    
    // Only update React step state if not using forceStep
    if (forceStep === undefined) {
      setStep((prev) => (prev + 1) % 6);
    }
  };

  const resetNetwork = () => {
    // 81→24→2 architecture: 81 inputs (9x9 grid), 24 hidden neurons, 2 output neurons
    const newWeights = Array.from({ length: 24 }, () => Array(81).fill(0).map(() => (Math.random() - 0.5) * 0.4));
    const newBiases = Array(24).fill(0).map(() => (Math.random() - 0.5) * 0.2);
    const newOutputWeights = Array.from({ length: 2 }, () => Array(24).fill(0).map(() => (Math.random() - 0.5) * 0.4));
    const newOutputBiases = Array(2).fill(0).map(() => (Math.random() - 0.5) * 0.2);
    
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
      outputErrors: Array(2).fill(0)
    };
    
    // Clear training history
    trainingHistoryStore.current = [];
    
    // Update React state
    setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
    setWeights(newWeights);
    setBiases(newBiases);
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
    setHiddenActivations(Array(4).fill(0));
    setOutputActivations(Array(2).fill(0));
    setLoss(0);
    setStep(0);
    setTrainingHistory([]);
    setSelectedWeightBox(null);
    setWeightDialogIteration(0);
  };

  // Dataset editor functions
  const addDatasetExample = () => {
    const newExample = {
      pattern: Array(9).fill(0).map(() => Array(9).fill(0)),
      label: 0
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
      updateExampleMutation.mutate({ 
        id: example.id, 
        example: { pattern, label } 
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
        example: { pattern: newPattern, label: example.label } 
      });
    } else {
      console.warn(`No valid example found at index ${exampleIndex}`, { example, trainingExamples });
    }
  };

  const saveDataset = () => {
    setShowDatasetEditor(false);
    setDatasetIndex(0);
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

  // Automated training functions
  const runToNextSample = () => {
    if (trainingExamples.length === 0) return;
    
    console.log('Starting runToNextSample for training index:', currentTrainingIndex);
    setIsAutoTraining(true);
    
    // Load current training example
    const currentExample = trainingExamples[currentTrainingIndex];
    console.log('Loading example for runToNextSample:', currentExample.label);
    const pattern = currentExample.pattern as number[][] | number[];
    // Convert flat array to 2D grid if needed
    const grid = Array.isArray(pattern[0]) ? pattern as number[][] : flatToGrid(pattern as number[]);
    setPixelGrid(grid);
    setSelectedLabel(currentExample.label);
    setStep(0); // Start at step 0
    
    // Run through all 6 steps automatically using nextStep() with forced step numbers
    let stepCount = 0;
    const interval = setInterval(() => {
      if (stepCount < 6) {
        console.log('runToNextSample - calling nextStep(), step:', stepCount);
        nextStep(stepCount); // Force the step number to avoid React state timing issues
        stepCount++;
      } else {
        clearInterval(interval);
        console.log('runToNextSample completed all 6 steps. Training history length:', trainingHistoryStore.current.length);
        // Update React step state to final step and then complete
        setStep(0);
        // After completing all steps, move to next example
        setTimeout(() => {
          const nextIndex = (currentTrainingIndex + 1) % trainingExamples.length;
          setCurrentTrainingIndex(nextIndex);
          setIsAutoTraining(false);
        }, autoTrainingSpeed / 2);
      }
    }, autoTrainingSpeed);
  };

  // Process entire training set by calling runToNextSample() multiple times
  const processTrainingSet = () => {
    setIsEpochDialogOpen(true);
  };

  const startMultiEpochTraining = () => {
    if (trainingExamples.length === 0) return;
    
    console.log(`Starting processTrainingSet for ${numberOfEpochs} epoch(s) with ${trainingExamples.length} examples`);
    
    setIsAutoTraining(true);
    setCurrentTrainingIndex(0);
    setIsEpochDialogOpen(false);
    
    // Reset epoch loss tracking
    setEpochLossHistory([]);
    currentEpochLoss.current = [];
    
    let epochCount = 0;
    let currentExampleIndex = 0;
    setCurrentEpoch(0);
    
    const processNextExample = () => {
      if (currentExampleIndex >= trainingExamples.length) {
        // Finished one epoch - calculate and store average loss
        if (currentEpochLoss.current.length > 0) {
          const averageLoss = currentEpochLoss.current.reduce((sum, loss) => sum + loss, 0) / currentEpochLoss.current.length;
          console.log(`Epoch ${epochCount + 1} completed. Average loss: ${averageLoss.toFixed(6)}`);
          
          // Store epoch loss history
          setEpochLossHistory(prev => [...prev, { epoch: epochCount + 1, averageLoss }]);
          
          // Reset for next epoch
          currentEpochLoss.current = [];
        }
        
        epochCount++;
        currentExampleIndex = 0;
        setCurrentEpoch(epochCount);
        
        if (epochCount >= numberOfEpochs) {
          console.log(`Finished processing ${numberOfEpochs} epoch(s). Training history length:`, trainingHistoryStore.current.length);
          setIsAutoTraining(false);
          setStep(0);
          return;
        }
        
        console.log(`Starting epoch ${epochCount + 1} of ${numberOfEpochs}`);
      }
      
      console.log(`Epoch ${epochCount + 1}/${numberOfEpochs} - Processing example ${currentExampleIndex + 1} of ${trainingExamples.length}`);
      
      // Set the current training index and run to next sample
      setCurrentTrainingIndex(currentExampleIndex);
      
      // Use the same logic as runToNextSample but continue to next example when done
      const currentExample = trainingExamples[currentExampleIndex];
      setPixelGrid(currentExample.pattern as number[][]);
      setSelectedLabel(currentExample.label);
      setStep(0);
      
      // Run through all 6 steps using nextStep() with forced step numbers
      let stepCount = 0;
      const interval = setInterval(() => {
        if (stepCount < 6) {
          nextStep(stepCount); // Force the step number
          stepCount++;
          
          // After step 2 (loss calculation), capture the loss for epoch tracking
          if (stepCount === 3) { // After step 2 completes (0-indexed)
            const currentLoss = currentNetworkState.current.loss;
            currentEpochLoss.current.push(currentLoss);
            console.log(`Sample ${currentExampleIndex + 1}/${trainingExamples.length} - Loss: ${currentLoss.toFixed(6)}`);
          }
        } else {
          clearInterval(interval);
          // Move to next example immediately (no delay between examples)
          currentExampleIndex++;
          processNextExample();
        }
      }, autoTrainingSpeed);
    };
    
    // Start processing
    processNextExample();
  };

  // Inference mode function
  const runInference = () => {
    if (mode !== 'inference') return;
    
    const inputs = getPixelValues();
    
    // Forward pass only (no training)
    const hiddenSums = weights.map((neuronWeights, i) => 
      inputs.reduce((sum, input, j) => sum + input * neuronWeights[j], 0) + biases[i]
    );
    const hiddenOutputs = hiddenSums.map(sigmoid);
    
    const outputSums = outputWeights.map((neuronWeights, i) => 
      hiddenOutputs.reduce((sum, hidden, j) => sum + hidden * neuronWeights[j], 0) + outputBiases[i]
    );
    const outputs = outputSums.map(sigmoid);
    
    // Update activations for visualization
    setHiddenActivations(hiddenOutputs);
    setOutputActivations(outputs);
    
    // Determine prediction (higher output wins)
    const predictedDigit = outputs[0] > outputs[1] ? 0 : 1;
    const confidence = Math.max(...outputs);
    

    
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
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">🧠 Binary Digit Trainer</h1>
          <p className="text-gray-600">Step-by-step Neural Network Learning Simulator</p>
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
                <p className="text-xs text-gray-600 mb-4">Click and drag to draw. Hover over pixels to see values.</p>
              </div>
              
              <div className="space-y-3">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Target Label</h3>
                  <div className="flex gap-2 justify-center">
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

                {/* Mode Selection */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Mode</h3>
                  <div className="flex gap-2 justify-center">
                    {[
                      { value: 'training', label: 'Training' },
                      { value: 'inference', label: 'Predict' }
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
                <div className="text-xs text-gray-600 space-y-1">
                  <div>Learning Rate: {learningRate}</div>
                  <div>Architecture: 81 → 24 → 2</div>
                  <div>Activation: Sigmoid</div>
                  <div>Loss: Mean Squared Error</div>
                  <div>Dataset: {trainingExamples.length} examples</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Neural Network Diagram */}
          <Card className="col-span-2">
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Neural Network Diagram</h2>
              
              <div className="relative h-[550px] bg-gray-50 rounded-lg overflow-auto">
                <svg className="w-full" viewBox="0 0 750 1320" style={{ minHeight: '1320px', paddingTop: '0', marginTop: '0' }}>
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

                  {/* Legend */}
                  <g className="legend">
                    <text x="38" y="1200" fontSize="16" fill="#666" fontWeight="bold">Weight Details:</text>
                    <circle cx="50" cy="1220" r="8" fill="#10B981" stroke="#059669" strokeWidth="1.5"/>
                    <line x1="46" y1="1220" x2="54" y2="1220" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
                    <line x1="50" y1="1216" x2="50" y2="1224" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
                    <text x="65" y="1225" fontSize="13" fill="#666">Click green plus button to view detailed weights for each neuron</text>
                  </g>
                </svg>
              </div>

              {/* Network Summary */}
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Prediction:</span>
                    <span className="font-bold">
                      Digit {outputActivations[0] > outputActivations[1] ? 0 : 1}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Loss (MSE):</span>
                    <span className="font-mono">{loss.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Controls - Made 50% wider */}
          <Card className="lg:col-span-1 lg:min-w-[400px]">
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Training Steps</h2>
              
              {/* Training Mode Toggle */}
              <div className="mb-4 flex gap-2">
                <Button 
                  onClick={() => setTrainingMode('manual')}
                  variant={trainingMode === 'manual' ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                >
                  Manual Draw
                </Button>
                <Button 
                  onClick={() => setTrainingMode('dataset')}
                  variant={trainingMode === 'dataset' ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                >
                  Training Set
                </Button>
              </div>

              {/* Current Step Info - Show detailed info only when not auto-training */}
              {!isAutoTraining ? (
                <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                  <div className="text-sm font-medium text-blue-900 mb-2">
                    Step {step + 1} of 6: {STEP_DESCRIPTIONS[step] ? STEP_DESCRIPTIONS[step].name : 'Ready'}
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
              ) : (
                <div className="mb-4 p-4 bg-purple-50 rounded-lg">
                  <div className="text-sm font-medium text-purple-900 mb-2">
                    Automated Training in Progress
                  </div>
                  <div className="text-sm text-purple-800 mb-2">
                    {numberOfEpochs > 1 ? `Epoch ${currentEpoch + 1} of ${numberOfEpochs}` : 'Processing training examples automatically'}
                  </div>
                  
                  {/* Epoch Progress Bar (only show if multiple epochs) */}
                  {numberOfEpochs > 1 && (
                    <div className="mb-3">
                      <div className="w-full bg-purple-300 rounded-full h-1.5">
                        <div 
                          className="bg-purple-700 h-1.5 rounded-full transition-all duration-300" 
                          style={{ width: `${((currentEpoch + 1) / numberOfEpochs) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  {/* Sample Progress Bar */}
                  <div className="w-full bg-purple-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300" 
                      style={{ 
                        width: `${trainingExamples.length > 0 ? ((currentTrainingIndex + 1) / trainingExamples.length) * 100 : 0}%` 
                      }}
                    ></div>
                  </div>
                  <div className="text-xs text-purple-700 mt-2 text-center">
                    Sample {currentTrainingIndex + 1} of {trainingExamples.length}
                    {numberOfEpochs > 1 && ` • Epoch ${currentEpoch + 1}/${numberOfEpochs}`}
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
              )}

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
                    onClick={() => nextStep()}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                    size="sm"
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

              {/* Automated Training Controls - Only in Training Mode */}
              {mode === 'training' && trainingMode === 'dataset' && trainingExamples.length > 0 && (
                <div className="mt-4 space-y-2">
                  <div className="text-sm font-medium text-gray-700 mb-2">Automated Training</div>
                  <Button 
                    onClick={runToNextSample}
                    disabled={isAutoTraining}
                    size="sm"
                    className="w-full bg-green-600 hover:bg-green-700 text-white"
                  >
                    {isAutoTraining ? 'Training...' : 'Run to Next Sample'}
                  </Button>
                  <Button 
                    onClick={processTrainingSet}
                    disabled={isAutoTraining}
                    size="sm"
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white"
                  >
                    {isAutoTraining ? 'Processing Set...' : 'Process Training Set'}
                  </Button>
                  <div className="text-xs text-gray-600 text-center">
                    Sample {currentTrainingIndex + 1} of {trainingExamples.length} • Speed: {autoTrainingSpeed}ms
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
                      Example {datasetIndex + 1} of {trainingExamples.length} • Target: {trainingExamples[datasetIndex]?.label}
                      <br />
                      One-hot: [{trainingExamples[datasetIndex]?.label === 0 ? '1,0' : '0,1'}] (Neuron0: digit0, Neuron1: digit1)
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button 
                      onClick={() => setDatasetIndex(Math.max(0, datasetIndex - 1))}
                      disabled={datasetIndex === 0}
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      ← Prev Example
                    </Button>
                    <Button 
                      onClick={() => setDatasetIndex(Math.min(trainingExamples.length - 1, datasetIndex + 1))}
                      disabled={datasetIndex === trainingExamples.length - 1}
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
                    onClick={() => setSelectedWeightBox(null)}
                    variant="outline"
                    size="sm"
                  >
                    ×
                  </Button>
                </div>
                
                {trainingHistory.length > 0 && (
                  <div className="mb-4">
                    <label className="text-sm font-medium">Training Iteration: </label>
                    <input
                      type="range"
                      min="0"
                      max={trainingHistory.length - 1}
                      value={weightDialogIteration}
                      onChange={(e) => setWeightDialogIteration(parseInt(e.target.value))}
                      className="ml-2 w-32"
                    />
                    <span className="ml-2 text-sm">{weightDialogIteration + 1} / {trainingHistory.length}</span>
                  </div>
                )}
                
                {/* Weight Visualization */}
                <div className="bg-gray-50 p-4 rounded-lg">
                  {selectedWeightBox.type === 'hidden' && (
                    <div className="h-[400px] overflow-auto">
                      <svg viewBox="0 0 600 1850" className="w-full" style={{ minHeight: '1850px' }}>
                          {/* Background */}
                          <rect x="50" y="30" width="500" height="1800" fill="white" stroke="#9CA3AF" strokeWidth="2"/>
                          <line x1="300" y1="30" x2="300" y2="1830" stroke="#666" strokeWidth="2" opacity="0.5"/>
                          
                          {/* Weight bars */}
                          {(trainingHistory[weightDialogIteration]?.weights[selectedWeightBox.index] || weights[selectedWeightBox.index]).map((weight: number, i: number) => {
                            const barY = 45 + i * 22;
                            const barWidth = Math.abs(weight) * 250;
                            const barX = weight >= 0 ? 300 : 300 - barWidth;
                            return (
                              <g key={i}>
                                <rect
                                  x={barX}
                                  y={barY}
                                  width={barWidth}
                                  height="10"
                                  fill={weight > 0 ? "#10B981" : "#EF4444"}
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
                          
                          {/* Bias visualization */}
                          {(() => {
                            const bias = (trainingHistory[weightDialogIteration]?.biases && trainingHistory[weightDialogIteration]?.biases[selectedWeightBox.index]) || biases[selectedWeightBox.index];
                            const biasY = 45 + 81 * 22;
                            const biasWidth = Math.abs(bias) * 250;
                            const biasX = bias >= 0 ? 300 : 300 - biasWidth;
                            return (
                              <g>
                                <rect
                                  x={biasX}
                                  y={biasY}
                                  width={biasWidth}
                                  height="12"
                                  fill={bias > 0 ? "#8B5CF6" : "#EC4899"}
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
                          <text x="55" y="1845" fontSize="12" fill="#666">-1</text>
                          <text x="295" y="1845" fontSize="12" fill="#666">0</text>
                          <text x="535" y="1845" fontSize="12" fill="#666">+1</text>
                      </svg>
                    </div>
                  )}

                  {selectedWeightBox.type === 'output' && (
                    <svg width="100%" height="580" viewBox="0 0 600 580">
                      <g>
                        {/* Large weight box */}
                        <rect x="50" y="30" width="500" height="540" fill="white" stroke="#9CA3AF" strokeWidth="2"/>
                        <line x1="300" y1="30" x2="300" y2="570" stroke="#666" strokeWidth="2" opacity="0.5"/>
                        
                        {/* Weight bars */}
                        {(trainingHistory[weightDialogIteration]?.outputWeights[selectedWeightBox.index] || outputWeights[selectedWeightBox.index]).map((weight: number, i: number) => {
                          const barY = 50 + i * 22;
                          const barWidth = Math.abs(weight) * 250;
                          const barX = weight >= 0 ? 300 : 300 - barWidth;
                          return (
                            <g key={i}>
                              <rect
                                x={barX}
                                y={barY}
                                width={barWidth}
                                height="18"
                                fill={weight > 0 ? "#10B981" : "#EF4444"}
                                opacity="0.8"
                              />
                              <text x="20" y={barY + 14} fontSize="11" fill="#666">
                                Hidden {i + 1}:
                              </text>
                              <text x={weight >= 0 ? barX + barWidth + 5 : barX - 5} y={barY + 14} 
                                    fontSize="11" fill="#333" textAnchor={weight >= 0 ? "start" : "end"}>
                                {weight.toFixed(3)}
                              </text>
                            </g>
                          );
                        })}
                        
                        {/* Bias visualization */}
                        {(() => {
                          const bias = (trainingHistory[weightDialogIteration]?.outputBiases && trainingHistory[weightDialogIteration]?.outputBiases[selectedWeightBox.index]) || outputBiases[selectedWeightBox.index];
                          const biasY = 50 + 24 * 22;
                          const biasWidth = Math.abs(bias) * 250;
                          const biasX = bias >= 0 ? 300 : 300 - biasWidth;
                          return (
                            <g>
                              <rect
                                x={biasX}
                                y={biasY}
                                width={biasWidth}
                                height="18"
                                fill={bias > 0 ? "#8B5CF6" : "#EC4899"}
                                opacity="0.8"
                              />
                              <text x="20" y={biasY + 14} fontSize="11" fill="#666" fontWeight="bold">
                                Bias:
                              </text>
                              <text x={bias >= 0 ? biasX + biasWidth + 5 : biasX - 5} y={biasY + 14} 
                                    fontSize="11" fill="#333" textAnchor={bias >= 0 ? "start" : "end"} fontWeight="bold">
                                {bias.toFixed(3)}
                              </text>
                            </g>
                          );
                        })()}
                        
                        {/* Labels */}
                        <text x="55" y="575" fontSize="12" fill="#666">-1</text>
                        <text x="295" y="575" fontSize="12" fill="#666">0</text>
                        <text x="535" y="575" fontSize="12" fill="#666">+1</text>
                      </g>
                    </svg>
                  )}
                </div>
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
                  {trainingExamples.filter((ex: TrainingExample) => ex.label === 0).length} zeros, {trainingExamples.filter((ex: TrainingExample) => ex.label === 1).length} ones
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
                              value={example.label}
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
                          <div>Pattern: [{pixelValues.map(v => v.toString()).join(', ')}]</div>
                          <div className="mt-1">Click pixels to toggle. Target: {example.label}</div>
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
                <Label htmlFor="epochs">Number of Epochs:</Label>
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
                  onClick={startMultiEpochTraining}
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
      </div>
    </div>
  );
}
