import { useState, useEffect, useCallback, useRef } from 'react';
import { SimulatedNeuralNetwork } from '@/lib/neural-network';

interface TestResult {
  predicted: number;
  actual: number;
  confidence: number;
  timestamp: number;
  correct: boolean;
}

export function useNeuralNetwork() {
  const [binaryGrid, setBinaryGrid] = useState<number[]>(new Array(64).fill(0));
  const [targetLabel, setTargetLabel] = useState<number>(0);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [predictions, setPredictions] = useState<number[]>(new Array(10).fill(0.1));
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    loss: 1,
    epochs: 0,
    learningRate: 0.001,
    activations: {
      input: new Array(64).fill(0),
      hidden: new Array(32).fill(0),
      output: new Array(10).fill(0)
    },
    trainingTime: 0
  });

  const [trainingHistory, setTrainingHistory] = useState({
    accuracy: [0],
    loss: [1]
  });

  const networkRef = useRef<SimulatedNeuralNetwork>(new SimulatedNeuralNetwork());
  const trainingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const trainingStartTimeRef = useRef<number>(0);

  // Update predictions when grid changes
  useEffect(() => {
    const { hiddenActivations, outputActivations, predictions: newPredictions } = 
      networkRef.current.forward(binaryGrid);
    
    setPredictions(newPredictions);
    setMetrics(prev => ({
      ...prev,
      activations: {
        input: binaryGrid,
        hidden: hiddenActivations,
        output: outputActivations
      }
    }));
  }, [binaryGrid]);

  const startTraining = useCallback(() => {
    if (isTraining) return;
    
    setIsTraining(true);
    trainingStartTimeRef.current = Date.now();
    
    trainingIntervalRef.current = setInterval(() => {
      // Simulate training on random data
      const randomInput = Array.from({ length: 64 }, () => Math.random() > 0.5 ? 1 : 0);
      const randomLabel = Math.floor(Math.random() * 10);
      
      const loss = networkRef.current.train(randomInput, randomLabel);
      
      setMetrics(prev => {
        const newEpochs = prev.epochs + 1;
        const newAccuracy = Math.min(0.95, prev.accuracy + (Math.random() * 0.01));
        const newLoss = Math.max(0.05, prev.loss - (Math.random() * 0.005));
        const trainingTime = Math.floor((Date.now() - trainingStartTimeRef.current) / 1000);
        
        return {
          ...prev,
          accuracy: newAccuracy,
          loss: newLoss,
          epochs: newEpochs,
          trainingTime
        };
      });

      setTrainingHistory(prev => ({
        accuracy: [...prev.accuracy.slice(-49), metrics.accuracy],
        loss: [...prev.loss.slice(-49), metrics.loss]
      }));

      // Update predictions for current input
      const { hiddenActivations, outputActivations, predictions: newPredictions } = 
        networkRef.current.forward(binaryGrid);
      
      setPredictions(newPredictions);
      setMetrics(prev => ({
        ...prev,
        activations: {
          input: binaryGrid,
          hidden: hiddenActivations,
          output: outputActivations
        }
      }));
    }, 100);
  }, [isTraining, binaryGrid, metrics.accuracy, metrics.loss]);

  const stopTraining = useCallback(() => {
    setIsTraining(false);
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
  }, []);

  const resetNetwork = useCallback(() => {
    stopTraining();
    networkRef.current.reset();
    setMetrics({
      accuracy: 0,
      loss: 1,
      epochs: 0,
      learningRate: 0.001,
      activations: {
        input: new Array(64).fill(0),
        hidden: new Array(32).fill(0),
        output: new Array(10).fill(0)
      },
      trainingTime: 0
    });
    setTrainingHistory({
      accuracy: [0],
      loss: [1]
    });
    setPredictions(new Array(10).fill(0.1));
    setTestResults([]);
  }, [stopTraining]);

  const testCurrentInput = useCallback(() => {
    const { predictions: currentPredictions } = networkRef.current.forward(binaryGrid);
    const predictedDigit = currentPredictions.indexOf(Math.max(...currentPredictions));
    const confidence = Math.max(...currentPredictions);
    
    const testResult: TestResult = {
      predicted: predictedDigit,
      actual: targetLabel,
      confidence: confidence,
      timestamp: Date.now(),
      correct: predictedDigit === targetLabel
    };
    
    setTestResults(prev => [testResult, ...prev.slice(0, 9)]);
  }, [binaryGrid, targetLabel]);

  const loadSampleDigit = useCallback((digit: number) => {
    // Generate a sample pattern for the digit (simplified)
    const patterns: { [key: number]: number[] } = {
      0: [0,1,1,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      5: [1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
    };
    
    if (patterns[digit]) {
      setBinaryGrid(patterns[digit]);
      setTargetLabel(digit);
    }
  }, []);

  const generateRandomGrid = useCallback(() => {
    const newGrid = Array.from({ length: 64 }, () => Math.random() > 0.7 ? 1 : 0);
    setBinaryGrid(newGrid);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    };
  }, []);

  return {
    binaryGrid,
    setBinaryGrid,
    targetLabel,
    setTargetLabel,
    isTraining,
    startTraining,
    stopTraining,
    resetNetwork,
    predictions,
    metrics,
    trainingHistory,
    testResults,
    testCurrentInput,
    loadSampleDigit,
    generateRandomGrid
  };
}
