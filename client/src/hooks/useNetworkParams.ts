import { useState, useRef, useCallback } from "react";
import { initWeight } from "@/lib/nn-math";

/** Shape of the master ref that async training reads/writes */
export interface NetworkStateRef {
  weights: number[][];
  biases: number[];
  outputWeights: number[][];
  outputBiases: number[];
  hiddenActivations: number[];
  outputActivations: number[];
  hiddenPreActivations: number[];
  outputPreActivations: number[];
  loss: number;
  outputErrors: number[];
  currentTarget: number[];
  inputs: number[];
}

export interface UseNetworkParamsReturn {
  // State
  weights: number[][];
  biases: number[];
  outputWeights: number[][];
  outputBiases: number[];
  hiddenActivations: number[];
  outputActivations: number[];
  loss: number;
  step: number;
  trainingHistory: any[];
  learningRate: number;

  // Refs
  currentNetworkState: React.MutableRefObject<NetworkStateRef>;
  trainingHistoryStore: React.MutableRefObject<any[]>;
  selectedLabelRef: React.MutableRefObject<number>;

  // Setters
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

  // Methods
  resetNetwork: () => void;
}

const createInitialWeights = () =>
  Array.from({ length: 24 }, () =>
    Array(81)
      .fill(0)
      .map(() => initWeight(81, 24)),
  );

const createInitialBiases = () => Array(24).fill(0);

const createInitialOutputWeights = () =>
  Array.from({ length: 2 }, () =>
    Array(24)
      .fill(0)
      .map(() => initWeight(24, 2)),
  );

const createInitialOutputBiases = () => Array(2).fill(0);

function createInitialNetworkStateRef(): NetworkStateRef {
  return {
    weights: createInitialWeights(),
    biases: createInitialBiases(),
    outputWeights: createInitialOutputWeights(),
    outputBiases: createInitialOutputBiases(),
    hiddenActivations: Array(24).fill(0),
    outputActivations: Array(2).fill(0),
    hiddenPreActivations: Array(24).fill(0),
    outputPreActivations: Array(2).fill(0),
    loss: 0,
    outputErrors: Array(2).fill(0),
    currentTarget: [1, 0],
    inputs: Array(81).fill(0),
  };
}

export function useNetworkParams(): UseNetworkParamsReturn {
  const [weights, setWeights] = useState(createInitialWeights);
  const [biases, setBiases] = useState(createInitialBiases);
  const [outputWeights, setOutputWeights] = useState(createInitialOutputWeights);
  const [outputBiases, setOutputBiases] = useState(createInitialOutputBiases);
  const [hiddenActivations, setHiddenActivations] = useState(() => Array(24).fill(0));
  const [outputActivations, setOutputActivations] = useState(() => Array(2).fill(0));
  const [loss, setLoss] = useState(0);
  const [step, setStep] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [learningRate, setLearningRate] = useState(0.01);

  // The master ref that async training reads/writes
  const currentNetworkState = useRef<NetworkStateRef>(createInitialNetworkStateRef());

  // Persistent training history store - independent of React state
  const trainingHistoryStore = useRef<any[]>([]);

  // Ref for synchronous access to selected label during training
  const selectedLabelRef = useRef<number>(0);

  const resetNetwork = useCallback(() => {
    // 81->24->2 architecture: 81 inputs (9x9 grid), 24 hidden neurons, 2 output neurons
    const newWeights = createInitialWeights();
    const newBiases = createInitialBiases();
    const newOutputWeights = createInitialOutputWeights();
    const newOutputBiases = createInitialOutputBiases();

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
      currentTarget: [1, 0],
      inputs: Array(81).fill(0),
    };

    // Clear training history
    trainingHistoryStore.current = [];

    // Update React state
    setWeights(newWeights);
    setBiases(newBiases);
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
    setHiddenActivations(Array(24).fill(0));
    setOutputActivations(Array(2).fill(0));
    setLoss(0);
    setStep(0);
    setTrainingHistory([]);
  }, []);

  return {
    // State
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

    // Refs
    currentNetworkState,
    trainingHistoryStore,
    selectedLabelRef,

    // Setters
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

    // Methods
    resetNetwork,
  };
}
