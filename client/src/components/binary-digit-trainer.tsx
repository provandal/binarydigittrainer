import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const initialInput = Array(9).fill(0);
const initialWeights = Array.from({ length: 4 }, () => Array(9).fill(0).map(() => (Math.random() - 0.5) * 0.4));
const initialBiases = Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.2);
const initialOutputWeights = Array.from({ length: 2 }, () => Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.4));
const initialOutputBiases = Array(2).fill(0).map(() => (Math.random() - 0.5) * 0.2);

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const sigmoidDerivative = (x) => x * (1 - x);

const STEP_NAMES = [
  "Ready - Draw your digit",
  "Forward Pass - Input to Hidden",
  "Forward Pass - Hidden to Output", 
  "Calculate Loss",
  "Backpropagation - Output Layer",
  "Backpropagation - Hidden Layer"
];

export default function BinaryDigitTrainer() {
  const [input, setInput] = useState(initialInput);
  const [weights, setWeights] = useState(initialWeights);
  const [biases, setBiases] = useState(initialBiases);
  const [outputWeights, setOutputWeights] = useState(initialOutputWeights);
  const [outputBiases, setOutputBiases] = useState(initialOutputBiases);
  const [hiddenActivations, setHiddenActivations] = useState(Array(4).fill(0));
  const [outputActivations, setOutputActivations] = useState(Array(2).fill(0));
  const [selectedLabel, setSelectedLabel] = useState(0);
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(0);
  const [learningRate] = useState(0.5);

  const toggleInput = (i) => {
    const newInput = [...input];
    newInput[i] = newInput[i] === 0 ? 1 : 0;
    setInput(newInput);
    setStep(0); // Reset to first step when input changes
  };

  const forwardPassHidden = () => {
    const newActivations = weights.map((w, i) => {
      const z = w.reduce((sum, weight, j) => sum + weight * input[j], biases[i]);
      return sigmoid(z);
    });
    setHiddenActivations(newActivations);
  };

  const forwardPassOutput = () => {
    const newOutputActivations = outputWeights.map((w, i) => {
      const z = w.reduce((sum, weight, j) => sum + weight * hiddenActivations[j], outputBiases[i]);
      return sigmoid(z);
    });
    setOutputActivations(newOutputActivations);
  };

  const calculateLoss = () => {
    const target = [selectedLabel === 0 ? 1 : 0, selectedLabel === 1 ? 1 : 0];
    const mse = outputActivations.reduce((sum, output, i) => 
      sum + Math.pow(output - target[i], 2), 0) / 2;
    setLoss(mse);
  };

  const backpropagationOutput = () => {
    const target = [selectedLabel === 0 ? 1 : 0, selectedLabel === 1 ? 1 : 0];
    const outputErrors = outputActivations.map((output, i) => 
      (output - target[i]) * sigmoidDerivative(output));
    
    // Update output weights and biases
    const newOutputWeights = outputWeights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * outputErrors[i] * hiddenActivations[j]));
    const newOutputBiases = outputBiases.map((bias, i) => 
      bias - learningRate * outputErrors[i]);
    
    setOutputWeights(newOutputWeights);
    setOutputBiases(newOutputBiases);
  };

  const backpropagationHidden = () => {
    const target = [selectedLabel === 0 ? 1 : 0, selectedLabel === 1 ? 1 : 0];
    const outputErrors = outputActivations.map((output, i) => 
      (output - target[i]) * sigmoidDerivative(output));
    
    // Calculate hidden errors
    const hiddenErrors = hiddenActivations.map((activation, i) => {
      const error = outputErrors.reduce((sum, outputError, j) => 
        sum + outputError * outputWeights[j][i], 0);
      return error * sigmoidDerivative(activation);
    });
    
    // Update hidden weights and biases
    const newWeights = weights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * hiddenErrors[i] * input[j]));
    const newBiases = biases.map((bias, i) => 
      bias - learningRate * hiddenErrors[i]);
    
    setWeights(newWeights);
    setBiases(newBiases);
  };

  const nextStep = () => {
    switch (step) {
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
      case 4:
        backpropagationHidden();
        break;
      case 5:
        // Complete cycle, start over
        setStep(-1);
        break;
    }
    setStep((prev) => (prev + 1) % 6);
  };

  const resetNetwork = () => {
    setWeights(Array.from({ length: 4 }, () => Array(9).fill(0).map(() => (Math.random() - 0.5) * 0.4)));
    setBiases(Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.2));
    setOutputWeights(Array.from({ length: 2 }, () => Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.4)));
    setOutputBiases(Array(2).fill(0).map(() => (Math.random() - 0.5) * 0.2));
    setHiddenActivations(Array(4).fill(0));
    setOutputActivations(Array(2).fill(0));
    setLoss(0);
    setStep(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">🧠 Binary Digit Trainer</h1>
          <p className="text-gray-600">Step-by-step Neural Network Learning Simulator</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Input Grid */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Input Grid (3×3)</h2>
              <div className="grid grid-cols-3 gap-2 mb-4">
                {input.map((val, i) => (
                  <button
                    key={i}
                    onClick={() => toggleInput(i)}
                    className={`w-12 h-12 border-2 rounded-lg font-bold text-lg transition-all duration-200 ${
                      val ? "bg-gray-800 text-white border-gray-800" : "bg-white text-gray-800 border-gray-300 hover:border-gray-400"
                    }`}
                  >
                    {val}
                  </button>
                ))}
              </div>
              
              <div className="space-y-3">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Target Label</h3>
                  <div className="flex gap-2">
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
              </div>
            </CardContent>
          </Card>

          {/* Hidden Layer */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Hidden Layer (4)</h2>
              <div className="space-y-3">
                {hiddenActivations.map((activation, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full border-2 border-purple-300 bg-purple-100 flex items-center justify-center text-sm font-bold">
                      H{i+1}
                    </div>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-600">Activation</span>
                        <span className="text-xs font-mono">{activation.toFixed(3)}</span>
                      </div>
                      <div className="w-full h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-2 bg-purple-500 rounded-full transition-all duration-300"
                          style={{ width: `${Math.max(activation * 100, 2)}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Weight Visualization */}
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Weights to Hidden</h3>
                <div className="grid grid-cols-3 gap-1">
                  {weights[0] && weights[0].map((_, inputIdx) => (
                    <div key={inputIdx} className="text-center">
                      <div className="text-xs text-gray-500 mb-1">I{inputIdx+1}</div>
                      {weights.map((neuronWeights, neuronIdx) => (
                        <div
                          key={neuronIdx}
                          className={`w-6 h-1 mx-auto mb-1 rounded ${
                            neuronWeights[inputIdx] > 0 ? 'bg-green-500' : 'bg-red-500'
                          }`}
                          style={{ 
                            opacity: Math.min(Math.abs(neuronWeights[inputIdx]) * 2, 1),
                            width: `${Math.max(Math.abs(neuronWeights[inputIdx]) * 20, 2)}px`
                          }}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Output Layer */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Output Layer (2)</h2>
              <div className="space-y-4">
                {outputActivations.map((activation, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center text-sm font-bold ${
                      i === 0 ? 'border-green-300 bg-green-100' : 'border-blue-300 bg-blue-100'
                    }`}>
                      {i}
                    </div>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-600">Probability</span>
                        <span className="text-sm font-mono font-bold">
                          {(activation * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-3 bg-gray-200 rounded-full">
                        <div
                          className={`h-3 rounded-full transition-all duration-300 ${
                            i === 0 ? 'bg-green-500' : 'bg-blue-500'
                          }`}
                          style={{ width: `${Math.max(activation * 100, 2)}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Prediction:</span>
                  <span className="text-lg font-bold text-gray-900">
                    Digit {outputActivations[0] > outputActivations[1] ? 0 : 1}
                  </span>
                </div>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-sm text-gray-600">Loss (MSE):</span>
                  <span className="text-sm font-mono">{loss.toFixed(4)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Controls */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Training Steps</h2>
              
              <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                <div className="text-sm font-medium text-blue-900 mb-1">
                  Step {step + 1} of 6
                </div>
                <div className="text-sm text-blue-700">
                  {STEP_NAMES[step]}
                </div>
              </div>

              <div className="space-y-3">
                <Button 
                  onClick={nextStep}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                >
                  Next Step →
                </Button>
                
                <Button 
                  onClick={resetNetwork}
                  variant="outline"
                  className="w-full"
                >
                  Reset Network
                </Button>
              </div>

              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Network Info</h3>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>Learning Rate: {learningRate}</div>
                  <div>Architecture: 9 → 4 → 2</div>
                  <div>Activation: Sigmoid</div>
                  <div>Loss: Mean Squared Error</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
