import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

// Each pixel is a 3x3 grid of sub-pixels (9 total per pixel)
const initialPixelGrid = Array(9).fill(0).map(() => Array(9).fill(0)); // 9 pixels, each with 9 sub-pixels
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
  const [pixelGrid, setPixelGrid] = useState(initialPixelGrid);
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
  const [isDrawing, setIsDrawing] = useState(false);
  const [hoveredPixel, setHoveredPixel] = useState(null);

  // Calculate pixel values (0-1 based on how many sub-pixels are filled)
  const getPixelValues = () => {
    return pixelGrid.map(pixel => {
      const filledSubPixels = pixel.reduce((sum, subPixel) => sum + subPixel, 0);
      return filledSubPixels / 9; // Convert to 0-1 range
    });
  };

  const toggleSubPixel = (pixelIndex, subPixelIndex) => {
    const newPixelGrid = [...pixelGrid];
    newPixelGrid[pixelIndex] = [...newPixelGrid[pixelIndex]];
    newPixelGrid[pixelIndex][subPixelIndex] = newPixelGrid[pixelIndex][subPixelIndex] === 0 ? 1 : 0;
    setPixelGrid(newPixelGrid);
    setStep(0); // Reset to first step when input changes
  };

  const handleMouseDown = (pixelIndex, subPixelIndex) => {
    setIsDrawing(true);
    toggleSubPixel(pixelIndex, subPixelIndex);
  };

  const handleMouseEnter = (pixelIndex, subPixelIndex) => {
    if (isDrawing) {
      toggleSubPixel(pixelIndex, subPixelIndex);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handlePixelHover = (pixelIndex) => {
    setHoveredPixel(pixelIndex);
  };

  const handlePixelLeave = () => {
    setHoveredPixel(null);
  };

  const forwardPassHidden = () => {
    const pixelValues = getPixelValues();
    const newActivations = weights.map((w, i) => {
      const z = w.reduce((sum, weight, j) => sum + weight * pixelValues[j], biases[i]);
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
    const pixelValues = getPixelValues();
    
    // Calculate hidden errors
    const hiddenErrors = hiddenActivations.map((activation, i) => {
      const error = outputErrors.reduce((sum, outputError, j) => 
        sum + outputError * outputWeights[j][i], 0);
      return error * sigmoidDerivative(activation);
    });
    
    // Update hidden weights and biases
    const newWeights = weights.map((weights, i) => 
      weights.map((weight, j) => 
        weight - learningRate * hiddenErrors[i] * pixelValues[j]));
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
    setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
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
          {/* Drawing Canvas */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Drawing Canvas (3×3 pixels)</h2>
              <div 
                className="grid grid-cols-3 gap-0 mb-4 w-48 h-48 mx-auto border-2 border-gray-400 bg-gray-100"
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              >
                {pixelGrid.map((pixel, pixelIndex) => (
                  <div 
                    key={pixelIndex} 
                    className="relative border border-gray-300"
                    onMouseEnter={() => handlePixelHover(pixelIndex)}
                    onMouseLeave={handlePixelLeave}
                  >
                    {/* Pixel with 3x3 sub-pixels */}
                    <div className="grid grid-cols-3 gap-0 w-full h-full">
                      {pixel.map((subPixel, subPixelIndex) => (
                        <div
                          key={subPixelIndex}
                          onMouseDown={() => handleMouseDown(pixelIndex, subPixelIndex)}
                          onMouseEnter={() => handleMouseEnter(pixelIndex, subPixelIndex)}
                          className={`w-full h-full border border-gray-200 cursor-crosshair select-none transition-colors duration-100 ${
                            subPixel ? "bg-gray-800" : "bg-white hover:bg-gray-100"
                          }`}
                        />
                      ))}
                    </div>
                    
                    {/* Hover tooltip */}
                    {hoveredPixel === pixelIndex && (
                      <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white px-2 py-1 rounded text-xs font-mono whitespace-nowrap z-10">
                        Pixel {pixelIndex + 1}: {(pixel.reduce((sum, sub) => sum + sub, 0) / 9).toFixed(2)}
                      </div>
                    )}
                  </div>
                ))}
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
              </div>
            </CardContent>
          </Card>

          {/* Neural Network Diagram */}
          <Card className="col-span-2">
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Neural Network Diagram</h2>
              
              <div className="relative h-96 bg-gray-50 rounded-lg p-4 overflow-hidden">
                <svg className="w-full h-full" viewBox="0 0 400 340">
                  {/* Input Layer */}
                  <g className="input-layer">
                    <text x="20" y="20" fontSize="12" fill="#666" fontWeight="bold">Input (9)</text>
                    {getPixelValues().map((value, i) => (
                      <g key={`input-${i}`}>
                        <circle
                          cx="40"
                          cy={40 + i * 25}
                          r="10"
                          fill={value > 0.5 ? "#3B82F6" : "#E5E7EB"}
                          stroke="#9CA3AF"
                          strokeWidth="1"
                        />
                        <text x="40" y={44 + i * 25} fontSize="7" fill="#000" textAnchor="middle" fontWeight="bold">
                          {value.toFixed(2)}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Hidden Layer */}
                  <g className="hidden-layer">
                    <text x="130" y="20" fontSize="12" fill="#666" fontWeight="bold">Hidden (4)</text>
                    {hiddenActivations.map((activation, i) => (
                      <g key={`hidden-${i}`}>
                        {/* Bar graph for this hidden neuron - 9 bars for 9 inputs */}
                        <g className="bar-graph">
                          {getPixelValues().map((inputValue, inputIdx) => {
                            const weight = weights[i][inputIdx];
                            const barY = 50 + i * 50 + inputIdx * 3;
                            return (
                              <g key={`bar-${i}-${inputIdx}`}>
                                {/* Input value bar */}
                                <rect
                                  x="100"
                                  y={barY}
                                  width={Math.abs(inputValue) * 20}
                                  height="1"
                                  fill="#3B82F6"
                                  opacity="0.7"
                                />
                                {/* Weight bar */}
                                <rect
                                  x="100"
                                  y={barY + 1}
                                  width={Math.abs(weight) * 20}
                                  height="1"
                                  fill={weight > 0 ? "#10B981" : "#EF4444"}
                                  opacity="0.8"
                                />
                              </g>
                            );
                          })}
                        </g>
                        
                        <circle
                          cx="150"
                          cy={70 + i * 50}
                          r="12"
                          fill={activation > 0.5 ? "#8B5CF6" : "#E5E7EB"}
                          stroke="#9CA3AF"
                          strokeWidth="2"
                        />
                        <text x="150" y={75 + i * 50} fontSize="7" fill="#000" textAnchor="middle" fontWeight="bold">
                          {activation.toFixed(2)}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Output Layer */}
                  <g className="output-layer">
                    <text x="220" y="20" fontSize="12" fill="#666" fontWeight="bold">Output (2)</text>
                    {outputActivations.map((activation, i) => (
                      <g key={`output-${i}`}>
                        {/* Bar graph for this output neuron - 4 bars for 4 hidden inputs */}
                        <g className="bar-graph">
                          {hiddenActivations.map((hiddenValue, hiddenIdx) => {
                            const weight = outputWeights[i][hiddenIdx];
                            const barY = 80 + i * 80 + hiddenIdx * 8;
                            return (
                              <g key={`output-bar-${i}-${hiddenIdx}`}>
                                {/* Hidden activation value bar */}
                                <rect
                                  x="200"
                                  y={barY}
                                  width={Math.abs(hiddenValue) * 30}
                                  height="3"
                                  fill="#8B5CF6"
                                  opacity="0.7"
                                />
                                {/* Weight bar */}
                                <rect
                                  x="200"
                                  y={barY + 4}
                                  width={Math.abs(weight) * 30}
                                  height="3"
                                  fill={weight > 0 ? "#10B981" : "#EF4444"}
                                  opacity="0.8"
                                />
                              </g>
                            );
                          })}
                        </g>
                        
                        <circle
                          cx="260"
                          cy={110 + i * 80}
                          r="15"
                          fill={activation === Math.max(...outputActivations) ? "#10B981" : "#E5E7EB"}
                          stroke="#9CA3AF"
                          strokeWidth="2"
                        />
                        <text x="260" y={115 + i * 80} fontSize="8" fill="#000" textAnchor="middle" fontWeight="bold">
                          {activation.toFixed(2)}
                        </text>
                        <text x="280" y={115 + i * 80} fontSize="9" fill="#666" fontWeight="bold">
                          {i}: {(activation * 100).toFixed(0)}%
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Simple connection lines */}
                  {weights.map((hiddenWeights, hiddenIdx) =>
                    hiddenWeights.map((weight, inputIdx) => (
                      <line
                        key={`line-ih-${hiddenIdx}-${inputIdx}`}
                        x1="50"
                        y1={40 + inputIdx * 25}
                        x2="138"
                        y2={70 + hiddenIdx * 50}
                        stroke="#D1D5DB"
                        strokeWidth="0.5"
                        opacity="0.3"
                      />
                    ))
                  )}

                  {outputWeights.map((outputWeightArray, outputIdx) =>
                    outputWeightArray.map((weight, hiddenIdx) => (
                      <line
                        key={`line-ho-${outputIdx}-${hiddenIdx}`}
                        x1="162"
                        y1={70 + hiddenIdx * 50}
                        x2="245"
                        y2={110 + outputIdx * 80}
                        stroke="#D1D5DB"
                        strokeWidth="0.5"
                        opacity="0.3"
                      />
                    ))
                  )}

                  {/* Legend */}
                  <g className="legend">
                    <text x="20" y="310" fontSize="10" fill="#666" fontWeight="bold">Legend:</text>
                    <rect x="20" y="320" width="15" height="2" fill="#3B82F6" opacity="0.7"/>
                    <text x="40" y="323" fontSize="8" fill="#666">Input Values</text>
                    <rect x="110" y="320" width="15" height="2" fill="#10B981" opacity="0.8"/>
                    <text x="130" y="323" fontSize="8" fill="#666">Positive Weights</text>
                    <rect x="200" y="320" width="15" height="2" fill="#EF4444" opacity="0.8"/>
                    <text x="220" y="323" fontSize="8" fill="#666">Negative Weights</text>
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
