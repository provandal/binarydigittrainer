import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";


// Each pixel is a 3x3 grid of sub-pixels (9 total per pixel)
const initialPixelGrid = Array(9).fill(0).map(() => Array(9).fill(0)); // 9 pixels, each with 9 sub-pixels
const initialWeights = Array.from({ length: 4 }, () => Array(9).fill(0).map(() => (Math.random() - 0.5) * 0.4));
const initialBiases = Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.2);
const initialOutputWeights = Array.from({ length: 2 }, () => Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.4));
const initialOutputBiases = Array(2).fill(0).map(() => (Math.random() - 0.5) * 0.2);

const sigmoid = (x: number) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const sigmoidDerivative = (x: number) => x * (1 - x);

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
  const [hoveredPixel, setHoveredPixel] = useState<number | null>(null);
  const [selectedWeightBox, setSelectedWeightBox] = useState<{type: 'hidden' | 'output', index: number} | null>(null);
  const [weightDialogIteration, setWeightDialogIteration] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);

  // Calculate pixel values (0-1 based on how many sub-pixels are filled)
  const getPixelValues = () => {
    return pixelGrid.map(pixel => {
      const filledSubPixels = pixel.reduce((sum, subPixel) => sum + subPixel, 0);
      return filledSubPixels / 9; // Convert to 0-1 range
    });
  };

  const toggleSubPixel = (pixelIndex: number, subPixelIndex: number) => {
    const newPixelGrid = [...pixelGrid];
    newPixelGrid[pixelIndex] = [...newPixelGrid[pixelIndex]];
    newPixelGrid[pixelIndex][subPixelIndex] = newPixelGrid[pixelIndex][subPixelIndex] === 0 ? 1 : 0;
    setPixelGrid(newPixelGrid);
    setStep(0); // Reset to first step when input changes
  };

  const handleMouseDown = (pixelIndex: number, subPixelIndex: number) => {
    setIsDrawing(true);
    toggleSubPixel(pixelIndex, subPixelIndex);
  };

  const handleMouseEnter = (pixelIndex: number, subPixelIndex: number) => {
    if (isDrawing) {
      toggleSubPixel(pixelIndex, subPixelIndex);
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
    
    // Save training history snapshot
    const historySnapshot = {
      iteration: trainingHistory.length,
      weights: newWeights.map(w => [...w]),
      outputWeights: outputWeights.map(w => [...w]),
      loss: loss,
      hiddenActivations: [...hiddenActivations],
      outputActivations: [...outputActivations]
    };
    setTrainingHistory(prev => [...prev, historySnapshot]);
    
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
        // Complete cycle, start over - clear the canvas for next digit
        setPixelGrid(Array(9).fill(0).map(() => Array(9).fill(0)));
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
    setTrainingHistory([]);
    setSelectedWeightBox(null);
    setWeightDialogIteration(0);
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
              
              <div className="relative h-[420px] bg-gray-50 rounded-lg p-4 overflow-hidden">
                <svg className="w-full h-full" viewBox="0 0 750 563">
                  {/* Input Layer */}
                  <g className="input-layer">
                    <text x="38" y="38" fontSize="20" fill="#666" fontWeight="bold">Input (9)</text>
                    {getPixelValues().map((value, i) => (
                      <g key={`input-${i}`}>
                        <circle
                          cx="75"
                          cy={75 + i * 44}
                          r="19"
                          fill={value > 0.5 ? "#3B82F6" : "#E5E7EB"}
                          stroke="#9CA3AF"
                          strokeWidth="2.5"
                        />
                        <text x="75" y={83 + i * 44} fontSize="12" fill="#000" textAnchor="middle" fontWeight="bold">
                          {value.toFixed(2)}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Hidden Layer */}
                  <g className="hidden-layer">
                    <text x="250" y="38" fontSize="20" fill="#666" fontWeight="bold">Hidden (4)</text>
                    {hiddenActivations.map((activation, i) => (
                      <g key={`hidden-${i}`}>
                        <circle
                          cx="313"
                          cy={150 + i * 100}
                          r="23"
                          fill={activation > 0.5 ? "#8B5CF6" : "#E5E7EB"}
                          stroke="#9CA3AF"
                          strokeWidth="2.5"
                        />
                        <text x="313" y={159 + i * 100} fontSize="12" fill="#000" textAnchor="middle" fontWeight="bold">
                          {activation.toFixed(2)}
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Output Layer */}
                  <g className="output-layer">
                    <text x="525" y="38" fontSize="20" fill="#666" fontWeight="bold">Output (2)</text>
                    {outputActivations.map((activation, i) => (
                      <g key={`output-${i}`}>
                        <circle
                          cx="600"
                          cy={225 + i * 150}
                          r="28"
                          fill={activation === Math.max(...outputActivations) ? "#10B981" : "#E5E7EB"}
                          stroke="#9CA3AF"
                          strokeWidth="3.75"
                        />
                        <text x="600" y={235 + i * 150} fontSize="15" fill="#000" textAnchor="middle" fontWeight="bold">
                          {activation.toFixed(2)}
                        </text>
                        <text x="638" y={231 + i * 150} fontSize="15" fill="#666" fontWeight="bold">
                          {i}: {(activation * 100).toFixed(0)}%
                        </text>
                      </g>
                    ))}
                  </g>

                  {/* Connection lines */}
                  {weights.map((hiddenWeights, hiddenIdx) =>
                    hiddenWeights.map((weight, inputIdx) => (
                      <line
                        key={`line-ih-${hiddenIdx}-${inputIdx}`}
                        x1="94"
                        y1={75 + inputIdx * 44}
                        x2="290"
                        y2={150 + hiddenIdx * 100}
                        stroke="#9CA3AF"
                        strokeWidth="1.25"
                        opacity="0.4"
                      />
                    ))
                  )}

                  {outputWeights.map((outputWeightArray, outputIdx) =>
                    outputWeightArray.map((weight, hiddenIdx) => (
                      <line
                        key={`line-ho-${outputIdx}-${hiddenIdx}`}
                        x1="336"
                        y1={150 + hiddenIdx * 100}
                        x2="572"
                        y2={225 + outputIdx * 150}
                        stroke="#9CA3AF"
                        strokeWidth="1.25"
                        opacity="0.4"
                      />
                    ))
                  )}

                  {/* Weight bar graphs - rendered on top */}
                  {/* Hidden layer weight boxes */}
                  {hiddenActivations.map((activation, i) => (
                    <g key={`hidden-weight-box-${i}`} className="weight-box cursor-pointer" onClick={() => {
                      setSelectedWeightBox({type: 'hidden', index: i});
                      setWeightDialogIteration(trainingHistory.length === 0 ? 0 : Math.max(0, trainingHistory.length - 1));
                    }}>
                      {/* Box border - centered on neuron */}
                      <rect
                        x="188"
                        y={116 + i * 100}
                        width="100"
                        height="68"
                        fill="white"
                        stroke="#9CA3AF"
                        strokeWidth="2.5"
                        opacity="1"
                      />
                      {/* Zero line in middle */}
                      <line
                        x1="238"
                        y1={116 + i * 100}
                        x2="238"
                        y2={184 + i * 100}
                        stroke="#666"
                        strokeWidth="1.25"
                        opacity="0.5"
                      />
                      {/* Weight bars - 9 bars for 9 inputs */}
                      {weights[i].map((weight, inputIdx) => {
                        const barY = 123 + i * 100 + inputIdx * 6.9;
                        const barWidth = Math.abs(weight) * 50;
                        const barX = weight >= 0 ? 238 : 238 - barWidth;
                        return (
                          <rect
                            key={`weight-bar-${i}-${inputIdx}`}
                            x={barX}
                            y={barY}
                            width={barWidth}
                            height="5"
                            fill={weight > 0 ? "#10B981" : "#EF4444"}
                            opacity="0.9"
                          />
                        );
                      })}
                      {/* Labels */}
                      <text x="194" y="133" fontSize="10" fill="#666">-1</text>
                      <text x="231" y="133" fontSize="10" fill="#666">0</text>
                      <text x="275" y="133" fontSize="10" fill="#666">+1</text>
                    </g>
                  ))}

                  {/* Output layer weight boxes */}
                  {outputActivations.map((activation, i) => (
                    <g key={`output-weight-box-${i}`} className="weight-box cursor-pointer" onClick={() => {
                      setSelectedWeightBox({type: 'output', index: i});
                      setWeightDialogIteration(trainingHistory.length === 0 ? 0 : Math.max(0, trainingHistory.length - 1));
                    }}>
                      {/* Box border - centered on neuron */}
                      <rect
                        x="450"
                        y={195 + i * 150}
                        width="125"
                        height="60"
                        fill="white"
                        stroke="#9CA3AF"
                        strokeWidth="2.5"
                        opacity="1"
                      />
                      {/* Zero line in middle */}
                      <line
                        x1="513"
                        y1={195 + i * 150}
                        x2="513"
                        y2={255 + i * 150}
                        stroke="#666"
                        strokeWidth="1.25"
                        opacity="0.5"
                      />
                      {/* Weight bars - 4 bars for 4 hidden inputs */}
                      {outputWeights[i].map((weight, hiddenIdx) => {
                        const barY = 203 + i * 150 + hiddenIdx * 12.5;
                        const barWidth = Math.abs(weight) * 63;
                        const barX = weight >= 0 ? 513 : 513 - barWidth;
                        return (
                          <rect
                            key={`output-weight-bar-${i}-${hiddenIdx}`}
                            x={barX}
                            y={barY}
                            width={barWidth}
                            height="7.5"
                            fill={weight > 0 ? "#10B981" : "#EF4444"}
                            opacity="0.9"
                          />
                        );
                      })}
                      {/* Labels */}
                      <text x="456" y="211" fontSize="12" fill="#666">-1</text>
                      <text x="506" y="211" fontSize="12" fill="#666">0</text>
                      <text x="556" y="211" fontSize="12" fill="#666">+1</text>
                    </g>
                  ))}

                  {/* Legend */}
                  <g className="legend">
                    <text x="38" y="520" fontSize="18" fill="#666" fontWeight="bold">Weight Bar Graphs (click for details):</text>
                    <rect x="38" y="539" width="31" height="5" fill="#10B981" opacity="0.8"/>
                    <text x="75" y="545" fontSize="15" fill="#666">Positive Weights (extend right from center)</text>
                    <rect x="350" y="539" width="31" height="5" fill="#EF4444" opacity="0.8"/>
                    <text x="387" y="545" fontSize="15" fill="#666">Negative Weights (extend left from center)</text>
                    <text x="38" y="558" fontSize="12" fill="#999">Center line = 0, Right edge = +1, Left edge = -1</text>
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

          {/* Detailed Weight View */}
          {selectedWeightBox && (
            <Card className="col-span-3">
              <CardContent className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">
                    {selectedWeightBox.type === 'hidden' 
                      ? `Hidden Neuron ${selectedWeightBox.index + 1} Weights (9 input connections)`
                      : `Output Neuron ${selectedWeightBox.index} Weights (4 hidden connections)`}
                  </h2>
                  <Button 
                    onClick={() => setSelectedWeightBox(null)}
                    variant="outline"
                    size="sm"
                  >
                    ×
                  </Button>
                </div>
                
                {/* Weight Visualization */}
                <div className="bg-gray-50 p-4 rounded-lg">
                  <svg width="100%" height="220" viewBox="0 0 600 220">
                    {selectedWeightBox.type === 'hidden' && (
                      <g>
                        {/* Large weight box */}
                        <rect x="50" y="30" width="500" height="160" fill="white" stroke="#9CA3AF" strokeWidth="2"/>
                        <line x1="300" y1="30" x2="300" y2="190" stroke="#666" strokeWidth="2" opacity="0.5"/>
                        
                        {/* Weight bars */}
                        {(trainingHistory[weightDialogIteration]?.weights[selectedWeightBox.index] || weights[selectedWeightBox.index]).map((weight: number, i: number) => {
                          const barY = 45 + i * 16;
                          const barWidth = Math.abs(weight) * 250;
                          const barX = weight >= 0 ? 300 : 300 - barWidth;
                          return (
                            <g key={i}>
                              <rect
                                x={barX}
                                y={barY}
                                width={barWidth}
                                height="12"
                                fill={weight > 0 ? "#10B981" : "#EF4444"}
                                opacity="0.8"
                              />
                              <text x="20" y={barY + 9} fontSize="11" fill="#666">
                                Input {i + 1}:
                              </text>
                              <text x={weight >= 0 ? barX + barWidth + 5 : barX - 5} y={barY + 9} 
                                    fontSize="11" fill="#333" textAnchor={weight >= 0 ? "start" : "end"}>
                                {weight.toFixed(3)}
                              </text>
                            </g>
                          );
                        })}
                        
                        {/* Labels */}
                        <text x="55" y="205" fontSize="12" fill="#666">-1</text>
                        <text x="295" y="205" fontSize="12" fill="#666">0</text>
                        <text x="535" y="205" fontSize="12" fill="#666">+1</text>
                      </g>
                    )}

                    {selectedWeightBox.type === 'output' && (
                      <g>
                        {/* Large weight box */}
                        <rect x="50" y="30" width="500" height="120" fill="white" stroke="#9CA3AF" strokeWidth="2"/>
                        <line x1="300" y1="30" x2="300" y2="150" stroke="#666" strokeWidth="2" opacity="0.5"/>
                        
                        {/* Weight bars */}
                        {(trainingHistory[weightDialogIteration]?.outputWeights[selectedWeightBox.index] || outputWeights[selectedWeightBox.index]).map((weight: number, i: number) => {
                          const barY = 50 + i * 25;
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
                        
                        {/* Labels */}
                        <text x="55" y="165" fontSize="12" fill="#666">-1</text>
                        <text x="295" y="165" fontSize="12" fill="#666">0</text>
                        <text x="535" y="165" fontSize="12" fill="#666">+1</text>
                      </g>
                    )}
                  </svg>
                </div>
              </CardContent>
            </Card>
          )}

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
