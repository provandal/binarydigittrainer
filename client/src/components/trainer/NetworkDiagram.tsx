import React from "react";
import { Card, CardContent } from "@/components/ui/card";

export interface NetworkDiagramProps {
  pixelValues: number[];
  hiddenActivations: number[];
  outputActivations: number[];
  outputBiases: number[];
  weights: number[][];
  outputWeights: number[][];
  activeElements: string[];
  trainingHistoryLength: number;
  mode: "training" | "inference";
  loss: number;
  showDebugDialog: boolean;
  setShowDebugDialog: (v: boolean) => void;
  setSelectedWeightBox: (v: { type: "hidden" | "output"; index: number } | null) => void;
  setWeightDialogIteration: (v: number) => void;
  setTourWeightVisualizationOpened: (v: boolean) => void;
  tourWeightVisualizationOpenedRef: React.MutableRefObject<boolean>;
}

export function NetworkDiagram({
  pixelValues,
  hiddenActivations,
  outputActivations,
  outputBiases,
  weights,
  outputWeights,
  activeElements,
  trainingHistoryLength,
  mode,
  loss,
  showDebugDialog,
  setShowDebugDialog,
  setSelectedWeightBox,
  setWeightDialogIteration,
  setTourWeightVisualizationOpened,
  tourWeightVisualizationOpenedRef,
}: NetworkDiagramProps) {
  return (
    <Card className="lg:col-span-2">
      <CardContent className="p-3 sm:p-6">
        <h2 className="mb-2 text-base font-semibold sm:mb-4 sm:text-lg">Neural Network Diagram</h2>

        <div className="relative flex h-[350px] items-start overflow-auto rounded-lg bg-gray-50 sm:h-[550px]">
          <svg
            className="neural-network-diagram block w-full self-start"
            viewBox="0 -20 750 1340"
            preserveAspectRatio="xMinYMin meet"
            style={{ minHeight: 1340 }}
          >
            {/* Input Layer */}
            <g className="input-layer">
              <text x="38" y="5" fontSize="20" fill="#666" fontWeight="bold">
                Input (81)
              </text>
              {pixelValues.map((value, i) => (
                <g key={`input-${i}`}>
                  <circle
                    cx="75"
                    cy={25 + i * 20}
                    r="12"
                    fill={value > 0.5 ? "#3B82F6" : "#E5E7EB"}
                    stroke={activeElements.includes("input") ? "#F59E0B" : "#9CA3AF"}
                    strokeWidth={activeElements.includes("input") ? "2" : "1"}
                    className={activeElements.includes("input") ? "animate-pulse" : ""}
                  />
                  <text
                    x="75"
                    y={28 + i * 20}
                    fontSize="7"
                    fill="#000"
                    textAnchor="middle"
                    fontWeight="bold"
                  >
                    {value}
                  </text>
                </g>
              ))}
            </g>

            {/* Hidden Layer */}
            <g className="hidden-layer">
              <text x="250" y="5" fontSize="20" fill="#666" fontWeight="bold">
                Hidden (24)
              </text>
              {hiddenActivations.map((activation, i) => (
                <g key={`hidden-${i}`}>
                  <circle
                    cx="313"
                    cy={25 + i * 22}
                    r="12"
                    fill={activation > 0.5 ? "#8B5CF6" : "#E5E7EB"}
                    stroke={activeElements.includes("hidden") ? "#F59E0B" : "#9CA3AF"}
                    strokeWidth={activeElements.includes("hidden") ? "2" : "1"}
                    className={activeElements.includes("hidden") ? "animate-pulse" : ""}
                  />
                  <text
                    x="313"
                    y={29 + i * 22}
                    fontSize="8"
                    fill="#000"
                    textAnchor="middle"
                    fontWeight="bold"
                  >
                    {activation.toFixed(2)}
                  </text>
                </g>
              ))}
            </g>

            {/* Output Layer */}
            <g className="output-layer">
              <text x="525" y="5" fontSize="20" fill="#666" fontWeight="bold">
                Output (2)
              </text>
              {outputActivations.map((activation, i) => (
                <g key={`output-${i}`}>
                  <circle
                    cx="600"
                    cy={70 + i * 120}
                    r="25"
                    fill={activation === Math.max(...outputActivations) ? "#10B981" : "#E5E7EB"}
                    stroke={activeElements.includes("output") ? "#F59E0B" : "#9CA3AF"}
                    strokeWidth={activeElements.includes("output") ? "5" : "3.75"}
                    className={activeElements.includes("output") ? "animate-pulse" : ""}
                  />
                  <text
                    x="600"
                    y={77 + i * 120}
                    fontSize="15"
                    fill="#000"
                    textAnchor="middle"
                    fontWeight="bold"
                  >
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
                  stroke={activeElements.includes("connections") ? "#F59E0B" : "#9CA3AF"}
                  strokeWidth={activeElements.includes("connections") ? "1" : "0.3"}
                  opacity={activeElements.includes("connections") ? "0.8" : "0.2"}
                  className={activeElements.includes("connections") ? "animate-pulse" : ""}
                />
              )),
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
                  stroke={activeElements.includes("connections") ? "#F59E0B" : "#9CA3AF"}
                  strokeWidth={activeElements.includes("connections") ? "2.5" : "1.25"}
                  opacity={activeElements.includes("connections") ? "0.8" : "0.4"}
                  className={activeElements.includes("connections") ? "animate-pulse" : ""}
                />
              )),
            )}

            {/* Weight detail buttons - rendered on top */}
            {/* Hidden layer plus buttons */}
            {hiddenActivations.map((activation, i) => (
              <g
                key={`hidden-plus-${i}`}
                className="cursor-pointer"
                onClick={() => {
                  setSelectedWeightBox({ type: "hidden", index: i });
                  setWeightDialogIteration(
                    trainingHistoryLength === 0 ? 0 : Math.max(0, trainingHistoryLength - 1),
                  );
                  setTourWeightVisualizationOpened(true);
                  tourWeightVisualizationOpenedRef.current = true;
                }}
              >
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
              <g
                key={`output-plus-${i}`}
                className="cursor-pointer"
                data-tour-target={i === 0 ? "output-neuron-0-plus" : undefined}
                onClick={() => {
                  setSelectedWeightBox({ type: "output", index: i });
                  setWeightDialogIteration(
                    trainingHistoryLength === 0 ? 0 : Math.max(0, trainingHistoryLength - 1),
                  );
                  setTourWeightVisualizationOpened(true);
                  tourWeightVisualizationOpenedRef.current = true;
                }}
              >
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
            <g
              className="debug-icon cursor-pointer"
              onClick={() => {
                setShowDebugDialog(true);
              }}
            >
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
        <div className="mt-4 rounded-lg bg-gray-50 p-3">
          <div className="text-sm">
            <div className="mb-2 font-medium text-gray-700">Weight Details:</div>
            <div className="space-y-2 text-xs text-gray-600">
              <div className="flex items-center gap-2">
                <div className="flex h-4 w-4 items-center justify-center rounded-full bg-green-500">
                  <span className="text-xs font-bold text-white">+</span>
                </div>
                <span>Click green plus button to view detailed weights for each neuron</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex h-4 w-4 items-center justify-center rounded-full bg-gray-500">
                  <span className="text-xs font-bold text-white">i</span>
                </div>
                <span>Click info button (top right) to view network debug information</span>
              </div>
            </div>
          </div>
        </div>

        {/* Network Summary */}
        {mode === "training" && (
          <div className="mt-4 rounded-lg bg-gray-50 p-3">
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
  );
}
