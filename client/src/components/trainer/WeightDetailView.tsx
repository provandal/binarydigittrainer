import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { HelpIcon } from "@/components/HelpIcon";
import {
  type ColorScheme,
  getBarColor,
  getColorSchemeDescription,
  getPositiveColorName,
  vec81ToGrid9,
} from "@/lib/color-schemes";
import { getDecisionContribs, type DecisionContrib } from "@/lib/nn-helpers";
import { Heatmap9x9 } from "./Heatmap9x9";

export interface WeightDetailViewProps {
  selectedWeightBox: { type: "hidden" | "output"; index: number };
  setSelectedWeightBox: (v: { type: "hidden" | "output"; index: number } | null) => void;
  setTourWeightVisualizationOpened: (v: boolean) => void;
  tourWeightVisualizationOpenedRef: React.MutableRefObject<boolean>;
  weightDialogIteration: number;
  setWeightDialogIteration: (v: number) => void;
  trainingHistory: any[];
  weights: number[][];
  biases: number[];
  outputWeights: number[][];
  outputBiases: number[];
  currentNetworkState: React.MutableRefObject<any>;
  pixelGrid: number[][];
  showInputOverlay: boolean;
  setShowInputOverlay: (v: boolean) => void;
  useGlobalScale: boolean;
  setUseGlobalScale: (v: boolean) => void;
  colorScheme: ColorScheme;
  setColorScheme: (v: ColorScheme) => void;
  viewMode: "decision" | "logit";
  setViewMode: (v: "decision" | "logit") => void;
}

export function WeightDetailView({
  selectedWeightBox,
  setSelectedWeightBox,
  setTourWeightVisualizationOpened,
  tourWeightVisualizationOpenedRef,
  weightDialogIteration,
  setWeightDialogIteration,
  trainingHistory,
  weights,
  biases,
  outputWeights,
  outputBiases,
  currentNetworkState,
  pixelGrid,
  showInputOverlay,
  setShowInputOverlay,
  useGlobalScale,
  setUseGlobalScale,
  colorScheme,
  setColorScheme,
  viewMode,
  setViewMode,
}: WeightDetailViewProps) {
  return (
    <div className="mt-3 sm:mt-6">
      <Card data-tour-target="weight-dialog">
        <CardContent className="p-3 sm:p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {selectedWeightBox.type === "hidden"
                ? `Hidden Neuron ${selectedWeightBox.index + 1} Weights (81 input connections)`
                : `Output Neuron ${selectedWeightBox.index} Weights (24 hidden connections)`}
            </h2>
            <Button
              onClick={() => {
                setSelectedWeightBox(null);
                setTourWeightVisualizationOpened(true);
                tourWeightVisualizationOpenedRef.current = true;
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
                <span className="ml-2 text-sm">
                  {weightDialogIteration + 1} / {trainingHistory.length}
                </span>
              </>
            ) : (
              <span className="ml-2 text-sm text-gray-500">
                Training iteration information is not available for models loaded by checkpoint
              </span>
            )}
          </div>

          {/* Weight Visualization */}
          <div className="rounded-lg bg-gray-50 p-4">
            {selectedWeightBox.type === "hidden" && (
              <div className="flex flex-col gap-4 sm:flex-row sm:gap-6">
                {/* Left side: Activation Explorer with 9x9 heatmap */}
                <div className="flex-shrink-0">
                  <div className="mb-4">
                    <div className="mb-2 flex items-baseline justify-between">
                      <h3 className="flex items-center text-sm font-semibold">
                        Activation Explorer
                        <HelpIcon k="activationExplorer" />
                      </h3>
                      <div className="font-mono text-xs text-gray-600">
                        {(() => {
                          const z =
                            currentNetworkState.current.hiddenPreActivations?.[
                              selectedWeightBox.index
                            ] ?? 0;
                          const a =
                            currentNetworkState.current.hiddenActivations?.[
                              selectedWeightBox.index
                            ] ?? 0;
                          return `z: ${z.toFixed(3)} \u00A0\u00A0 a: ${a.toFixed(3)}`;
                        })()}
                      </div>
                    </div>

                    {/* Input Overlay Toggle */}
                    <div className="mb-2 flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="input-overlay"
                        checked={showInputOverlay}
                        onChange={(e) => setShowInputOverlay(e.target.checked)}
                        className="rounded"
                      />
                      <label
                        htmlFor="input-overlay"
                        className="cursor-pointer text-xs text-gray-600"
                      >
                        Show input overlay
                      </label>
                    </div>

                    {/* Global Scale Toggle */}
                    <div className="mb-2 flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="global-scale"
                        checked={useGlobalScale}
                        onChange={(e) => setUseGlobalScale(e.target.checked)}
                        className="rounded"
                      />
                      <label
                        htmlFor="global-scale"
                        className="cursor-pointer text-xs text-gray-600"
                      >
                        Use global scale
                      </label>
                    </div>

                    {/* Color Scheme Selector */}
                    <div className="mb-3">
                      <label className="mb-1 block text-xs text-gray-600">Color scheme:</label>
                      <select
                        value={colorScheme}
                        onChange={(e) => setColorScheme(e.target.value as any)}
                        className="rounded border border-gray-300 bg-white px-2 py-1 text-xs"
                      >
                        <option value="blue-red">Blue/Red (default)</option>
                        <option value="blue-orange">Blue/Orange</option>
                        <option value="green-purple">Green/Purple</option>
                        <option value="high-contrast">High contrast</option>
                      </select>
                      <div className="mt-2 rounded bg-gray-50 p-2 text-xs text-gray-600">
                        {getColorSchemeDescription(colorScheme)}
                      </div>
                    </div>

                    {/* Weight template as heatmap */}
                    <div className="mb-3">
                      {(() => {
                        const w81 =
                          trainingHistory[weightDialogIteration]?.weights?.[
                            selectedWeightBox.index
                          ] ?? weights[selectedWeightBox.index];
                        const grid = vec81ToGrid9(w81);
                        const inputGrid = showInputOverlay ? pixelGrid : null;

                        let globalMaxAbs = null;
                        if (useGlobalScale) {
                          const allWeights =
                            trainingHistory[weightDialogIteration]?.weights ?? weights;
                          globalMaxAbs = allWeights.reduce(
                            (max: number, neuronWeights: number[]) => {
                              const neuronMax = neuronWeights.reduce(
                                (m: number, w: number) => Math.max(m, Math.abs(w)),
                                0,
                              );
                              return Math.max(max, neuronMax);
                            },
                            0,
                          );
                        }

                        return (
                          <Heatmap9x9
                            grid={grid}
                            showInputOverlay={showInputOverlay}
                            inputGrid={inputGrid}
                            globalMaxAbs={globalMaxAbs}
                            colorScheme={colorScheme}
                          />
                        );
                      })()}
                    </div>

                    <p className="max-w-[200px] text-xs text-gray-600">
                      The colors visualize this neuron's input weights (blue=positive,
                      red=negative). Blue pixels push activation up when the corresponding input
                      pixel is on; red pulls it down. Scrub the "Training Iteration" slider above to
                      watch this template evolve.
                    </p>
                  </div>
                </div>

                {/* Right side: Weight bars (fixed spacing and bias visibility) */}
                <div className="min-w-0 flex-grow">
                  <h3 className="mb-2 text-sm font-semibold">Weight Details</h3>
                  <div className="h-[300px] overflow-auto sm:h-[400px]">
                    <svg viewBox="0 0 600 1350" className="w-full" style={{ minHeight: "1350px" }}>
                      {/* Background */}
                      <rect
                        x="50"
                        y="30"
                        width="500"
                        height="1300"
                        fill="white"
                        stroke="#9CA3AF"
                        strokeWidth="2"
                      />
                      <line
                        x1="300"
                        y1="30"
                        x2="300"
                        y2="1330"
                        stroke="#666"
                        strokeWidth="2"
                        opacity="0.5"
                      />

                      {/* Weight bars - reduced spacing from 22px to 15px */}
                      {(
                        trainingHistory[weightDialogIteration]?.weights[selectedWeightBox.index] ||
                        weights[selectedWeightBox.index]
                      ).map((weight: number, i: number) => {
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
                              fill={getBarColor(weight, colorScheme)}
                              opacity="0.8"
                            />
                            <text x="20" y={barY + 8} fontSize="8" fill="#666">
                              Input {i + 1}:
                            </text>
                            <text
                              x={weight >= 0 ? barX + barWidth + 5 : barX - 5}
                              y={barY + 8}
                              fontSize="8"
                              fill="#333"
                              textAnchor={weight >= 0 ? "start" : "end"}
                            >
                              {weight.toFixed(3)}
                            </text>
                          </g>
                        );
                      })}

                      {/* Bias visualization - fixed positioning */}
                      {(() => {
                        const bias =
                          (trainingHistory[weightDialogIteration]?.biases &&
                            trainingHistory[weightDialogIteration]?.biases[
                              selectedWeightBox.index
                            ]) ||
                          biases[selectedWeightBox.index];
                        const biasY = 45 + 81 * 15 + 10;
                        const biasWidth = Math.abs(bias) * 250;
                        const biasX = bias >= 0 ? 300 : 300 - biasWidth;
                        return (
                          <g>
                            <rect
                              x={biasX}
                              y={biasY}
                              width={biasWidth}
                              height="12"
                              fill={getBarColor(bias, colorScheme)}
                              opacity="0.8"
                            />
                            <text x="20" y={biasY + 9} fontSize="10" fill="#666" fontWeight="bold">
                              Bias:
                            </text>
                            <text
                              x={bias >= 0 ? biasX + biasWidth + 5 : biasX - 5}
                              y={biasY + 9}
                              fontSize="10"
                              fill="#333"
                              textAnchor={bias >= 0 ? "start" : "end"}
                              fontWeight="bold"
                            >
                              {bias.toFixed(3)}
                            </text>
                          </g>
                        );
                      })()}

                      {/* Labels */}
                      <text x="55" y="1325" fontSize="12" fill="#666">
                        -1
                      </text>
                      <text x="295" y="1325" fontSize="12" fill="#666">
                        0
                      </text>
                      <text x="535" y="1325" fontSize="12" fill="#666">
                        +1
                      </text>
                    </svg>
                  </div>
                </div>
              </div>
            )}

            {selectedWeightBox.type === "output" && (
              <div>
                <div className="flex flex-col gap-4 sm:flex-row sm:gap-6">
                  {/* Left side: Top Contributors with mini thumbnails */}
                  <div className="flex-shrink-0 sm:w-1/3">
                    <div className="mb-4 flex items-baseline justify-between">
                      <h3 className="text-sm font-semibold">
                        {viewMode === "decision"
                          ? "Decision Contributors (0 vs 1)"
                          : `Output ${selectedWeightBox.index} — Logit (z₀${selectedWeightBox.index === 0 ? "₀" : "₁"})`}
                      </h3>
                    </div>

                    {(() => {
                      let globalMaxAbs = null;
                      if (useGlobalScale) {
                        const allWeights =
                          trainingHistory[weightDialogIteration]?.weights ?? weights;
                        globalMaxAbs = allWeights.reduce((max: number, neuronWeights: number[]) => {
                          const neuronMax = neuronWeights.reduce(
                            (m: number, w: number) => Math.max(m, Math.abs(w)),
                            0,
                          );
                          return Math.max(max, neuronMax);
                        }, 0);
                      }

                      if (viewMode === "decision") {
                        const hiddenActivs =
                          trainingHistory[weightDialogIteration]?.hiddenActivations ??
                          currentNetworkState.current.hiddenActivations;
                        const outputWeightsData = [
                          trainingHistory[weightDialogIteration]?.outputWeights?.[0] ??
                            outputWeights[0],
                          trainingHistory[weightDialogIteration]?.outputWeights?.[1] ??
                            outputWeights[1],
                        ];

                        const decisionContribs = getDecisionContribs(
                          hiddenActivs,
                          outputWeightsData,
                        );

                        const helpsZero = decisionContribs
                          .filter((c) => c.contrib > 0)
                          .sort((a, b) => b.contrib - a.contrib)
                          .slice(0, 6);

                        const helpsOne = decisionContribs
                          .filter((c) => c.contrib < 0)
                          .sort((a, b) => a.contrib - b.contrib)
                          .slice(0, 6);

                        const renderDecisionGrid = (
                          contributors: DecisionContrib[],
                          title: string,
                          description: string,
                        ) => (
                          <div className="mb-6">
                            <div className="mb-3">
                              <h4 className="mb-1 text-sm font-medium text-gray-800">{title}</h4>
                              <p className="text-xs text-gray-600">{description}</p>
                            </div>
                            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                              {contributors.map(({ idx, contrib, w0, w1, h }) => {
                                const w81 =
                                  trainingHistory[weightDialogIteration]?.weights?.[idx] ??
                                  weights[idx];
                                const grid = vec81ToGrid9(w81);
                                return (
                                  <div
                                    key={idx}
                                    className="w-fit cursor-pointer rounded border bg-white p-2 transition-colors hover:bg-gray-50"
                                    onClick={() =>
                                      setSelectedWeightBox({ type: "hidden", index: idx })
                                    }
                                    title="Click to view detailed analysis of this hidden neuron"
                                  >
                                    <div className="mb-1 text-xs">
                                      <span className="font-medium">Hidden {idx + 1}</span>
                                    </div>
                                    <Heatmap9x9
                                      grid={grid}
                                      cell={12}
                                      showInputOverlay={showInputOverlay}
                                      inputGrid={showInputOverlay ? pixelGrid : null}
                                      globalMaxAbs={globalMaxAbs}
                                      colorScheme={colorScheme}
                                    />
                                    <div className="mt-1 text-center text-xs">
                                      <div className="font-mono font-medium text-blue-600">
                                        Contrib: {contrib.toFixed(3)}
                                      </div>
                                      <div className="text-xs text-gray-500">
                                        h={h.toFixed(2)}, w₀-w₁={(w0 - w1).toFixed(3)}
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
                            {helpsZero.length > 0 &&
                              renderDecisionGrid(
                                helpsZero,
                                "Helps classify as 0",
                                "These patterns push the decision toward digit 0 (positive contributions)",
                              )}

                            {helpsOne.length > 0 &&
                              renderDecisionGrid(
                                helpsOne,
                                "Helps classify as 1",
                                "These patterns push the decision toward digit 1 (negative contributions)",
                              )}

                            {(() => {
                              const b0 =
                                trainingHistory[weightDialogIteration]?.outputBiases?.[0] ??
                                outputBiases[0];
                              const b1 =
                                trainingHistory[weightDialogIteration]?.outputBiases?.[1] ??
                                outputBiases[1];
                              const biasDelta = b0 - b1;
                              return (
                                <div className="mb-6 border-t pt-4">
                                  <div className="mb-3">
                                    <h4 className="mb-1 text-sm font-medium text-gray-800">
                                      Bias Contribution
                                    </h4>
                                    <p className="text-xs text-gray-600">
                                      How much the bias terms contribute to the decision
                                    </p>
                                  </div>
                                  <div className="rounded bg-gray-50 p-3">
                                    <div className="text-sm">
                                      <span className="font-medium">Decision bias (b₀ - b₁): </span>
                                      <span
                                        className={`font-mono ${biasDelta >= 0 ? "text-blue-600" : "text-red-600"}`}
                                      >
                                        {biasDelta.toFixed(3)}
                                      </span>
                                    </div>
                                    <div className="mt-1 text-xs text-gray-600">
                                      b₀={b0.toFixed(3)}, b₁={b1.toFixed(3)}
                                    </div>
                                  </div>
                                </div>
                              );
                            })()}
                          </div>
                        );
                      } else {
                        const k = selectedWeightBox.index;
                        const ow =
                          trainingHistory[weightDialogIteration]?.outputWeights?.[k] ??
                          outputWeights[k];

                        const positiveWeights = ow
                          .map((w: number, i: number) => ({ i, w }))
                          .filter(({ w }: { w: number }) => w > 0)
                          .sort((a: { w: number }, b: { w: number }) => b.w - a.w)
                          .slice(0, 6);

                        const negativeWeights = ow
                          .map((w: number, i: number) => ({ i, w }))
                          .filter(({ w }: { w: number }) => w < 0)
                          .sort((a: { w: number }, b: { w: number }) => a.w - b.w)
                          .slice(0, 6);

                        const renderContributorGrid = (
                          contributors: Array<{ i: number; w: number }>,
                          title: string,
                          description: string,
                        ) => (
                          <div className="mb-6">
                            <div className="mb-3">
                              <h4 className="mb-1 text-sm font-medium text-gray-800">{title}</h4>
                              <p className="text-xs text-gray-600">{description}</p>
                            </div>
                            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                              {contributors.map(({ i, w }) => {
                                const w81 =
                                  trainingHistory[weightDialogIteration]?.weights?.[i] ??
                                  weights[i];
                                const grid = vec81ToGrid9(w81);
                                return (
                                  <div
                                    key={i}
                                    className="w-fit cursor-pointer rounded border bg-white p-2 transition-colors hover:bg-gray-50"
                                    onClick={() =>
                                      setSelectedWeightBox({ type: "hidden", index: i })
                                    }
                                    title="Click to view detailed analysis of this hidden neuron"
                                  >
                                    <div className="mb-1 text-xs">
                                      <span className="font-medium">Hidden {i + 1}</span>
                                    </div>
                                    <Heatmap9x9
                                      grid={grid}
                                      cell={12}
                                      showInputOverlay={showInputOverlay}
                                      inputGrid={showInputOverlay ? pixelGrid : null}
                                      globalMaxAbs={globalMaxAbs}
                                      colorScheme={colorScheme}
                                    />
                                    <div className="mt-1 text-center text-xs text-gray-600">
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
                            {positiveWeights.length > 0 &&
                              renderContributorGrid(
                                positiveWeights,
                                "Excitatory Contributors",
                                `Patterns consisting of strongly positive values (${getPositiveColorName(colorScheme)}) make the neuron more likely to fire`,
                              )}

                            {negativeWeights.length > 0 &&
                              renderContributorGrid(
                                negativeWeights,
                                "Inhibitory Contributors",
                                `Patterns consisting of strongly positive values (${getPositiveColorName(colorScheme)}) make the neuron less likely to fire`,
                              )}
                          </div>
                        );
                      }
                    })()}

                    {/* Controls moved below both classes */}
                    <div className="mt-4 border-t pt-4">
                      {/* View Mode Toggle */}
                      <div className="mb-2">
                        <h4 className="mb-2 flex items-center text-sm font-medium text-gray-700">
                          View Mode
                          <HelpIcon k="decisionVsLogit" />
                        </h4>
                      </div>
                      <div className="mb-4 rounded-lg bg-gray-50 p-3">
                        <div className="mb-2 flex flex-wrap items-center gap-2 sm:gap-4">
                          <label className="flex items-center gap-2">
                            <input
                              type="radio"
                              name="viewMode"
                              value="logit"
                              checked={viewMode === "logit"}
                              onChange={(e) => setViewMode("logit")}
                              className="text-blue-600"
                            />
                            <span className="text-sm font-medium">
                              Output score (before probability)
                            </span>
                          </label>
                          <label className="flex items-center gap-2">
                            <input
                              type="radio"
                              name="viewMode"
                              value="decision"
                              checked={viewMode === "decision"}
                              onChange={(e) => setViewMode("decision")}
                              className="text-blue-600"
                            />
                            <span className="text-sm font-medium">Decision (which class wins)</span>
                          </label>
                        </div>
                        <div className="text-xs text-gray-600">
                          {viewMode === "decision"
                            ? "Shows contributions to z₀−z₁, which controls the 0-vs-1 choice"
                            : "Shows contributions to z_k = Σ w_{j→k}h_j + b_k before probabilities"}
                        </div>
                      </div>

                      {/* Input Overlay Toggle for Output View */}
                      <div className="mb-2 flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="output-input-overlay"
                          checked={showInputOverlay}
                          onChange={(e) => setShowInputOverlay(e.target.checked)}
                          className="rounded"
                        />
                        <label
                          htmlFor="output-input-overlay"
                          className="cursor-pointer text-xs text-gray-600"
                        >
                          Show input overlay
                        </label>
                      </div>

                      {/* Global Scale Toggle for Output View */}
                      <div className="mb-2 flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="output-global-scale"
                          checked={useGlobalScale}
                          onChange={(e) => setUseGlobalScale(e.target.checked)}
                          className="rounded"
                        />
                        <label
                          htmlFor="output-global-scale"
                          className="cursor-pointer text-xs text-gray-600"
                        >
                          Use global scale
                        </label>
                      </div>

                      {/* Color Scheme Selector for Output View */}
                      <div className="mb-3">
                        <label className="mb-1 block text-xs text-gray-600">Color scheme:</label>
                        <select
                          value={colorScheme}
                          onChange={(e) => setColorScheme(e.target.value as any)}
                          className="rounded border border-gray-300 bg-white px-2 py-1 text-xs"
                        >
                          <option value="blue-red">Blue/Red (default)</option>
                          <option value="blue-orange">Blue/Orange</option>
                          <option value="green-purple">Green/Purple</option>
                          <option value="high-contrast">High contrast</option>
                        </select>
                        <div className="mt-2 rounded bg-gray-50 p-2 text-xs text-gray-600">
                          {getColorSchemeDescription(colorScheme)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right side: Weight Details */}
                  <div className="flex-grow">
                    <h3 className="mb-4 text-sm font-semibold">Weight Details</h3>
                    <svg width="100%" height="500" viewBox="0 0 600 500">
                      <g>
                        {/* Large weight box */}
                        <rect
                          x="50"
                          y="30"
                          width="500"
                          height="460"
                          fill="white"
                          stroke="#9CA3AF"
                          strokeWidth="2"
                        />
                        <line
                          x1="300"
                          y1="30"
                          x2="300"
                          y2="490"
                          stroke="#666"
                          strokeWidth="2"
                          opacity="0.5"
                        />

                        {/* Weight bars - reduced spacing from 22px to 16px */}
                        {(
                          trainingHistory[weightDialogIteration]?.outputWeights[
                            selectedWeightBox.index
                          ] || outputWeights[selectedWeightBox.index]
                        ).map((weight: number, i: number) => {
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
                                fill={getBarColor(weight, colorScheme)}
                                opacity="0.8"
                              />
                              <text x="20" y={barY + 9} fontSize="10" fill="#666">
                                Hidden {i + 1}:
                              </text>
                              <text
                                x={weight >= 0 ? barX + barWidth + 5 : barX - 5}
                                y={barY + 9}
                                fontSize="10"
                                fill="#333"
                                textAnchor={weight >= 0 ? "start" : "end"}
                              >
                                {weight.toFixed(3)}
                              </text>
                            </g>
                          );
                        })}

                        {/* Bias visualization - fixed positioning */}
                        {(() => {
                          const bias =
                            (trainingHistory[weightDialogIteration]?.outputBiases &&
                              trainingHistory[weightDialogIteration]?.outputBiases[
                                selectedWeightBox.index
                              ]) ||
                            outputBiases[selectedWeightBox.index];
                          const biasY = 50 + 24 * 16 + 10;
                          const biasWidth = Math.abs(bias) * 250;
                          const biasX = bias >= 0 ? 300 : 300 - biasWidth;
                          return (
                            <g>
                              <rect
                                x={biasX}
                                y={biasY}
                                width={biasWidth}
                                height="14"
                                fill={getBarColor(bias, colorScheme)}
                                opacity="0.8"
                              />
                              <text
                                x="20"
                                y={biasY + 10}
                                fontSize="11"
                                fill="#666"
                                fontWeight="bold"
                              >
                                Bias:
                              </text>
                              <text
                                x={bias >= 0 ? biasX + biasWidth + 5 : biasX - 5}
                                y={biasY + 10}
                                fontSize="11"
                                fill="#333"
                                textAnchor={bias >= 0 ? "start" : "end"}
                                fontWeight="bold"
                              >
                                {bias.toFixed(3)}
                              </text>
                            </g>
                          );
                        })()}

                        {/* Labels */}
                        <text x="55" y="485" fontSize="12" fill="#666">
                          -1
                        </text>
                        <text x="295" y="485" fontSize="12" fill="#666">
                          0
                        </text>
                        <text x="535" y="485" fontSize="12" fill="#666">
                          +1
                        </text>
                      </g>
                    </svg>

                    <p className="mt-4 max-w-[500px] text-xs text-gray-600 sm:ml-20">
                      Bars show connection strength from each hidden unit to this output. Thumbnails
                      show each hidden unit's input template. Together they explain how mid-level
                      patterns combine to vote for "0" or "1". Click a thumbnail to view that hidden
                      neuron's detailed analysis.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
