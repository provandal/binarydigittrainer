import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { HelpIcon } from "@/components/HelpIcon";
import { flatToGrid } from "@/lib/nn-helpers";

export interface DrawingCanvasProps {
  pixelGrid: number[][];
  canvasRef: React.RefObject<HTMLDivElement>;
  handleMouseUp: () => void;
  handleTouchStart: (e: React.TouchEvent) => void;
  handleTouchMove: (e: React.TouchEvent) => void;
  handleTouchEnd: (e: React.TouchEvent) => void;
  handleMouseDown: (row: number, col: number) => void;
  handleMouseEnter: (row: number, col: number) => void;
  clearCanvas: () => void;
  mode: "training" | "inference";
  selectedLabel: number;
  setSelectedLabel: (label: number) => void;
  setMode: (mode: "training" | "inference") => void;
  modeRef: React.MutableRefObject<string>;
  setStep: (step: number) => void;
  setPixelGrid: (grid: number[][]) => void;
  setTourInferenceModeEnabled: (v: boolean) => void;
  tourInferenceModeEnabledRef: React.MutableRefObject<boolean>;
  prediction: { digit: number; confidence: number } | null;
  setPrediction: (p: { digit: number; confidence: number } | null) => void;
  learningRate: number;
  setLearningRate: (rate: number) => void;
  trainingExamplesCount: number;
}

export function DrawingCanvas({
  pixelGrid,
  canvasRef,
  handleMouseUp,
  handleTouchStart,
  handleTouchMove,
  handleTouchEnd,
  handleMouseDown,
  handleMouseEnter,
  clearCanvas,
  mode,
  selectedLabel,
  setSelectedLabel,
  setMode,
  modeRef,
  setStep,
  setPixelGrid,
  setTourInferenceModeEnabled,
  tourInferenceModeEnabledRef,
  prediction,
  setPrediction,
  learningRate,
  setLearningRate,
  trainingExamplesCount,
}: DrawingCanvasProps) {
  return (
    <Card>
      <CardContent className="p-3 sm:p-6">
        <h2 className="mb-2 text-base font-semibold sm:mb-4 sm:text-lg">
          Drawing Canvas (9×9 pixels)
        </h2>
        <div
          ref={canvasRef}
          className="mx-auto mb-4 grid h-48 w-48 touch-none grid-cols-9 gap-0 border-2 border-gray-400 bg-gray-100"
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        >
          {(Array.isArray(pixelGrid[0]) ? pixelGrid : flatToGrid(pixelGrid as any)).map(
            (row, rowIndex) =>
              row.map((pixel, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                  onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                  className={`h-full w-full cursor-crosshair select-none border border-gray-200 transition-colors duration-100 ${
                    pixel ? "bg-gray-800" : "bg-white hover:bg-gray-100"
                  }`}
                />
              )),
          )}
        </div>

        <div className="text-center">
          <p className="mb-2 text-xs text-gray-600">
            Click and drag to draw. Hover over pixels to see values.
          </p>
          <Button onClick={clearCanvas} variant="outline" size="sm" className="text-xs">
            Clear Canvas
          </Button>
        </div>

        <div className="space-y-3">
          {/* Target Label - Only show in training mode */}
          {mode === "training" && (
            <div>
              <h3 className="mb-2 text-sm font-medium text-gray-700">Target Label</h3>
              <div className="flex justify-center gap-2" data-tour-target="label-selector">
                {[0, 1].map((label) => (
                  <label key={label} className="flex cursor-pointer items-center gap-2">
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
            <h3 className="mb-2 flex items-center text-sm font-medium text-gray-700">
              Mode
              <HelpIcon k="mode" />
            </h3>
            <div className="flex justify-center gap-2">
              {[
                { value: "training", label: "Training" },
                { value: "inference", label: "Inference" },
              ].map((modeOption) => (
                <label key={modeOption.value} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    name="mode"
                    value={modeOption.value}
                    checked={mode === modeOption.value}
                    onChange={() => {
                      setMode(modeOption.value as "training" | "inference");
                      modeRef.current = modeOption.value as "training" | "inference";
                      if (modeOption.value === "inference") {
                        setStep(0);
                        setPrediction(null);
                        setPixelGrid(
                          Array(9)
                            .fill(0)
                            .map(() => Array(9).fill(0)),
                        );
                        setTourInferenceModeEnabled(true);
                        tourInferenceModeEnabledRef.current = true;
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
          {mode === "inference" && prediction && (
            <div
              className="rounded-lg border border-green-200 bg-green-50 p-3"
              data-tour-target="prediction-display"
            >
              <h4 className="mb-1 text-sm font-medium text-green-800">Prediction</h4>
              <div className="text-2xl font-bold text-green-700">Digit: {prediction.digit}</div>
              <div className="text-xs text-green-600">
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>

        {/* Network Info */}
        <div className="mt-6">
          <h3 className="mb-2 text-sm font-medium text-gray-700">Network Info</h3>
          <div className="space-y-2 text-xs text-gray-600">
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
                className="w-16 rounded border border-gray-300 px-1 py-0.5 text-xs focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>Architecture: 81 → 24 → 2</div>
            <div>Hidden: Sigmoid, Output: Softmax</div>
            <div>Loss: Cross-Entropy</div>
            <div>Dataset: {trainingExamplesCount} examples</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
