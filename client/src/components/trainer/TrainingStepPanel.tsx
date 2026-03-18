import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Edit3, Upload, ChevronDown, ChevronRight, Save, FolderOpen } from "lucide-react";
import { HelpIcon } from "@/components/HelpIcon";
import { STEP_DESCRIPTIONS } from "@/lib/nn-helpers";
import { MINI_TUTORIALS } from "@/lib/tutorials";
import type { TrainingExample } from "@shared/schema";

export interface TrainingStepPanelProps {
  // Training mode
  trainingMode: "manual" | "dataset";
  setTrainingMode: React.Dispatch<React.SetStateAction<"manual" | "dataset">>;
  setTrainingCompleted: (v: boolean) => void;
  setTourDatasetLoaded: (v: boolean) => void;
  datasetLoadedRef: React.MutableRefObject<boolean>;
  tourTriggerRef: React.MutableRefObject<(() => void) | null>;

  // Step info
  step: number;
  setStep: (v: number) => void;
  isAutoTraining: boolean;
  trainingCompleted: boolean;
  numberOfEpochs: number;
  currentEpoch: number;
  currentExampleIndex: number;
  trainingExamples: TrainingExample[];
  epochLossHistory: Array<{ epoch: number; averageLoss: number }>;

  // Navigation
  nextStep: () => void;
  setTourStepExecuted: (v: boolean) => void;
  resetNetwork: () => void;
  setShowDatasetEditor: (v: boolean) => void;
  handleFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;

  // Model management
  showModelManagement: boolean;
  setShowModelManagement: (v: boolean) => void;
  showModelManagementRef: React.MutableRefObject<boolean>;
  handleExportCheckpoint: () => void;
  handleImportCheckpointFile: (event: React.ChangeEvent<HTMLInputElement>) => void;
  lrDecayEnabled: boolean;
  setLrDecayEnabled: (v: boolean) => void;
  lrDecayRate: number;
  setLrDecayRate: (v: number) => void;
  minLR: number;
  setMinLR: (v: number) => void;
  lrHistory: Array<{ epoch: number; learningRate: number }>;
  completedEpochs: number;
  examplesSeen: number;
  learningRate: number;
  lastEpochAvgLoss: number | null;
  normalizeEnabled: boolean;
  targetSize: number;
  lastCheckpointLoaded: string | null;

  // Automated training
  mode: "training" | "inference";
  autoTrainingSpeed: number;
  runToNextSample: () => void;
  processTrainingSet: () => void;
  stopTraining: () => void;
  setCurrentExampleIndex: (v: number) => void;
}

export function TrainingStepPanel({
  trainingMode,
  setTrainingMode,
  setTrainingCompleted,
  setTourDatasetLoaded,
  datasetLoadedRef,
  tourTriggerRef,
  step,
  setStep,
  isAutoTraining,
  trainingCompleted,
  numberOfEpochs,
  currentEpoch,
  currentExampleIndex,
  trainingExamples,
  epochLossHistory,
  nextStep,
  setTourStepExecuted,
  resetNetwork,
  setShowDatasetEditor,
  handleFileUpload,
  showModelManagement,
  setShowModelManagement,
  showModelManagementRef,
  handleExportCheckpoint,
  handleImportCheckpointFile,
  lrDecayEnabled,
  setLrDecayEnabled,
  lrDecayRate,
  setLrDecayRate,
  minLR,
  setMinLR,
  lrHistory,
  completedEpochs,
  examplesSeen,
  learningRate,
  lastEpochAvgLoss,
  normalizeEnabled,
  targetSize,
  lastCheckpointLoaded,
  mode,
  autoTrainingSpeed,
  runToNextSample,
  processTrainingSet,
  stopTraining,
  setCurrentExampleIndex,
}: TrainingStepPanelProps) {
  return (
    <Card>
      <CardContent className="p-3 sm:p-6">
        <h2 className="mb-2 text-base font-semibold sm:mb-4 sm:text-lg">Training Steps</h2>

        {/* Training Mode Toggle */}
        <div className="mb-2">
          <h3 className="mb-2 flex items-center text-sm font-medium text-gray-700">
            Training Mode
            <HelpIcon k="trainingMode" />
          </h3>
        </div>
        <div className="mb-4 flex gap-2">
          <Button
            onClick={() => {
              setTrainingMode("manual");
              setTrainingCompleted(false);
            }}
            variant={trainingMode === "manual" ? "default" : "outline"}
            size="sm"
            className="flex-1"
          >
            Manual Draw
          </Button>
          <Button
            onClick={() => {
              setTrainingMode("dataset");
              setTrainingCompleted(false);
              setTourDatasetLoaded(true);
              datasetLoadedRef.current = true;
              console.log("🎯 TOUR: Training Set clicked! Setting both state and ref to true");
              setTimeout(() => {
                if (tourTriggerRef.current) {
                  console.log("🔔 TOUR: Triggering validation check after Training Set click");
                  tourTriggerRef.current();
                }
              }, 100);
            }}
            variant={trainingMode === "dataset" ? "default" : "outline"}
            size="sm"
            className="flex-1"
            data-tour-target="dataset-button"
          >
            Training Set
          </Button>
        </div>

        {/* Current Step Info - Show detailed info only when not auto-training and not completed */}
        {!isAutoTraining && !trainingCompleted ? (
          <div className="training-step-display mb-4 rounded-lg bg-blue-50 p-4">
            <div className="mb-2 flex items-center text-sm font-medium text-blue-900">
              Step {step + 1} of 6:{" "}
              {STEP_DESCRIPTIONS[step] ? STEP_DESCRIPTIONS[step].name : "Ready"}
              {step < 6 && <HelpIcon k={`step${step + 1}` as keyof typeof MINI_TUTORIALS} />}
            </div>

            {/* Concept Explanation */}
            <div className="mb-3 text-sm text-blue-800">
              <strong>Concept:</strong>{" "}
              {STEP_DESCRIPTIONS[step]
                ? STEP_DESCRIPTIONS[step].concept
                : "Ready to begin training"}
            </div>

            {/* Mathematical Formula */}
            <div className="rounded bg-blue-100 p-2 font-mono text-xs text-blue-700">
              <strong>Formula:</strong>{" "}
              {STEP_DESCRIPTIONS[step]
                ? STEP_DESCRIPTIONS[step].formula
                : "Click Next Step to begin"}
            </div>
          </div>
        ) : isAutoTraining || trainingCompleted ? (
          <div className="mb-4 rounded-lg bg-purple-50 p-4">
            <div className="mb-2 text-sm font-medium text-purple-900">
              {isAutoTraining ? "Automated Training in Progress" : "Training Complete"}
            </div>
            <div className="mb-2 text-sm text-purple-800">
              {isAutoTraining
                ? numberOfEpochs > 1
                  ? `Epoch ${currentEpoch} of ${numberOfEpochs}`
                  : "Processing training examples automatically"
                : `Completed ${numberOfEpochs} epoch(s) with ${trainingExamples.length} samples`}
            </div>

            {/* Epoch Progress Bar (only show if multiple epochs) */}
            {numberOfEpochs > 1 && (
              <div className="mb-3">
                <div className="h-1.5 w-full rounded-full bg-purple-300">
                  <div
                    className="h-1.5 rounded-full bg-purple-700 transition-all duration-300"
                    style={{ width: `${(currentEpoch / numberOfEpochs) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* Sample Progress Bar */}
            <div className="h-2 w-full rounded-full bg-purple-200">
              <div
                className="h-2 rounded-full bg-purple-600 transition-all duration-300"
                style={{
                  width: `${trainingExamples.length > 0 ? ((currentExampleIndex + 1) / trainingExamples.length) * 100 : 0}%`,
                }}
              ></div>
            </div>
            <div className="mt-2 text-center text-xs text-purple-700">
              Sample {currentExampleIndex + 1} of {trainingExamples.length}
              {numberOfEpochs > 1 && ` • Epoch ${currentEpoch}/${numberOfEpochs}`}
            </div>

            {/* Epoch Loss History */}
            {epochLossHistory.length > 0 && (
              <div className="mt-3 border-t border-purple-200 pt-3">
                <div className="mb-2 text-xs font-medium text-purple-900">Learning Progress</div>
                <div className="max-h-20 space-y-1 overflow-y-auto">
                  {epochLossHistory.slice(-5).map((epochData, index) => (
                    <div
                      key={epochData.epoch}
                      className="flex justify-between text-xs text-purple-800"
                    >
                      <span>Epoch {epochData.epoch}:</span>
                      <span className="font-mono">{epochData.averageLoss.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
                {epochLossHistory.length > 1 && (
                  <div className="mt-2 text-center text-xs text-purple-700">
                    {epochLossHistory[epochLossHistory.length - 1].averageLoss <
                    epochLossHistory[0].averageLoss
                      ? "📉 Loss decreasing!"
                      : "📈 Loss trend varies"}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : null}

        {/* Navigation Controls */}
        <div className="mb-4 space-y-2">
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
              onClick={() => {
                nextStep();
                setTourStepExecuted(true);
              }}
              className="flex-1 bg-blue-600 text-white hover:bg-blue-700"
              size="sm"
              data-tour-target="next-step-button"
            >
              Next Step →
            </Button>
          </div>

          <Button onClick={resetNetwork} variant="outline" className="w-full" size="sm">
            Reset Network
          </Button>

          <Button
            onClick={() => setShowDatasetEditor(true)}
            variant="outline"
            className="w-full"
            size="sm"
          >
            <Edit3 className="mr-2 h-4 w-4" />
            Edit Training Set
          </Button>

          <div className="relative">
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
              id="bulk-upload-input"
            />
            <Button variant="outline" className="w-full" size="sm">
              <Upload className="mr-2 h-4 w-4" />
              Upload Training Set
            </Button>
          </div>
        </div>

        {/* Model Management Section */}
        <div className="mt-4">
          <Button
            onClick={() => {
              const next = !showModelManagement;
              setShowModelManagement(next);
              showModelManagementRef.current = next;
            }}
            variant="ghost"
            size="sm"
            className="h-auto w-full justify-between p-2"
            data-tour-target="model-management-toggle"
          >
            <span className="text-sm font-medium text-gray-700">Model Management</span>
            {showModelManagement ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>

          {showModelManagement && (
            <div className="mt-2 space-y-3 rounded-lg bg-gray-50 p-3">
              {/* Checkpoint Export/Import */}
              <div className="space-y-2">
                <div className="flex items-center text-xs font-medium uppercase tracking-wide text-gray-600">
                  Checkpoints
                  <HelpIcon k="checkpoints" />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={handleExportCheckpoint}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    data-tour-target="save-checkpoint-button"
                  >
                    <Save className="mr-1 h-3 w-3" />
                    Export
                  </Button>

                  <div className="relative">
                    <input
                      type="file"
                      accept=".json"
                      onChange={handleImportCheckpointFile}
                      className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                    />
                    <Button variant="outline" size="sm" className="w-full text-xs">
                      <FolderOpen className="mr-1 h-3 w-3" />
                      Import
                    </Button>
                  </div>
                </div>
              </div>

              {/* Learning Rate Decay Controls */}
              <div className="space-y-2">
                <div className="text-xs font-medium uppercase tracking-wide text-gray-600">
                  Learning Rate Decay
                </div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="lr-decay-enabled"
                      checked={lrDecayEnabled}
                      onChange={(e) => setLrDecayEnabled(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="lr-decay-enabled" className="text-xs text-gray-700">
                      Enable decay (per epoch)
                    </label>
                  </div>

                  {lrDecayEnabled && (
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="space-y-1">
                        <label className="text-gray-600">Decay rate</label>
                        <input
                          type="number"
                          step="0.001"
                          min="0.90"
                          max="0.999"
                          value={lrDecayRate}
                          onChange={(e) => setLrDecayRate(parseFloat(e.target.value) || 0.99)}
                          className="w-full rounded border px-2 py-1 text-xs"
                        />
                      </div>
                      <div className="space-y-1">
                        <label className="text-gray-600">Min LR</label>
                        <input
                          type="number"
                          step="0.0001"
                          min="0.0001"
                          max="0.1"
                          value={minLR}
                          onChange={(e) => setMinLR(parseFloat(e.target.value) || 0.0005)}
                          className="w-full rounded border px-2 py-1 text-xs"
                        />
                      </div>
                    </div>
                  )}

                  {/* Learning Rate History Chart */}
                  {lrHistory.length > 1 && (
                    <div className="mt-2 rounded border bg-white p-2">
                      <div className="mb-1 text-xs text-gray-600">LR over epochs</div>
                      <div className="flex h-12 items-end gap-0.5">
                        {lrHistory.slice(-10).map((point, i) => {
                          const maxLR = Math.max(...lrHistory.map((p) => p.learningRate));
                          const height = (point.learningRate / maxLR) * 100;
                          return (
                            <div
                              key={i}
                              className="min-w-1 flex-1 bg-blue-200"
                              style={{ height: `${height}%` }}
                              title={`Epoch ${point.epoch}: ${point.learningRate.toFixed(5)}`}
                            ></div>
                          );
                        })}
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        Last 10 epochs (hover for values)
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Model Stats */}
              <div className="space-y-2">
                <div className="text-xs font-medium uppercase tracking-wide text-gray-600">
                  Model Stats
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Architecture:</span>
                    <span className="font-mono">81→24→2</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Epochs:</span>
                    <span className="font-mono">{completedEpochs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Examples:</span>
                    <span className="font-mono">{examplesSeen}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Current LR:</span>
                    <span className="font-mono">{learningRate.toFixed(5)}</span>
                  </div>
                  <div className="col-span-2 flex justify-between">
                    <span className="text-gray-600">Last Loss:</span>
                    <span className="font-mono">
                      {lastEpochAvgLoss !== null ? lastEpochAvgLoss.toFixed(4) : "N/A"}
                    </span>
                  </div>
                  <div className="col-span-2 flex justify-between">
                    <span className="text-gray-600">Normalization:</span>
                    <span className="font-mono">
                      {normalizeEnabled ? `Enabled (${targetSize}x${targetSize})` : "Disabled"}
                    </span>
                  </div>
                  {lrDecayEnabled && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Decay Rate:</span>
                        <span className="font-mono">{lrDecayRate}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Min LR:</span>
                        <span className="font-mono">{minLR}</span>
                      </div>
                    </>
                  )}
                  {lastCheckpointLoaded && (
                    <div className="col-span-2 flex justify-between border-t pt-1">
                      <span className="text-gray-600">Last Loaded:</span>
                      <span
                        className="max-w-24 truncate font-mono text-xs"
                        title={lastCheckpointLoaded}
                      >
                        {lastCheckpointLoaded}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Automated Training Controls - Only in Training Mode */}
        {mode === "training" && trainingMode === "dataset" && trainingExamples.length > 0 && (
          <div className="mt-4 space-y-2">
            <div className="mb-2 text-sm font-medium text-gray-700">Automated Training</div>
            <Button
              onClick={runToNextSample}
              disabled={isAutoTraining}
              size="sm"
              className="w-full bg-green-600 text-white hover:bg-green-700"
              data-tour-target="run-next-sample-button"
            >
              {isAutoTraining ? "Training..." : "Run to Next Sample"}
            </Button>
            <Button
              onClick={processTrainingSet}
              disabled={isAutoTraining}
              size="sm"
              className="w-full bg-purple-600 text-white hover:bg-purple-700"
              data-tour-target="multi-epoch-button"
            >
              {isAutoTraining ? "Processing Set..." : "Process Training Set"}
            </Button>

            {/* Stop Training Button - Only show when training is active */}
            {isAutoTraining && (
              <Button
                onClick={stopTraining}
                size="sm"
                variant="outline"
                className="w-full border-red-500 text-red-600 hover:border-red-600 hover:bg-red-50 hover:text-red-700"
                data-tour-target="stop-training-button"
              >
                🛑 Stop Training
              </Button>
            )}
            <div className="text-center text-xs text-gray-600">
              Sample {currentExampleIndex + 1} of {trainingExamples.length} • Speed:{" "}
              {autoTrainingSpeed}ms
            </div>
          </div>
        )}

        {/* Dataset Info and Navigation */}
        {trainingMode === "dataset" && (
          <div className="mt-4 space-y-3">
            <div className="rounded-lg bg-green-50 p-3">
              <div className="mb-1 text-sm font-medium text-green-900">Training Dataset</div>
              <div className="text-xs text-green-700">
                Example {currentExampleIndex + 1} of {trainingExamples.length} • Target:{" "}
                {Array.isArray(trainingExamples[currentExampleIndex]?.label)
                  ? (trainingExamples[currentExampleIndex]?.label as number[])?.[0] === 1
                    ? "0"
                    : "1"
                  : String(trainingExamples[currentExampleIndex]?.label || 0)}
                <br />
                One-hot: [
                {Array.isArray(trainingExamples[currentExampleIndex]?.label)
                  ? (trainingExamples[currentExampleIndex]?.label as number[]).join(",")
                  : trainingExamples[currentExampleIndex]?.label === 0
                    ? "1,0"
                    : "0,1"}
                ] (Neuron0: digit0, Neuron1: digit1)
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => setCurrentExampleIndex(Math.max(0, currentExampleIndex - 1))}
                disabled={currentExampleIndex === 0}
                variant="outline"
                size="sm"
                className="flex-1"
              >
                ← Prev Example
              </Button>
              <Button
                onClick={() =>
                  setCurrentExampleIndex(
                    Math.min(trainingExamples.length - 1, currentExampleIndex + 1),
                  )
                }
                disabled={currentExampleIndex === trainingExamples.length - 1}
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
  );
}
