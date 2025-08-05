import { Brain } from "lucide-react";
import BinaryGrid from "./binary-grid";
import NetworkVisualization from "./network-visualization";
import TrainingControls from "./training-controls";
import NetworkMetrics from "./network-metrics";
import PredictionResults from "./prediction-results";
import TrainingProgress from "./training-progress";
import TestInterface from "./test-interface";
import { useNeuralNetwork } from "@/hooks/use-neural-network";

export default function BinaryDigitTrainer() {
  const {
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
  } = useNeuralNetwork();

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
                <Brain className="text-white text-lg w-5 h-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Binary Digit Trainer</h1>
                <p className="text-sm text-slate-600">Neural Network Simulator</p>
              </div>
            </div>
            <div className="hidden sm:flex items-center space-x-4">
              <div className="text-sm text-slate-600">
                <span className="font-medium">Status:</span>
                <span className={`ml-1 font-medium ${isTraining ? 'text-accent' : 'text-secondary'}`}>
                  {isTraining ? 'Training' : 'Ready'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Panel: Binary Input & Controls */}
          <div className="lg:col-span-4 space-y-6">
            <BinaryGrid 
              grid={binaryGrid} 
              onGridChange={setBinaryGrid}
              onGenerateRandom={generateRandomGrid}
            />
            <TrainingControls
              targetLabel={targetLabel}
              onTargetLabelChange={setTargetLabel}
              isTraining={isTraining}
              onStartTraining={startTraining}
              onStopTraining={stopTraining}
              onResetNetwork={resetNetwork}
            />
            <NetworkMetrics metrics={metrics} />
          </div>

          {/* Center Panel: Neural Network Visualization */}
          <div className="lg:col-span-5 space-y-6">
            <NetworkVisualization 
              isTraining={isTraining}
              activations={metrics.activations}
            />
            <PredictionResults predictions={predictions} />
          </div>

          {/* Right Panel: Training Progress & Testing */}
          <div className="lg:col-span-3 space-y-6">
            <TrainingProgress 
              trainingHistory={trainingHistory}
              metrics={metrics}
            />
            <TestInterface
              onTestCurrent={testCurrentInput}
              onLoadSample={loadSampleDigit}
              testResults={testResults}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
