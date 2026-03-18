import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export interface DebugHistoryEntry {
  iteration: number;
  label: number[];
  outputActivations: number[];
  outputErrors: number[];
  outputBiases: number[];
  loss: number;
  step: number;
  timestamp: Date;
}

export interface DebugHistoryPanelProps {
  debugHistory: DebugHistoryEntry[];
  setShowDebugDialog: (v: boolean) => void;
}

export function DebugHistoryPanel({ debugHistory, setShowDebugDialog }: DebugHistoryPanelProps) {
  return (
    <div className="mt-3 sm:mt-6">
      <Card>
        <CardContent className="p-3 sm:p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Debug History ({debugHistory.length} entries)</h2>
            <Button onClick={() => setShowDebugDialog(false)} variant="outline" size="sm">
              ×
            </Button>
          </div>

          {debugHistory.length === 0 ? (
            <div className="p-4 text-center text-sm italic text-gray-600">
              No debug data captured yet. Run some training steps to see debug information.
            </div>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-gray-200">
              {/* Fixed Table Headers */}
              <div className="min-w-[600px] border-b border-gray-200 bg-gray-50">
                <div className="grid grid-cols-8 gap-2 p-3 text-xs font-medium text-gray-700">
                  <div className="text-center">Iteration #</div>
                  <div className="text-center">Time</div>
                  <div className="text-center">Loss</div>
                  <div className="text-center">Output Activations</div>
                  <div className="text-center">Output Errors</div>
                  <div className="text-center">Output Biases</div>
                  <div className="text-center">Step #</div>
                  <div className="text-center">Label</div>
                </div>
              </div>

              {/* Scrollable Data Rows */}
              <div className="max-h-96 min-w-[600px] overflow-y-auto">
                {debugHistory.map((entry, index) => (
                  <div
                    key={index}
                    className={`grid grid-cols-8 gap-2 border-b border-gray-100 p-3 text-xs ${
                      index % 2 === 0 ? "bg-white" : "bg-gray-50"
                    }`}
                  >
                    <div className="text-center font-mono">{entry.iteration}</div>
                    <div className="text-center font-mono text-xs">
                      {entry.timestamp.toLocaleTimeString()}
                    </div>
                    <div className="text-center font-mono">{entry.loss.toFixed(6)}</div>
                    <div className="text-center font-mono text-xs">
                      [{entry.outputActivations.map((a: number) => a.toFixed(3)).join(", ")}]
                    </div>
                    <div className="text-center font-mono text-xs">
                      [{entry.outputErrors.map((e: number) => e.toFixed(3)).join(", ")}]
                    </div>
                    <div className="text-center font-mono text-xs">
                      [{entry.outputBiases.map((b: number) => b.toFixed(3)).join(", ")}]
                    </div>
                    <div className="text-center font-mono">{entry.step}</div>
                    <div className="text-center font-mono">
                      [{Array.isArray(entry.label) ? entry.label.join(",") : entry.label}]
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
