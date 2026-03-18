import React from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Trash2, Plus } from "lucide-react";
import { flatToGrid } from "@/lib/nn-helpers";
import type { TrainingExample } from "@shared/schema";

export interface DatasetEditorDialogProps {
  showDatasetEditor: boolean;
  setShowDatasetEditor: (v: boolean) => void;
  trainingExamples: TrainingExample[];
  addDatasetExample: () => void;
  removeDatasetExample: (index: number) => void;
  updateDatasetExample: (index: number, pattern: number[][] | number[], label: number) => void;
  handleEditorMouseDown: (exampleIndex: number, rowIndex: number, colIndex: number) => void;
  handleEditorMouseEnter: (exampleIndex: number, rowIndex: number, colIndex: number) => void;
  handleEditorMouseUp: () => void;
  saveDataset: () => void;
  getPatternPreview: (pattern: number[][] | number[]) => number[];
}

export function DatasetEditorDialog({
  showDatasetEditor,
  setShowDatasetEditor,
  trainingExamples,
  addDatasetExample,
  removeDatasetExample,
  updateDatasetExample,
  handleEditorMouseDown,
  handleEditorMouseEnter,
  handleEditorMouseUp,
  saveDataset,
  getPatternPreview,
}: DatasetEditorDialogProps) {
  return (
    <Dialog open={showDatasetEditor} onOpenChange={setShowDatasetEditor}>
      <DialogContent
        className="max-h-[90vh] max-w-[95vw] overflow-y-auto sm:max-w-4xl"
        onMouseUp={handleEditorMouseUp}
      >
        <DialogHeader>
          <DialogTitle>Edit Training Dataset</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-600">
              {trainingExamples.length} examples total •
              {
                trainingExamples.filter((ex: TrainingExample) => {
                  const label = ex.label as number[];
                  return Array.isArray(label) && label[0] === 1;
                }).length
              }{" "}
              zeros,{" "}
              {
                trainingExamples.filter((ex: TrainingExample) => {
                  const label = ex.label as number[];
                  return Array.isArray(label) && label[1] === 1;
                }).length
              }{" "}
              ones
            </p>
            <Button onClick={addDatasetExample} size="sm">
              <Plus className="mr-2 h-4 w-4" />
              Add Example
            </Button>
          </div>

          <div className="grid max-h-96 gap-4 overflow-y-auto">
            {trainingExamples.map((example: TrainingExample, index: number) => {
              const pixelValues = getPatternPreview(example.pattern as number[][]);
              return (
                <div key={example.id} className="rounded-lg border bg-gray-50 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium">Example {index + 1}</span>
                      <div className="flex items-center gap-2">
                        <Label htmlFor={`label-${index}`} className="text-sm">
                          Label:
                        </Label>
                        <select
                          id={`label-${index}`}
                          value={
                            Array.isArray(example.label)
                              ? (example.label as number[])?.[0] === 1
                                ? "0"
                                : "1"
                              : String(example.label || 0)
                          }
                          onChange={(e) =>
                            updateDatasetExample(
                              index,
                              example.pattern as number[][] | number[],
                              parseInt(e.target.value),
                            )
                          }
                          className="rounded border px-2 py-1 text-sm"
                        >
                          <option value={0}>0</option>
                          <option value={1}>1</option>
                        </select>
                      </div>
                    </div>
                    <Button
                      onClick={() => removeDatasetExample(index)}
                      variant="outline"
                      size="sm"
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="flex items-center gap-4">
                    {/* 9x9 pixel grid */}
                    <div className="grid h-32 w-32 grid-cols-9 gap-0 border-2 border-gray-400 bg-gray-100">
                      {(() => {
                        const pattern = example.pattern as number[][] | number[];
                        const grid = Array.isArray(pattern[0])
                          ? (pattern as number[][])
                          : flatToGrid(pattern as number[]);
                        return grid.map((row: number[], rowIndex: number) =>
                          row.map((pixel: number, colIndex: number) => (
                            <div
                              key={`${rowIndex}-${colIndex}`}
                              className={`h-full w-full cursor-crosshair select-none border border-gray-200 transition-colors duration-100 ${
                                pixel ? "bg-gray-800" : "bg-white hover:bg-gray-100"
                              }`}
                              onMouseDown={() => handleEditorMouseDown(index, rowIndex, colIndex)}
                              onMouseEnter={() => handleEditorMouseEnter(index, rowIndex, colIndex)}
                            />
                          )),
                        );
                      })()}
                    </div>

                    <div className="text-xs text-gray-600">
                      <div>
                        Pattern (81 pixels): [
                        {(() => {
                          const pattern = example.pattern as number[][] | number[];
                          const flatPattern = Array.isArray(pattern[0])
                            ? (pattern as number[][]).flat()
                            : (pattern as number[]);
                          return (
                            flatPattern
                              .slice(0, 12)
                              .map((v) => v.toString())
                              .join(",") + (flatPattern.length > 12 ? "..." : "")
                          );
                        })()}
                        ]
                      </div>
                      <div className="mt-1">
                        Click pixels to toggle. Target:{" "}
                        {Array.isArray(example.label)
                          ? (example.label as number[])?.[0] === 1
                            ? "0"
                            : "1"
                          : String(example.label)}
                      </div>
                      <div className="mt-1">Each pixel is 0 (white) or 1 (black)</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex gap-3 border-t pt-4">
            <Button onClick={saveDataset} className="flex-1">
              Save Changes
            </Button>
            <Button
              onClick={() => {
                setShowDatasetEditor(false);
              }}
              variant="outline"
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
