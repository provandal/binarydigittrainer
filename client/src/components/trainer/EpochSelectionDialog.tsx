import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { HelpIcon } from "@/components/HelpIcon";

export interface EpochSelectionDialogProps {
  isEpochDialogOpen: boolean;
  setIsEpochDialogOpen: (v: boolean) => void;
  trainingExamplesCount: number;
  numberOfEpochs: number;
  setNumberOfEpochs: (v: number) => void;
  startMultiEpochTraining: () => void;
  setTourMultiEpochStarted: (v: boolean) => void;
}

export function EpochSelectionDialog({
  isEpochDialogOpen,
  setIsEpochDialogOpen,
  trainingExamplesCount,
  numberOfEpochs,
  setNumberOfEpochs,
  startMultiEpochTraining,
  setTourMultiEpochStarted,
}: EpochSelectionDialogProps) {
  return (
    <Dialog open={isEpochDialogOpen} onOpenChange={setIsEpochDialogOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Select Number of Epochs</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="text-sm text-gray-600">
            An epoch is one complete pass through all {trainingExamplesCount} training examples.
            Multiple epochs help the neural network learn patterns better.
          </div>

          <div className="space-y-2">
            <Label htmlFor="epochs" className="flex items-center">
              Number of Epochs:
              <HelpIcon k="epochs" />
            </Label>
            <Input
              id="epochs"
              type="number"
              min="1"
              max="100"
              value={numberOfEpochs}
              onChange={(e) => setNumberOfEpochs(Math.max(1, parseInt(e.target.value) || 1))}
              className="w-full"
            />
          </div>

          <div className="text-xs text-gray-500">
            Total training steps: {numberOfEpochs} × {trainingExamplesCount} ={" "}
            {numberOfEpochs * trainingExamplesCount}
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              onClick={() => {
                startMultiEpochTraining();
                setTourMultiEpochStarted(true);
              }}
              className="flex-1 bg-purple-600 hover:bg-purple-700"
            >
              Start Training
            </Button>
            <Button
              onClick={() => setIsEpochDialogOpen(false)}
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
