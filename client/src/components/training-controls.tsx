import { Play, Pause, RotateCcw } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface TrainingControlsProps {
  targetLabel: number;
  onTargetLabelChange: (label: number) => void;
  isTraining: boolean;
  onStartTraining: () => void;
  onStopTraining: () => void;
  onResetNetwork: () => void;
}

export default function TrainingControls({
  targetLabel,
  onTargetLabelChange,
  isTraining,
  onStartTraining,
  onStopTraining,
  onResetNetwork
}: TrainingControlsProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Training Controls</h2>
        
        {/* Target Label Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">Target Label</label>
          <Select value={targetLabel.toString()} onValueChange={(value) => onTargetLabelChange(parseInt(value))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
                <SelectItem key={digit} value={digit.toString()}>
                  Digit {digit}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Control Buttons */}
        <div className="grid grid-cols-2 gap-3">
          {!isTraining ? (
            <Button 
              onClick={onStartTraining}
              className="bg-primary hover:bg-blue-700 text-white"
            >
              <Play className="w-4 h-4 mr-2" />
              Start Training
            </Button>
          ) : (
            <Button 
              onClick={onStopTraining}
              variant="secondary"
              className="bg-slate-500 hover:bg-slate-600 text-white"
            >
              <Pause className="w-4 h-4 mr-2" />
              Pause
            </Button>
          )}
          <Button 
            onClick={onResetNetwork}
            variant="outline"
            className="bg-slate-200 hover:bg-slate-300 text-slate-700"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </div>
        
        <Button 
          onClick={onResetNetwork}
          variant="outline"
          className="w-full mt-3"
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Reset Network
        </Button>
      </CardContent>
    </Card>
  );
}
