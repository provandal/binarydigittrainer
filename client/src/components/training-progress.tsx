import { Card, CardContent } from "@/components/ui/card";

interface TrainingProgressProps {
  trainingHistory: {
    accuracy: number[];
    loss: number[];
  };
  metrics: {
    accuracy: number;
    loss: number;
    trainingTime: number;
  };
}

export default function TrainingProgress({ trainingHistory, metrics }: TrainingProgressProps) {
  const bestAccuracy = Math.max(...trainingHistory.accuracy, 0);
  const lowestLoss = Math.min(...trainingHistory.loss, 1);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Training Progress</h2>
        
        {/* Mini Chart Visualization */}
        <div className="relative h-32 bg-slate-50 rounded-lg p-4 mb-4">
          <svg className="w-full h-full" viewBox="0 0 200 80">
            {/* Accuracy line */}
            <polyline 
              fill="none" 
              stroke="#10B981" 
              strokeWidth="2"
              points={trainingHistory.accuracy.map((acc, i) => 
                `${(i / (trainingHistory.accuracy.length - 1)) * 200},${80 - (acc * 60)}`
              ).join(' ')}
            />
            {/* Loss line */}
            <polyline 
              fill="none" 
              stroke="#EF4444" 
              strokeWidth="2" 
              opacity="0.7"
              points={trainingHistory.loss.map((loss, i) => 
                `${(i / (trainingHistory.loss.length - 1)) * 200},${80 - ((1 - loss) * 60)}`
              ).join(' ')}
            />
          </svg>
          
          {/* Chart Labels */}
          <div className="absolute top-2 left-4 flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-secondary rounded-full" />
              <span className="text-slate-600">Accuracy</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-red-500 rounded-full" />
              <span className="text-slate-600">Loss</span>
            </div>
          </div>
        </div>

        {/* Progress Stats */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-slate-600">Best Accuracy</span>
            <span className="text-sm font-bold text-secondary">
              {(bestAccuracy * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-slate-600">Lowest Loss</span>
            <span className="text-sm font-bold text-accent">
              {lowestLoss.toFixed(3)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-slate-600">Training Time</span>
            <span className="text-sm font-medium">
              {formatTime(metrics.trainingTime)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
