import { Card, CardContent } from "@/components/ui/card";

interface NetworkMetricsProps {
  metrics: {
    accuracy: number;
    loss: number;
    epochs: number;
    learningRate: number;
  };
}

export default function NetworkMetrics({ metrics }: NetworkMetricsProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Network Metrics</h2>
        
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">Accuracy</span>
              <span className="text-sm font-bold text-secondary">
                {(metrics.accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2">
              <div 
                className="bg-secondary h-2 rounded-full transition-all duration-300"
                style={{ width: `${metrics.accuracy * 100}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">Training Loss</span>
              <span className="text-sm font-bold text-accent">
                {metrics.loss.toFixed(3)}
              </span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2">
              <div 
                className="bg-accent h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(metrics.loss * 100, 100)}%` }}
              />
            </div>
          </div>
          
          <div className="pt-2 border-t border-slate-200">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-600">Epochs</span>
              <span className="text-sm font-medium">{metrics.epochs.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center mt-1">
              <span className="text-sm text-slate-600">Learning Rate</span>
              <span className="text-sm font-medium">{metrics.learningRate.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
