import { Card, CardContent } from "@/components/ui/card";

interface PredictionResultsProps {
  predictions: number[];
}

export default function PredictionResults({ predictions }: PredictionResultsProps) {
  const maxPrediction = Math.max(...predictions);
  const predictedDigit = predictions.indexOf(maxPrediction);

  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Current Prediction</h2>
        
        <div className="grid grid-cols-5 gap-3 mb-4">
          {predictions.slice(0, 5).map((pred, index) => (
            <div key={index} className="text-center">
              <div className="text-xs text-slate-600 mb-1">{index}</div>
              <div className="h-16 bg-slate-100 rounded flex items-end justify-center">
                <div 
                  className={`w-full rounded transition-all duration-300 ${
                    pred === maxPrediction ? 'bg-secondary' : 'bg-slate-300'
                  }`}
                  style={{ height: `${Math.max(pred * 100, 5)}%` }}
                />
              </div>
              <div className={`text-xs mt-1 ${
                pred === maxPrediction ? 'font-bold text-secondary' : 'text-slate-500'
              }`}>
                {pred.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
        
        <div className="grid grid-cols-5 gap-3 mb-4">
          {predictions.slice(5, 10).map((pred, index) => (
            <div key={index + 5} className="text-center">
              <div className="text-xs text-slate-600 mb-1">{index + 5}</div>
              <div className="h-16 bg-slate-100 rounded flex items-end justify-center">
                <div 
                  className={`w-full rounded transition-all duration-300 ${
                    pred === maxPrediction ? 'bg-secondary' : 'bg-slate-300'
                  }`}
                  style={{ height: `${Math.max(pred * 100, 5)}%` }}
                />
              </div>
              <div className={`text-xs mt-1 ${
                pred === maxPrediction ? 'font-bold text-secondary' : 'text-slate-500'
              }`}>
                {pred.toFixed(2)}
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between pt-4 border-t border-slate-200">
          <div>
            <span className="text-sm text-slate-600">Predicted:</span>
            <span className="ml-2 text-lg font-bold text-secondary">{predictedDigit}</span>
          </div>
          <div>
            <span className="text-sm text-slate-600">Confidence:</span>
            <span className="ml-2 text-lg font-bold text-secondary">
              {(maxPrediction * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
