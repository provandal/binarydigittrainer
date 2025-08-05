import { PlayCircle, Check, X, ArrowRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface TestResult {
  predicted: number;
  actual: number;
  confidence: number;
  timestamp: number;
  correct: boolean;
}

interface TestInterfaceProps {
  onTestCurrent: () => void;
  onLoadSample: (digit: number) => void;
  testResults: TestResult[];
}

const sampleDigits = [
  { digit: 0, pattern: [0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0] },
  { digit: 5, pattern: [1,1,1,0,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0] }
];

export default function TestInterface({ onTestCurrent, onLoadSample, testResults }: TestInterfaceProps) {
  const formatTimeAgo = (timestamp: number) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    return `${Math.floor(seconds / 60)}m ago`;
  };

  const testAccuracy = testResults.length > 0 
    ? (testResults.filter(r => r.correct).length / testResults.length * 100).toFixed(0)
    : 0;

  return (
    <>
      <Card>
        <CardContent className="p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Test Network</h2>
          
          {/* Sample Test Cases */}
          <div className="space-y-3 mb-4">
            <div className="text-sm font-medium text-slate-700 mb-2">Sample Digits</div>
            
            {sampleDigits.map((sample, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-3 border border-slate-200 rounded-lg hover:border-primary cursor-pointer transition-colors duration-200"
                onClick={() => onLoadSample(sample.digit)}
              >
                <div className="grid grid-cols-5 gap-px w-16 h-16">
                  {sample.pattern.slice(0, 20).map((bit, i) => (
                    <div 
                      key={i}
                      className={`rounded-sm ${bit ? 'bg-slate-800' : 'bg-slate-300'}`}
                    />
                  ))}
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-slate-900">{sample.digit}</div>
                  <div className="text-xs text-slate-500">Sample</div>
                </div>
                <Button variant="ghost" size="sm">
                  <ArrowRight className="w-4 h-4 text-primary" />
                </Button>
              </div>
            ))}
          </div>

          <Button 
            onClick={onTestCurrent}
            className="w-full bg-primary hover:bg-blue-700 text-white"
          >
            <PlayCircle className="w-4 h-4 mr-2" />
            Test Current Input
          </Button>
        </CardContent>
      </Card>

      {/* Recent Results */}
      <Card>
        <CardContent className="p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Recent Tests</h2>
          
          <div className="space-y-3">
            {testResults.slice(0, 3).map((test, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-slate-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    test.correct ? 'bg-secondary' : 'bg-red-500'
                  }`}>
                    {test.correct ? (
                      <Check className="text-white w-4 h-4" />
                    ) : (
                      <X className="text-white w-4 h-4" />
                    )}
                  </div>
                  <div>
                    <div className="text-sm font-medium">
                      {test.correct 
                        ? `Digit ${test.predicted}` 
                        : `Digit ${test.actual} → ${test.predicted}`
                      }
                    </div>
                    <div className="text-xs text-slate-500">
                      {(test.confidence * 100).toFixed(0)}% confidence
                    </div>
                  </div>
                </div>
                <div className="text-xs text-slate-500">
                  {formatTimeAgo(test.timestamp)}
                </div>
              </div>
            ))}
            
            {testResults.length === 0 && (
              <div className="text-center text-slate-500 py-4">
                No tests performed yet
              </div>
            )}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-200">
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-600">Test Accuracy</span>
              <span className="font-bold text-secondary">{testAccuracy}%</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
