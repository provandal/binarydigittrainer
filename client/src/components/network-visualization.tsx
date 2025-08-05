import { Card, CardContent } from "@/components/ui/card";

interface NetworkVisualizationProps {
  isTraining: boolean;
  activations: {
    input: number[];
    hidden: number[];
    output: number[];
  };
}

export default function NetworkVisualization({ isTraining, activations }: NetworkVisualizationProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-slate-900">Neural Network</h2>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isTraining ? 'bg-secondary animate-pulse' : 'bg-slate-300'}`} />
            <span className="text-sm text-slate-600">
              {isTraining ? 'Training Active' : 'Ready'}
            </span>
          </div>
        </div>
        
        {/* Network Layers Visualization */}
        <div className="flex justify-between items-center h-64 relative">
          {/* Input Layer */}
          <div className="flex flex-col space-y-2">
            <div className="text-xs font-medium text-slate-600 text-center mb-2">Input (64)</div>
            {activations.input.slice(0, 9).map((activation, index) => (
              <div 
                key={index}
                className="w-3 h-3 rounded-full transition-all duration-300"
                style={{
                  backgroundColor: activation > 0.5 ? '#2563EB' : '#CBD5E1',
                  opacity: 0.3 + (activation * 0.7)
                }}
              />
            ))}
            <div className="w-2 h-2 bg-slate-400 rounded-full mx-auto my-1" />
            <div className="w-2 h-2 bg-slate-400 rounded-full mx-auto" />
            <div className="w-2 h-2 bg-slate-400 rounded-full mx-auto" />
          </div>

          {/* Connection Lines */}
          <div className="flex-1 mx-4 relative">
            <svg className="w-full h-full absolute top-0 left-0" viewBox="0 0 100 100" preserveAspectRatio="none">
              {/* Animated connection lines */}
              {activations.input.slice(0, 5).map((_, i) => (
                <line 
                  key={i}
                  x1="0" 
                  y1={20 + i * 15} 
                  x2="50" 
                  y2={30 + i * 10} 
                  stroke="#2563EB" 
                  strokeWidth="1" 
                  opacity={0.3 + Math.random() * 0.4}
                />
              ))}
            </svg>
          </div>

          {/* Hidden Layer */}
          <div className="flex flex-col space-y-3">
            <div className="text-xs font-medium text-slate-600 text-center mb-2">Hidden (32)</div>
            {activations.hidden.slice(0, 7).map((activation, index) => (
              <div 
                key={index}
                className="w-4 h-4 rounded-full transition-all duration-300"
                style={{
                  backgroundColor: '#8B5CF6',
                  opacity: 0.3 + (activation * 0.7)
                }}
              />
            ))}
            <div className="w-3 h-3 bg-slate-400 rounded-full mx-auto my-1" />
            <div className="w-3 h-3 bg-slate-400 rounded-full mx-auto" />
          </div>

          {/* More Connection Lines */}
          <div className="flex-1 mx-4 relative">
            <svg className="w-full h-full absolute top-0 left-0" viewBox="0 0 100 100" preserveAspectRatio="none">
              {activations.hidden.slice(0, 3).map((_, i) => (
                <line 
                  key={i}
                  x1="0" 
                  y1={20 + i * 20} 
                  x2="100" 
                  y2={30 + i * 20} 
                  stroke="#10B981" 
                  strokeWidth="2" 
                  opacity={0.5 + Math.random() * 0.3}
                />
              ))}
            </svg>
          </div>

          {/* Output Layer */}
          <div className="flex flex-col space-y-3">
            <div className="text-xs font-medium text-slate-600 text-center mb-2">Output (10)</div>
            {activations.output.map((activation, index) => (
              <div 
                key={index}
                className={`w-4 h-4 rounded-full transition-all duration-300 ${
                  activation === Math.max(...activations.output) && activation > 0.5 
                    ? 'animate-pulse' 
                    : ''
                }`}
                style={{
                  backgroundColor: activation === Math.max(...activations.output) ? '#10B981' : '#CBD5E1',
                  opacity: 0.2 + (activation * 0.8)
                }}
              />
            ))}
          </div>
        </div>

        {/* Layer Labels */}
        <div className="flex justify-between mt-4 text-xs text-slate-500">
          <span>64 neurons</span>
          <span>32 neurons</span>
          <span>10 neurons</span>
        </div>
      </CardContent>
    </Card>
  );
}
