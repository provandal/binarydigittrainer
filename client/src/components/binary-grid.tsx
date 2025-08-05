import { Dice6 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface BinaryGridProps {
  grid: number[];
  onGridChange: (grid: number[]) => void;
  onGenerateRandom: () => void;
}

export default function BinaryGrid({ grid, onGridChange, onGenerateRandom }: BinaryGridProps) {
  const toggleBit = (index: number) => {
    const newGrid = [...grid];
    newGrid[index] = newGrid[index] === 0 ? 1 : 0;
    onGridChange(newGrid);
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-900">Binary Input</h2>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onGenerateRandom}
            className="text-slate-500 hover:text-slate-700"
          >
            <Dice6 className="w-4 h-4 mr-1 text-primary" />
            Random
          </Button>
        </div>
        
        {/* 8x8 Binary Grid */}
        <div className="grid grid-cols-8 gap-1 bg-slate-100 p-3 rounded-lg">
          {grid.map((bit, index) => (
            <button
              key={index}
              onClick={() => toggleBit(index)}
              className={`aspect-square border border-slate-300 rounded cursor-pointer hover:opacity-80 flex items-center justify-center text-sm font-mono font-bold transition-all duration-200 ${
                bit === 1 
                  ? 'bg-slate-800 text-white hover:bg-slate-700' 
                  : 'bg-white text-slate-700 hover:bg-slate-50'
              }`}
            >
              {bit}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
