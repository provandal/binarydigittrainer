// BinaryDigitTrainer.jsx
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const initialInput = Array(9).fill(0);
const initialWeights = Array.from({ length: 4 }, () => Array(9).fill(0.1));
const initialBiases = Array(4).fill(0.0);

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export default function BinaryDigitTrainer() {
  const [input, setInput] = useState(initialInput);
  const [weights, setWeights] = useState(initialWeights);
  const [biases, setBiases] = useState(initialBiases);
  const [activations, setActivations] = useState(Array(4).fill(0));
  const [selectedLabel, setSelectedLabel] = useState(null);
  const [step, setStep] = useState(0);

  const toggleInput = (i) => {
    const newInput = [...input];
    newInput[i] = newInput[i] === 0 ? 1 : 0;
    setInput(newInput);
  };

  const feedforward = () => {
    const newActivations = weights.map((w, i) => {
      const z = w.reduce((sum, weight, j) => sum + weight * input[j], biases[i]);
      return sigmoid(z);
    });
    setActivations(newActivations);
  };

  const handleLabelSelect = (label) => {
    setSelectedLabel(label);
  };

  const nextStep = () => {
    if (step === 0) feedforward();
    setStep((prev) => (prev + 1) % 6); // Loop through 6 steps
  };

  return (
    <div className="grid grid-cols-4 gap-4 p-4">
      <Card className="col-span-1">
        <CardContent>
          <h2 className="text-lg font-semibold mb-2">Input Grid</h2>
          <div className="grid grid-cols-3 gap-1">
            {input.map((val, i) => (
              <button
                key={i}
                onClick={() => toggleInput(i)}
                className={`w-8 h-8 border ${val ? "bg-black" : "bg-white"}`}
              ></button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-1">
        <CardContent>
          <h2 className="text-lg font-semibold mb-2">Hidden Layer</h2>
          <div className="space-y-2">
            {activations.map((a, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full border flex items-center justify-center">
                  {a.toFixed(2)}
                </div>
                <div className="flex-grow h-2 bg-gray-200">
                  <div
                    className="h-2 bg-blue-500"
                    style={{ width: `${a * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-1">
        <CardContent>
          <h2 className="text-lg font-semibold mb-2">Label</h2>
          <div className="space-y-2">
            {[0, 1].map((label) => (
              <label key={label} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="label"
                  value={label}
                  checked={selectedLabel === label}
                  onChange={() => handleLabelSelect(label)}
                />
                Digit {label}
              </label>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-1">
        <CardContent>
          <h2 className="text-lg font-semibold mb-2">Controls</h2>
          <Button onClick={nextStep}>Step</Button>
          <p className="mt-2 text-sm">Current Step: {step}</p>
        </CardContent>
      </Card>
    </div>
  );
}
