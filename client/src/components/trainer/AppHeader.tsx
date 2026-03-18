import React from "react";

export interface AppHeaderProps {
  onOpenGuidedTour: () => void;
  onOpenAbout: () => void;
}

export function AppHeader({ onOpenGuidedTour, onOpenAbout }: AppHeaderProps) {
  return (
    <div className="mb-4 text-center sm:mb-8">
      <h1 className="mb-1 text-xl font-bold text-gray-900 sm:mb-2 sm:text-3xl">
        🧠 Binary Digit Trainer
      </h1>
      <p className="text-xs text-gray-600 sm:text-base">
        Step-by-step Neural Network Learning Simulator
      </p>

      {/* Action buttons */}
      <div className="mt-2 flex justify-center gap-2 sm:absolute sm:right-0 sm:top-0 sm:mt-0">
        <button
          onClick={onOpenGuidedTour}
          className="rounded-md bg-blue-600 px-2 py-1 text-xs text-white transition-colors hover:bg-blue-700 sm:px-3 sm:text-sm"
        >
          Guided Tour
        </button>
        <button
          onClick={onOpenAbout}
          className="rounded-md bg-gray-600 px-2 py-1 text-xs text-white transition-colors hover:bg-gray-700 sm:px-3 sm:text-sm"
        >
          About
        </button>
        <a
          href="https://github.com/provandal/binarydigittrainer/discussions"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-md bg-green-600 px-2 py-1 text-xs text-white transition-colors hover:bg-green-700 sm:px-3 sm:text-sm"
        >
          Feedback
        </a>
      </div>
    </div>
  );
}
