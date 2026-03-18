import React from "react";
import { type ColorScheme, weightColor } from "@/lib/color-schemes";

export interface Heatmap9x9Props {
  grid: number[][];
  cell?: number;
  showInputOverlay?: boolean;
  inputGrid?: number[][] | null;
  globalMaxAbs?: number | null;
  colorScheme: ColorScheme;
}

export function Heatmap9x9({
  grid,
  cell = 18,
  showInputOverlay = false,
  inputGrid = null,
  globalMaxAbs = null,
  colorScheme,
}: Heatmap9x9Props) {
  const flat = grid.flat();
  const localMaxAbs = flat.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const maxAbs = globalMaxAbs !== null ? globalMaxAbs : localMaxAbs;
  return (
    <div className="inline-grid" style={{ gridTemplateColumns: `repeat(9, ${cell}px)` }}>
      {grid.map((row, r) =>
        row.map((v, c) => {
          const isInputActive = inputGrid?.[r]?.[c] === 1;
          const baseStyle = {
            width: cell,
            height: cell,
            background: weightColor(v, maxAbs, colorScheme),
            opacity: showInputOverlay && !isInputActive ? 0.3 : 1,
            border:
              showInputOverlay && isInputActive
                ? "2px solid #000"
                : "1px solid rgba(255,255,255,0.4)",
          };
          return (
            <div
              key={`${r}-${c}`}
              title={`Weight: ${v.toFixed(3)}${showInputOverlay ? `, Input: ${isInputActive ? "1" : "0"}` : ""}`}
              style={baseStyle}
              className="transition-opacity duration-200"
            />
          );
        }),
      )}
    </div>
  );
}
