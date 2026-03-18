import { useState, useRef, useCallback } from "react";
import { flatToGrid } from "@/lib/nn-helpers";

export interface UseCanvasDrawingParams {
  tourTriggerRef: React.MutableRefObject<(() => void) | null>;
  onPixelChange: () => void;
  pixelGridRef: React.MutableRefObject<number[][]>;
}

export interface UseCanvasDrawingReturn {
  // State
  pixelGrid: number[][];
  isDrawing: boolean;
  hoveredPixel: number | null;

  // Refs
  canvasRef: React.RefObject<HTMLDivElement>;
  pixelGridRef: React.MutableRefObject<number[][]>;
  isDrawingRef: React.MutableRefObject<boolean>;
  changedCellsRef: React.MutableRefObject<number>;

  // Methods
  togglePixel: (rowIndex: number, colIndex: number) => void;
  handleMouseDown: (rowIndex: number, colIndex: number) => void;
  handleMouseEnter: (rowIndex: number, colIndex: number) => void;
  handleMouseUp: () => void;
  handleTouchStart: (e: React.TouchEvent) => void;
  handleTouchMove: (e: React.TouchEvent) => void;
  handleTouchEnd: (e: React.TouchEvent) => void;
  handlePixelHover: (pixelIndex: number) => void;
  handlePixelLeave: () => void;
  clearCanvas: () => void;
  setPixelGrid: (grid: number[][] | number[]) => void;
  getTouchCell: (touch: React.Touch) => [number, number] | null;
}

const initialPixelGrid = Array(9)
  .fill(0)
  .map(() => Array(9).fill(0));

export function useCanvasDrawing({
  tourTriggerRef,
  onPixelChange,
  pixelGridRef,
}: UseCanvasDrawingParams): UseCanvasDrawingReturn {
  const [pixelGrid, setPixelGridState] = useState(initialPixelGrid);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hoveredPixel, setHoveredPixel] = useState<number | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);
  const isDrawingRef = useRef(false);
  const changedCellsRef = useRef(0);

  // Safe setter that updates both ref (immediate) and React state (async, UI only)
  const setPixelGrid = useCallback((grid: number[][] | number[]) => {
    const normalized = Array.isArray(grid[0]) ? (grid as number[][]) : flatToGrid(grid as number[]);
    pixelGridRef.current = normalized; // immediate, used by training logic
    setPixelGridState(normalized); // async, UI only
  }, []);

  const togglePixel = useCallback(
    (rowIndex: number, colIndex: number) => {
      setPixelGridState((prev) => {
        const next = prev.map((row, r) =>
          r === rowIndex ? row.map((v, c) => (c === colIndex ? (v ? 0 : 1) : v)) : row,
        );
        // Update ref immediately for training logic
        pixelGridRef.current = next;
        changedCellsRef.current += 1;
        return next;
      });
      onPixelChange();
    },
    [onPixelChange],
  );

  const handleMouseDown = useCallback(
    (rowIndex: number, colIndex: number) => {
      setIsDrawing(true);
      isDrawingRef.current = true;
      changedCellsRef.current = 0;
      togglePixel(rowIndex, colIndex);
    },
    [togglePixel],
  );

  const handleMouseEnter = useCallback(
    (rowIndex: number, colIndex: number) => {
      if (isDrawing) {
        togglePixel(rowIndex, colIndex);
      }
    },
    [isDrawing, togglePixel],
  );

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false);
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      // Only signal the tour if something actually changed
      if (changedCellsRef.current > 0) {
        tourTriggerRef.current?.();
      }
    }
  }, [tourTriggerRef]);

  const getTouchCell = useCallback((touch: React.Touch): [number, number] | null => {
    const el = canvasRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    const col = Math.floor((x / rect.width) * 9);
    const row = Math.floor((y / rect.height) * 9);
    if (row < 0 || row > 8 || col < 0 || col > 8) return null;
    return [row, col];
  }, []);

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      e.preventDefault();
      const cell = getTouchCell(e.touches[0]);
      if (cell) handleMouseDown(cell[0], cell[1]);
    },
    [getTouchCell, handleMouseDown],
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      e.preventDefault();
      const cell = getTouchCell(e.touches[0]);
      if (cell) handleMouseEnter(cell[0], cell[1]);
    },
    [getTouchCell, handleMouseEnter],
  );

  const handleTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      e.preventDefault();
      handleMouseUp();
    },
    [handleMouseUp],
  );

  const handlePixelHover = useCallback((pixelIndex: number) => {
    setHoveredPixel(pixelIndex);
  }, []);

  const handlePixelLeave = useCallback(() => {
    setHoveredPixel(null);
  }, []);

  const clearCanvas = useCallback(() => {
    setPixelGrid(
      Array(9)
        .fill(0)
        .map(() => Array(9).fill(0)),
    );
  }, [setPixelGrid]);

  return {
    // State
    pixelGrid,
    isDrawing,
    hoveredPixel,

    // Refs
    canvasRef,
    pixelGridRef,
    isDrawingRef,
    changedCellsRef,

    // Methods
    togglePixel,
    handleMouseDown,
    handleMouseEnter,
    handleMouseUp,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
    handlePixelHover,
    handlePixelLeave,
    clearCanvas,
    setPixelGrid,
    getTouchCell,
  };
}
