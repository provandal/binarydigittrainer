// Visualization color helpers — pure functions

export type ColorScheme = "blue-red" | "blue-orange" | "green-purple" | "high-contrast";

export const getBarColor = (weight: number, scheme: ColorScheme = "blue-red"): string => {
  const isPositive = weight > 0;

  switch (scheme) {
    case "blue-red":
      return isPositive ? "#3B82F6" : "#EF4444";
    case "blue-orange":
      return isPositive ? "#3B82F6" : "#F97316";
    case "green-purple":
      return isPositive ? "#22C55E" : "#A855F7";
    case "high-contrast":
      return isPositive ? "#6B7280" : "#1F2937";
    default:
      return isPositive ? "#3B82F6" : "#EF4444";
  }
};

export const weightColor = (
  x: number,
  maxAbs: number,
  scheme: ColorScheme = "blue-red",
): string => {
  const a = maxAbs || 1e-6;
  const t = Math.max(-1, Math.min(1, x / a));
  const intensity = Math.abs(t);
  const c = Math.round(255 * intensity);

  switch (scheme) {
    case "blue-red":
      return t >= 0 ? `rgb(${255 - c}, ${255 - c}, 255)` : `rgb(255, ${255 - c}, ${255 - c})`;
    case "blue-orange":
      return t >= 0
        ? `rgb(${255 - c}, ${255 - c}, 255)`
        : `rgb(255, ${255 - Math.round(c * 0.6)}, ${255 - c})`;
    case "green-purple":
      return t >= 0
        ? `rgb(${255 - c}, 255, ${255 - c})`
        : `rgb(${255 - Math.round(c * 0.3)}, ${255 - c}, 255)`;
    case "high-contrast":
      return t >= 0 ? `rgb(${255 - c}, ${255 - c}, ${255 - c})` : `rgb(${c}, ${c}, ${c})`;
    default:
      return t >= 0 ? `rgb(${255 - c}, ${255 - c}, 255)` : `rgb(255, ${255 - c}, ${255 - c})`;
  }
};

export const getColorSchemeDescription = (scheme: string): string => {
  switch (scheme) {
    case "blue-red":
      return "Blue indicates positive values, the darker the color the more positive the value. Red indicates negative values, the darker the color the more negative the value.";
    case "blue-orange":
      return "Blue indicates positive values, the darker the color the more positive the value. Orange indicates negative values, the darker the color the more negative the value.";
    case "green-purple":
      return "Green indicates positive values, the darker the color the more positive the value. Purple indicates negative values, the darker the color the more negative the value.";
    case "high-contrast":
      return "Light gray indicates positive values, the darker the color the more positive the value. Dark gray indicates negative values, the darker the color the more negative the value.";
    default:
      return "Blue indicates positive values, the darker the color the more positive the value. Red indicates negative values, the darker the color the more negative the value.";
  }
};

export const getPositiveColorName = (scheme: ColorScheme = "blue-red"): string => {
  switch (scheme) {
    case "blue-red":
    case "blue-orange":
      return "blue";
    case "green-purple":
      return "green";
    case "high-contrast":
      return "light gray";
    default:
      return "blue";
  }
};

export const vec81ToGrid9 = (v: number[]): number[][] => {
  const g: number[][] = [];
  for (let r = 0; r < 9; r++) g.push(v.slice(r * 9, (r + 1) * 9));
  return g;
};
