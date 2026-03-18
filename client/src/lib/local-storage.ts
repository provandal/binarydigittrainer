import type { TrainingExample, InsertTrainingExample } from "@shared/schema";

const STORAGE_KEY = "binary-digit-trainer-examples";

function loadExamples(): TrainingExample[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveExamples(examples: TrainingExample[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(examples));
}

function nextId(examples: TrainingExample[]): number {
  if (examples.length === 0) return 1;
  return Math.max(...examples.map((e) => e.id)) + 1;
}

export function getTrainingExamples(): TrainingExample[] {
  return loadExamples();
}

export function createTrainingExample(data: InsertTrainingExample): TrainingExample {
  const examples = loadExamples();
  const newExample: TrainingExample = {
    ...data,
    id: nextId(examples),
    createdAt: new Date(),
  };
  examples.push(newExample);
  saveExamples(examples);
  return newExample;
}

export function updateTrainingExample(
  id: number,
  data: InsertTrainingExample
): TrainingExample | null {
  const examples = loadExamples();
  const idx = examples.findIndex((e) => e.id === id);
  if (idx === -1) return null;
  examples[idx] = { ...examples[idx], ...data };
  saveExamples(examples);
  return examples[idx];
}

export function deleteTrainingExample(id: number): boolean {
  const examples = loadExamples();
  const filtered = examples.filter((e) => e.id !== id);
  if (filtered.length === examples.length) return false;
  saveExamples(filtered);
  return true;
}

export function clearTrainingExamples(): void {
  saveExamples([]);
}

export function bulkUploadTrainingExamples(
  data: { input: number[]; target: number[] }[]
): number {
  const examples = loadExamples();
  let id = nextId(examples);
  const newExamples: TrainingExample[] = data.map((item) => ({
    id: id++,
    pattern: item.input,
    label: item.target,
    createdAt: new Date(),
  }));
  saveExamples([...examples, ...newExamples]);
  return newExamples.length;
}
