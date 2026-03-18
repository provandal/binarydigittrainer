import { z } from "zod";

// Training example types (localStorage-backed, no database)
export const insertTrainingExampleSchema = z.object({
  pattern: z.union([z.array(z.number()), z.array(z.array(z.number()))]),
  label: z.array(z.number()),
});

export type InsertTrainingExample = z.infer<typeof insertTrainingExampleSchema>;

export type TrainingExample = InsertTrainingExample & {
  id: number;
  createdAt: Date;
};
