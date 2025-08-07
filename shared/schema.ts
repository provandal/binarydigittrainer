import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, json, timestamp, serial } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const trainingExamples = pgTable("training_examples", {
  id: serial("id").primaryKey(),
  pattern: json("pattern").notNull(),
  label: json("label").notNull(), // Changed to json to store one-hot arrays [1,0] or [0,1]
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertTrainingExampleSchema = createInsertSchema(trainingExamples).omit({ id: true, createdAt: true });

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertTrainingExample = z.infer<typeof insertTrainingExampleSchema>;
export type TrainingExample = typeof trainingExamples.$inferSelect;
