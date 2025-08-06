import { type User, type InsertUser, type TrainingExample, type InsertTrainingExample, users, trainingExamples } from "@shared/schema";
import { randomUUID } from "crypto";
import { db } from "./db";
import { eq, asc, sql } from "drizzle-orm";

// modify the interface with any CRUD methods
// you might need

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  getTrainingExamples(): Promise<TrainingExample[]>;
  createTrainingExample(example: InsertTrainingExample): Promise<TrainingExample>;
  updateTrainingExample(id: number, example: InsertTrainingExample): Promise<TrainingExample | undefined>;
  deleteTrainingExample(id: number): Promise<boolean>;
  clearTrainingExamples(): Promise<void>;
  bulkCreateTrainingExamples(examples: InsertTrainingExample[]): Promise<TrainingExample[]>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private trainingExamples: Map<number, TrainingExample>;
  private nextExampleId: number = 1;

  constructor() {
    this.users = new Map();
    this.trainingExamples = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async getTrainingExamples(): Promise<TrainingExample[]> {
    return Array.from(this.trainingExamples.values());
  }

  async createTrainingExample(example: InsertTrainingExample): Promise<TrainingExample> {
    const id = this.nextExampleId++;
    const trainingExample: TrainingExample = {
      ...example,
      id,
      createdAt: new Date(),
    };
    this.trainingExamples.set(id, trainingExample);
    return trainingExample;
  }

  async updateTrainingExample(id: number, example: InsertTrainingExample): Promise<TrainingExample | undefined> {
    const existing = this.trainingExamples.get(id);
    if (!existing) return undefined;
    
    const updated: TrainingExample = {
      ...existing,
      ...example,
    };
    this.trainingExamples.set(id, updated);
    return updated;
  }

  async deleteTrainingExample(id: number): Promise<boolean> {
    return this.trainingExamples.delete(id);
  }

  async clearTrainingExamples(): Promise<void> {
    this.trainingExamples.clear();
    this.nextExampleId = 1;
  }

  async bulkCreateTrainingExamples(examples: InsertTrainingExample[]): Promise<TrainingExample[]> {
    const results: TrainingExample[] = [];
    for (const example of examples) {
      const result = await this.createTrainingExample(example);
      results.push(result);
    }
    return results;
  }
}

export class DatabaseStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(insertUser)
      .returning();
    return user;
  }

  async getTrainingExamples(): Promise<TrainingExample[]> {
    return await db.select().from(trainingExamples).orderBy(asc(trainingExamples.id));
  }

  async createTrainingExample(example: InsertTrainingExample): Promise<TrainingExample> {
    const [trainingExample] = await db
      .insert(trainingExamples)
      .values(example)
      .returning();
    return trainingExample;
  }

  async updateTrainingExample(id: number, example: InsertTrainingExample): Promise<TrainingExample | undefined> {
    const [updated] = await db
      .update(trainingExamples)
      .set(example)
      .where(eq(trainingExamples.id, id))
      .returning();
    return updated || undefined;
  }

  async deleteTrainingExample(id: number): Promise<boolean> {
    const result = await db
      .delete(trainingExamples)
      .where(eq(trainingExamples.id, id))
      .returning();
    return result.length > 0;
  }

  async clearTrainingExamples(): Promise<void> {
    await db.delete(trainingExamples);
    // Reset the sequence to start from 1
    await db.execute(sql`SELECT setval('training_examples_id_seq', 1, false)`);
  }

  async bulkCreateTrainingExamples(examples: InsertTrainingExample[]): Promise<TrainingExample[]> {
    const results = await db
      .insert(trainingExamples)
      .values(examples)
      .returning();
    return results;
  }
}

export const storage = new DatabaseStorage();
