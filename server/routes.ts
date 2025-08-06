import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertTrainingExampleSchema } from "@shared/schema";
import { z } from "zod";
import { autoBackup, restoreTrainingExamples } from "./backup";

export async function registerRoutes(app: Express): Promise<Server> {
  // Training examples API routes
  
  // Get all training examples
  app.get("/api/training-examples", async (req, res) => {
    try {
      const examples = await storage.getTrainingExamples();
      res.json(examples);
    } catch (error) {
      console.error("Error fetching training examples:", error);
      res.status(500).json({ error: "Failed to fetch training examples" });
    }
  });

  // Create a new training example
  app.post("/api/training-examples", async (req, res) => {
    try {
      const validatedData = insertTrainingExampleSchema.parse(req.body);
      const example = await storage.createTrainingExample(validatedData);
      res.status(201).json(example);
      // Backup when creating new examples
      setImmediate(() => autoBackup());
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid data", details: error.errors });
      } else {
        console.error("Error creating training example:", error);
        res.status(500).json({ error: "Failed to create training example" });
      }
    }
  });

  // Update a training example
  app.put("/api/training-examples/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const validatedData = insertTrainingExampleSchema.parse(req.body);
      const example = await storage.updateTrainingExample(id, validatedData);
      
      if (!example) {
        res.status(404).json({ error: "Training example not found" });
        return;
      }
      
      res.json(example);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid data", details: error.errors });
      } else {
        console.error("Error updating training example:", error);
        res.status(500).json({ error: "Failed to update training example" });
      }
    }
  });

  // Delete a training example
  app.delete("/api/training-examples/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const deleted = await storage.deleteTrainingExample(id);
      
      if (!deleted) {
        res.status(404).json({ error: "Training example not found" });
        return;
      }
      
      res.status(204).send();
      // Backup when deleting examples
      setImmediate(() => autoBackup());
    } catch (error) {
      console.error("Error deleting training example:", error);
      res.status(500).json({ error: "Failed to delete training example" });
    }
  });

  // Clear all training examples
  app.delete("/api/training-examples", async (req, res) => {
    try {
      await storage.clearTrainingExamples();
      res.status(204).send();
      // Backup when clearing all examples
      setImmediate(() => autoBackup());
    } catch (error) {
      console.error("Error clearing training examples:", error);
      res.status(500).json({ error: "Failed to clear training examples" });
    }
  });

  // Restore training examples from backup
  app.post("/api/training-examples/restore", async (req, res) => {
    try {
      const count = await restoreTrainingExamples();
      res.json({ message: `Restored ${count} training examples from backup` });
    } catch (error) {
      console.error("Error restoring training examples:", error);
      res.status(500).json({ error: "Failed to restore training examples" });
    }
  });

  // Bulk upload training examples
  app.post("/api/training-examples/bulk-upload", async (req, res) => {
    try {
      const bulkData = z.array(z.object({
        input: z.array(z.number()).length(81),
        target: z.array(z.number()).length(2)
      })).parse(req.body);

      // Convert one-hot encoded targets to integer labels
      const examples = bulkData.map(item => ({
        pattern: item.input,
        // Convert [1,0] to 0 (digit "0"), [0,1] to 1 (digit "1")
        label: item.target[0] === 1 ? 0 : 1
      }));

      // Clear existing data first
      await storage.clearTrainingExamples();
      
      // Bulk insert new data
      const results = await storage.bulkCreateTrainingExamples(examples);
      
      res.json({ 
        message: `Successfully uploaded ${results.length} training examples`,
        count: results.length 
      });
      
      // Backup after bulk upload
      setImmediate(() => autoBackup());
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid data format", details: error.errors });
      } else {
        console.error("Error bulk uploading training examples:", error);
        res.status(500).json({ error: "Failed to bulk upload training examples" });
      }
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
