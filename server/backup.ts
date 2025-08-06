import { writeFileSync, readFileSync, existsSync } from 'fs';
import { db } from './db';
import { trainingExamples } from '@shared/schema';

const BACKUP_FILE = './training_examples_backup.json';

// Backup training examples to local file
export async function backupTrainingExamples() {
  try {
    const examples = await db.select().from(trainingExamples);
    writeFileSync(BACKUP_FILE, JSON.stringify(examples, null, 2));
    console.log(`Backed up ${examples.length} training examples to ${BACKUP_FILE}`);
    return examples.length;
  } catch (error) {
    console.error('Failed to backup training examples:', error);
    return 0;
  }
}

// Restore training examples from local file
export async function restoreTrainingExamples() {
  try {
    if (!existsSync(BACKUP_FILE)) {
      console.log('No backup file found');
      return 0;
    }

    const backupData = JSON.parse(readFileSync(BACKUP_FILE, 'utf8'));
    if (!Array.isArray(backupData) || backupData.length === 0) {
      console.log('No backup data to restore');
      return 0;
    }

    // Clear existing data and restore from backup
    await db.delete(trainingExamples);
    
    for (const example of backupData) {
      await db.insert(trainingExamples).values({
        pattern: example.pattern,
        label: example.label
      });
    }

    console.log(`Restored ${backupData.length} training examples from backup`);
    return backupData.length;
  } catch (error) {
    console.error('Failed to restore training examples:', error);
    return 0;
  }
}

// Auto-backup on data changes (with throttling to prevent excessive backups)
let lastBackupTime = 0;
const BACKUP_THROTTLE_MS = 2000; // Only backup once every 2 seconds

export async function autoBackup() {
  const now = Date.now();
  if (now - lastBackupTime < BACKUP_THROTTLE_MS) {
    return; // Skip backup if too recent
  }
  lastBackupTime = now;
  await backupTrainingExamples();
}