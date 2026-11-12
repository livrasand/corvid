
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import * as tf from '@tensorflow/tfjs-node';
import type { TransformerLM } from '../model/transformer.js';

export interface CheckpointMetadata {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export interface Checkpoint {
  metadata: CheckpointMetadata;
  variables: Array<{ shape: number[]; data: number[] }>;
}

export async function saveCheckpoint(
  path: string,
  model: TransformerLM,
  metadata: CheckpointMetadata,
): Promise<void> {
  const variables = [];
  for (const variable of model.variables) {
    const data = await variable.data();

    variables.push({
      shape: [...variable.shape],
      data: Array.from(data, Number),
    });
  }

  const checkpoint: Checkpoint = { metadata, variables };

  await mkdir(dirname(path), { recursive: true });
  await writeFile(
    path,
    `${JSON.stringify(checkpoint, null, 2)}\n`,
    'utf8',
  );
}

export async function loadCheckpoint(path: string): Promise<Checkpoint> {
  const content = await readFile(path, 'utf8');
  const data: unknown = JSON.parse(content);

  if (
    typeof data !== 'object'
    || data === null
    || !Array.isArray((data as Record<string, unknown>).variables)
    || typeof (data as Record<string, unknown>).metadata !== 'object'
  ) {
    throw new Error(`Invalid checkpoint file: ${path}`);
  }

  return data as Checkpoint;
}

export async function restoreCheckpoint(
  path: string,
  model: TransformerLM,
): Promise<CheckpointMetadata> {
  const checkpoint = await loadCheckpoint(path);

  if (checkpoint.variables.length !== model.variables.length) {
    throw new Error(
      `Checkpoint has ${checkpoint.variables.length} variables but the model has ${model.variables.length}`,
    );
  }

  for (const [index, variable] of model.variables.entries()) {
    const saved = checkpoint.variables[index]!;

    if (
      saved.shape.length !== variable.shape.length
      || saved.shape.some((dimension, i) => dimension !== variable.shape[i])
    ) {
      throw new Error(
        `Checkpoint shape [${saved.shape.join(', ')}] does not match model shape [${variable.shape.join(', ')}] at variable ${index}`,
      );
    }

    variable.assign(tf.tensor(saved.data, saved.shape));
  }

  return checkpoint.metadata;
}
