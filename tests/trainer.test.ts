
import * as tf from '@tensorflow/tfjs-node';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { CorvidConfig } from '../src/core/config.js';
import { trainModel } from '../src/training/trainer.js';
import type { TrainingResult } from '../src/training/trainer.js';

const config: CorvidConfig = {
  model: {
    vocabSize: 256,
    maxSeqLen: 32,
    dModel: 32,
    nHeads: 4,
    nLayers: 1,
    dFF: 64,
    dropout: 0.1,
    tieEmbeddings: true,
  },
  training: {
    batchSize: 2,
    learningRate: 0.001,
    weightDecay: 0.01,
    epochs: 2,
    seed: 42,
  },
};

const trainIds = Array.from({ length: 200 }, (_, i) => i % 256);
const valIds = Array.from({ length: 40 }, (_, i) => (i * 7) % 256);

const results: TrainingResult[] = [];
const temporaryDirectories: string[] = [];

afterEach(async () => {
  while (results.length > 0) {
    results.pop()!.finalModel.dispose();
  }
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

describe('trainModel', () => {
  it('trains for the configured epochs and records metrics', async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), 'corvid-train-'));
    temporaryDirectories.push(checkpointDir);

    const epochsSeen: number[] = [];
    const result = await trainModel({
      config,
      trainIds,
      valIds,
      checkpointDir,
      onEpochEnd: (metrics) => {
        epochsSeen.push(metrics.epoch);
      },
    });
    results.push(result);

    expect(result.history.length).toBe(config.training.epochs);
    expect(epochsSeen).toEqual([1, 2]);
    expect(result.steps).toBeGreaterThan(0);

    for (const metrics of result.history) {
      expect(Number.isFinite(metrics.trainLoss)).toBe(true);
      expect(Number.isFinite(metrics.valLoss)).toBe(true);
      expect(metrics.trainLoss).toBeGreaterThan(0);
    }
  });

  it('saves one checkpoint per epoch', async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), 'corvid-train-'));
    temporaryDirectories.push(checkpointDir);

    await trainModel({ config, trainIds, valIds, checkpointDir }).then((result) => results.push(result));

    const files = (await readdir(checkpointDir)).sort();
    expect(files).toEqual([
      'checkpoint-epoch1.json',
      'checkpoint-epoch2.json',
    ]);
  });

  it('is reproducible with the same seed', async () => {
    const first = await trainModel({ config, trainIds, valIds });
    results.push(first);
    const second = await trainModel({ config, trainIds, valIds });
    results.push(second);

    expect(second.history).toEqual(first.history);
    expect(second.steps).toBe(first.steps);
  });

  it('throws when there are not enough training tokens', async () => {
    await expect(
      trainModel({ config, trainIds: [1, 2], valIds: [] }),
    ).rejects.toThrow(/Not enough training data/);
  });

  it('reports NaN valLoss when the validation split is empty', async () => {
    const result = await trainModel({ config, trainIds, valIds: [] });
    results.push(result);

    expect(result.steps).toBeGreaterThan(0);
    expect(Number.isFinite(result.history[0]!.trainLoss)).toBe(true);
    expect(Number.isNaN(result.history[0]!.valLoss)).toBe(true);
  });
});
