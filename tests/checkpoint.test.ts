
import * as tf from '@tensorflow/tfjs-node';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { createTransformerLM, type TransformerLM } from '../src/model/transformer.js';
import {
  loadCheckpoint,
  restoreCheckpoint,
  saveCheckpoint,
} from '../src/training/checkpoint.js';

const baseConfig = {
  vocabSize: 256,
  maxSeqLen: 32,
  dModel: 32,
  nHeads: 4,
  nLayers: 1,
  dFF: 64,
  dropout: 0.1,
  tieEmbeddings: true,
};

const models: TransformerLM[] = [];
const temporaryDirectories: string[] = [];

afterEach(async () => {
  while (models.length > 0) {
    models.pop()!.dispose();
  }
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

async function temporaryDir(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'corvid-checkpoint-'));
  temporaryDirectories.push(directory);

  return directory;
}

function idsTensor(values: number[][]): tf.Tensor {
  return tf.tensor2d(values, [values.length, values[0]!.length], 'int32');
}

describe('checkpoint persistence', () => {
  it('saves and restores weights, reproducing identical logits', async () => {
    const path = join(await temporaryDir(), 'checkpoint.json');
    const modelA = createTransformerLM(baseConfig, { seed: 3 });
    models.push(modelA);
    const metadata = { epoch: 2, trainLoss: 1.5, valLoss: 1.7 };

    await saveCheckpoint(path, modelA, metadata);

    const modelB = createTransformerLM(baseConfig, { seed: 99 }); // pesos distintos
    models.push(modelB);
    const ids = idsTensor([[1, 2, 3, 4]]);

    const [logitsA, logitsBefore] = tf.tidy(() => [
      modelA.forward(ids) as tf.Tensor3D,
      modelB.forward(ids) as tf.Tensor3D,
    ]);
    const restored = await restoreCheckpoint(path, modelB);

    expect(restored).toEqual(metadata);

    // modelB's original weights (seed 99) produce different logits;
    // after restore they must match modelA's (seed 3).
    const valuesBefore = Array.from(logitsBefore!.dataSync());
    expect(Array.from(logitsA!.dataSync())).not.toEqual(valuesBefore);

    return tf.tidy(() => {
      const logitsAfter = modelB.forward(ids);
      const valuesA = Array.from(logitsA!.dataSync());
      const valuesAfter = Array.from(logitsAfter.dataSync());

      expect(valuesAfter).toEqual(valuesA);
    });
  });

  it('round-trips through loadCheckpoint', async () => {
    const path = join(await temporaryDir(), 'checkpoint.json');
    const model = createTransformerLM(baseConfig, { seed: 3 });
    models.push(model);

    await saveCheckpoint(path, model, { epoch: 1, trainLoss: 2, valLoss: 2.1 });
    const checkpoint = await loadCheckpoint(path);

    expect(checkpoint.metadata.epoch).toBe(1);
    expect(checkpoint.variables.length).toBe(model.variables.length);
  });

  it('rejects malformed checkpoint files', async () => {
    const { writeFile } = await import('node:fs/promises');
    const path = join(await temporaryDir(), 'checkpoint.json');
    await writeFile(path, '{"nope": true}', 'utf8');

    await expect(loadCheckpoint(path)).rejects.toThrow();
  });

  it('rejects restoring into a model with different variable count', async () => {
    const path = join(await temporaryDir(), 'checkpoint.json');
    const modelA = createTransformerLM(baseConfig, { seed: 3 });
    models.push(modelA);
    await saveCheckpoint(path, modelA, { epoch: 1, trainLoss: 1, valLoss: 1 });

    const modelB = createTransformerLM(
      { ...baseConfig, nLayers: 3 },
      { seed: 4 },
    );
    models.push(modelB);

    await expect(restoreCheckpoint(path, modelB)).rejects.toThrow();
  });

  it('rejects restoring into a model with different shapes', async () => {
    const path = join(await temporaryDir(), 'checkpoint.json');
    const modelA = createTransformerLM(baseConfig, { seed: 3 });
    models.push(modelA);
    await saveCheckpoint(path, modelA, { epoch: 1, trainLoss: 1, valLoss: 1 });

    const modelB = createTransformerLM({ ...baseConfig, dModel: 64 }, { seed: 4 });
    models.push(modelB);

    await expect(restoreCheckpoint(path, modelB)).rejects.toThrow();
  });
});
