
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { CorvidConfig } from '../src/core/config.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { saveTokenizer } from '../src/core/save-tokenizer.js';
import { trainModel } from '../src/training/trainer.js';
import { exportModelBundle } from '../src/training/export-model.js';
import { main as generateMain } from '../scripts/generate.js';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

async function temporaryDir(prefix: string): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), prefix));
  temporaryDirectories.push(directory);

  return directory;
}

describe('generate CLI', () => {
  it('restores a checkpoint and generates text from a prompt', async () => {
    const root = await temporaryDir('corvid-generate-');
    const rawDir = join(root, 'raw');
    await mkdir(rawDir);

    const corpus = 'hello world, the quick brown fox jumps over the lazy dog. '.repeat(30);
    await writeFile(join(rawDir, 'corpus.txt'), corpus);

    const tokenizer = CharacterTokenizer.fromCorpus(corpus);
    const tokenizerPath = join(root, 'tokenizer.json');
    await saveTokenizer(tokenizer, tokenizerPath);

    const config: CorvidConfig = {
      model: {
        // The model must cover the entire tokenizer vocabulary so
        // that decode does not receive out-of-range IDs.
        vocabSize: Math.max(256, tokenizer.vocabSize),
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
        epochs: 1,
        seed: 42,
      },
    };
    const configPath = join(root, 'config.json');
    await writeFile(configPath, JSON.stringify(config));

    const checkpointDir = join(root, 'checkpoints');
    const training = await trainModel({
      config,
      trainIds: Array.from({ length: 300 }, (_, index) => index % tokenizer.vocabSize),
      valIds: Array.from({ length: 60 }, (_, index) => index % tokenizer.vocabSize),
      checkpointDir,
    });
    training.finalModel.dispose();

    const logs: string[] = [];
    const printed: string[] = [];

    const text = await generateMain({
      configPath,
      tokenizerPath,
      checkpointDir,
      prompt: 'hola',
      maxNewTokens: 10,
      temperature: 1,
      seed: 42,
      log: (message) => logs.push(message),
      print: (message) => printed.push(message),
    });

    expect(typeof text).toBe('string');
    expect(logs.join('\n')).toContain('Restaurando checkpoint');
    expect(logs.join('\n')).toContain('checkpoint-epoch1.json');
    expect(printed.length).toBeGreaterThan(0);
  });

  it('throws when the checkpoint directory has no checkpoints', async () => {
    const root = await temporaryDir('corvid-generate-empty-');
    const emptyDir = join(root, 'checkpoints');
    await mkdir(emptyDir);

    // Valid config so the flow reaches checkpoint search.
    const configPath = join(root, 'config.json');
    await writeFile(configPath, JSON.stringify({
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
        epochs: 1,
        seed: 42,
      },
    }));
    const tokenizerPath = join(root, 'tokenizer.json');
    await saveTokenizer(
      CharacterTokenizer.fromCorpus(Array.from({ length: 300 }, (_, i) => String.fromCharCode(33 + i)).join('')),
      tokenizerPath,
    );

    await expect(
      generateMain({
        checkpointDir: emptyDir,
        configPath,
        tokenizerPath,
      }),
    ).rejects.toThrow(/No checkpoints found/);
  });

  it('generates text from an exported model bundle (modelDir)', async () => {
    const root = await temporaryDir('corvid-generate-bundle-');
    const rawDir = join(root, 'raw');
    await mkdir(rawDir);

    const corpus = 'hello world, the quick brown fox jumps over the lazy dog. '.repeat(30);
    await writeFile(join(rawDir, 'corpus.txt'), corpus);

    const tokenizer = CharacterTokenizer.fromCorpus(corpus);
    const tokenizerPath = join(root, 'tokenizer.json');
    await saveTokenizer(tokenizer, tokenizerPath);

    const config: CorvidConfig = {
      model: {
        vocabSize: Math.max(256, tokenizer.vocabSize),
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
        epochs: 1,
        seed: 42,
      },
    };
    const configPath = join(root, 'config.json');
    await writeFile(configPath, JSON.stringify(config));

    const checkpointDir = join(root, 'checkpoints');
    const training = await trainModel({
      config,
      trainIds: Array.from({ length: 300 }, (_, index) => index % tokenizer.vocabSize),
      valIds: Array.from({ length: 60 }, (_, index) => index % tokenizer.vocabSize),
      checkpointDir,
    });
    training.finalModel.dispose();

    const bundleDir = join(root, 'bundle');
    await exportModelBundle({
      name: 'corvid-bundle',
      outputDir: bundleDir,
      checkpointPath: join(checkpointDir, 'checkpoint-epoch1.json'),
      tokenizerPath,
      configPath,
    });

    const logs: string[] = [];
    const printed: string[] = [];

    const text = await generateMain({
      modelDir: bundleDir,
      prompt: 'hola',
      maxNewTokens: 10,
      temperature: 1,
      seed: 42,
      log: (message) => logs.push(message),
      print: (message) => printed.push(message),
    });

    expect(typeof text).toBe('string');
    expect(logs.join('\n')).toContain('Bundle');
    expect(printed.length).toBeGreaterThan(0);
  });
});
