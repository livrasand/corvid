
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, describe, expect, it } from 'vitest';
import type { CorvidConfig } from '../src/core/config.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { saveTokenizer } from '../src/core/save-tokenizer.js';
import { trainModel } from '../src/training/trainer.js';
import { saveCheckpoint } from '../src/training/checkpoint.js';
import { exportModelBundle } from '../src/training/export-model.js';

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

describe('exportModelBundle', () => {
  async function setupBundle(root: string) {
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

    const checkpointPath = join(checkpointDir, 'checkpoint-epoch1.json');
    const outputDir = join(root, 'bundle');

    return { tokenizer, tokenizerPath, config, configPath, checkpointPath, outputDir };
  }

  it('writes a portable bundle with manifest and copied files', async () => {
    const root = await temporaryDir('corvid-export-');
    const setup = await setupBundle(root);

    const manifest = await exportModelBundle({
      name: 'corvid-test',
      outputDir: setup.outputDir,
      checkpointPath: setup.checkpointPath,
      tokenizerPath: setup.tokenizerPath,
      configPath: setup.configPath,
    });

    expect(manifest.name).toBe('corvid-test');
    expect(manifest.format).toBe(1);
    expect(manifest.checkpoint.epoch).toBe(1);
    expect(manifest.files).toEqual({
      checkpoint: 'checkpoint.json',
      tokenizer: 'tokenizer.json',
      config: 'config.json',
    });

    const names = await (await import('node:fs/promises')).readdir(setup.outputDir);
    expect(names).toContain('model.json');
    expect(names).toContain('checkpoint.json');
    expect(names).toContain('tokenizer.json');
    expect(names).toContain('config.json');

    const writtenManifest = JSON.parse(await readFile(join(setup.outputDir, 'model.json'), 'utf8'));
    expect(writtenManifest.name).toBe('corvid-test');
  });

  it('throws when the tokenizer has more tokens than the model vocabSize', async () => {
    const root = await temporaryDir('corvid-export-big-');
    const rawDir = join(root, 'raw');
    await mkdir(rawDir);

    const corpus = 'hello world. '.repeat(30);
    await writeFile(join(rawDir, 'corpus.txt'), corpus);

    // Tokenizer with more than 256 tokens.
    const bigCorpus = Array.from({ length: 300 }, (_, i) => String.fromCharCode(33 + i)).join('');
    const tokenizer = CharacterTokenizer.fromCorpus(bigCorpus);
    const tokenizerPath = join(root, 'tokenizer.json');
    await saveTokenizer(tokenizer, tokenizerPath);

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
        epochs: 1,
        seed: 42,
      },
    };
    const configPath = join(root, 'config.json');
    await writeFile(configPath, JSON.stringify(config));

    const checkpointDir = join(root, 'checkpoints');
    const training = await trainModel({
      config,
      trainIds: Array.from({ length: 300 }, (_, index) => index % 256),
      valIds: Array.from({ length: 60 }, (_, index) => index % 256),
      checkpointDir,
    });
    training.finalModel.dispose();

    await expect(
      exportModelBundle({
        name: 'corvid-big',
        outputDir: join(root, 'bundle'),
        checkpointPath: join(checkpointDir, 'checkpoint-epoch1.json'),
        tokenizerPath,
        configPath,
      }),
    ).rejects.toThrow(/Tokenizer has .* tokens but the model vocabSize is/);
  });

  it('throws when the checkpoint shape does not match the config', async () => {
    const root = await temporaryDir('corvid-export-bad-');
    const setup = await setupBundle(root);

    const checkpoint = JSON.parse(await readFile(setup.checkpointPath, 'utf8'));
    checkpoint.variables[0].shape = [999, 32];
    const badCheckpointPath = join(root, 'bad-checkpoint.json');
    await writeFile(badCheckpointPath, JSON.stringify(checkpoint));

    await expect(
      exportModelBundle({
        name: 'corvid-bad',
        outputDir: join(root, 'bundle'),
        checkpointPath: badCheckpointPath,
        tokenizerPath: setup.tokenizerPath,
        configPath: setup.configPath,
      }),
    ).rejects.toThrow(/does not match expected shape/);
  });
});
