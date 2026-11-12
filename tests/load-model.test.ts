
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, describe, expect, it } from 'vitest';
import type { CorvidConfig } from '../src/core/config.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { saveTokenizer } from '../src/core/save-tokenizer.js';
import { trainModel } from '../src/training/trainer.js';
import { exportModelBundle } from '../src/training/export-model.js';
import { loadTrainedModel } from '../src/inference/load-model.js';
import { generateText } from '../src/inference/generate.js';

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

describe('loadTrainedModel', () => {
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

    const outputDir = join(root, 'bundle');
    await exportModelBundle({
      name: 'corvid-load',
      outputDir,
      checkpointPath: join(checkpointDir, 'checkpoint-epoch1.json'),
      tokenizerPath,
      configPath,
    });

    return { outputDir, tokenizer };
  }

  it('restores a model ready for inference from a bundle', async () => {
    const root = await temporaryDir('corvid-load-');
    const setup = await setupBundle(root);

    const bundle = await loadTrainedModel(setup.outputDir);

    expect(bundle.manifest.name).toBe('corvid-load');
    expect(bundle.config.model.vocabSize).toBeGreaterThanOrEqual(256);
    expect(bundle.tokenizer.vocabSize).toBe(setup.tokenizer.vocabSize);
    expect(bundle.model.config.vocabSize).toBe(bundle.config.model.vocabSize);

    const text = generateText(bundle.model, bundle.tokenizer, {
      prompt: 'hola',
      maxNewTokens: 5,
      seed: 42,
    });
    expect(typeof text).toBe('string');

    bundle.model.dispose();
  });

  it('throws on an invalid manifest', async () => {
    const root = await temporaryDir('corvid-load-bad-');
    const dir = join(root, 'bad-bundle');
    await (await import('node:fs/promises')).mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'model.json'), JSON.stringify({ format: 1 }));

    await expect(loadTrainedModel(dir)).rejects.toThrow(/Invalid model bundle manifest/);
  });

  it('throws on an unsupported format', async () => {
    const root = await temporaryDir('corvid-load-format-');
    const setup = await setupBundle(root);
    const manifestPath = join(setup.outputDir, 'model.json');
    const manifest = JSON.parse(await (await import('node:fs/promises')).readFile(manifestPath, 'utf8'));
    manifest.format = 999;
    await writeFile(manifestPath, JSON.stringify(manifest));

    await expect(loadTrainedModel(setup.outputDir)).rejects.toThrow(/Unsupported model bundle format/);
  });
});
