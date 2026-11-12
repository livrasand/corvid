
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, describe, expect, it } from 'vitest';
import type { CorvidConfig } from '../src/core/config.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { saveTokenizer } from '../src/core/save-tokenizer.js';
import { trainModel } from '../src/training/trainer.js';
import { main as exportMain } from '../scripts/export-model.js';

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

describe('export-model CLI', () => {
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

    return { root, tokenizer, tokenizerPath, configPath, checkpointDir };
  }

  it('exports a bundle using an explicit checkpoint path', async () => {
    const root = await temporaryDir('corvid-export-cli-');
    const setup = await setupBundle(root);

    const logs: string[] = [];
    const manifest = await exportMain({
      name: 'corvid-cli',
      outputDir: join(root, 'bundle'),
      checkpointPath: join(setup.checkpointDir, 'checkpoint-epoch1.json'),
      tokenizerPath: setup.tokenizerPath,
      configPath: setup.configPath,
      log: (message) => logs.push(message),
    });

    expect(manifest.name).toBe('corvid-cli');
    expect(manifest.checkpoint.epoch).toBe(1);
    const { readdir } = await import('node:fs/promises');
    expect(await readdir(join(root, 'bundle'))).toContain('model.json');
    expect(logs.join('\n')).toContain('listo');
  });

  it('picks the highest epoch when checkpointPath is omitted', async () => {
    const root = await temporaryDir('corvid-export-cli-latest-');
    const setup = await setupBundle(root);

    // Simular un checkpoint de epoch 3 copiando y ajustando la metadata.
    const source = JSON.parse(await (await import('node:fs/promises')).readFile(
      join(setup.checkpointDir, 'checkpoint-epoch1.json'),
      'utf8',
    ));
    source.metadata.epoch = 3;
    await writeFile(
      join(setup.checkpointDir, 'checkpoint-epoch3.json'),
      JSON.stringify(source),
    );

    const manifest = await exportMain({
      name: 'corvid-latest',
      outputDir: join(root, 'bundle'),
      checkpointDir: setup.checkpointDir,
      tokenizerPath: setup.tokenizerPath,
      configPath: setup.configPath,
    });

    expect(manifest.checkpoint.epoch).toBe(3);
  });
});
