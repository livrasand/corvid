
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { main } from '../scripts/train.js';

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

async function setupProject(): Promise<{
  root: string;
  configPath: string;
  rawDir: string;
  outDir: string;
  tokenizerPath: string;
  checkpointDir: string;
}> {
  const root = await temporaryDir('corvid-cli-');
  const rawDir = join(root, 'raw');
  await mkdir(rawDir);
  await writeFile(
    join(rawDir, 'corpus.txt'),
    'hello world, the quick brown fox jumps over the lazy dog. '.repeat(40),
  );

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

  return {
    root,
    configPath,
    rawDir,
    outDir: join(root, 'splits'),
    tokenizerPath: join(root, 'tokenizer.json'),
    checkpointDir: join(root, 'checkpoints'),
  };
}

describe('train CLI', () => {
  it('runs the full pipeline and writes tokenizer, splits and checkpoints', async () => {
    const setup = await setupProject();
    const logs: string[] = [];

    await main({ ...setup, log: (message) => logs.push(message) });

    const checkpointFiles = await readdir(setup.checkpointDir);
    expect(checkpointFiles).toEqual(['checkpoint-epoch1.json']);

    const logsText = logs.join('\n');
    expect(logsText).toContain('Tokenizer nuevo guardado');
    expect(logsText).toContain('Splits:');
    expect(logsText).toContain('Época 1:');
    expect(logsText).toContain('Listo:');

    // The persisted tokenizer exists and the splits too.
    await expect(readdir(setup.outDir)).resolves.toEqual([
      'test.json',
      'train.json',
      'val.json',
    ]);
  });

  it('reuses an existing tokenizer instead of rebuilding it', async () => {
    const setup = await setupProject();
    const logs: string[] = [];

    const options = { ...setup, log: (message: string) => logs.push(message) };
    await main(options);
    logs.length = 0;
    await main(options);

    const secondRun = logs.join('\n');
    expect(secondRun).toContain('Tokenizer cargado desde');
  });

  it('rebuilds an invalid tokenizer that only has special tokens', async () => {
    const setup = await setupProject();
    const logs: string[] = [];

    // Degenerate tokenizer: only <pad>/<unk>/<bos>/<eos> (empty previous corpus).
    await writeFile(setup.tokenizerPath, JSON.stringify({
      specials: ['<pad>', '<unk>', '<bos>', '<eos>'],
      chars: [],
    }));

    await main({ ...setup, log: (message) => logs.push(message) });

    const logsText = logs.join('\n');
    expect(logsText).toContain('is invalid');
    expect(logsText).toContain('rebuilt');

    // Later generation with this tokenizer cannot be empty
    // due to lack of characters: the vocabulary must exceed the specials.
    const persisted = JSON.parse(await readFile(setup.tokenizerPath, 'utf8')) as {
      specials: string[];
      chars: string[];
    };
    expect(persisted.chars.length).toBeGreaterThan(0);
  });
});
