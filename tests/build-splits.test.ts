
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { loadSplitIds, buildSplits } from '../src/data/build-splits.js';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

async function temporaryDir(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'corvid-splits-'));
  temporaryDirectories.push(directory);

  return directory;
}

describe('buildSplits', () => {
  it('writes train, val and test splits as token ids', async () => {
    const root = await temporaryDir();
    const rawDir = join(root, 'raw');
    const outDir = join(root, 'splits');

    await mkdir(rawDir);
    await writeFile(join(rawDir, 'corpus.txt'), 'hello world');

    const tokenizer = CharacterTokenizer.fromCorpus('hello world');
    const counts = await buildSplits({ rawDir, outDir, tokenizer });

    const files = (await readdir(outDir)).sort();
    expect(files).toEqual(['test.json', 'train.json', 'val.json']);

    expect(counts.train).toBe(9); // 'hello worl'
    expect(counts.val).toBe(0);
    expect(counts.test).toBe(1); // 'd'

    const trainIds = await loadSplitIds(join(outDir, 'train.json'));
    expect(tokenizer.decode(trainIds)).toBe('hello worl');

    const testIds = await loadSplitIds(join(outDir, 'test.json'));
    expect(tokenizer.decode(testIds)).toBe('d');
  });

  it('respects custom ratios', async () => {
    const root = await temporaryDir();
    const rawDir = join(root, 'raw');
    const outDir = join(root, 'splits');

    await mkdir(rawDir);
    await writeFile(join(rawDir, 'corpus.txt'), 'abcdefghij');

    const tokenizer = CharacterTokenizer.fromCorpus('abcdefghij');
    const counts = await buildSplits({
      rawDir,
      outDir,
      tokenizer,
      trainRatio: 0.5,
      valRatio: 0.3,
      testRatio: 0.2,
    });

    expect(counts.train).toBe(5);
    expect(counts.val).toBe(3);
    expect(counts.test).toBe(2);
  });

  it('reads multiple raw files in sorted order', async () => {
    const root = await temporaryDir();
    const rawDir = join(root, 'raw');
    const outDir = join(root, 'splits');

    await mkdir(rawDir);
    await writeFile(join(rawDir, 'b.txt'), 'mundo');
    await writeFile(join(rawDir, 'a.txt'), 'hola');

    const tokenizer = CharacterTokenizer.fromCorpus('hola\nmundo');
    await buildSplits({ rawDir, outDir, tokenizer });

    const trainIds = await loadSplitIds(join(outDir, 'train.json'));
    expect(tokenizer.decode(trainIds)).toContain('hola\n');
  });

  it('loadSplitIds rejects files without an ids array', async () => {
    const dir = await temporaryDir();
    const path = join(dir, 'bad.json');
    await writeFile(path, '{"tokens": []}', 'utf8');

    await expect(loadSplitIds(path)).rejects.toThrow();
  });
});
