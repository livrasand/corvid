
import { mkdir, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { loadCorpus } from '../src/data/load-corpus.js';

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
  const directory = await mkdtemp(join(tmpdir(), 'corvid-corpus-'));
  temporaryDirectories.push(directory);

  return directory;
}

describe('loadCorpus', () => {
  it('reads files in sorted name order joined with newlines', async () => {
    const dir = await temporaryDir();
    await writeFile(join(dir, 'b.txt'), 'segundo');
    await writeFile(join(dir, 'a.txt'), 'primero');

    expect(await loadCorpus(dir)).toBe('primero\nsegundo');
  });

  it('ignores subdirectories and non-files', async () => {
    const dir = await temporaryDir();
    await mkdir(join(dir, 'nested'));
    await writeFile(join(dir, 'nested', 'inner.txt'), 'oculto');
    await writeFile(join(dir, 'main.txt'), 'visible');

    expect(await loadCorpus(dir)).toBe('visible');
  });

  it('returns an empty string for an empty directory', async () => {
    const dir = await temporaryDir();

    expect(await loadCorpus(dir)).toBe('');
  });
});
