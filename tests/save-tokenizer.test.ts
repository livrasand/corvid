
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import {
  loadTokenizer,
  saveTokenizer,
} from '../src/core/save-tokenizer.js';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

async function temporaryPath(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'corvid-tokenizer-'));
  temporaryDirectories.push(directory);

  return join(directory, 'processed', 'tokenizer.json');
}

describe('tokenizer persistence', () => {
  it('saves and loads a tokenizer, creating parent directories', async () => {
    const path = await temporaryPath();
    const tokenizer = CharacterTokenizer.fromCorpus('hello world');
    const text = 'mundo hola';

    await saveTokenizer(tokenizer, path);
    const loaded = await loadTokenizer(path);

    expect(loaded.vocabSize).toBe(tokenizer.vocabSize);
    expect(loaded.encode(text)).toEqual(tokenizer.encode(text));
    expect(loaded.decode(loaded.encode(text))).toBe(text);
  });

  it('writes pretty-printed JSON with a trailing newline', async () => {
    const path = await temporaryPath();
    await saveTokenizer(CharacterTokenizer.fromCorpus('hola'), path);

    const content = await readFile(path, 'utf8');
    expect(content.endsWith('\n')).toBe(true);
    expect(JSON.parse(content)).toHaveProperty('specials');
  });

  it('rejects files that do not contain a valid tokenizer', async () => {
    const path = await temporaryPath();
    await saveTokenizer(CharacterTokenizer.fromCorpus('hola'), path);

    await writeFile(path, '{"specials": []}', 'utf8');

    await expect(loadTokenizer(path)).rejects.toThrow();
  });

  it('rejects reading a missing file', async () => {
    await expect(loadTokenizer('data/processed/nonexistent.json'))
      .rejects.toThrow();
  });
});
