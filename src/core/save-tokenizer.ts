
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { CharacterTokenizer } from './tokenizer.js';

export async function saveTokenizer(
  tokenizer: CharacterTokenizer,
  path: string,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(tokenizer.toJSON(), null, 2)}\n`, 'utf8');
}

export async function loadTokenizer(path: string): Promise<CharacterTokenizer> {
  const content = await readFile(path, 'utf8');
  const data: unknown = JSON.parse(content);

  return CharacterTokenizer.fromJSON(data);
}
