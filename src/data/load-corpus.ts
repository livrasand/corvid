
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

export async function loadCorpus(dir: string): Promise<string> {
  const entries = await readdir(dir, { withFileTypes: true });

  const files = entries
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();

  const contents: string[] = [];
  for (const name of files) {
    contents.push(await readFile(join(dir, name), 'utf8'));
  }

  return contents.join('\n');
}
