
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { CharacterTokenizer } from '../core/tokenizer.js';
import { loadCorpus } from './load-corpus.js';
import { splitText, type SplitRatios } from './splits.js';

export interface BuildSplitsOptions {
  rawDir: string;
  outDir: string;
  tokenizer: CharacterTokenizer;
  trainRatio?: number;
  valRatio?: number;
  testRatio?: number;
}

export interface SplitCounts {
  train: number;
  val: number;
  test: number;
}

export async function buildSplits(
  options: BuildSplitsOptions,
): Promise<SplitCounts> {
  const corpus = await loadCorpus(options.rawDir);
  const ratios: SplitRatios = {};
  if (options.trainRatio !== undefined) {
    ratios.trainRatio = options.trainRatio;
  }
  if (options.valRatio !== undefined) {
    ratios.valRatio = options.valRatio;
  }
  if (options.testRatio !== undefined) {
    ratios.testRatio = options.testRatio;
  }

  const splits = splitText(corpus, ratios);

  await mkdir(options.outDir, { recursive: true });

  const counts: SplitCounts = { train: 0, val: 0, test: 0 };

  for (const [name, text] of [
    ['train', splits.train],
    ['val', splits.val],
    ['test', splits.test],
  ] as const) {
    const ids = options.tokenizer.encode(text);
    counts[name] = ids.length;
    await writeFile(
      join(options.outDir, `${name}.json`),
      `${JSON.stringify({ ids }, null, 2)}\n`,
      'utf8',
    );
  }

  return counts;
}

export async function loadSplitIds(path: string): Promise<number[]> {
  const content = await readFile(path, 'utf8');
  const data: unknown = JSON.parse(content);

  if (
    typeof data !== 'object'
    || data === null
    || !Array.isArray((data as Record<string, unknown>).ids)
  ) {
    throw new Error(`Split file must contain an ids array: ${path}`);
  }

  return (data as { ids: number[] }).ids;
}
