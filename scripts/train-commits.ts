import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import { existsSync, mkdirSync } from 'node:fs';
import { join, basename } from 'node:path';
import { loadConfig } from '../src/core/load-config.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { saveTokenizer } from '../src/core/save-tokenizer.js';
import { trainModel } from '../src/training/trainer.js';

export interface TrainCommitsOptions {
  csvPath?: string;
  configPath?: string;
  outDir?: string;
  tokenizerPath?: string;
  checkpointDir?: string;
  maxSamples?: number;
  log?: (message: string) => void;
}

function parseCsvLine(line: string): { diff: string; message: string } | null {
  const firstComma = line.indexOf(',');
  if (firstComma === -1) return null;

  let rest = line.slice(firstComma + 1);

  const diffEndMarker = ',"';
  const diffEndIdx = rest.indexOf(diffEndMarker);
  if (diffEndIdx === -1) return null;

  let diff = rest.slice(0, diffEndIdx);
  let afterDiff = rest.slice(diffEndIdx + diffEndMarker.length);

  const messageEndIdx = afterDiff.indexOf('",');
  if (messageEndIdx === -1) return null;

  const message = afterDiff.slice(0, messageEndIdx);

  diff = diff.replace(/""/g, '"');
  const messageUnescaped = message.replace(/""/g, '"');

  return { diff, message: messageUnescaped };
}

function buildTrainingExample(diff: string, message: string): string {
  return `<bos>diff:\n${diff}\nmessage:\n${message}<eos>\n`;
}

async function* readCommitsCsv(
  csvPath: string,
  maxSamples?: number,
): AsyncGenerator<{ diff: string; message: string }> {
  const fileStream = createReadStream(csvPath, { encoding: 'utf8' });
  const rl = createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  let firstLine = true;
  let count = 0;

  for await (const line of rl) {
    if (firstLine) {
      firstLine = false;
      continue;
    }

    if (!line.trim()) continue;

    const parsed = parseCsvLine(line);
    if (parsed) {
      yield parsed;
      count += 1;
      if (maxSamples && count >= maxSamples) break;
    }
  }
}

export async function main(options: TrainCommitsOptions = {}): Promise<void> {
  const log = options.log ?? ((message: string) => console.log(message));
  const csvPath = options.csvPath ?? 'data/source/commitbench.csv';
  const configPath = options.configPath ?? 'configs/default.json';
  const outDir = options.outDir ?? 'models/corvid-commits';
  const tokenizerPath = options.tokenizerPath ?? join(outDir, 'tokenizer.json');
  const checkpointDir = options.checkpointDir ?? join(outDir, 'checkpoints');
  const maxSamples = options.maxSamples;

  if (!existsSync(csvPath)) {
    throw new Error(`CSV not found: ${csvPath}`);
  }

  mkdirSync(outDir, { recursive: true });
  mkdirSync(checkpointDir, { recursive: true });

  log(`Loading configuration from ${configPath}`);
  const config = await loadConfig(configPath);

  log(`Reading commits from ${csvPath}...`);
  const examples: string[] = [];
  let sampleCount = 0;

  for await (const commit of readCommitsCsv(csvPath, maxSamples)) {
    const example = buildTrainingExample(commit.diff, commit.message);
    examples.push(example);
    sampleCount += 1;

    if (sampleCount % 10000 === 0) {
      log(`  Processed ${sampleCount} commits...`);
    }
  }

  log(`Total commits for training: ${sampleCount}`);

  const corpus = examples.join('\n');
  log(`Corpus size: ${(corpus.length / 1024 / 1024).toFixed(2)} MB`);

  log('Building tokenizer from commit corpus...');
  const tokenizer = CharacterTokenizer.fromCorpus(corpus, {
    maxVocabSize: config.model.vocabSize,
  });
  await saveTokenizer(tokenizer, tokenizerPath);
  log(`Tokenizer saved at ${tokenizerPath} (${tokenizer.vocabSize} tokens)`);

  log('Encoding corpus as token IDs...');
  const allIds: number[] = [];
  for (const example of examples) {
    allIds.push(...tokenizer.encode(example, { bos: false, eos: true }));
  }

  log(`Total tokens: ${allIds.length}`);

  log('Creating train/val splits...');
  const splitIdx = Math.floor(allIds.length * 0.9);
  const trainIds = allIds.slice(0, splitIdx);
  const valIds = allIds.slice(splitIdx);

  log(`Train tokens: ${trainIds.length}, Val tokens: ${valIds.length}`);

  log('Training model for commit message generation...');
  const result = await trainModel({
    config,
    trainIds,
    valIds,
    checkpointDir,
    onEpochEnd: (metrics) => {
      log(
        `Epoch ${metrics.epoch}: trainLoss=${metrics.trainLoss.toFixed(4)} `
        + `valLoss=${metrics.valLoss.toFixed(4)}`,
      );
    },
  });

  result.finalModel.dispose();
  log(`Done: ${result.steps} steps in ${result.history.length} epochs.`);
  log(`Model saved at ${checkpointDir}`);
}

const invokedDirectly = process.argv[1] !== undefined
  && basename(process.argv[1]) === 'train-commits.ts';

if (invokedDirectly) {
  const csvPath = process.argv[2];
  const maxSamples = process.argv[3] ? Number(process.argv[3]) : undefined;
  await main(csvPath === undefined ? {} : { csvPath, maxSamples });
}
