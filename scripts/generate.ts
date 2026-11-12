
import { readdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { loadConfig } from '../src/core/load-config.js';
import { loadTokenizer } from '../src/core/save-tokenizer.js';
import { TransformerLM } from '../src/model/transformer.js';
import { generateText } from '../src/inference/generate.js';
import { loadCheckpoint, restoreCheckpoint } from '../src/training/checkpoint.js';
import { loadTrainedModel } from '../src/inference/load-model.js';
import type { CharacterTokenizer } from '../src/core/tokenizer.js';

export interface GenerateCliOptions {
  /** Path to an exported model bundle. If defined, ignores config/tokenizer/checkpoints. */
  modelDir?: string;
  configPath?: string;
  tokenizerPath?: string;
  /** Explicit checkpoint path. If omitted, uses the latest epoch checkpoint in checkpointDir. */
  checkpointPath?: string;
  checkpointDir?: string;
  prompt?: string;
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  seed?: number;
  log?: (message: string) => void;
  print?: (message: string) => void;
}

export async function findLatestCheckpoint(checkpointDir: string): Promise<string> {
  const files = await readdir(checkpointDir);
  const epochs = files
    .map((file) => /^checkpoint-epoch(\d+)\.json$/.exec(file))
    .filter((match): match is RegExpExecArray => match !== null)
    .map((match) => ({ epoch: Number(match[1]), file: match[0] }))
    .sort((a, b) => b.epoch - a.epoch);

  if (epochs.length === 0) {
    throw new Error(`No checkpoints found in ${checkpointDir}`);
  }

  return join(checkpointDir, epochs[0]!.file);
}

export async function main(
  options: GenerateCliOptions = {},
): Promise<string> {
  const log = options.log ?? ((message: string) => console.log(message));
  const print = options.print ?? ((message: string) => console.log(message));

  let model: TransformerLM;
  let tokenizer: CharacterTokenizer;

  if (options.modelDir !== undefined) {
    log(`Loading model bundle from ${options.modelDir}`);
    const bundle = await loadTrainedModel(options.modelDir);
    log(
      `Bundle '${bundle.manifest.name}' restaurado (epoch ${bundle.manifest.checkpoint.epoch}, valLoss ${bundle.manifest.checkpoint.valLoss})`,
    );
    model = bundle.model;
    tokenizer = bundle.tokenizer;
  } else {
    const configPath = options.configPath ?? 'configs/default.json';
    const tokenizerPath = options.tokenizerPath ?? 'data/processed/tokenizer.json';
    const checkpointDir = options.checkpointDir ?? 'checkpoints';

    log(`Loading configuration from ${configPath}`);
    const config = await loadConfig(configPath);

    log(`Loading tokenizer from ${tokenizerPath}`);
    const loadedTokenizer = await loadTokenizer(tokenizerPath);

    const checkpointPath = options.checkpointPath
      ?? await findLatestCheckpoint(checkpointDir);
    log(`Restoring checkpoint ${checkpointPath}`);
    await loadCheckpoint(checkpointPath);
    const newModel = new TransformerLM(config.model);
    await restoreCheckpoint(checkpointPath, newModel);

    model = newModel;
    tokenizer = loadedTokenizer;
  }

  const text = generateText(model, tokenizer, {
    prompt: options.prompt ?? '',
    maxNewTokens: options.maxNewTokens ?? 200,
    temperature: options.temperature,
    topK: options.topK,
    seed: options.seed,
  });

  print(`\n${text}\n`);
  model.dispose();

  return text;
}

const invokedDirectly = process.argv[1] !== undefined
  && basename(process.argv[1]) === 'generate.ts';

if (invokedDirectly) {
  const args = process.argv.slice(2);
  const modelDirIndex = args.indexOf('--model');

  if (modelDirIndex !== -1) {
    const modelDir = args[modelDirIndex + 1];
    const rest = args.filter((_, index) => index !== modelDirIndex && index !== modelDirIndex + 1);
    const promptArgument = rest[0];

    await main({
      ...(modelDir === undefined ? {} : { modelDir }),
      ...(promptArgument === undefined ? {} : { prompt: promptArgument }),
    });
  } else {
    const [configPath, promptArgument] = args;
    await main({
      ...(configPath === undefined ? {} : { configPath }),
      ...(promptArgument === undefined ? {} : { prompt: promptArgument }),
    });
  }
}
