
import { existsSync } from 'node:fs';
import { basename, join } from 'node:path';
import { loadConfig } from '../src/core/load-config.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import { loadTokenizer, saveTokenizer } from '../src/core/save-tokenizer.js';
import { loadCorpus } from '../src/data/load-corpus.js';
import { buildSplits, loadSplitIds } from '../src/data/build-splits.js';
import { trainModel } from '../src/training/trainer.js';

export interface TrainCliOptions {
  configPath?: string;
  rawDir?: string;
  outDir?: string;
  tokenizerPath?: string;
  checkpointDir?: string;
  log?: (message: string) => void;
}

export async function main(options: TrainCliOptions = {}): Promise<void> {
  const log = options.log ?? ((message: string) => console.log(message));
  const configPath = options.configPath ?? 'configs/default.json';
  const rawDir = options.rawDir ?? 'data/raw';
  const outDir = options.outDir ?? 'data/splits';
  const tokenizerPath = options.tokenizerPath ?? 'data/processed/tokenizer.json';
  const checkpointDir = options.checkpointDir ?? 'checkpoints';

  log(`Loading configuration from ${configPath}`);
  const config = await loadConfig(configPath);

  log('Loading corpus...');
  const corpus = await loadCorpus(rawDir);

  let tokenizer: CharacterTokenizer;
  if (existsSync(tokenizerPath)) {
    tokenizer = await loadTokenizer(tokenizerPath);

    // A tokenizer with vocabSize <= 4 only has special tokens
    // (likely built from an empty corpus): not usable, rebuilds
    // from the current corpus.
    if (tokenizer.vocabSize <= 4) {
      log(
        `The tokenizer at ${tokenizerPath} is invalid `
        + `(${tokenizer.vocabSize} tokens): will rebuild from the corpus`,
      );
      tokenizer = CharacterTokenizer.fromCorpus(corpus, {
        maxVocabSize: config.model.vocabSize,
      });
      await saveTokenizer(tokenizer, tokenizerPath);
      log(`Tokenizer rebuilt and saved at ${tokenizerPath} (${tokenizer.vocabSize} tokens)`);
    } else {
      log(`Tokenizer loaded from ${tokenizerPath} (${tokenizer.vocabSize} tokens)`);
    }
  } else {
    tokenizer = CharacterTokenizer.fromCorpus(corpus, {
      maxVocabSize: config.model.vocabSize,
    });
    await saveTokenizer(tokenizer, tokenizerPath);
    log(`New tokenizer saved at ${tokenizerPath} (${tokenizer.vocabSize} tokens)`);
  }

  log('Building splits...');
  const counts = await buildSplits({ rawDir, outDir, tokenizer });
  log(`Splits: train=${counts.train} val=${counts.val} test=${counts.test} tokens`);

  const [trainIds, valIds] = await Promise.all([
    loadSplitIds(join(outDir, 'train.json')),
    loadSplitIds(join(outDir, 'val.json')),
  ]);

  log('Training...');
  const result = await trainModel({
    config,
    trainIds,
    valIds,
    checkpointDir,
    onEpochEnd: (metrics) => {
      log(
        `Epoch ${metrics.epoch}: trainLoss=${metrics.trainLoss.toFixed(4)} `
        + `valLoss=${metrics.valLoss.toFixed(4)} (checkpoint at ${checkpointDir})`,
      );
    },
  });

  result.finalModel.dispose();
  log(`Done: ${result.steps} steps in ${result.history.length} epochs.`);
}

const invokedDirectly = process.argv[1] !== undefined
  && basename(process.argv[1]) === 'train.ts';

if (invokedDirectly) {
  const configPath = process.argv[2];
  await main(configPath === undefined ? {} : { configPath });
}
