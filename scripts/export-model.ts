
import { basename, join } from 'node:path';
import { findLatestCheckpoint } from './generate.js';
import { exportModelBundle } from '../src/training/export-model.js';
import type { ModelBundleManifest } from '../src/training/export-model.js';

export interface ExportCliOptions {
  name?: string;
  outputDir?: string;
  checkpointPath?: string;
  checkpointDir?: string;
  tokenizerPath?: string;
  configPath?: string;
  log?: (message: string) => void;
}

export async function main(
  options: ExportCliOptions = {},
): Promise<ModelBundleManifest> {
  const log = options.log ?? ((message: string) => console.log(message));
  const name = options.name ?? 'corvid-xxs';
  const outputDir = options.outputDir ?? join('models', name);
  const configPath = options.configPath ?? 'configs/default.json';
  const tokenizerPath = options.tokenizerPath ?? 'data/processed/tokenizer.json';
  const checkpointPath = options.checkpointPath
    ?? await findLatestCheckpoint(options.checkpointDir ?? 'checkpoints');

  log(`Exportando bundle '${name}' desde ${checkpointPath}`);
  const manifest = await exportModelBundle({
    name,
    outputDir,
    checkpointPath,
    tokenizerPath,
    configPath,
    log,
  });
  log(`Bundle '${name}' listo en ${outputDir} (epoch ${manifest.checkpoint.epoch})`);

  return manifest;
}

const invokedDirectly = process.argv[1] !== undefined
  && basename(process.argv[1]) === 'export-model.ts';

if (invokedDirectly) {
  await main();
}
