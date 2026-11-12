
import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { loadConfig } from '../core/load-config.js';
import type { CorvidConfig, ModelConfig, TrainingConfig } from '../core/config.js';
import { loadTokenizer } from '../core/save-tokenizer.js';
import {
  loadCheckpoint,
  type Checkpoint,
  type CheckpointMetadata,
} from './checkpoint.js';

export const MODEL_BUNDLE_FORMAT = 1;

/** Nombre del archivo de manifiesto dentro del bundle. */
export const MANIFEST_FILENAME = 'model.json';

export interface ModelBundleFiles {
  checkpoint: string;
  tokenizer: string;
  config: string;
}

export interface ModelBundleManifest {
  name: string;
  format: number;
  createdAt: string;
  model: ModelConfig;
  training: TrainingConfig;
  checkpoint: CheckpointMetadata;
  files: ModelBundleFiles;
}

export interface ExportModelBundleOptions {
  /** Public name of the package (e.g. 'corvid-xxs'). */
  name: string;
  /** Destination directory of the bundle. */
  outputDir: string;
  checkpointPath: string;
  tokenizerPath: string;
  configPath: string;
  log?: (message: string) => void;
}

/**
 * Formas esperadas de las variables del modelo en el orden en que
 * TransformerLM las expone: 2 globales + 12 por bloque + 2 finales.
 */
export function expectedModelShapes(config: ModelConfig): number[][] {
  const shapes: number[][] = [
    [config.vocabSize, config.dModel],
    [config.maxSeqLen, config.dModel],
  ];

  for (let layer = 0; layer < config.nLayers; layer += 1) {
    shapes.push(
      [config.dModel, config.dModel],
      [config.dModel, config.dModel],
      [config.dModel, config.dModel],
      [config.dModel, config.dModel],
      [config.dModel],
      [config.dModel],
      [config.dModel],
      [config.dModel],
      [config.dModel, config.dFF],
      [config.dFF],
      [config.dFF, config.dModel],
      [config.dModel],
    );
  }

  shapes.push([config.dModel], [config.dModel]);

  return shapes;
}

function validateCheckpointShapes(
  checkpoint: Checkpoint,
  config: ModelConfig,
  checkpointPath: string,
): void {
  const shapes = expectedModelShapes(config);

  if (checkpoint.variables.length !== shapes.length) {
    throw new Error(
      `Checkpoint has ${checkpoint.variables.length} variables but the config expects ${shapes.length}`,
    );
  }

  for (const [index, variable] of checkpoint.variables.entries()) {
    const expected = shapes[index]!;

    if (
      variable.shape.length !== expected.length
      || variable.shape.some((dimension, i) => dimension !== expected[i])
    ) {
      throw new Error(
        `Checkpoint shape [${variable.shape.join(', ')}] does not match expected shape [${expected.join(', ')}] at variable ${index}`,
      );
    }
  }
}

/**
 * Packages an already trained model into a portable directory: copies the
 * checkpoint, the tokenizer and the config, and writes a manifest with the
 * training metrics. Does not require the model in memory.
 */
export async function exportModelBundle(
  options: ExportModelBundleOptions,
): Promise<ModelBundleManifest> {
  const { name, outputDir, checkpointPath, tokenizerPath, configPath } = options;
  const log = options.log ?? (() => {});

  const config: CorvidConfig = await loadConfig(configPath);
  log(`Config cargada desde ${configPath}`);

  const checkpoint = await loadCheckpoint(checkpointPath);
  log(`Checkpoint epoch ${checkpoint.metadata.epoch} (valLoss ${checkpoint.metadata.valLoss})`);
  validateCheckpointShapes(checkpoint, config.model, checkpointPath);

  const tokenizer = await loadTokenizer(tokenizerPath);

  if (tokenizer.vocabSize > config.model.vocabSize) {
    throw new Error(
      `Tokenizer has ${tokenizer.vocabSize} tokens but the model vocabSize is ${config.model.vocabSize}`,
    );
  }

  const bundleCheckpointPath = join(outputDir, 'checkpoint.json');
  const bundleTokenizerPath = join(outputDir, 'tokenizer.json');
  const bundleConfigPath = join(outputDir, 'config.json');

  await mkdir(outputDir, { recursive: true });
  await copyFile(checkpointPath, bundleCheckpointPath);
  await copyFile(tokenizerPath, bundleTokenizerPath);
  await copyFile(configPath, bundleConfigPath);

  const manifest: ModelBundleManifest = {
    name,
    format: MODEL_BUNDLE_FORMAT,
    createdAt: new Date().toISOString(),
    model: config.model,
    training: config.training,
    checkpoint: checkpoint.metadata,
    files: {
      checkpoint: 'checkpoint.json',
      tokenizer: 'tokenizer.json',
      config: 'config.json',
    },
  };

  await writeFile(
    join(outputDir, MANIFEST_FILENAME),
    `${JSON.stringify(manifest, null, 2)}\n`,
    'utf8',
  );

  log(`Bundle escrito en ${outputDir}`);

  return manifest;
}
