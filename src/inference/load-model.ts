
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { CorvidConfig } from '../core/config.js';
import { loadConfig } from '../core/load-config.js';
import { loadTokenizer } from '../core/save-tokenizer.js';
import type { CharacterTokenizer } from '../core/tokenizer.js';
import { TransformerLM } from '../model/transformer.js';
import { restoreCheckpoint } from '../training/checkpoint.js';
import {
  MANIFEST_FILENAME,
  MODEL_BUNDLE_FORMAT,
  type ModelBundleManifest,
} from '../training/export-model.js';

export interface LoadedBundle {
  manifest: ModelBundleManifest;
  config: CorvidConfig;
  tokenizer: CharacterTokenizer;
  model: TransformerLM;
}

function isManifest(data: unknown): data is ModelBundleManifest {
  if (typeof data !== 'object' || data === null) {
    return false;
  }

  const record = data as Record<string, unknown>;
  const files = record.files;
  const checkpoint = record.checkpoint;

  return (
    typeof record.name === 'string'
    && typeof record.format === 'number'
    && typeof record.model === 'object'
    && record.model !== null
    && typeof record.training === 'object'
    && record.training !== null
    && typeof files === 'object'
    && files !== null
    && typeof (files as Record<string, unknown>).checkpoint === 'string'
    && typeof (files as Record<string, unknown>).tokenizer === 'string'
    && typeof (files as Record<string, unknown>).config === 'string'
    && typeof checkpoint === 'object'
    && checkpoint !== null
    && typeof (checkpoint as Record<string, unknown>).epoch === 'number'
  );
}

/**
 * Carga un bundle de modelo entrenado (exportado con exportModelBundle):
 * lee el manifiesto, restaura el checkpoint en un TransformerLM nuevo y
 * devuelve el modelo con su tokenizer y config listos para inferencia.
 */
export async function loadTrainedModel(bundleDir: string): Promise<LoadedBundle> {
  const manifestPath = join(bundleDir, MANIFEST_FILENAME);
  const content = await readFile(manifestPath, 'utf8');
  const parsed: unknown = JSON.parse(content);

  if (!isManifest(parsed)) {
    throw new Error(`Invalid model bundle manifest: ${manifestPath}`);
  }

  if (parsed.format !== MODEL_BUNDLE_FORMAT) {
    throw new Error(
      `Unsupported model bundle format ${parsed.format} (expected ${MODEL_BUNDLE_FORMAT})`,
    );
  }

  const config = await loadConfig(join(bundleDir, parsed.files.config));
  const tokenizer = await loadTokenizer(join(bundleDir, parsed.files.tokenizer));

  const model = new TransformerLM(config.model);
  const metadata = await restoreCheckpoint(
    join(bundleDir, parsed.files.checkpoint),
    model,
  );

  if (metadata.epoch !== parsed.checkpoint.epoch) {
    model.dispose();
    throw new Error(
      `Checkpoint epoch ${metadata.epoch} does not match manifest epoch ${parsed.checkpoint.epoch}`,
    );
  }

  return { manifest: parsed, config, tokenizer, model };
}
