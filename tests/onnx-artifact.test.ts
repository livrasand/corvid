
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const bundleDir = join(process.cwd(), 'models', 'corvid-xxs');
const onnxPath = join(bundleDir, 'model.onnx');

describe('Corvid XXS ONNX artifact', () => {
  it.skipIf(!existsSync(onnxPath))('ships a single-file ONNX model', () => {
    const stat = statSync(onnxPath);
    // The model has ~1.08M float32 parameters (~4 MB); a file < 1 MB
    // would indicate the weights were left in an external .data file.
    expect(stat.size).toBeGreaterThan(1_000_000);

    const manifest = JSON.parse(
      readFileSync(join(bundleDir, 'model.json'), 'utf8'),
    );
    expect(manifest.files.onnx).toBe('model.onnx');
  });
});
