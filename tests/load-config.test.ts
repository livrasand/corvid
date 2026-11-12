
import { describe, expect, it } from 'vitest';
import { loadConfig } from '../src/core/load-config.js';

describe('loadConfig', () => {
  it('loads and validates the default configuration', async () => {
    const config = await loadConfig('configs/default.json');

    expect(config.model.dModel).toBe(128);
    expect(config.model.nHeads).toBe(4);
    expect(config.training.seed).toBe(42);
  });

  it('rejects invalid configuration files', async () => {
    await expect(
      loadConfig('package.json'),
    ).rejects.toThrow();
  });

  it('rejects files that do not exist', async () => {
    await expect(
      loadConfig('configs/nonexistent.json'),
    ).rejects.toThrow();
  });
});
