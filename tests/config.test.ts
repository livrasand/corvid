
import { describe, expect, it } from 'vitest';
import {
  validateConfig,
  type CorvidConfig,
} from '../src/core/config.js';

const validConfig: CorvidConfig = {
  model: {
    vocabSize: 2048,
    maxSeqLen: 256,
    dModel: 128,
    nHeads: 4,
    nLayers: 4,
    dFF: 512,
    dropout: 0.1,
    tieEmbeddings: true,
  },
  training: {
    batchSize: 4,
    learningRate: 0.0003,
    weightDecay: 0.01,
    epochs: 10,
    seed: 42,
  },
};

describe('Corvid configuration', () => {
  it('accepts a valid configuration', () => {
    expect(validateConfig(validConfig)).toEqual(validConfig);
  });

  it('rejects incompatible attention dimensions', () => {
    const config = structuredClone(validConfig);
    config.model.dModel = 128;
    config.model.nHeads = 3;

    expect(() => validateConfig(config)).toThrow();
  });

  it('rejects invalid vocabulary sizes', () => {
    const config = structuredClone(validConfig);
    config.model.vocabSize = 10;

    expect(() => validateConfig(config)).toThrow();
  });

  it('rejects invalid dropout values', () => {
    const config = structuredClone(validConfig);
    config.model.dropout = 0.9;

    expect(() => validateConfig(config)).toThrow();
  });
});
