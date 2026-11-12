
import { afterEach, describe, expect, it } from 'vitest';
import type { TransformerLM } from '../src/model/transformer.js';
import { createTransformerLM } from '../src/model/transformer.js';
import { CharacterTokenizer } from '../src/core/tokenizer.js';
import {
  generateText,
  generateTokenIds,
  sampleFromDistribution,
  topKFilter,
} from '../src/inference/generate.js';

// The schema requires vocabSize >= 256; the tokenizer must cover that range
// so decode does not fail on IDs outside its vocabulary.
const alphabet = Array.from(
  { length: 300 },
  (_, index) => String.fromCharCode(33 + index),
).join('');
const tokenizer = CharacterTokenizer.fromCorpus(alphabet);

const modelConfig = {
  vocabSize: 256,
  maxSeqLen: 32,
  dModel: 32,
  nHeads: 4,
  nLayers: 1,
  dFF: 64,
  dropout: 0.1,
  tieEmbeddings: true,
};

const models: TransformerLM[] = [];

afterEach(() => {
  while (models.length > 0) {
    models.pop()!.dispose();
  }
});

function makeModel(): TransformerLM {
  const model = createTransformerLM(modelConfig, { seed: 11 });
  models.push(model);

  return model;
}

describe('sampleFromDistribution', () => {
  it('samples deterministically from the cumulative distribution', () => {
    const random = () => 0.5;

    // [0.4, 0.4, 0.2]: threshold 0.5 falls on the second element.
    expect(sampleFromDistribution([0.4, 0.4, 0.2], random)).toBe(1);
    // Threshold 0 falls on the first.
    expect(sampleFromDistribution([0.4, 0.4, 0.2], () => 0)).toBe(0);
    // Very high threshold falls on the last (fallback).
    expect(sampleFromDistribution([0.4, 0.4, 0.2], () => 0.999)).toBe(2);
  });
});

describe('topKFilter', () => {
  it('keeps the k highest probabilities and renormalizes', () => {
    const result = topKFilter([0.1, 0.5, 0.2, 0.2], 2);

    // The two most probable are ids 1 (0.5) and 2 (0.2).
    expect(result.ids).toEqual([1, 2]);
    // Renormalizados: 0.5/0.7 y 0.2/0.7.
    expect(result.probabilities[0]).toBeCloseTo(5 / 7, 6);
    expect(result.probabilities[1]).toBeCloseTo(2 / 7, 6);
  });

  it('clamps k to the vocabulary size', () => {
    const result = topKFilter([0.5, 0.3, 0.2], 10);

    expect(result.ids.length).toBe(3);
    expect(result.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it('falls back to a uniform distribution when probabilities are all zero', () => {
    const result = topKFilter([0, 0, 0], 2);

    expect(result.probabilities).toEqual([0.5, 0.5]);
  });

  it('throws for non-positive k', () => {
    expect(() => topKFilter([1], 0)).toThrow();
    expect(() => topKFilter([1], -1)).toThrow();
  });

  it('returns a single deterministic winner with k = 1', () => {
    const result = topKFilter([0.2, 0.8], 1);

    expect(result.ids).toEqual([1]);
    expect(result.probabilities).toEqual([1]);
  });
});

describe('generateTokenIds', () => {
  it('is deterministic with the same seed and differs across seeds', () => {
    const model = makeModel();
    const options = { prompt: 'abc', maxNewTokens: 10, seed: 42 };

    const first = generateTokenIds(model, tokenizer, options);
    const second = generateTokenIds(model, tokenizer, options);
    const third = generateTokenIds(model, tokenizer, { ...options, seed: 7 });

    expect(first).toEqual(second);
    expect(third).not.toEqual(first);
  });

  it('greedy sampling (temperature 0) is deterministic regardless of seed', () => {
    const model = makeModel();

    const first = generateTokenIds(model, tokenizer, {
      prompt: 'abc',
      maxNewTokens: 10,
      temperature: 0,
      seed: 1,
    });
    const second = generateTokenIds(model, tokenizer, {
      prompt: 'abc',
      maxNewTokens: 10,
      temperature: 0,
      seed: 999,
    });

    expect(first).toEqual(second);
  });

  it('produces at most maxNewTokens ids within the vocabulary range', () => {
    const model = makeModel();
    const ids = generateTokenIds(model, tokenizer, {
      prompt: 'abc',
      maxNewTokens: 25,
      seed: 5,
    });

    // The untrained model may sample <eos> and stop early.
    expect(ids.length).toBeGreaterThan(0);
    expect(ids.length).toBeLessThanOrEqual(25);
    expect(ids.every((id) => id >= 0 && id < modelConfig.vocabSize)).toBe(true);
  });

  it('generates exactly maxNewTokens when stopAtEos is disabled', () => {
    const model = makeModel();
    const ids = generateTokenIds(model, tokenizer, {
      prompt: 'abc',
      maxNewTokens: 25,
      seed: 5,
      stopAtEos: false,
    });

    expect(ids.length).toBe(25);
  });

  it('truncates the context to maxSeqLen without throwing on long prompts', () => {
    const model = makeModel();
    const longPrompt = alphabet.slice(0, 40); // 40 > maxSeqLen 32

    const ids = generateTokenIds(model, tokenizer, {
      prompt: longPrompt,
      maxNewTokens: 5,
      seed: 3,
    });

    expect(ids.length).toBe(5);
  });

  it('starts from <bos> when the prompt is empty', () => {
    const model = makeModel();
    // With the same seed, empty prompt and '<bos>-wrong' prompt differ:
    // we check at least that empty prompt has no error and generates tokens.
    const ids = generateTokenIds(model, tokenizer, {
      prompt: '',
      maxNewTokens: 8,
      seed: 42,
    });

    expect(ids.length).toBe(8);
    // The initial <bos> should not appear in the output.
    expect(ids).not.toContain(tokenizer.bosId);
  });

  it('throws when maxNewTokens is less than 1', () => {
    const model = makeModel();

    expect(() =>
      generateTokenIds(model, tokenizer, { prompt: 'a', maxNewTokens: 0 }),
    ).toThrow();
  });

  it('works with topK sampling and returns ids within the vocabulary range', () => {
    const model = makeModel();
    const ids = generateTokenIds(model, tokenizer, {
      prompt: 'abc',
      maxNewTokens: 15,
      topK: 10,
      seed: 21,
    });

    expect(ids.length).toBe(15);
    expect(ids.every((id) => id >= 0 && id < modelConfig.vocabSize)).toBe(true);
  });
});

describe('generateText', () => {
  it('returns decodable text of one character per generated token', () => {
    const model = makeModel();
    const prompt = 'abc';
    const ids = generateTokenIds(model, tokenizer, {
      prompt,
      maxNewTokens: 12,
      seed: 42,
    });
    const text = generateText(model, tokenizer, {
      prompt,
      maxNewTokens: 12,
      seed: 42,
    });

    expect(text.length).toBe(ids.length);
  });

  it('only uses characters from the tokenizer vocabulary', () => {
    const model = makeModel();
    const text = generateText(model, tokenizer, {
      prompt: 'abc',
      maxNewTokens: 30,
      seed: 8,
    });

    for (const char of text) {
      expect(alphabet.includes(char)).toBe(true);
    }
  });
});
