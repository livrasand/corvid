
import * as tf from '@tensorflow/tfjs-node';
import type { CharacterTokenizer } from '../core/tokenizer.js';
import type { TransformerLM } from '../model/transformer.js';
import { mulberry32 } from '../training/shuffle.js';

export interface GenerateOptions {
  /** Starting text. If empty, generation starts from <bos>. */
  prompt: string;
  /** Maximum number of new tokens to generate. */
  maxNewTokens: number;
  /** Sampling temperature. 0 or lower = greedy (argmax). Default 1. */
  temperature?: number | undefined;
  /** Top-k sampling: only the k most probable tokens participate. */
  topK?: number | undefined;
  /** Seed for reproducible sampling. */
  seed?: number | undefined;
  /** Stop generation when sampling <eos>. Default true. */
  stopAtEos?: boolean;
}

export function sampleFromDistribution(
  probabilities: readonly number[],
  random: () => number,
): number {
  const threshold = random();
  let cumulative = 0;

  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index]!;
    if (threshold < cumulative) {
      return index;
    }
  }

  return probabilities.length - 1;
}

export interface TopKResult {
  /** Original IDs of the k most probable tokens, in descending order. */
  ids: number[];
  /** Renormalized probabilities of those k tokens. */
  probabilities: number[];
}

/**
 * Filters the distribution to the k most probable tokens and renormalizes.
 * Implemented in pure JS because the tf.topk kernel of tfjs-node is
 * broken in this version (util.isNullOrUndefined does not exist).
 */
export function topKFilter(
  probabilities: readonly number[],
  k: number,
): TopKResult {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error(`k must be a positive integer, got ${k}`);
  }

  const limit = Math.min(k, probabilities.length);
  const indexed = probabilities.map((probability, id) => ({ probability, id }));
  indexed.sort((a, b) => b.probability - a.probability);
  const top = indexed.slice(0, limit);
  const total = top.reduce((sum, entry) => sum + entry.probability, 0);

  return {
    ids: top.map((entry) => entry.id),
    probabilities: total > 0
      ? top.map((entry) => entry.probability / total)
      : top.map(() => 1 / limit),
  };
}

/**
 * Generates tokens autoregressively and returns ONLY the generated tokens
 * (without the prompt and without <eos> if stopped by it).
 */
export function generateTokenIds(
  model: TransformerLM,
  tokenizer: CharacterTokenizer,
  options: GenerateOptions,
): number[] {
  const {
    prompt,
    maxNewTokens,
    temperature = 1,
    topK,
    seed,
    stopAtEos = true,
  } = options;

  if (maxNewTokens < 1) {
    throw new Error('maxNewTokens must be at least 1');
  }

  // Only tokens known by the tokenizer are sampled: if the model has a
  // vocabSize larger than the real vocabulary, surplus IDs could never
  // be decoded.
  const sampleVocab = Math.min(
    model.config.vocabSize,
    tokenizer.vocabSize,
  );
  const maxSeqLen = model.config.maxSeqLen;
  const random = seed === undefined ? Math.random : mulberry32(seed);

  const ids = prompt === ''
    ? [tokenizer.bosId]
    : tokenizer.encode(prompt);
  const generated: number[] = [];

  for (let step = 0; step < maxNewTokens; step += 1) {
    // Context window: the last maxSeqLen tokens.
    const context = ids.slice(-maxSeqLen);

    const nextId = tf.tidy(() => {
      const input = tf.tensor2d(
        [context],
        [1, context.length],
        'int32',
      );
      const logits = model.forward(input) as tf.Tensor3D;
      const lastPosition = logits.shape[1]! - 1;
      const lastLogits = logits
        .slice([0, lastPosition, 0], [1, 1, sampleVocab])
        .reshape([sampleVocab]) as tf.Tensor1D;

      if (temperature <= 0) {
        return tf.argMax(lastLogits).dataSync()[0]!;
      }

      const scaled = lastLogits.div(temperature);
      const probabilities = Array.from(
        (tf.softmax(scaled) as tf.Tensor1D).dataSync(),
      );

      if (topK !== undefined) {
        const k = Math.min(Math.max(Math.trunc(topK), 1), sampleVocab);
        const filtered = topKFilter(probabilities, k);
        const localIndex = sampleFromDistribution(filtered.probabilities, random);

        return filtered.ids[localIndex]!;
      }

      return sampleFromDistribution(probabilities, random);
    });

    if (stopAtEos && nextId === tokenizer.eosId) {
      break;
    }

    ids.push(nextId);
    generated.push(nextId);
  }

  return generated;
}

/** Genera texto autoregresivo a partir de un prompt, decodificando los tokens nuevos. */
export function generateText(
  model: TransformerLM,
  tokenizer: CharacterTokenizer,
  options: GenerateOptions,
): string {
  const generatedIds = generateTokenIds(model, tokenizer, options);

  return tokenizer.decode(generatedIds, { skipSpecials: true });
}
