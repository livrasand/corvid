
import * as tf from '@tensorflow/tfjs-node';
import { afterEach, describe, expect, it } from 'vitest';
import { createTransformerLM } from '../src/model/transformer.js';
import type { TransformerLM } from '../src/model/transformer.js';

const baseConfig = {
  vocabSize: 256,
  maxSeqLen: 32,
  dModel: 32,
  nHeads: 4,
  nLayers: 2,
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

function makeModel(
  overrides: Partial<typeof baseConfig> = {},
  options: { seed?: number } = {},
): TransformerLM {
  const model = createTransformerLM({ ...baseConfig, ...overrides }, options);
  models.push(model);

  return model;
}

function idsTensor(values: number[][]): tf.Tensor {
  return tf.tensor2d(values, [values.length, values[0]!.length], 'int32');
}

function isAllClose(a: tf.Tensor, b: tf.Tensor, tolerance = 1e-5): boolean {
  const valuesA = Array.from(a.dataSync());
  const valuesB = Array.from(b.dataSync());

  if (valuesA.length !== valuesB.length) {
    return false;
  }

  return valuesA.every(
    (value, index) => Math.abs(value - valuesB[index]!) <= tolerance,
  );
}

describe('TransformerLM', () => {
  it('produces logits with shape [batch, seqLen, vocabSize]', () => {
    const model = makeModel();
    const ids = idsTensor([[1, 2, 3, 4], [5, 6, 7, 8]]);

    return tf.tidy(() => {
      const logits = model.forward(ids);

      expect(logits.shape).toEqual([2, 4, baseConfig.vocabSize]);
    });
  });

  it('produces finite logits', () => {
    const model = makeModel();
    const ids = idsTensor([[1, 2, 3]]);

    return tf.tidy(() => {
      const logits = model.forward(ids);
      const values = Array.from(logits.dataSync());

      expect(values.every((value) => Number.isFinite(value))).toBe(true);
    });
  });

  it('does not attend to future tokens (causal mask)', () => {
    const model = makeModel();
    const idsA = idsTensor([[1, 2, 3, 4]]);
    const idsB = idsTensor([[1, 2, 3, 9]]); // only changes the last token

    return tf.tidy(() => {
      const logitsA = model.forward(idsA);
      const logitsB = model.forward(idsB);

      const prefix = (t: tf.Tensor3D) => t.slice([0, 0, 0], [1, 3, baseConfig.vocabSize]);
      const last = (t: tf.Tensor3D) => t.slice([0, 3, 0], [1, 1, baseConfig.vocabSize]);

      expect(isAllClose(prefix(logitsA), prefix(logitsB))).toBe(true);
      expect(isAllClose(last(logitsA), last(logitsB))).toBe(false);
    });
  });

  it('is reproducible with the same seed', () => {
    const modelA = makeModel({}, { seed: 7 });
    const modelB = makeModel({}, { seed: 7 });
    const ids = idsTensor([[1, 2, 3, 4, 5]]);

    return tf.tidy(() => {
      const logitsA = modelA.forward(ids);
      const logitsB = modelB.forward(ids);

      expect(isAllClose(logitsA, logitsB)).toBe(true);
    });
  });

  it('applies dropout only in training mode', () => {
    const model = makeModel({ dropout: 0.5 });
    const ids = idsTensor([[1, 2, 3, 4]]);

    return tf.tidy(() => {
      const evalLogits = model.forward(ids, { training: false });
      const trainLogits = model.forward(ids, { training: true });

      expect(isAllClose(evalLogits, trainLogits)).toBe(false);
    });
  });

  it('exposes one variable per expected parameter group', () => {
    const model = makeModel();
    // Per block: 4 projections + 2 layernorms (gamma+beta) + 4 FFN = 12.
    // Global: token embedding + position embedding + final gamma/beta = 4.
    expect(model.variables.length).toBe(12 * baseConfig.nLayers + 4);
  });

  it('rejects configurations where dModel is not divisible by nHeads', () => {
    expect(() => makeModel({ nHeads: 3 })).toThrow();
  });

  it('rejects sequences longer than maxSeqLen', () => {
    const model = makeModel(); // maxSeqLen 32
    const ids = idsTensor([Array.from({ length: 40 }, (_, i) => i % 256)]);

    expect(() => model.forward(ids)).toThrow();
  });

  it('throws when forward is called after dispose', () => {
    const model = makeModel();
    const ids = idsTensor([[1, 2, 3]]);
    model.dispose();

    expect(() => model.forward(ids)).toThrow();
  });
});
