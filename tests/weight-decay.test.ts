
import * as tf from '@tensorflow/tfjs-node';
import { afterEach, describe, expect, it } from 'vitest';
import type { TransformerLM } from '../src/model/transformer.js';
import { createTransformerLM } from '../src/model/transformer.js';
import {
  applyWeightDecay,
  isDecayable,
} from '../src/training/weight-decay.js';

const variables: tf.Variable[] = [];
const models: TransformerLM[] = [];

afterEach(() => {
  while (variables.length > 0) {
    variables.pop()!.dispose();
  }
  while (models.length > 0) {
    models.pop()!.dispose();
  }
});

function makeVariable(data: number[], shape: number[]): tf.Variable {
  const variable = tf.variable(tf.tensor(data, shape));
  variables.push(variable);

  return variable;
}

describe('isDecayable', () => {
  it('marks rank-2 variables (matrices) as decayable', () => {
    const matrix = makeVariable([1, 2, 3, 4], [2, 2]);

    expect(isDecayable(matrix)).toBe(true);
  });

  it('excludes rank-1 variables (biases, layernorms)', () => {
    const vector = makeVariable([1, 2, 3], [3]);
    const scalar = makeVariable([5], [1]);

    expect(isDecayable(vector)).toBe(false);
    expect(isDecayable(scalar)).toBe(false);
  });
});

describe('applyWeightDecay', () => {
  it('shrinks rank-2 variables by the factor 1 - lr * lambda', () => {
    const matrix = makeVariable([1, -2, 3, -4], [2, 2]);

    applyWeightDecay([matrix], { weightDecay: 0.1, learningRate: 0.001 });

    const factor = 1 - 0.001 * 0.1;
    expect(matrix.dataSync()[0]).toBeCloseTo(1 * factor, 6);
    expect(matrix.dataSync()[1]).toBeCloseTo(-2 * factor, 6);
    expect(matrix.dataSync()[2]).toBeCloseTo(3 * factor, 6);
    expect(matrix.dataSync()[3]).toBeCloseTo(-4 * factor, 6);
  });

  it('leaves rank-1 variables untouched', () => {
    const vector = makeVariable([7, -8, 9], [3]);
    const matrix = makeVariable([1, 2, 3, 4], [2, 2]);

    applyWeightDecay([vector, matrix], { weightDecay: 0.5, learningRate: 0.1 });

    expect(Array.from(vector.dataSync())).toEqual([7, -8, 9]);
    // The factor in this case: 1 - 0.1 * 0.5 = 0.95.
    expect(matrix.dataSync()[0]).toBeCloseTo(0.95, 6);
  });

  it('is a no-op when weightDecay is zero', () => {
    const matrix = makeVariable([1.5, 2.5, 3.5, 4.5], [2, 2]);

    applyWeightDecay([matrix], { weightDecay: 0, learningRate: 0.1 });

    expect(Array.from(matrix.dataSync())).toEqual([1.5, 2.5, 3.5, 4.5]);
  });

  it('applied twice composes the factors multiplicatively', () => {
    const matrix = makeVariable([2, 4, 6, 8], [2, 2]);
    const options = { weightDecay: 0.1, learningRate: 0.1 }; // factor 0.99

    applyWeightDecay([matrix], options);
    applyWeightDecay([matrix], options);

    expect(matrix.dataSync()[0]).toBeCloseTo(2 * 0.99 * 0.99, 6);
  });
});

describe('model parameter classification', () => {
  it('decays exactly the matmul weights and embeddings of the model', () => {
    const model = createTransformerLM(
      {
        vocabSize: 256,
        maxSeqLen: 32,
        dModel: 32,
        nHeads: 4,
        nLayers: 2,
        dFF: 64,
        dropout: 0.1,
        tieEmbeddings: true,
      },
      { seed: 3 },
    );
    models.push(model);

    // Per block: 4 attention projections + 2 FFN matrices = 6.
    // Global: token embedding + position embedding = 2.
    const decayable = model.variables.filter(isDecayable);

    expect(decayable.length).toBe(6 * model.config.nLayers + 2);
    for (const variable of decayable) {
      expect(variable.rank).toBe(2);
    }
  });
});
