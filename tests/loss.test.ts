
import * as tf from '@tensorflow/tfjs-node';
import { describe, expect, it } from 'vitest';
import { lossFromLogits } from '../src/training/loss.js';

describe('lossFromLogits', () => {
  it('returns ln(vocabSize) for uniform logits', () => {
    return tf.tidy(() => {
      const logits = tf.zeros([1, 2, 4]) as tf.Tensor3D;
      const targets = tf.tensor2d([[0, 3]], [1, 2], 'int32');
      const loss = lossFromLogits(logits, targets);

      expect(Math.abs(loss.dataSync()[0]! - Math.log(4))).toBeLessThan(1e-5);
    });
  });

  it('returns zero for perfectly confident predictions', () => {
    return tf.tidy(() => {
      // Class 0 at position 0, class 3 at position 1.
      const logits = tf.tensor3d(
        [
          [[100, -100, -100, -100], [-100, -100, -100, 100]],
        ],
        [1, 2, 4],
      );
      const targets = tf.tensor2d([[0, 3]], [1, 2], 'int32');
      const loss = lossFromLogits(logits, targets);

      expect(loss.dataSync()[0]!).toBeLessThan(1e-4);
    });
  });

  it('averages over batch and sequence positions', () => {
    return tf.tidy(() => {
      // batch 2, seq 1: primera muestra perfecta, segunda uniforme.
      const logits = tf.tensor3d(
        [
          [[100, -100, -100, -100]],
          [[0, 0, 0, 0]],
        ],
        [2, 1, 4],
      );
      const targets = tf.tensor2d([[0], [1]], [2, 1], 'int32');
      const loss = lossFromLogits(logits, targets);

      expect(Math.abs(loss.dataSync()[0]! - Math.log(4) / 2)).toBeLessThan(1e-5);
    });
  });

  it('returns a finite positive scalar', () => {
    return tf.tidy(() => {
      const logits = tf.randomNormal([2, 3, 8]) as tf.Tensor3D;
      const targets = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
      const loss = lossFromLogits(logits, targets);

      expect(Number.isFinite(loss.dataSync()[0]!)).toBe(true);
      expect(loss.dataSync()[0]!).toBeGreaterThan(0);
    });
  });
});
