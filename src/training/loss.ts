
import * as tf from '@tensorflow/tfjs-node';

export function lossFromLogits(
  logits: tf.Tensor3D,
  targets: tf.Tensor,
): tf.Scalar {
  const [batch = 0, seqLen = 0, vocabSize = 0] = logits.shape as [
    number,
    number,
    number,
  ];

  const logProbs = tf.logSoftmax(logits).reshape([
    batch * seqLen,
    vocabSize,
  ]) as tf.Tensor2D;
  const oneHotTargets = tf.oneHot(
    tf.cast(targets.reshape([batch * seqLen]), 'int32'),
    vocabSize,
  ) as tf.Tensor2D;
  const pickedLogProbs = tf.sum(tf.mul(logProbs, oneHotTargets), -1);

  return tf.neg(tf.mean(pickedLogProbs)) as tf.Scalar;
}
