
import * as tf from '@tensorflow/tfjs-node';

export interface WeightDecayOptions {
  /** Factor lambda del decoupled weight decay (AdamW). 0 desactiva el decaimiento. */
  weightDecay: number;
  /** Learning rate del optimizador en el paso actual. */
  learningRate: number;
}

/**
 * Parameter classification in nanoGPT style: only matrices (rank 2) decay.
 * Rank 1 vectors (biases, LayerNorm gammas and betas) are exempt.
 */
export function isDecayable(variable: tf.Variable): boolean {
  return variable.rank === 2;
}

/**
 * Applies decoupled weight decay (AdamW): theta *= 1 - lr * lambda,
 * without going through Adam normalization. Executes after each optimizer
 * step, directly on the model variables.
 */
export function applyWeightDecay(
  variables: readonly tf.Variable[],
  options: WeightDecayOptions,
): void {
  const { weightDecay, learningRate } = options;

  if (weightDecay <= 0 || learningRate <= 0) {
    return;
  }

  const factor = 1 - learningRate * weightDecay;

  // tf.tidy evita fugar los tensores intermedios del mul; el tensor
  // asignado a la variable sobrevive al scope (assign lo protege).
  tf.tidy(() => {
    for (const variable of variables) {
      if (isDecayable(variable)) {
        variable.assign(variable.mul(factor));
      }
    }
  });
}
