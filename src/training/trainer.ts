
import { join } from 'node:path';
import * as tf from '@tensorflow/tfjs-node';
import type { CorvidConfig } from '../core/config.js';
import { TransformerLM } from '../model/transformer.js';
import { createBatches } from '../data/batches.js';
import { lossFromLogits } from './loss.js';
import { seededShuffle } from './shuffle.js';
import { saveCheckpoint } from './checkpoint.js';
import { applyWeightDecay } from './weight-decay.js';

export interface EpochMetrics {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export interface TrainerOptions {
  config: CorvidConfig;
  trainIds: readonly number[];
  valIds: readonly number[];
  checkpointDir?: string;
  onEpochEnd?: (metrics: EpochMetrics) => void | Promise<void>;
}

export interface TrainingResult {
  history: EpochMetrics[];
  steps: number;
  finalModel: TransformerLM;
}

function idsToTensor(rows: number[][], seqLen: number): tf.Tensor2D {
  return tf.tensor2d(rows, [rows.length, seqLen], 'int32');
}

export function evaluateLoss(
  model: TransformerLM,
  ids: readonly number[],
  config: CorvidConfig,
): number {
  const { maxSeqLen } = config.model;

  return tf.tidy(() => {
    const batches = createBatches(ids, {
      seqLen: maxSeqLen,
      batchSize: config.training.batchSize,
    });

    if (batches.length === 0) {
      return Number.NaN;
    }

    let total = 0;
    for (const batch of batches) {
      const inputs = idsToTensor(batch.inputs, maxSeqLen);
      const targets = idsToTensor(batch.targets, maxSeqLen);
      const logits = model.forward(inputs, { training: false });
      const loss = lossFromLogits(logits, targets);
      total += loss.dataSync()[0]!;
    }

    return total / batches.length;
  });
}

export async function trainModel(options: TrainerOptions): Promise<TrainingResult> {
  const { config } = options;
  const { maxSeqLen } = config.model;
  const { batchSize, epochs, seed } = config.training;

  const trainBatches = createBatches(options.trainIds, {
    seqLen: maxSeqLen,
    batchSize,
    dropLast: true,
  });

  if (trainBatches.length === 0) {
    const minimumTokens = (maxSeqLen + 1) * batchSize;

    throw new Error(
      `Not enough training data: ${options.trainIds.length} tokens produce `
      + `no training batches with maxSeqLen=${maxSeqLen} and batchSize=${batchSize} `
      + `(need at least ${minimumTokens} tokens). Check that data/raw contains text.`,
    );
  }

  const model = new TransformerLM(config.model, { seed });
  const optimizer = tf.train.adam(config.training.learningRate);
  const decayOptions = {
    weightDecay: config.training.weightDecay,
    learningRate: config.training.learningRate,
  };

  const history: EpochMetrics[] = [];
  let steps = 0;

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const batches = seededShuffle(trainBatches, seed + epoch);

    let trainLossSum = 0;
    let trainSteps = 0;

    for (const batch of batches) {
      const inputs = idsToTensor(batch.inputs, maxSeqLen);
      const targets = idsToTensor(batch.targets, maxSeqLen);

      const loss = optimizer.minimize(
        () => lossFromLogits(model.forward(inputs, { training: true }), targets),
        true,
        model.variables,
      ) as tf.Scalar;

      trainLossSum += loss.dataSync()[0]!;
      trainSteps += 1;
      steps += 1;

      // Decoupled weight decay (AdamW): after the Adam step.
      applyWeightDecay(model.variables, decayOptions);

      loss.dispose();
      inputs.dispose();
      targets.dispose();
    }

    const metrics: EpochMetrics = {
      epoch,
      trainLoss: trainSteps > 0 ? trainLossSum / trainSteps : Number.NaN,
      valLoss: evaluateLoss(model, options.valIds, config),
    };
    history.push(metrics);

    if (options.onEpochEnd) {
      await options.onEpochEnd(metrics);
    }

    if (options.checkpointDir !== undefined) {
      await saveCheckpoint(
        join(options.checkpointDir, `checkpoint-epoch${epoch}.json`),
        model,
        metrics,
      );
    }
  }

  return { history, steps, finalModel: model };
}
