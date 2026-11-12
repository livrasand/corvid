
import * as tf from '@tensorflow/tfjs-node';
import { ModelConfigSchema } from '../core/config.js';
import type { ModelConfig } from '../core/config.js';

const LAYER_NORM_EPSILON = 1e-5;
const INIT_STDDEV = 0.02;

export interface TransformerOptions {
  seed?: number;
}

export interface ForwardOptions {
  training?: boolean;
}

export interface LayerNormParams {
  gamma: tf.Variable;
  beta: tf.Variable;
}

function randomVariable(shape: number[], seed?: number): tf.Variable {
  return tf.variable(tf.randomNormal(shape, 0, INIT_STDDEV, 'float32', seed));
}

function zerosVariable(shape: number[]): tf.Variable {
  return tf.variable(tf.zeros(shape));
}

function onesVariable(shape: number[]): tf.Variable {
  return tf.variable(tf.ones(shape));
}

function layerNorm(x: tf.Tensor, params: LayerNormParams): tf.Tensor {
  const axis = x.shape.length - 1;
  const { mean, variance } = tf.moments(x, axis, true);
  const normalized = tf.div(
    tf.sub(x, mean),
    tf.sqrt(tf.add(variance, LAYER_NORM_EPSILON)),
  );

  return tf.add(tf.mul(normalized, params.gamma), params.beta);
}

function gelu(x: tf.Tensor): tf.Tensor {
  const coefficient = Math.sqrt(2 / Math.PI);
  const inner = tf.add(x, tf.mul(0.044715, tf.pow(x, 3)));

  return tf.mul(0.5, tf.mul(x, tf.add(1, tf.tanh(tf.mul(coefficient, inner)))));
}

function applyDropout(
  x: tf.Tensor,
  rate: number,
  training: boolean,
  getSeed?: () => number | undefined,
): tf.Tensor {
  if (!training || rate <= 0) {
    return x;
  }

  return tf.dropout(x, rate, undefined, getSeed?.());
}

function causalMask(seqLen: number): tf.Tensor4D {
  const band = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0);
  // Posiciones futuras (j > i) reciben -1e9 antes del softmax.
  const mask = tf.sub(band, 1).mul(1e9);

  return mask.reshape([1, 1, seqLen, seqLen]) as tf.Tensor4D;
}

export class TransformerBlock {
  private readonly wq: tf.Variable;
  private readonly wk: tf.Variable;
  private readonly wv: tf.Variable;
  private readonly wo: tf.Variable;
  private readonly norm1: LayerNormParams;
  private readonly norm2: LayerNormParams;
  private readonly ffW1: tf.Variable;
  private readonly ffB1: tf.Variable;
  private readonly ffW2: tf.Variable;
  private readonly ffB2: tf.Variable;

  private readonly dropoutSeed: number | undefined;
  private dropoutCounter = 0;

  constructor(
    private readonly config: ModelConfig,
    seed?: number,
  ) {
    this.dropoutSeed = seed;

    let counter = 0;
    const nextSeed = () => (seed === undefined ? undefined : seed + counter++);
    const { dModel, dFF } = config;

    this.wq = randomVariable([dModel, dModel], nextSeed());
    this.wk = randomVariable([dModel, dModel], nextSeed());
    this.wv = randomVariable([dModel, dModel], nextSeed());
    this.wo = randomVariable([dModel, dModel], nextSeed());

    this.norm1 = { gamma: onesVariable([dModel]), beta: zerosVariable([dModel]) };
    this.norm2 = { gamma: onesVariable([dModel]), beta: zerosVariable([dModel]) };

    this.ffW1 = randomVariable([dModel, dFF], nextSeed());
    this.ffB1 = zerosVariable([dFF]);
    this.ffW2 = randomVariable([dFF, dModel], nextSeed());
    this.ffB2 = zerosVariable([dModel]);
  }

  forward(
    x: tf.Tensor3D,
    mask: tf.Tensor4D,
    options: ForwardOptions = {},
  ): tf.Tensor3D {
    const { dModel, nHeads, dFF, dropout } = this.config;
    const [batch, seqLen] = x.shape;
    const dHead = dModel / nHeads;
    const training = options.training ?? false;
    // Seed per call so dropout masks are reproducible
    // without repeating the same mask at each step.
    const nextDropoutSeed = () => {
      if (this.dropoutSeed === undefined) {
        return undefined;
      }
      const seed = this.dropoutSeed + this.dropoutCounter;
      this.dropoutCounter += 1;

      return seed;
    };

    const linear = (input: tf.Tensor3D, weight: tf.Variable): tf.Tensor3D => {
      return tf
        .matMul(
          input.reshape([batch * seqLen, dModel]) as tf.Tensor2D,
          weight as tf.Tensor2D,
        )
        .reshape([batch, seqLen, dModel]) as tf.Tensor3D;
    };

    // --- Multi-head attention with causal mask ---
    const h = layerNorm(x, this.norm1) as tf.Tensor3D;
    const splitHeads = (t: tf.Tensor3D): tf.Tensor4D => {
      return t
        .reshape([batch, seqLen, nHeads, dHead])
        .transpose([0, 2, 1, 3]) as tf.Tensor4D;
    };

    const q = splitHeads(linear(h, this.wq));
    const k = splitHeads(linear(h, this.wk));
    const v = splitHeads(linear(h, this.wv));

    let scores = tf
      .matMul(q, k, false, true)
      .div(Math.sqrt(dHead)) as tf.Tensor4D;
    scores = tf.add(scores, mask) as tf.Tensor4D;
    let attention = tf.softmax(scores, -1) as tf.Tensor4D;
    attention = applyDropout(attention, dropout, training, nextDropoutSeed) as tf.Tensor4D;

    const context = tf.matMul(attention, v) as tf.Tensor4D;
    const merged = context
      .transpose([0, 2, 1, 3])
      .reshape([batch, seqLen, dModel]) as tf.Tensor3D;
    const attended = applyDropout(
      linear(merged, this.wo),
      dropout,
      training,
      nextDropoutSeed,
    ) as tf.Tensor3D;
    x = tf.add(x, attended) as tf.Tensor3D;

    // --- Feed-forward ---
    const h2 = layerNorm(x, this.norm2) as tf.Tensor3D;
    const hidden = gelu(
      tf.add(
        tf.matMul(h2.reshape([batch * seqLen, dModel]) as tf.Tensor2D, this.ffW1 as tf.Tensor2D),
        this.ffB1,
      ),
    );
    const projected = tf
      .add(
        tf.matMul(hidden as tf.Tensor2D, this.ffW2 as tf.Tensor2D),
        this.ffB2,
      )
      .reshape([batch, seqLen, dModel]) as tf.Tensor3D;
    x = tf.add(x, applyDropout(projected, dropout, training, nextDropoutSeed)) as tf.Tensor3D;

    return x;
  }

  get variables(): tf.Variable[] {
    return [
      this.wq,
      this.wk,
      this.wv,
      this.wo,
      this.norm1.gamma,
      this.norm1.beta,
      this.norm2.gamma,
      this.norm2.beta,
      this.ffW1,
      this.ffB1,
      this.ffW2,
      this.ffB2,
    ];
  }

  dispose(): void {
    for (const variable of this.variables) {
      variable.dispose();
    }
  }
}

export class TransformerLM {
  readonly config: ModelConfig;
  readonly tokenEmbedding: tf.Variable;
  readonly positionEmbedding: tf.Variable;
  readonly blocks: TransformerBlock[];
  readonly finalNorm: LayerNormParams;

  private disposed = false;

  constructor(config: ModelConfig, options: TransformerOptions = {}) {
    const { seed } = options;
    let counter = 0;
    const nextSeed = () => (seed === undefined ? undefined : seed + counter++);

    this.config = config;
    this.tokenEmbedding = randomVariable([config.vocabSize, config.dModel], nextSeed());
    this.positionEmbedding = randomVariable([config.maxSeqLen, config.dModel], nextSeed());
    this.blocks = Array.from(
      { length: config.nLayers },
      (_, index) =>
        new TransformerBlock(
          config,
          seed === undefined ? undefined : seed + 1000 * index,
        ),
    );
    this.finalNorm = {
      gamma: onesVariable([config.dModel]),
      beta: zerosVariable([config.dModel]),
    };
  }

  forward(ids: tf.Tensor, options: ForwardOptions = {}): tf.Tensor3D {
    if (this.disposed) {
      throw new Error('TransformerLM has been disposed');
    }

    const { dModel, maxSeqLen, vocabSize } = this.config;
    const [batch = 0, seqLen = 0] = ids.shape;

    if (seqLen > maxSeqLen) {
      throw new Error(
        `Sequence length ${seqLen} exceeds maxSeqLen ${maxSeqLen}`,
      );
    }

    const training = options.training ?? false;

    let x = tf.gather(this.tokenEmbedding, tf.cast(ids, 'int32')) as tf.Tensor3D;
    const positions = tf
      .slice(this.positionEmbedding, [0, 0], [seqLen, dModel])
      .reshape([1, seqLen, dModel]) as tf.Tensor3D;
    x = tf.add(x, positions) as tf.Tensor3D;

    const mask = causalMask(seqLen);
    for (const block of this.blocks) {
      x = block.forward(x, mask, { training });
    }

    const h = layerNorm(x, this.finalNorm) as tf.Tensor3D;

    // Output projection tied to embedding: logits = h @ E^T.
    return tf
      .matMul(
        h.reshape([batch * seqLen, dModel]) as tf.Tensor2D,
        this.tokenEmbedding as tf.Tensor2D,
        false,
        true,
      )
      .reshape([batch, seqLen, vocabSize]) as tf.Tensor3D;
  }

  get variables(): tf.Variable[] {
    return [
      this.tokenEmbedding,
      this.positionEmbedding,
      ...this.blocks.flatMap((block) => block.variables),
      this.finalNorm.gamma,
      this.finalNorm.beta,
    ];
  }

  dispose(): void {
    for (const variable of this.variables) {
      variable.dispose();
    }

    this.disposed = true;
  }
}

export function createTransformerLM(
  rawConfig: unknown,
  options: TransformerOptions = {},
): TransformerLM {
  const config: ModelConfig = ModelConfigSchema.parse(rawConfig);

  return new TransformerLM(config, options);
}
