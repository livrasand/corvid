
import { z } from 'zod';

export const ModelConfigSchema = z.object({
  vocabSize: z.number().int().min(256).max(32768),
  maxSeqLen: z.number().int().min(32).max(2048),
  dModel: z.number().int().min(32).max(1024),
  nHeads: z.number().int().min(1).max(16),
  nLayers: z.number().int().min(1).max(24),
  dFF: z.number().int().min(64).max(4096),
  dropout: z.number().min(0).max(0.5),
  tieEmbeddings: z.boolean(),
}).superRefine((config, ctx) => {
  if (config.dModel % config.nHeads !== 0) {
    ctx.addIssue({
      code: 'custom',
      path: ['nHeads'],
      message: 'dModel must be divisible by nHeads',
    });
  }
});

export const TrainingConfigSchema = z.object({
  batchSize: z.number().int().min(1).max(32),
  learningRate: z.number().positive().max(0.01),
  weightDecay: z.number().min(0).max(1),
  epochs: z.number().int().min(1).max(1000),
  seed: z.number().int().nonnegative(),
});

export const CorvidConfigSchema = z.object({
  model: ModelConfigSchema,
  training: TrainingConfigSchema,
});

export type ModelConfig = z.infer<typeof ModelConfigSchema>;
export type TrainingConfig = z.infer<typeof TrainingConfigSchema>;
export type CorvidConfig = z.infer<typeof CorvidConfigSchema>;

export function validateConfig(input: unknown): CorvidConfig {
  return CorvidConfigSchema.parse(input);
}
