
export interface Window {
  inputs: number[];
  targets: number[];
}

export interface Batch {
  inputs: number[][];
  targets: number[][];
}

export interface BatchOptions {
  seqLen: number;
  batchSize: number;
  dropLast?: boolean;
}

export function createWindows(
  ids: readonly number[],
  seqLen: number,
): Window[] {
  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new Error(`seqLen must be a positive integer, got ${seqLen}`);
  }

  const windows: Window[] = [];
  const lastStart = ids.length - seqLen - 1;

  for (let start = 0; start <= lastStart; start += seqLen) {
    windows.push({
      inputs: ids.slice(start, start + seqLen) as number[],
      targets: ids.slice(start + 1, start + seqLen + 1) as number[],
    });
  }

  return windows;
}

export function createBatches(
  ids: readonly number[],
  options: BatchOptions,
): Batch[] {
  const { seqLen, batchSize, dropLast = false } = options;

  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error(`batchSize must be a positive integer, got ${batchSize}`);
  }

  const windows = createWindows(ids, seqLen);
  const batches: Batch[] = [];

  for (let start = 0; start < windows.length; start += batchSize) {
    const group = windows.slice(start, start + batchSize);

    if (dropLast && group.length < batchSize) {
      break;
    }

    batches.push({
      inputs: group.map((window) => window.inputs),
      targets: group.map((window) => window.targets),
    });
  }

  return batches;
}
