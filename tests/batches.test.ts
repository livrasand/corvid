
import { describe, expect, it } from 'vitest';
import {
  createBatches,
  createWindows,
} from '../src/data/batches.js';

describe('createWindows', () => {
  it('creates non-overlapping windows with targets shifted by one', () => {
    const ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const windows = createWindows(ids, 3);

    expect(windows).toEqual([
      { inputs: [0, 1, 2], targets: [1, 2, 3] },
      { inputs: [3, 4, 5], targets: [4, 5, 6] },
      { inputs: [6, 7, 8], targets: [7, 8, 9] },
    ]);
  });

  it('discards leftover tokens that cannot fill a full window', () => {
    const ids = [0, 1, 2, 3, 4, 5];
    const windows = createWindows(ids, 4);

    expect(windows).toEqual([
      { inputs: [0, 1, 2, 3], targets: [1, 2, 3, 4] },
    ]);
  });

  it('returns no windows when ids are shorter than seqLen + 1', () => {
    expect(createWindows([0, 1, 2], 3)).toEqual([]);
    expect(createWindows([], 3)).toEqual([]);
  });

  it('rejects an invalid seqLen', () => {
    expect(() => createWindows([0, 1], 0)).toThrow();
    expect(() => createWindows([0, 1], 1.5)).toThrow();
  });
});

describe('createBatches', () => {
  const ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  it('groups consecutive windows into batches', () => {
    const batches = createBatches(ids, { seqLen: 2, batchSize: 2 });

    expect(batches).toEqual([
      {
        inputs: [[0, 1], [2, 3]],
        targets: [[1, 2], [3, 4]],
      },
      {
        inputs: [[4, 5], [6, 7]],
        targets: [[5, 6], [7, 8]],
      },
    ]);
  });

  it('keeps the last partial batch by default', () => {
    const batches = createBatches(ids, { seqLen: 2, batchSize: 3 });

    expect(batches).toHaveLength(2);
    expect(batches[1]).toEqual({
      inputs: [[6, 7]],
      targets: [[7, 8]],
    });
  });

  it('drops the last partial batch with dropLast', () => {
    const batches = createBatches(ids, {
      seqLen: 2,
      batchSize: 3,
      dropLast: true,
    });

    expect(batches).toHaveLength(1);
    expect(batches[0]).toEqual({
      inputs: [[0, 1], [2, 3], [4, 5]],
      targets: [[1, 2], [3, 4], [5, 6]],
    });
  });

  it('returns no batches when there are no windows', () => {
    expect(createBatches([0], { seqLen: 4, batchSize: 2 })).toEqual([]);
  });

  it('rejects an invalid batchSize', () => {
    expect(() => createBatches(ids, { seqLen: 2, batchSize: 0 })).toThrow();
    expect(() => createBatches(ids, { seqLen: 2, batchSize: 2.5 })).toThrow();
  });
});
