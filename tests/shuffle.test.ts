
import { describe, expect, it } from 'vitest';
import { mulberry32, seededShuffle } from '../src/training/shuffle.js';

describe('mulberry32', () => {
  it('is deterministic for the same seed', () => {
    const first = mulberry32(42);
    const second = mulberry32(42);
    const valuesA = Array.from({ length: 10 }, () => first());
    const valuesB = Array.from({ length: 10 }, () => second());

    expect(valuesA).toEqual(valuesB);
  });

  it('produces values in [0, 1)', () => {
    const random = mulberry32(7);
    const values = Array.from({ length: 1000 }, () => random());

    expect(values.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it('differs across seeds', () => {
    const first = mulberry32(1);
    const second = mulberry32(2);

    expect(first()).not.toBe(second());
  });
});

describe('seededShuffle', () => {
  const items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  it('is deterministic for the same seed', () => {
    expect(seededShuffle(items, 42)).toEqual(seededShuffle(items, 42));
  });

  it('preserves all elements', () => {
    const shuffled = seededShuffle(items, 42);

    expect([...shuffled].sort((a, b) => a - b)).toEqual(items);
  });

  it('does not mutate the input', () => {
    const copy = [...items];
    seededShuffle(items, 42);

    expect(items).toEqual(copy);
  });

  it('typically changes the order', () => {
    expect(seededShuffle(items, 42)).not.toEqual(items);
  });

  it('handles empty arrays', () => {
    expect(seededShuffle([], 42)).toEqual([]);
  });
});
