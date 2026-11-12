
import { describe, expect, it } from 'vitest';
import { splitText } from '../src/data/splits.js';

describe('splitText', () => {
  it('splits with default 90/5/5 ratios', () => {
    const text = 'abcdefghijklmnopqrst'; // 20 caracteres
    const splits = splitText(text);

    expect(splits.train).toBe('abcdefghijklmnopqr');
    expect(splits.val).toBe('s');
    expect(splits.test).toBe('t');
  });

  it('splits with custom ratios', () => {
    const text = 'abcdefghij'; // 10 caracteres
    const splits = splitText(text, {
      trainRatio: 0.5,
      valRatio: 0.3,
      testRatio: 0.2,
    });

    expect(splits.train).toBe('abcde');
    expect(splits.val).toBe('fgh');
    expect(splits.test).toBe('ij');
  });

  it('produces contiguous, non-overlapping splits covering the text', () => {
    const text = 'the-quick-brown-fox-jumps-over-the-lazy-dog-and-today';
    const splits = splitText(text, { trainRatio: 0.6, valRatio: 0.2, testRatio: 0.2 });

    expect(splits.train + splits.val + splits.test).toBe(text);
  });

  it('handles an empty text', () => {
    const splits = splitText('');

    expect(splits.train).toBe('');
    expect(splits.val).toBe('');
    expect(splits.test).toBe('');
  });

  it('rejects ratios that do not sum to 1', () => {
    expect(() => splitText('abc', {
      trainRatio: 0.5,
      valRatio: 0.2,
      testRatio: 0.2,
    })).toThrow();
  });

  it('rejects negative ratios', () => {
    expect(() => splitText('abc', {
      trainRatio: 1.1,
      valRatio: -0.1,
      testRatio: 0,
    })).toThrow();
  });
});
