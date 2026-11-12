
import { describe, expect, it } from 'vitest';
import {
  CharacterTokenizer,
  SPECIAL_TOKENS,
} from '../src/core/tokenizer.js';

describe('CharacterTokenizer', () => {
  it('builds a deterministic vocabulary from a corpus', () => {
    const first = CharacterTokenizer.buildVocab('banana');
    const second = CharacterTokenizer.buildVocab('banana');

    expect(first).toEqual(second);
    expect(first.slice(0, SPECIAL_TOKENS.length)).toEqual([...SPECIAL_TOKENS]);
    // Frequency: 'a' (3) and 'n' (2) before 'b' (1); tie-break by code.
    expect(first.slice(SPECIAL_TOKENS.length)).toEqual(['a', 'n', 'b']);
  });

  it('round-trips text through encode and decode', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('hello world');
    const text = 'hello world';
    const ids = tokenizer.encode(text);

    expect(tokenizer.decode(ids)).toBe(text);
  });

  it('maps unknown characters to the unk token', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('hola');
    const ids = tokenizer.encode('hola 😺');

    expect(ids.at(-1)).toBe(tokenizer.unkId);
    expect(tokenizer.decode(ids)).toBe('hola');
  });

  it('adds bos and eos tokens when requested', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('hola');
    const ids = tokenizer.encode('hola', { bos: true, eos: true });

    expect(ids[0]).toBe(tokenizer.bosId);
    expect(ids.at(-1)).toBe(tokenizer.eosId);
    expect(tokenizer.decode(ids)).toBe('hola');
  });

  it('can decode without skipping special tokens', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('hola');
    const ids = tokenizer.encode('h', { eos: true });

    expect(tokenizer.decode(ids, { skipSpecials: false })).toBe('h<eos>');
  });

  it('keeps only the most frequent characters within maxVocabSize', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('aaabbc', { maxVocabSize: 5 });

    expect(tokenizer.vocabSize).toBe(5);
    // Only 'a' (the most frequent) enters the vocabulary; b and c go to <unk>.
    expect(tokenizer.encode('abc')).toEqual([4, tokenizer.unkId, tokenizer.unkId]);
    expect(tokenizer.decode(tokenizer.encode('abc'))).toBe('a');
  });

  it('rejects a maxVocabSize smaller than the special tokens', () => {
    expect(() => CharacterTokenizer.buildVocab('hola', { maxVocabSize: 2 }))
      .toThrow();
  });

  it('rejects invalid token lists on construction', () => {
    expect(() => new CharacterTokenizer(['<pad>', '<unk>'])).toThrow();
    expect(() => new CharacterTokenizer(['<pad>', '<unk>', '<eos>', '<bos>']))
      .toThrow();
    expect(() => new CharacterTokenizer([...SPECIAL_TOKENS, 'a', 'a']))
      .toThrow();
  });

  it('round-trips through JSON', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('hello world');
    const restored = CharacterTokenizer.fromJSON(tokenizer.toJSON());

    expect(restored.vocabSize).toBe(tokenizer.vocabSize);
    expect(restored.encode('hello world')).toEqual(tokenizer.encode('hello world'));
    expect(restored.decode(tokenizer.encode('hello world'))).toBe('hello world');
  });

  it('rejects malformed JSON data', () => {
    expect(() => CharacterTokenizer.fromJSON(null)).toThrow();
    expect(() => CharacterTokenizer.fromJSON({ specials: [] })).toThrow();
  });

  it('rejects out-of-range ids when decoding', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('hola');

    expect(() => tokenizer.decode([tokenizer.vocabSize])).toThrow(RangeError);
    expect(() => tokenizer.decode([-1])).toThrow(RangeError);
    expect(() => tokenizer.decode([1.5])).toThrow(RangeError);
  });

  it('handles a corpus with no characters', () => {
    const tokenizer = CharacterTokenizer.fromCorpus('');

    expect(tokenizer.vocabSize).toBe(SPECIAL_TOKENS.length);
    expect(tokenizer.encode('x')).toEqual([tokenizer.unkId]);
  });
});
