
export const SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>'] as const;

export interface TokenizerJSON {
  specials: string[];
  chars: string[];
}

export interface BuildVocabOptions {
  maxVocabSize?: number;
}

export interface EncodeOptions {
  bos?: boolean;
  eos?: boolean;
}

export interface DecodeOptions {
  skipSpecials?: boolean;
}

export class CharacterTokenizer {
  private readonly charToId: Map<string, number>;
  private readonly idToChar: string[];

  constructor(tokens: readonly string[]) {
    if (tokens.length < SPECIAL_TOKENS.length) {
      throw new Error(
        `Tokenizer vocabulary must contain at least ${SPECIAL_TOKENS.length} special tokens`,
      );
    }

    for (let index = 0; index < SPECIAL_TOKENS.length; index += 1) {
      if (tokens[index] !== SPECIAL_TOKENS[index]) {
        throw new Error(
          `Expected special token '${SPECIAL_TOKENS[index]}' at index ${index}`,
        );
      }
    }

    const seen = new Set<string>();
    this.charToId = new Map();
    this.idToChar = [];

    for (const token of tokens) {
      if (seen.has(token)) {
        throw new Error(`Duplicate token in vocabulary: '${token}'`);
      }

      seen.add(token);
      this.charToId.set(token, this.idToChar.length);
      this.idToChar.push(token);
    }
  }

  static buildVocab(corpus: string, options: BuildVocabOptions = {}): string[] {
    const { maxVocabSize = Number.POSITIVE_INFINITY } = options;

    if (maxVocabSize < SPECIAL_TOKENS.length) {
      throw new Error(
        `maxVocabSize must be at least ${SPECIAL_TOKENS.length} (special tokens)`,
      );
    }

    const counts = new Map<string, number>();
    for (const char of Array.from(corpus)) {
      counts.set(char, (counts.get(char) ?? 0) + 1);
    }

    const chars = Array.from(counts.keys())
      .sort((a, b) => {
        const byCount = (counts.get(b) ?? 0) - (counts.get(a) ?? 0);
        if (byCount !== 0) {
          return byCount;
        }
        return a < b ? -1 : a > b ? 1 : 0;
      })
      .slice(0, maxVocabSize - SPECIAL_TOKENS.length);

    return [...SPECIAL_TOKENS, ...chars];
  }

  static fromCorpus(corpus: string, options: BuildVocabOptions = {}): CharacterTokenizer {
    return new CharacterTokenizer(CharacterTokenizer.buildVocab(corpus, options));
  }

  static fromJSON(data: unknown): CharacterTokenizer {
    if (typeof data !== 'object' || data === null) {
      throw new Error('Tokenizer JSON must be an object');
    }

    const { specials, chars } = data as Record<string, unknown>;
    if (!Array.isArray(specials) || !Array.isArray(chars)) {
      throw new Error('Tokenizer JSON must contain specials and chars arrays');
    }

    return new CharacterTokenizer([...specials, ...chars]);
  }

  get vocabSize(): number {
    return this.idToChar.length;
  }

  get padId(): number {
    return 0;
  }

  get unkId(): number {
    return 1;
  }

  get bosId(): number {
    return 2;
  }

  get eosId(): number {
    return 3;
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    const { bos = false, eos = false } = options;
    const ids: number[] = [];

    if (bos) {
      ids.push(this.bosId);
    }

    for (const char of Array.from(text)) {
      ids.push(this.charToId.get(char) ?? this.unkId);
    }

    if (eos) {
      ids.push(this.eosId);
    }

    return ids;
  }

  decode(ids: readonly number[], options: DecodeOptions = {}): string {
    const { skipSpecials = true } = options;
    const specialIds = new Set(
      Array.from(SPECIAL_TOKENS, (_, index) => index),
    );

    let text = '';
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= this.idToChar.length) {
        throw new RangeError(`Token id out of range: ${id}`);
      }

      if (skipSpecials && specialIds.has(id)) {
        continue;
      }

      text += this.idToChar[id]!;
    }

    return text;
  }

  toJSON(): TokenizerJSON {
    return {
      specials: [...SPECIAL_TOKENS],
      chars: this.idToChar.slice(SPECIAL_TOKENS.length),
    };
  }
}
