
import { describe, expect, it } from 'vitest';
import { CsvColumnExtractor } from '../src/data/extract-csv.js';

function extractAll(
  text: string,
  options: { columnIndex: number; skipHeader?: boolean; limit?: number },
): string[] {
  const extractor = new CsvColumnExtractor(options);
  const emitted = extractor.push(text);
  const pending = extractor.finish();

  return [...emitted, ...pending];
}

describe('CsvColumnExtractor', () => {
  it('extracts the requested column from simple rows', () => {
    const rows = extractAll('a,b,c\n1,2,3\n', { columnIndex: 1 });

    expect(rows).toEqual(['2']);
  });

  it('skips the header by default', () => {
    const rows = extractAll('hash,diff,message,project\nh1,d1,m1,p1\n', {
      columnIndex: 2,
    });

    expect(rows).toEqual(['m1']);
  });

  it('supports keeping the header', () => {
    const rows = extractAll('hash,message\nh,m\n', {
      columnIndex: 1,
      skipHeader: false,
    });

    expect(rows).toEqual(['message', 'm']);
  });

  it('handles quoted fields with embedded commas', () => {
    const rows = extractAll('name,message\n"doe, john","fix bug, add tests"\n', {
      columnIndex: 1,
    });

    expect(rows).toEqual(['fix bug, add tests']);
  });

  it('handles escaped double quotes inside quoted fields', () => {
    const rows = extractAll('a,message\n1,"said ""hello"" there"\n', {
      columnIndex: 1,
    });

    expect(rows).toEqual(['said "hello" there']);
  });

  it('handles embedded newlines inside quoted fields', () => {
    const rows = extractAll(
      'a,message\n1,"first line\nsecond line"\n2,"other"\n',
      { columnIndex: 1 },
    );

    expect(rows).toEqual(['first line\nsecond line', 'other']);
  });

  it('handles CRLF line endings', () => {
    const rows = extractAll('a,message\r\n1,first\r\n2,second\r\n', {
      columnIndex: 1,
    });

    expect(rows).toEqual(['first', 'second']);
  });

  it('preserves \r inside quoted fields', () => {
    const rows = extractAll('a,message\n1,"line\rbreak"\n', { columnIndex: 1 });

    expect(rows).toEqual(['line\rbreak']);
  });

  it('emits the same rows when the text is split into chunks', () => {
    const text = 'a,message\n1,"fix the\nbug"\n2,plain\n3,"quoted, comma"\n';
    const expected = ['fix the\nbug', 'plain', 'quoted, comma'];

    // Chunks de 3 caracteres fuerzan estados a cruzar fronteras.
    const extractor = new CsvColumnExtractor({ columnIndex: 1 });
    const emitted: string[] = [];
    for (let index = 0; index < text.length; index += 3) {
      emitted.push(...extractor.push(text.slice(index, index + 3)));
    }
    emitted.push(...extractor.finish());

    expect(emitted).toEqual(expected);
    expect(extractAll(text, { columnIndex: 1 })).toEqual(expected);
  });

  it('respects the limit', () => {
    const rows = extractAll('a,message\n1,one\n2,two\n3,three\n', {
      columnIndex: 1,
      limit: 2,
    });

    expect(rows).toEqual(['one', 'two']);
  });

  it('returns nothing when the input is empty', () => {
    const rows = extractAll('', { columnIndex: 1 });

    expect(rows).toEqual([]);
  });

  it('emits the last row even without a trailing newline', () => {
    const rows = extractAll('a,message\n1,final', { columnIndex: 1 });

    expect(rows).toEqual(['final']);
  });

  it('throws for a negative columnIndex', () => {
    expect(() => new CsvColumnExtractor({ columnIndex: -1 })).toThrow();
  });
});
