
export interface CsvColumnExtractorOptions {
  /** Column index (0-based) to extract. */
  columnIndex: number;
  /** Skip the first row (header). Default true. */
  skipHeader?: boolean | undefined;
  /** Maximum number of rows to emit. No limit if omitted. */
  limit?: number | undefined;
}

type State = 'field-start' | 'unquoted' | 'quoted' | 'quoted-quote';

/**
 * Incremental CSV column extractor with a state machine.
 * Supports quoted fields, escaped quotes (`""`), and embedded newlines
 * inside quoted fields. It is fed by chunks and emits the requested
 * column value for each complete row.
 */
export class CsvColumnExtractor {
  private state: State = 'field-start';
  private field = '';
  private fields: string[] = [];
  private rowIndex = 0;
  private emittedCount = 0;
  private readonly columnIndex: number;
  private readonly skipHeader: boolean;
  private readonly limit: number | undefined;

  constructor(options: CsvColumnExtractorOptions) {
    if (!Number.isInteger(options.columnIndex) || options.columnIndex < 0) {
      throw new Error(`columnIndex must be a non-negative integer, got ${options.columnIndex}`);
    }

    this.columnIndex = options.columnIndex;
    this.skipHeader = options.skipHeader ?? true;
    this.limit = options.limit;
  }

  /** Feeds a chunk and returns the column values emitted by it. */
  push(chunk: string): string[] {
    const emitted: string[] = [];

    for (const char of chunk) {
      const value = this.consume(char);

      if (value !== null) {
        emitted.push(value);
      }
    }

    return emitted;
  }

  /** Closes parsing: emits the last row if it did not end in newline. */
  finish(): string[] {
    if (this.field === '' && this.fields.length === 0) {
      return [];
    }

    const value = this.completeRow();

    return value === null ? [] : [value];
  }

  debugState(): { state: string; field: string; fields: number; rowIndex: number } {
    return { state: this.state, field: this.field, fields: this.fields.length, rowIndex: this.rowIndex };
  }

  private consume(char: string): string | null {
    switch (this.state) {
      case 'field-start':
        return this.consumeFieldStart(char);
      case 'unquoted':
        return this.consumeUnquoted(char);
      case 'quoted':
        return this.consumeQuoted(char);
      case 'quoted-quote':
        return this.consumeQuotedQuote(char);
    }
  }

  private consumeFieldStart(char: string): string | null {
    if (char === '"') {
      this.state = 'quoted';

      return null;
    }

    if (char === ',') {
      return this.completeField();
    }

    if (char === '\r') {
      return null; // Line endings CRLF: the \r is discarded outside quotes.
    }

    if (char === '\n') {
      return this.completeRow();
    }

    this.field += char;
    this.state = 'unquoted';

    return null;
  }

  private consumeUnquoted(char: string): string | null {
    if (char === ',') {
      return this.completeField();
    }

    if (char === '\r') {
      return null;
    }

    if (char === '\n') {
      return this.completeRow();
    }

    this.field += char;

    return null;
  }

  private consumeQuoted(char: string): string | null {
    if (char === '"') {
      this.state = 'quoted-quote';

      return null;
    }

    // Newlines and \r inside quotes are part of the data.
    this.field += char;

    return null;
  }

  private consumeQuotedQuote(char: string): string | null {
    if (char === '"') {
      this.field += '"';
      this.state = 'quoted';

      return null;
    }

    // Closing quote: goes back to field state (in case there is garbage).
    this.state = 'unquoted';

    return this.consume(char);
  }

  private completeField(): string | null {
    this.fields.push(this.field);
    this.field = '';
    this.state = 'field-start';

    return null;
  }

  private completeRow(): string | null {
    this.fields.push(this.field);
    this.field = '';
    this.state = 'field-start';

    const row = this.fields;
    this.fields = [];
    this.rowIndex += 1;

    if (this.skipHeader && this.rowIndex === 1) {
      return null;
    }

    if (this.limit !== undefined && this.emittedCount >= this.limit) {
      return null;
    }

    const value = row[this.columnIndex];

    if (value === undefined) {
      return null;
    }

    this.emittedCount += 1;

    return value;
  }
}
