
import { createReadStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { basename, dirname } from 'node:path';
import { CsvColumnExtractor } from '../src/data/extract-csv.js';

export interface ExtractCorpusOptions {
  csvPath: string;
  outPath: string;
  /** Index of the column to extract. Default 2 (message). */
  columnIndex?: number | undefined;
  /** Maximum number of rows to extract. Default no limit. */
  limit?: number | undefined;
  /** Chunk size in bytes for reading. Default 4 MiB. */
  chunkSize?: number | undefined;
  log?: ((message: string) => void) | undefined;
}

export async function extractCorpus(
  options: ExtractCorpusOptions,
): Promise<number> {
  const {
    csvPath,
    outPath,
    columnIndex = 2,
    limit,
    chunkSize = 4 * 1024 * 1024,
    log = () => {},
  } = options;

  const extractor = new CsvColumnExtractor({ columnIndex, limit });
  const pieces: string[] = [];
  let rows = 0;

  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(csvPath, {
      encoding: 'utf8',
      highWaterMark: chunkSize,
    });

    stream.on('data', (chunk) => {
      if (typeof chunk !== 'string') {
        return; // With utf8 encoding this should not happen, but we guarantee it.
      }

      const emitted = extractor.push(chunk);

      if (emitted.length > 0) {
        pieces.push(emitted.join('\n'));
        rows += emitted.length;
      }

      if (limit !== undefined && rows >= limit) {
        stream.destroy();
      }
    });

    stream.on('close', () => {
      resolve();
    });

    stream.on('error', (error) => {
      reject(error);
    });
  });

  for (const value of extractor.finish()) {
    pieces.push(value);
    rows += 1;
  }

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${pieces.join('\n')}\n`, 'utf8');

  log(`Extracted ${rows} rows from '${basename(csvPath)}' to ${outPath}`);

  return rows;
}

const invokedDirectly = process.argv[1] !== undefined
  && basename(process.argv[1]) === 'extract-corpus.ts';

if (invokedDirectly) {
  const [csvPath, outPath, limitArgument] = process.argv.slice(2);

  if (csvPath === undefined || outPath === undefined) {
    console.error('Uso: npx tsx scripts/extract-corpus.ts <csvPath> <outPath> [limit]');
    process.exit(1);
  }

  await extractCorpus({
    csvPath,
    outPath,
    ...(limitArgument === undefined ? {} : { limit: Number(limitArgument) }),
    log: (message) => console.log(message),
  });
}
