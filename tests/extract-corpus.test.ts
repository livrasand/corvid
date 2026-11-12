
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { extractCorpus } from '../scripts/extract-corpus.js';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

describe('extractCorpus', () => {
  it('extracts the message column into a plain-text corpus', async () => {
    const root = await mkdtemp(join(tmpdir(), 'corvid-extract-'));
    temporaryDirectories.push(root);

    const csvPath = join(root, 'commits.csv');
    // Embedded newlines in the diff field to exercise the state machine.
    await writeFile(csvPath, [
      'hash,diff,message,project,split',
      '"h1","diff --git a/x.py b/x.py\n@@ -1 +1 @@","fix phase diagram bug","pymatgen","train"',
      '"h2","diff --git a/y.py b/y.py\n+print(1)","add new test for lexer","pygments","val"',
    ].join('\n'));

    const outPath = join(root, 'raw', 'corpus.txt');
    const rows = await extractCorpus({ csvPath, outPath });

    expect(rows).toBe(2);
    await expect(readFile(outPath, 'utf8')).resolves.toBe(
      'fix phase diagram bug\nadd new test for lexer\n',
    );
  });

  it('stops early when the limit is reached', async () => {
    const root = await mkdtemp(join(tmpdir(), 'corvid-extract-'));
    temporaryDirectories.push(root);

    const csvPath = join(root, 'commits.csv');
    await writeFile(csvPath, 'hash,message,project\n1,one,p1\n2,two,p2\n3,three,p3\n4,four,p4\n');

    const outPath = join(root, 'corpus.txt');
    const rows = await extractCorpus({ csvPath, outPath, limit: 2 });

    expect(rows).toBe(2);
    // The default column (index 2) is 'project' in this 3-column CSV.
    await expect(readFile(outPath, 'utf8')).resolves.toBe('p1\np2\n');
  });

  it('honors the column index option', async () => {
    const root = await mkdtemp(join(tmpdir(), 'corvid-extract-'));
    temporaryDirectories.push(root);

    const csvPath = join(root, 'commits.csv');
    await writeFile(csvPath, 'hash,message,project\n1,m1,p1\n2,m2,p2\n');

    const outPath = join(root, 'projects.txt');
    const rows = await extractCorpus({ csvPath, outPath, columnIndex: 2 });

    expect(rows).toBe(2);
    await expect(readFile(outPath, 'utf8')).resolves.toBe('p1\np2\n');
  });

  it('creates the output directory if missing', async () => {
    const root = await mkdtemp(join(tmpdir(), 'corvid-extract-'));
    temporaryDirectories.push(root);

    const csvPath = join(root, 'commits.csv');
    await writeFile(csvPath, 'hash,message,project\n1,hello,p1\n');

    const outPath = join(root, 'nested', 'deep', 'corpus.txt');
    await extractCorpus({ csvPath, outPath });

    await expect(readFile(outPath, 'utf8')).resolves.toBe('p1\n');
  });

  it('rejects when the CSV file does not exist', async () => {
    const root = await mkdtemp(join(tmpdir(), 'corvid-extract-'));
    temporaryDirectories.push(root);

    await expect(
      extractCorpus({
        csvPath: join(root, 'missing.csv'),
        outPath: join(root, 'corpus.txt'),
      }),
    ).rejects.toThrow();
  });
});
