import { main } from './scripts/generate.js';

const PROMPTS: Array<{ prompt: string; temperature: number; topK?: number; seed?: number }> = [
  { prompt: 'fix', temperature: 0.7, topK: 40, seed: 1 },
  { prompt: 'add', temperature: 0.7, topK: 40, seed: 2 },
  { prompt: 'update', temperature: 0.7, topK: 40, seed: 3 },
  { prompt: 'remove', temperature: 0.7, topK: 40, seed: 4 },
  { prompt: 'Merge pull request', temperature: 0.7, topK: 40, seed: 5 },
  { prompt: 'refactor', temperature: 0.7, topK: 40, seed: 6 },
  { prompt: 'Bump', temperature: 0.7, topK: 40, seed: 7 },
  { prompt: 'fix the bug', temperature: 0.5, topK: 40, seed: 8 },
];

async function demo(): Promise<void> {
  for (const options of PROMPTS) {
    await main({
      ...options,
      maxNewTokens: 220,
      log: () => {},
      print: (message) => {
        console.log(`─── prompt: "${options.prompt}" (t=${options.temperature}, topK=${options.topK ?? '-'}, seed=${options.seed ?? '-'})`);
        console.log(`    ${message.replaceAll('\n', '\n    ')}`);
        console.log();
      },
    });
  }
}

await demo();
