import { readFile } from 'node:fs/promises';
import { validateConfig, type CorvidConfig } from './config.js';

export async function loadConfig(path: string): Promise<CorvidConfig> {
  const content = await readFile(path, 'utf8');
  const parsed: unknown = JSON.parse(content);

  return validateConfig(parsed);
}
