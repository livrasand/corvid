import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";

const script = readFileSync(new URL("./train-commits.ts", import.meta.url), "utf-8");

describe("train-commits.ts", () => {
  it("defines the buildTrainingExample function", () => {
    expect(script).toContain("function buildTrainingExample");
  });

  it("defines the parseCsvLine function", () => {
    expect(script).toContain("function parseCsvLine");
  });

  it("defines the readCommitsCsv generator", () => {
    expect(script).toContain("async function* readCommitsCsv");
  });

  it("exports the main function", () => {
    expect(script).toContain("export async function main");
  });

  it("uses CharacterTokenizer.fromCorpus", () => {
    expect(script).toContain("CharacterTokenizer.fromCorpus");
  });

  it("uses trainModel from the trainer", () => {
    expect(script).toContain("trainModel");
  });

  it("saves the tokenizer with saveTokenizer", () => {
    expect(script).toContain("saveTokenizer");
  });

  it("builds examples with bos/diff/message/eos format", () => {
    expect(script).toContain("<bos>");
    expect(script).toContain("<eos>");
    expect(script).toContain("diff:");
    expect(script).toContain("message:");
  });

  it("reads the CSV line by line with readline", () => {
    expect(script).toContain("createInterface");
    expect(script).toContain("createReadStream");
  });

  it("has configuration options for CSV, output, and checkpoints", () => {
    expect(script).toContain("csvPath");
    expect(script).toContain("outDir");
    expect(script).toContain("checkpointDir");
    expect(script).toContain("tokenizerPath");
  });

  it("supports maxSamples to limit the number of commits", () => {
    expect(script).toContain("maxSamples");
  });
});
