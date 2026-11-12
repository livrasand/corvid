
export interface TextSplits {
  train: string;
  val: string;
  test: string;
}

export interface SplitRatios {
  trainRatio?: number;
  valRatio?: number;
  testRatio?: number;
}

const RATIO_TOLERANCE = 1e-9;

export function splitText(text: string, ratios: SplitRatios = {}): TextSplits {
  const {
    trainRatio = 0.9,
    valRatio = 0.05,
    testRatio = 0.05,
  } = ratios;

  for (const ratio of [trainRatio, valRatio, testRatio]) {
    if (!Number.isFinite(ratio) || ratio < 0) {
      throw new Error(`Ratios must be non-negative numbers, got ${ratio}`);
    }
  }

  if (Math.abs(trainRatio + valRatio + testRatio - 1) > RATIO_TOLERANCE) {
    throw new Error(
      `Ratios must sum to 1, got ${trainRatio} + ${valRatio} + ${testRatio}`,
    );
  }

  const trainEnd = Math.floor(text.length * trainRatio);
  const valEnd = trainEnd + Math.floor(text.length * valRatio);

  return {
    train: text.slice(0, trainEnd),
    val: text.slice(trainEnd, valEnd),
    test: text.slice(valEnd),
  };
}
