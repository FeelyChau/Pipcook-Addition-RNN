
import { DatasetPool, DatasourceEntry, ScriptContext, DataCook } from '@pipcook/core';
import type * as Datacook from '@pipcook/datacook';

type Sample = DataCook.Dataset.Types.Sample<string, string>;
type DatasetMeta = DatasetPool.Types.DatasetMeta;

/**
 * The options for current script
 */
interface ScriptOption {
  // Maximum number of digits of each operand of the
  digits?: number;
  // Number of examples to generate.
  numExamples?: number
}
/**
 * The options for current script
 */
interface ScriptOption {
  // All the option could be undefined if user not passed in, so we should define it to be optional.
  url?: string;
}

/**
 * Generate examples.
 *
 * Each example consists of a question, e.g., '123+456' and and an
 * answer, e.g., '579'.
 *
 * @param digits Maximum number of digits of each operand of the
 * @param numExamples Number of examples to generate.
 * @param invert Whether to invert the strings in the question.
 * @returns The generated examples.
 */
function generateData(digits: number, numExamples: number, invert = false): string[][] {
  const digitArray = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  const arraySize = digitArray.length;

  const output = [];
  const maxLen = digits + 1 + digits;

  const f = () => {
    let str = '';
    while (str.length < digits) {
      const index = Math.floor(Math.random() * arraySize);
      str += digitArray[index];
    }
    return Number.parseInt(str);
  };

  const seen = new Set();
  while (output.length < numExamples) {
    const a = f();
    const b = f();
    const sorted = b > a ? [a, b] : [b, a];
    const key = sorted[0] + '`' + sorted[1];
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);

    // Pad the data with spaces such that it is always maxLen.
    const q = `${a}+${b}`;
    const query = q + ' '.repeat(maxLen - q.length);
    let ans = (a + b).toString();
    // Answer can be of maximum size `digits + 1`.
    ans += ' '.repeat(digits + 1 - ans.length);

    if (invert) {
      throw new Error('invert is not implemented yet');
    }
    output.push([query, ans]);
  }
  return output;
}

/**
 * This is the entry of datasource script
 */
export const datasource: DatasourceEntry<Sample, DatasetMeta> =
async (option: ScriptOption, context: ScriptContext): Promise<DatasetPool.Types.DatasetPool<Sample, DatasetMeta>> => {
  let {
    digits = '2',
    numExamples = '100'
  } = option;
  digits = Number.parseInt(digits as string);
  numExamples = Number.parseInt(numExamples as string);

  const data = generateData(digits, numExamples);
  const split = Math.floor(numExamples * 0.9);
  const trainData = data.slice(0, split).map(item => ({ label: item[1], data: item[0] }));
  const testData = data.slice(split).map(item => ({ label: item[1], data: item[0] }));
  const meta = {
    // todo: custum type
    type: DataCook.Dataset.Types.DatasetType.General,
    size: {
      test: testData.length,
      train: trainData.length
    }
  };
  
  return DatasetPool.ArrayDatasetPoolImpl.from({
    trainData,
    testData
  });
};
