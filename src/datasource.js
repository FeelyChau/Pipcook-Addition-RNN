let dataCook = null;

/**
 * Generate examples.
 *
 * Each example consists of a question, e.g., '123+456' and and an
 * answer, e.g., '579'.
 *
 * @param digits Maximum number of digits of each operand of the
 * @param numExamples Number of examples to generate.
 * @param invert Whether to invert the strings in the question.
 * @returns The generated examples and digit array.
 */
function generateData(digits, numExamples, invert = false) {
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
    return parseInt(str);
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
module.exports = async (options, context) => {
  let {
    digits = '2',
    numExamples = '100'
  } = options;
  digits = parseInt(digits);
  numExamples = parseInt(numExamples);
  dataCook = context.dataCook;

  const data = generateData(digits, numExamples);
  const split = Math.floor(numExamples * 0.9);
  const trainData = data.slice(0, split).map(item => ({ label: item[1], data: item[0] }));
  const testData = data.slice(split).map(item => ({ label: item[1], data: item[0] }));
  const meta = {
    type: dataCook.Dataset.Types.DatasetType.General,
    size: {
      test: testData.length,
      train: trainData.length
    }
  };
  return dataCook.Dataset.makeDataset({ trainData, testData }, meta);
};
