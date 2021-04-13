let tf = null;
let datacook = null;

class CharacterTable {
  /**
   * Constructor of CharacterTable.
   * @param chars A string that contains the characters that can appear
   *   in the input.
   */
  constructor(chars) {
    this.chars = chars;
    this.charIndices = {};
    this.size = this.chars.length;
    for (let i = 0; i < this.size; ++i) {
      const char = this.chars[i];
      if (this.charIndices[char] != null) {
        throw new Error(`Duplicate character '${char}'`);
      }
      this.charIndices[this.chars[i]] = i;
    }
  }

  /**
   * Convert a string into a one-hot encoded tensor.
   *
   * @param str The input string.
   * @param numRows Number of rows of the output tensor.
   * @returns The one-hot encoded 2D tensor.
   * @throws If `str` contains any characters outside the `CharacterTable`'s
   *   vocabulary.
   */
  encode(str, numRows) {
    const buf = tf.buffer([numRows, this.size]);
    for (let i = 0; i < str.length; ++i) {
      const char = str[i];
      if (this.charIndices[char] == null) {
        throw new Error(`Unknown character: '${char}'`);
      }
      buf.set(1, i, this.charIndices[char]);
    }
    return buf.toTensor().as2D(numRows, this.size);
  }
}

module.exports =
async (dataset, options, context) => {
  let { digits } = options;
  datacook = context.dataCook;
  digits = parseInt(digits);
  tf = await context.importJS('@tensorflow/tfjs');
  const characterTable = new CharacterTable('0123456789+ ');
  return datacook.Dataset.transformSampleInDataset(async (sample) => {
    return {
      label: characterTable.encode(sample.label, digits + 1),
      data: characterTable.encode(sample.data, digits + 1 + digits)
    };
  }, dataset);
};
