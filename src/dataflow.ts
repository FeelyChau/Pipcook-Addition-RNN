
import { DataCook, ScriptContext, DataflowEntry, DatasetPool } from '@pipcook/core';
import { Tensor2D, Tensor3D } from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';

type Sample = DataCook.Dataset.Types.Sample<string>;
type TensorSample = DataCook.Dataset.Types.Sample<Tensor2D>;
type DataSourceMeta = DatasetPool.Types.DatasetMeta;

class CharacterTable {
  charIndices: Record<string, any>;
  indicesChar: Record<string, any>;
  size: number;
  /**
   * Constructor of CharacterTable.
   * @param chars A string that contains the characters that can appear
   *   in the input.
   */
  constructor(
    private chars: string
  ) {
    this.chars = chars;
    this.charIndices = {};
    this.indicesChar = {};
    this.size = this.chars.length;
    for (let i = 0; i < this.size; ++i) {
      const char = this.chars[i];
      if (this.charIndices[char] != null) {
        throw new Error(`Duplicate character '${char}'`);
      }
      this.charIndices[this.chars[i]] = i;
      this.indicesChar[i] = this.chars[i];
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
  encode(str: string, numRows: number): Tensor2D {
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

  encodeBatch(strings: string[], numRows: number): Tensor3D {
    const numExamples = strings.length;
    const buf = tf.buffer([numExamples, numRows, this.size]);
    for (let n = 0; n < numExamples; ++n) {
      const str = strings[n];
      for (let i = 0; i < str.length; ++i) {
        const char = str[i];
        if (this.charIndices[char] == null) {
          throw new Error(`Unknown character: '${char}'`);
        }
        buf.set(1, n, i, this.charIndices[char]);
      }
    }
    return buf.toTensor().as3D(numExamples, numRows, this.size);
  }
}

export const dataflow: DataflowEntry<Sample, DataSourceMeta, TensorSample> =
async (dataset: DatasetPool.Types.DatasetPool<Sample, DataSourceMeta>, options: Record<string, any>, context: ScriptContext)
  : Promise<DatasetPool.Types.DatasetPool<TensorSample, DataSourceMeta>> => {
  let { digits } = options;
  digits = Number.parseInt(digits as string);
  
  const characterTable = new CharacterTable('0123456789+ ');
  return dataset.transform(async (sample: Sample): Promise<TensorSample> => {
    return {
      label: characterTable.encode(sample.label, digits + 1),
      data: characterTable.encode(sample.data, digits + 1 + digits)
    };
  });
};
