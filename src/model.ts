import { DataCook, ScriptContext, Runtime, ModelEntry, DatasetPool } from '@pipcook/core';
import { Tensor2D } from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';

type TensorSample = DataCook.Dataset.Types.Sample<Tensor2D>;
type DatasetMeta = DatasetPool.Types.DatasetMeta;

type RT = Runtime<TensorSample, DatasetMeta>;

function createAndCompileModel(
  hiddenSize: number, rnnType: string, digits: number, vocabularySize: number) {
  const maxLen = digits + 1 + digits;
  const model = tf.sequential();
  switch (rnnType) {
    case 'SimpleRNN':
      model.add(tf.layers.simpleRNN({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        inputShape: [maxLen, vocabularySize]
      }));
      break;
    case 'GRU':
      model.add(tf.layers.gru({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        inputShape: [maxLen, vocabularySize]
      }));
      break;
    case 'LSTM':
      model.add(tf.layers.lstm({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        inputShape: [maxLen, vocabularySize]
      }));
      break;
    default:
      throw new Error(`Unsupported RNN type: '${rnnType}'`);
  }
  model.add(tf.layers.repeatVector({n: digits + 1}));
  switch (rnnType) {
    case 'SimpleRNN':
      model.add(tf.layers.simpleRNN({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true
      }));
      break;
    case 'GRU':
      model.add(tf.layers.gru({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true
      }));
      break;
    case 'LSTM':
      model.add(tf.layers.lstm({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true
      }));
      break;
    default:
      throw new Error(`Unsupported RNN type: '${rnnType}'`);
  }
  model.add(tf.layers.timeDistributed(
      {layer: tf.layers.dense({units: vocabularySize})}));
  model.add(tf.layers.activation({activation: 'softmax'}));
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam',
    metrics: ['accuracy']
  });
  return model;
}

async function train(runtime: RT, model: tf.Sequential, iterations: number, batchSize: number) {
  for (let i = 0; i < iterations; ++i) {
    console.log('iterations', i);
    runtime.dataset.train.seek(0);
    while (true) {
      const samples = await runtime.dataset.train.nextBatch(batchSize);
      if (!samples || samples.length === 0) {
        break;
      }
      const xs: Tensor2D[] = [];
      const ys: Tensor2D[] = [];
      samples.forEach(sample => {
        xs.push(sample.data);
        ys.push(sample.label as any);
      });
      const xs3D = tf.stack(xs);
      const ys3D = tf.stack(ys);
      runtime.dataset.test.shuffle();
      let testSamples = await runtime.dataset.test.nextBatch(batchSize);
      if (testSamples.length === 0) {
        await runtime.dataset.test.seek(0);
        testSamples = await runtime.dataset.test.nextBatch(batchSize);
        if (testSamples.length === 0) {
          throw new TypeError('invalid test dataset');
        }
      }
      const testXs: Tensor2D[] = [];
      const testYs: Tensor2D[] = [];
      testSamples.forEach(sample => {
        testYs.push(sample.label as any);
        testXs.push(sample.data);
      });
      const testXs3D = tf.stack(testXs);
      const testYs3D = tf.stack(testYs);
      // const beginMs = performance.now();
      const history = await model.fit(xs3D, ys3D, {
        epochs: 1,
        batchSize,
        validationData: [testXs3D, testYs3D],
        yieldEvery: 'epoch'
      });
      // console.log('history', history);
    }
  }
}

export const model: ModelEntry<TensorSample, DatasetMeta> = async (
  api: Runtime<TensorSample, DatasetMeta>, options: Record<string, any>, context: ScriptContext
): Promise<void> => {
  let {
    rnnLayerSize,
    rnnType = 'SimpleRNN',
    digits = '2',
    vocabularySize = '12',
    iterations = '10',
    batchSize = '32'
  } = options;
  digits = Number.parseInt(digits as string);
  rnnLayerSize = Number.parseInt(rnnLayerSize as string);
  vocabularySize = Number.parseInt(vocabularySize as string);
  iterations = Number.parseInt(iterations as string);
  batchSize = Number.parseInt(batchSize as string);
  const model = createAndCompileModel(rnnLayerSize, rnnType, digits, vocabularySize);
  await train(api, model, iterations, batchSize);
  // const handler = tf.io.fileSystem(context.workspace.modelDir);
  await model.save(`file://${context.workspace.modelDir}`);
  await api.saveModel(context.workspace.modelDir, '');
  return;
};
