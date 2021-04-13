let tf = null;

function createAndCompileModel(
  hiddenSize, rnnType, digits, vocabularySize
) {
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
  model.add(tf.layers.repeatVector({
    n: digits + 1
  }));
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
  model.add(tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: vocabularySize
    })
  }));
  model.add(tf.layers.activation({
    activation: 'softmax'
  }));
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam',
    metrics: ['accuracy']
  });
  return model;
}

async function train(rt, model, epochs, batchSize) {
  const {
    dataSource,
    notifyProgress
  } = rt;
  const meta = await dataSource.getDatasetMeta();
  await dataSource.train.seek(0);
  const trainSamples = await dataSource.train.nextBatch(meta.size.train);
  if (!trainSamples || trainSamples.length === 0) {
    throw new TypeError('no train data.');
  }
  const xs = [];
  const ys = [];
  trainSamples.forEach(sample => {
    xs.push(sample.data);
    ys.push(sample.label);
  });
  const xs3D = tf.stack(xs);
  const ys3D = tf.stack(ys);

  dataSource.test.shuffle();
  let testSamples = await dataSource.test.nextBatch(meta.size.test);
  if (testSamples.length === 0) {
    throw new TypeError('invalid test dataset');
  }
  const testXs = [];
  const testYs = [];
  testSamples.forEach(sample => {
    testYs.push(sample.label);
    testXs.push(sample.data);
  });
  const testXs3D = tf.stack(testXs);
  const testYs3D = tf.stack(testYs);

  await model.fit(xs3D, ys3D, {
    epochs,
    batchSize,
    validationData: [ testXs3D, testYs3D ],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        notifyProgress({
          value: ((epoch + 1) / epochs) * 100,
          extendData: {
            epoch,
            logs
          }
        });
      }
    }
  });
}

module.exports = async (rt, options, context) => {
  let {
    digits = '2',
    rnnLayerSize,
    rnnType = 'SimpleRNN',
    vocabularySize = '12',
    iterations = '10',
    batchSize = '32'
  } = options;
  digits = parseInt(digits);
  rnnLayerSize = parseInt(rnnLayerSize);
  vocabularySize = parseInt(vocabularySize);
  iterations = parseInt(iterations);
  batchSize = parseInt(batchSize);
  tf = await context.importJS('@tensorflow/tfjs-node');
  const model = createAndCompileModel(rnnLayerSize, rnnType, digits, vocabularySize);
  await train(rt, model, iterations, batchSize);
  await model.save(`file://${context.workspace.modelDir}`);
  await rt.saveModel(context.workspace.modelDir);
};
