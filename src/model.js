const { DataCook, DatasetPool } = require('@pipcook/core');
const tf = require('@tensorflow/tfjs-node');

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
model.add(tf.layers.flatten());
model.add(tf.layers.dropout({rate: 0.25}));
model.add(tf.layers.dense({units: 512, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.5}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

const optimizer = 'rmsprop';
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});
 
async function datasetToTensors(dataset, meta) {
  const allSamples = await dataset.nextBatch(-1);
  const size = allSamples.length;

  // Only create one big array to hold batch of images.
  const imagesShape = [size, meta.height, meta.width, 1];
  const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
  const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

  let imageOffset = 0;
  let labelOffset = 0;
  for (let i = 0; i < size; ++i) {
    images.set(allSamples[i].data.buf, imageOffset);
    labels.set(allSamples[i].label, labelOffset);
    imageOffset += meta.imageFlatSize;
    labelOffset += 1;
  }

  return {
    images: tf.tensor4d(images, imagesShape),
    labels: tf.oneHot(tf.tensor1d(labels, 'int32'), meta.labelFlatSize).toFloat()
  };
}

module.exports = async (api, options, context) => {
  let {
    epochs = 20,
    batchSize = 128
  } = options;
  model.summary();
  const { images: trainImages, labels: trainLabels } = await datasetToTensors(api.dataset.train, api.dataset.meta);
  const validationSplit = 0.15;
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('batch:', batch);
        console.log(logs);
      },
    }
  });
  const { images: testImages, labels: testLabels } = await datasetToTensors(api.dataset.test, api.dataset.meta);
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  await model.save(`file://${context.workspace.modelDir}`);
  await api.saveModel(context.workspace.modelDir, '');
  return;
};
