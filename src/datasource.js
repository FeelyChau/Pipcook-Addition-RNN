
const { DatasetPool } = require('@pipcook/core');
const minstData = require('./data');

/**
 * This is the entry of datasource script
 */
module.exports =
async (option, context) => {
  await minstData.loadData(context.workspace.dataDir);
  const trainData = [];
  const testData = [];
  for (let i = 0; i < minstData.trainSize; ++i) {
    trainData.push({
      data: { buf: minstData.dataset[0][i] },
      label: minstData.dataset[1][i]
    });
  }
  for (let i = 0; i < minstData.testSize; ++i) {
    testData.push({
      data: { buf: minstData.dataset[2][i] },
      label: minstData.dataset[3][i]
    });
  }
  return DatasetPool.ArrayDatasetPoolImpl.from({
    trainData,
    testData
  }, {
    height: 28,
    width: 28,
    imageFlatSize: 28 * 28,
    labelFlatSize: 10
  });
};
