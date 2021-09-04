// import test from 'ava';
// import * as sinon from 'sinon';
// import * as script from './datasource';
// import * as Datacook from '@pipcook/datacook';

// test('datasource entry', async (t) => {
//   const dataset = await script.datasource({}, {
//     boa: null,
//     Datacook,
//     workspace: {
//       dataDir: '/tmp/',
//       modelDir: '/tmp/',
//       cacheDir: '/tmp/'
//     },
//     importPY: sinon.stub().resolves(),
//     importJS: sinon.stub().resolves()
//   });
//   const meta = await dataset.getDatasetMeta();
//   t.truthy(await dataset.getDatasetMeta(), 'should get dataset meta');
//   for (let i = 0; i < meta.size.train; ++i) {
//     const sample = await dataset.train.next();
//     console.log('train sample', i, sample);
//   }
//   for (let i = 0; i < meta.size.test; ++i) {
//     const sample = await dataset.test.next();
//     console.log('test sample', i, sample);
//   }
//   dataset.test.shuffle();
// });
