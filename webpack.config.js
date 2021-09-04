const path = require('path');

module.exports = {
  target: 'node',
  entry: {
    app: [ './src/index.ts' ]
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      }
    ],
  },
  resolve: {
    extensions: [ '.ts', '.js' ]
  },
  externals: {
    '@pipcook/core': 'commonjs2 @pipcook/core',
    '@tensorflow/tfjs-node': 'commonjs2 @tensorflow/tfjs-node',
    '@tensorflow/tfjs-node-gpu': 'commonjs2 @tensorflow/tfjs-node-gpu',
    '@node-rs/jieba': 'commonjs2 @node-rs/jieba',
  },
  output: {
    path: path.resolve(__dirname, './build'),
    filename: 'script.js',
    libraryTarget: 'umd',
  },
  mode: 'development'
};
