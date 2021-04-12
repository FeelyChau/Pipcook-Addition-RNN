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
  output: {
    path: path.resolve(__dirname, './build'),
    filename: 'script.js',
    libraryTarget: 'umd',
  },
  mode: 'development'
};
