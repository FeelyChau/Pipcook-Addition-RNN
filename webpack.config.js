const path = require('path');
const EsmWebpackPlugin = require('@purtuga/esm-webpack-plugin');

module.exports = {
  target: 'node',
  entry: {
    app: [ './src/index' ]
  },
  resolve: {
    extensions: [ '.mjs' ]
  },
  output: {
    path: path.resolve(__dirname, './build'),
    filename: 'script.js',
    library:"pipcookScript",
    libraryTarget:"var",
  },
  mode: "development"
};
