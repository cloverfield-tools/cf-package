var webpack = require('webpack');
var path = require('path');

// Set Environement variables
process.env.NODE_ENV = 'production';
process.env.MINIFY = '1';

// returns a Compiler instance with configuration file webpack.config.js
var compiler = webpack(require(path.join(process.cwd(), 'webpack.config.js')));
// Execute webpack
compiler.run(function (err, stats) {
    console.log(stats.toString({colors: true}));
});
