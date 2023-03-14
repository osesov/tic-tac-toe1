const path = require("path");

module.exports = {
    entry: "./browser/browser.ts",
    devtool: 'inline-source-map',
    mode: 'development',
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                loader: "ts-loader",
                exclude: /node_modules/,
                options: {
                    configFile: 'browser.tsconfig.json'
                }
            },
        ],
    },
    resolve: {
        extensions: [".tsx", ".ts", ".js"],
    },
    output: {
        filename: "bundle.js",
        path: path.resolve(__dirname, "index"),
    },
    externals: {
        '@tensorflow/tfjs': 'tf',
    },

    devServer: {
        static: './index',
        hot: true,
    },
};
