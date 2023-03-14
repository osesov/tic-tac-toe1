import typescript from "@rollup/plugin-typescript";
import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import serve from 'rollup-plugin-serve';
import livereload from 'rollup-plugin-livereload';

export default {
    input: "browser/browser.ts",
    external: ["@tensorflow/tfjs", "jquery"],
    output: {
        file: ".build/rollup/bundle.js",
        format: "iife",
        sourcemap: true,
        name: "main",
        globals: {
            "@tensorflow/tfjs": "tf",
            jquery: "$",
        },
    },
    plugins: [
        typescript({
            tsconfig: "browser.tsconfig.json",
            compilerOptions: {
                outDir: ".build/rollup.ts",
            },
        }),
        resolve(),
        commonjs(),

        serve({
            open: false,
            verbose: true,
            contentBase: ['index', '.build/rollup'],
            onListening: function (server) {
                const address = server.address()
                const host = address.address === '::' ? 'localhost' : address.address
                // by using a bound function, we can access options as `this`
                const protocol = this.https ? 'https' : 'http'
                console.log(`Server listening at ${protocol}://${host}:${address.port}/`)
              }
        }),
        livereload({
            watch: [
                'index',
                '.build/rollup'
            ],
            verbose: true, // Disable console output

            // other livereload options
            port: 12345,
            delay: 300,
        }),

    ],
    watch: {
        buildDelay: 1000,
        skipWrite: false,
        clearScreen: false,
        // include: [ "lib", "browser" ],
    },
};
