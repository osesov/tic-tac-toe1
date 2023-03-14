set -ex -o pipefail

build_dir="$(pwd)/.build"
ts_output_dir="${build_dir}/ts"
browserify_build_dir="${build_dir}/bfy"
output_dir="index"

mkdir -p "${ts_output_dir}" "${browserify_build_dir}"

tsc --project browser.tsconfig.json --outDir "${ts_output_dir}"
npx browserify --no-bundle-external "${ts_output_dir}/browser/browser.js" --debug | npx exorcist --root ../../ ${browserify_build_dir}/bundle.js.map > ${browserify_build_dir}/bundle.js
npx sorcery --input ${browserify_build_dir}/bundle.js --output "${output_dir}/bundle.js"
