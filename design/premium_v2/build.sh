#!/usr/bin/env bash
# Rebuild the Premium V2 static assets (compiled Tailwind CSS + data-free React
# component bundles + vendored React UMD).  Requires Node + npm (network for the
# npm registry).  Output is consumed by src/premium_renderer.py.
set -euo pipefail
cd "$(dirname "$0")"

npm install --no-audit --no-fund \
  react@18 react-dom@18 @babel/core @babel/cli @babel/preset-react tailwindcss@3.4.17

# Classic JSX runtime → React.createElement (works with the global React UMD;
# the automatic runtime would emit an `import` that breaks a classic <script>).
echo '{ "presets": [["@babel/preset-react", { "runtime": "classic" }]] }' > babel.config.json

# 1) Tailwind → static CSS (scans the JSX for used utilities; custom theme in tailwind.config.js)
echo "@tailwind base;@tailwind components;@tailwind utilities;" > tailwind.in.css
npx tailwindcss -c tailwind.config.js -i tailwind.in.css -o report.compiled.css --minify

# 2) Component bundles — DATA-FREE (the data file is injected at render time as
#    window.DEEP / window.PORTFOLIO).  Concatenate in dependency order, app LAST.
cat deep/deep-icons.jsx deep/deep-charts.jsx deep/deep-overview.jsx deep/deep-holdings.jsx \
    deep/deep-factors.jsx deep/deep-stress-regime.jsx deep/deep-plan.jsx deep/deep-cove.jsx \
    deep/deep-app.jsx > .deep.jsx
npx babel .deep.jsx -o deep-components.js

cat portfolio-icons.jsx portfolio-charts.jsx portfolio-overview.jsx portfolio-holdings.jsx \
    portfolio-performance.jsx portfolio-ideas.jsx portfolio-app.jsx > .base.jsx
npx babel .base.jsx -o base-components.js

# 3) Vendor the React UMD production builds (inlined by the renderer → no CDN)
cp node_modules/react/umd/react.production.min.js .
cp node_modules/react-dom/umd/react-dom.production.min.js .
rm -f .deep.jsx .base.jsx

echo "Built: report.compiled.css · deep-components.js · base-components.js · react(-dom).production.min.js"

# 4) Sync the RUNTIME subset into src/premium_assets/ so it ships in the deployed
#    container (Dockerfile COPYs src/ but not design/).  premium_renderer reads here.
mkdir -p ../../src/premium_assets
for f in react.production.min.js react-dom.production.min.js deep-components.js \
         base-components.js report.compiled.css custom.css \
         deep-data.sample.json base-data.sample.json; do
  cp "$f" "../../src/premium_assets/$f"
done
echo "Synced runtime assets → src/premium_assets/"
