# sparkbench.dev

Source for [sparkbench.dev](https://sparkbench.dev) — GB10 model leaderboard, benchmark results, and what runs on DGX Spark.

## What this is

A static site generated from data in the [sparkbench](https://github.com/shawnmarck/sparkbench) tool repo. No backend. Rebuilds nightly and on every push.

**Looking for the tool?** → [github.com/shawnmarck/sparkbench](https://github.com/shawnmarck/sparkbench)

## Stack

- Python + Jinja2 build script (`build.py`)
- Plain CSS (`public/style.css`)
- GitHub Pages via Actions

## Local build

```bash
pip install pyyaml jinja2

# Option A: use data from the tool repo clone
DATA_DIR=../sparkbench/data python build.py

# Option B: fetch from GitHub
mkdir -p data
BASE=https://raw.githubusercontent.com/shawnmarck/sparkbench/main/data
curl -fsSL $BASE/model-verification.yaml -o data/model-verification.yaml
curl -fsSL $BASE/model-catalog.yaml      -o data/model-catalog.yaml
curl -fsSL $BASE/golden-recipes.yaml     -o data/golden-recipes.yaml
python build.py

# Output is in site/
```

## Data pipeline

The CI workflow fetches three YAML files from the `sparkbench` tool repo on every build:

| File | What it contains |
|------|-----------------|
| `data/model-verification.yaml` | tok/s, engine, spark_status per model |
| `data/model-catalog.yaml` | name, params, capabilities, HF repo |
| `data/golden-recipes.yaml` | which model maps to which golden profile |

To update the leaderboard: merge new benchmark data into the tool repo. The site rebuilds overnight or trigger `workflow_dispatch`.

## Disclaimer

Not affiliated with or endorsed by NVIDIA Corporation.
