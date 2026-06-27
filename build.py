#!/usr/bin/env python3
"""Build sparkbench.dev static site from tool repo data files."""

import json
import os
import shutil
import yaml
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader

DATA_DIR = os.environ.get("DATA_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", "site")
TOOL_REPO = "https://github.com/shawnmarck/sparkbench"
HF_BASE = "https://huggingface.co"


def load_data():
    with open(f"{DATA_DIR}/model-verification.yaml") as f:
        verification = yaml.safe_load(f)["models"]
    with open(f"{DATA_DIR}/model-catalog.yaml") as f:
        catalog_raw = yaml.safe_load(f)["models"]
    with open(f"{DATA_DIR}/golden-recipes.yaml") as f:
        golden_raw = yaml.safe_load(f)

    catalog = {m["id"]: m for m in catalog_raw}
    golden = golden_raw.get("golden", {})

    models = []
    for inv_path, v in verification.items():
        if v.get("spark_status") != "works":
            continue
        cat = catalog.get(inv_path, {})
        lab, slug = inv_path.split("/", 1) if "/" in inv_path else ("", inv_path)
        hf_repo = cat.get("hf_repo") or inv_path
        models.append({
            "id": inv_path,
            "name": cat.get("name") or slug,
            "lab": cat.get("lab") or lab,
            "slug": slug,
            "param_b": cat.get("param_b"),
            "hf_url": f"{HF_BASE}/{hf_repo}",
            "hf_repo": hf_repo,
            "engine": v.get("engine") or v.get("tok_s_engine", ""),
            "tok_s": v.get("tok_s"),
            "capabilities": cat.get("capabilities", []),
            "golden_profile": golden.get(inv_path),
            "updated_at": v.get("updated_at", ""),
            "note": v.get("note", ""),
        })

    models.sort(key=lambda m: m["tok_s"] or 0, reverse=True)
    return models


def build():
    models = load_data()
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copytree("public", f"{OUT_DIR}/public", dirs_exist_ok=True)

    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)

    ctx = {
        "models": models,
        "built_at": built_at,
        "tool_repo": TOOL_REPO,
        "model_count": len(models),
    }

    index_tpl = env.get_template("index.html")
    with open(f"{OUT_DIR}/index.html", "w") as f:
        f.write(index_tpl.render(**ctx))

    for m in models:
        safe = m["id"].replace("/", "_")
        os.makedirs(f"{OUT_DIR}/models/{safe}", exist_ok=True)
        detail_tpl = env.get_template("model.html")
        with open(f"{OUT_DIR}/models/{safe}/index.html", "w") as f:
            f.write(detail_tpl.render(model=m, **{k: v for k, v in ctx.items() if k != "models"}))

    print(f"Built {len(models)} models → {OUT_DIR}/")


if __name__ == "__main__":
    build()
