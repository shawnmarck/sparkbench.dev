#!/usr/bin/env python3
"""Build sparkbench.dev static site from tool repo data files."""

import os
import re
import shutil
import yaml
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader

DATA_DIR = os.environ.get("DATA_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", "site")
SITE_URL = os.environ.get("SITE_URL", "https://sparkbench.dev").rstrip("/")
TOOL_REPO = "https://github.com/shawnmarck/sparkbench"
HF_BASE = "https://huggingface.co"


CAPABILITY_LABELS = {
    "agentic": "Agents",
    "coder": "Code",
    "coding": "Code",
    "reasoning": "Reasoning",
    "vision": "Multimodal",
    "multimodal": "Multimodal",
    "moe": "MoE",
    "dense": "Dense",
}


def _clean_note(note):
    """Drop noisy auto-generated 'Model Lab:' dumps; keep editorial notes."""
    if not note:
        return ""
    if note.startswith("Model Lab:"):
        return ""
    return note.strip()


_NAME_TOKEN_RE = re.compile(r"[a-z0-9]+")


def derive_use_cases(model):
    """Infer use-case tags from name + capabilities.

    Match on tokenized words and the explicit capability list, never on raw
    substrings — otherwise "vllm" matches "vl" and tags every vLLM model
    as Multimodal.
    """
    name = (model.get("name") or "").lower()
    inv = (model.get("id") or "").lower()
    caps = {c.lower() for c in model.get("capabilities", [])}
    tokens = set(_NAME_TOKEN_RE.findall(f"{name} {inv}"))

    def has(*needles):
        return any(n in tokens or n in caps for n in needles)

    tags = set()
    if has("coder", "coding", "code"):
        tags.add("Code")
    if has("agentic", "agent", "tool-calling"):
        tags.add("Agents")
    if has("reasoning", "thinking", "r1", "o1"):
        tags.add("Reasoning")
    if has("vision", "multimodal", "vl", "vlm"):
        tags.add("Multimodal")
    if not tags:
        tags.add("General")
    return sorted(tags)


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
        m = {
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
            "note": _clean_note(v.get("note", "")),
            "why_downloaded": cat.get("why_downloaded", "").strip(),
        }
        m["use_cases"] = derive_use_cases(m)
        models.append(m)

    models.sort(key=lambda m: m["tok_s"] or 0, reverse=True)
    return models


def compute_stats(models):
    tok_values = [m["tok_s"] for m in models if m["tok_s"]]
    engines = sorted({m["engine"] for m in models if m["engine"]})
    return {
        "count": len(models),
        "peak_tok_s": max(tok_values) if tok_values else 0,
        "median_tok_s": sorted(tok_values)[len(tok_values) // 2] if tok_values else 0,
        "engines": engines,
    }


def group_by_use_case(models):
    """Group models by their primary use case."""
    groups = {}
    for m in models:
        for uc in m["use_cases"]:
            groups.setdefault(uc, []).append(m)
    order = ["Code", "Agents", "Reasoning", "Multimodal", "General"]
    return [(uc, groups[uc][:6]) for uc in order if uc in groups]


def build():
    models = load_data()
    stats = compute_stats(models)
    use_case_groups = group_by_use_case(models)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Compute bar widths relative to peak
    peak = stats["peak_tok_s"] or 1
    for m in models:
        m["tok_s_pct"] = round((m["tok_s"] or 0) / peak * 100, 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copytree("public", f"{OUT_DIR}/public", dirs_exist_ok=True)

    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)

    ctx = {
        "models": models,
        "stats": stats,
        "use_case_groups": use_case_groups,
        "built_at": built_at,
        "tool_repo": TOOL_REPO,
        "site_url": SITE_URL,
    }

    index_tpl = env.get_template("index.html")
    with open(f"{OUT_DIR}/index.html", "w") as f:
        f.write(index_tpl.render(**ctx))

    for m in models:
        safe = m["id"].replace("/", "_")
        os.makedirs(f"{OUT_DIR}/models/{safe}", exist_ok=True)
        detail_tpl = env.get_template("model.html")
        with open(f"{OUT_DIR}/models/{safe}/index.html", "w") as f:
            sub_ctx = {k: v for k, v in ctx.items() if k != "models"}
            f.write(detail_tpl.render(model=m, **sub_ctx))

    write_sitemap(models)
    write_robots()

    print(f"Built {len(models)} models → {OUT_DIR}/")
    print(f"  peak: {stats['peak_tok_s']} tok/s, engines: {', '.join(stats['engines'])}")


def write_sitemap(models):
    """Emit a sitemap.xml so model detail URLs are discoverable."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
             f"  <url><loc>{SITE_URL}/</loc><changefreq>daily</changefreq><priority>1.0</priority></url>"]
    for m in models:
        slug = m["id"].replace("/", "_")
        lines.append(
            f"  <url><loc>{SITE_URL}/models/{slug}/</loc>"
            f"<changefreq>weekly</changefreq><priority>0.7</priority></url>"
        )
    lines.append("</urlset>")
    with open(f"{OUT_DIR}/sitemap.xml", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_robots():
    body = f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n"
    with open(f"{OUT_DIR}/robots.txt", "w") as f:
        f.write(body)


if __name__ == "__main__":
    build()
