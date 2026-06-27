#!/usr/bin/env python3
"""Build sparkbench.dev static site from tool repo data files."""

import os
import re
import shutil
import urllib.error
import urllib.request
import yaml
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader

DATA_DIR = os.environ.get("DATA_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", "site")
SITE_URL = os.environ.get("SITE_URL", "https://sparkbench.dev").rstrip("/")
TOOL_REPO = "https://github.com/shawnmarck/sparkbench"
HF_BASE = "https://huggingface.co"
EDITORS_PICK_ID = "qwen/qwen3.6-27b"

ENGINE_LABELS = {
    "eugr": "vLLM",
}


def engine_label(engine: str) -> str:
    return ENGINE_LABELS.get(engine or "", engine or "—")


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
_PROFILE_CTX_SUFFIX_RE = re.compile(r"(?:^|-)(\d+)(k|m)(?:-|$)", re.I)


def format_ctx_label(ctx: int) -> str:
    """Human label for a context window size (e.g. 32768 → 32k)."""
    if ctx >= 1_048_576:
        if ctx % 1_048_576 == 0:
            return f"{ctx // 1_048_576}M"
        return f"{ctx / 1_048_576:.1f}M"
    if ctx >= 1024:
        k = ctx / 1024
        if abs(k - round(k)) < 0.05:
            return f"{int(round(k))}k"
        return f"{k:.0f}k"
    return str(ctx)


def parse_ctx_from_profile_id(profile_id: str) -> int | None:
    m = _PROFILE_CTX_SUFFIX_RE.search(profile_id or "")
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    return n * (1_000_000 if unit == "m" else 1000)


def load_profile_bench_context() -> dict[str, dict]:
    path = f"{DATA_DIR}/profile-bench-context.yaml"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("profiles") or {}
    return raw if isinstance(raw, dict) else {}


def load_inference_benchmarks() -> dict[str, dict]:
    path = f"{DATA_DIR}/inference-benchmarks.yaml"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("profiles") or {}
    return raw if isinstance(raw, dict) else {}


def resolve_profile_ctx(profile_id: str, profile_ctx: dict[str, dict]) -> int | None:
    entry = profile_ctx.get(profile_id) if profile_id else None
    if isinstance(entry, dict) and entry.get("ctx"):
        return int(entry["ctx"])
    return parse_ctx_from_profile_id(profile_id)


def resolve_tok_s_ctx(v: dict, profile_ctx: dict[str, dict]) -> int | None:
    """Context window used for the tok/s measurement."""
    if v.get("tok_s_ctx"):
        return int(v["tok_s_ctx"])
    profile_id = v.get("tok_s_profile") or ""
    entry = profile_ctx.get(profile_id) if profile_id else None
    if isinstance(entry, dict) and entry.get("ctx"):
        return int(entry["ctx"])
    return parse_ctx_from_profile_id(profile_id)


def format_throughput(tok_s, ctx: int | None) -> str:
    if not tok_s:
        return "—"
    base = f"{tok_s} t/s"
    if ctx:
        return f"{base} @ {format_ctx_label(ctx)}"
    return base


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


def attach_max_ctx_bench(m: dict, v: dict, profile_ctx: dict, benchmarks: dict) -> None:
    """Golden max-fit profile throughput — may differ from the headline tok/s."""
    golden_profile = m.get("golden_profile")
    if not golden_profile:
        m["max_ctx_tok_s"] = m.get("tok_s")
        m["max_ctx_ctx"] = m.get("tok_s_ctx")
        m["max_ctx_ctx_label"] = m.get("tok_s_ctx_label")
        m["max_ctx_has_bench"] = bool(m.get("tok_s"))
        m["max_ctx_pending"] = False
        return

    ctx = resolve_profile_ctx(golden_profile, profile_ctx)
    bench = benchmarks.get(golden_profile) or {}
    tok_s = bench.get("tok_s")
    if tok_s is None and v.get("tok_s_profile") == golden_profile:
        tok_s = v.get("tok_s")

    m["max_ctx_profile"] = golden_profile
    m["max_ctx_ctx"] = ctx
    m["max_ctx_ctx_label"] = format_ctx_label(ctx) if ctx else None
    m["max_ctx_tok_s"] = tok_s
    m["max_ctx_has_bench"] = tok_s is not None
    m["max_ctx_pending"] = not m["max_ctx_has_bench"] and ctx is not None


def load_data():
    with open(f"{DATA_DIR}/model-verification.yaml") as f:
        verification = yaml.safe_load(f)["models"]
    with open(f"{DATA_DIR}/model-catalog.yaml") as f:
        catalog_raw = yaml.safe_load(f)["models"]
    with open(f"{DATA_DIR}/golden-recipes.yaml") as f:
        golden_raw = yaml.safe_load(f)
    use_case_overrides = {}
    uc_path = f"{DATA_DIR}/use-cases.yaml"
    if os.path.exists(uc_path):
        with open(uc_path) as f:
            uc_doc = yaml.safe_load(f) or {}
        use_case_overrides = (uc_doc.get("models") or {})

    catalog = {m["id"]: m for m in catalog_raw}
    golden = golden_raw.get("golden", {})

    # Fallback index: for verification keys that don't match a catalog id
    # exactly (e.g. inventory `nvidia/qwen3.6-35b-a3b` vs catalog
    # `nvidia/qwen3.6-35b-a3b-nvfp4`), pick the first catalog row in the
    # same lab whose slug shares the verification slug as a prefix.
    by_lab = {}
    for cat_row in catalog_raw:
        cid = cat_row["id"]
        if "/" in cid:
            clab, cslug = cid.split("/", 1)
            by_lab.setdefault(clab, []).append((cslug, cat_row))

    def resolve_catalog(inv_path):
        if inv_path in catalog:
            return catalog[inv_path]
        if "/" not in inv_path:
            return {}
        lab, slug = inv_path.split("/", 1)
        for cslug, row in by_lab.get(lab, []):
            if cslug == slug or cslug.startswith(slug + "-") or slug.startswith(cslug + "-"):
                return row
        return {}

    profile_ctx = load_profile_bench_context()
    benchmarks = load_inference_benchmarks()

    models = []
    for inv_path, v in verification.items():
        if v.get("spark_status") != "works":
            continue
        cat = resolve_catalog(inv_path)
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
            "tok_s_profile": v.get("tok_s_profile"),
            "capabilities": cat.get("capabilities", []),
            "golden_profile": golden.get(inv_path),
            "updated_at": v.get("updated_at", ""),
            "note": _clean_note(v.get("note", "")),
            "why_downloaded": cat.get("why_downloaded", "").strip(),
        }
        override = use_case_overrides.get(inv_path)
        if override:
            m["use_cases"] = sorted({str(t) for t in override})
        else:
            m["use_cases"] = derive_use_cases(m)
        ctx = resolve_tok_s_ctx(v, profile_ctx)
        m["tok_s_ctx"] = ctx
        m["tok_s_ctx_label"] = format_ctx_label(ctx) if ctx else None
        m["throughput"] = format_throughput(m["tok_s"], ctx)
        attach_max_ctx_bench(m, v, profile_ctx, benchmarks)
        models.append(m)

    models.sort(key=lambda m: m["tok_s"] or 0, reverse=True)
    return models


def compute_stats(models):
    tok_values = [m["tok_s"] for m in models if m["tok_s"]]
    engines = sorted({m["engine"] for m in models if m["engine"]})
    peak_model = max(models, key=lambda m: m["tok_s"] or 0) if models else None
    editors_pick = next((m for m in models if m["id"] == EDITORS_PICK_ID), None)
    golden_models = [m for m in models if m.get("golden_profile")]
    max_ctx_benched = [m for m in golden_models if m.get("max_ctx_has_bench")]
    return {
        "count": len(models),
        "peak_tok_s": max(tok_values) if tok_values else 0,
        "peak_throughput": peak_model["throughput"] if peak_model and peak_model.get("tok_s") else None,
        "median_tok_s": sorted(tok_values)[len(tok_values) // 2] if tok_values else 0,
        "engines": engines,
        "editors_pick": editors_pick,
        "max_ctx_golden_count": len(golden_models),
        "max_ctx_bench_count": len(max_ctx_benched),
    }


def group_by_use_case(models):
    """Group models by their primary use case."""
    groups = {}
    for m in models:
        for uc in m["use_cases"]:
            groups.setdefault(uc, []).append(m)
    order = ["General", "Agents", "Reasoning", "Code", "Multimodal"]
    out = []
    for uc in order:
        if uc not in groups:
            continue
        items = sorted(groups[uc], key=lambda m: m["tok_s"] or 0, reverse=True)
        out.append((uc, items[:6]))
    return out


def check_hf_link(url, timeout=8):
    """HEAD-check a HuggingFace URL. Returns True for 200, False for 4xx/5xx/errors."""
    req = urllib.request.Request(
        url, method="HEAD",
        headers={"User-Agent": "sparkbench.dev/build (link-check)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def verify_links(models):
    """HEAD-check every model's hf_url. Annotate `hf_ok` on each model.

    Models with `hf_ok=False` render WITHOUT a HuggingFace link rather than
    shipping a 401/404. Skipped entirely when SKIP_LINK_CHECK=1 (fast local builds).
    """
    if os.environ.get("SKIP_LINK_CHECK") == "1":
        for m in models:
            m["hf_ok"] = True
        print("  link check: skipped (SKIP_LINK_CHECK=1)")
        return
    broken = []
    for m in models:
        ok = check_hf_link(m["hf_url"])
        m["hf_ok"] = ok
        if not ok:
            broken.append(m)
    if broken:
        print(f"  link check: {len(broken)} model(s) without a public HF link:")
        for m in broken:
            print(f"    - {m['id']:55s} → {m['hf_url']}")
    else:
        print(f"  link check: all {len(models)} HF links resolve")


def build():
    models = load_data()
    verify_links(models)
    stats = compute_stats(models)
    use_case_groups = group_by_use_case(models)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Compute bar widths relative to peak (both ranking modes)
    peak = stats["peak_tok_s"] or 1
    max_ctx_peak = max((m["max_ctx_tok_s"] or 0) for m in models) or 1
    for m in models:
        m["tok_s_pct"] = round((m["tok_s"] or 0) / peak * 100, 1)
        m["max_ctx_tok_s_pct"] = round((m["max_ctx_tok_s"] or 0) / max_ctx_peak * 100, 1)
        m["editors_pick"] = m["id"] == EDITORS_PICK_ID

    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copytree("public", f"{OUT_DIR}/public", dirs_exist_ok=True)

    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["engine_label"] = engine_label

    ctx = {
        "models": models,
        "stats": stats,
        "use_case_groups": use_case_groups,
        "built_at": built_at,
        "tool_repo": TOOL_REPO,
        "site_url": SITE_URL,
        "root": "",
    }

    index_tpl = env.get_template("index.html")
    with open(f"{OUT_DIR}/index.html", "w") as f:
        f.write(index_tpl.render(**ctx))

    model_ctx = {**ctx, "root": "../../"}
    for m in models:
        safe = m["id"].replace("/", "_")
        os.makedirs(f"{OUT_DIR}/models/{safe}", exist_ok=True)
        detail_tpl = env.get_template("model.html")
        with open(f"{OUT_DIR}/models/{safe}/index.html", "w") as f:
            f.write(detail_tpl.render(model=m, **model_ctx))

    open(f"{OUT_DIR}/.nojekyll", "a").close()
    if "sparkbench.dev" in SITE_URL:
        with open(f"{OUT_DIR}/CNAME", "w") as f:
            f.write("sparkbench.dev\n")

    write_sitemap(models)
    write_robots()

    print(f"Built {len(models)} models → {OUT_DIR}/")
    print(f"  peak: {stats['peak_tok_s']} tok/s, engines: {', '.join(stats['engines'])}")
    print(f"  max-ctx golden: {stats['max_ctx_bench_count']}/{stats['max_ctx_golden_count']} benched")


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
