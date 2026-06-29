#!/usr/bin/env python3
"""Build sparkbench.dev static site from tool repo data files."""

import json
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
EDITORS_PICK_ID = "nvidia/qwen3.6-35b-a3b"

PRODUCT_ENGINES = ["eugr", "llamacpp", "ds4"]

ENGINE_LABELS = {
    "eugr": "vLLM",
    "llamacpp": "llama.cpp",
    "ds4": "ds4",
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
_GOLDEN_NOTE_CTX_RE = re.compile(r"golden\s+(\d+)([kKmM])/", re.I)
_GOLDEN_NOTE_KV_RE = re.compile(r"golden\s+\d+[kKmM]/(\S+)", re.I)
_RECIPE_NAME_PREFIX_RE = re.compile(r"^OpenCode\s*[·\.]\s*", re.I)


def public_model_name(name: str, inv_path: str) -> str:
    """Drop internal recipe branding from catalog names shown on the site."""
    cleaned = name or ""
    if inv_path == EDITORS_PICK_ID:
        cleaned = _RECIPE_NAME_PREFIX_RE.sub("", cleaned).strip()
    return cleaned


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


def load_inference_benchmark_history() -> dict[str, dict]:
    path = f"{DATA_DIR}/inference-benchmark-history.yaml"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("profiles") or {}
    return raw if isinstance(raw, dict) else {}


def parse_golden_ctx_from_note(note: str) -> int | None:
    """Parse context from Sparky golden headline notes, e.g. 'golden 256k/fp8 @ …'."""
    m = _GOLDEN_NOTE_CTX_RE.search(note or "")
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return n * 1_048_576
    return n * 1024


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


def _fmt_param_b(n: float | int | None) -> str | None:
    if n is None:
        return None
    val = float(n)
    if abs(val - round(val)) < 0.05:
        return f"{int(round(val))}B"
    return f"{val:g}B"


def parse_param_b(name: str, slug: str) -> float | None:
    """Infer total parameter count (billions) from catalog name/slug."""
    text = f"{name} {slug}"
    if re.search(r"coder-next", text, re.I):
        return 80.0
    if re.search(r"\bphi-4\b", text, re.I):
        return 14.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?[Bb](?:\b|[-/])", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def parse_active_param_b(name: str, slug: str) -> float | None:
    """Infer MoE active/forward parameter count (billions)."""
    text = f"{name} {slug}"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[- ]?[Bb]\s*[-/]\s*[Aa]?(\d+(?:\.\d+)?)\s*[Bb]",
        text,
        re.I,
    )
    if m:
        try:
            return float(m.group(2))
        except ValueError:
            pass
    m = re.search(r"-a(\d+(?:\.\d+)?)b\b", slug, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if re.search(r"coder-next", text, re.I):
        return 3.0
    if re.search(r"deepseek-v4", text, re.I):
        return 13.0
    return None


def infer_architecture(
    *,
    capabilities: list | None,
    param_b: float | None,
    param_active_b: float | None,
    name: str = "",
    slug: str = "",
) -> str | None:
    caps = {str(c).lower() for c in (capabilities or [])}
    if param_active_b or "moe" in caps:
        return "moe"
    if "dense" in caps:
        return "dense"
    text = f"{name} {slug}".lower()
    if re.search(r"a\d+b|moe", text):
        return "moe"
    if param_b is not None:
        return "dense"
    return None


def attach_model_params(m: dict, cat: dict) -> None:
    """Dense: total params. MoE: active-forward / total (matches portal inventory UI)."""
    name = cat.get("name") or m.get("name") or ""
    slug = cat.get("slug") or m.get("slug") or ""
    caps = cat.get("capabilities") or m.get("capabilities") or []
    param_b = cat.get("param_b") or parse_param_b(name, slug)
    param_active_b = cat.get("param_active_b") or parse_active_param_b(name, slug)
    arch = infer_architecture(
        capabilities=caps,
        param_b=param_b,
        param_active_b=param_active_b,
        name=name,
        slug=slug,
    )
    if arch == "dense" and param_b is not None and not param_active_b:
        param_active_b = param_b

    is_moe = bool(
        param_active_b
        and param_b
        and param_active_b < param_b
    )
    if is_moe:
        active = _fmt_param_b(param_active_b)
        total = _fmt_param_b(param_b)
        m["params_label"] = f"{active} / {total}"
        m["params_detail"] = f"{active} active / {total} total"
    elif param_b is not None:
        label = _fmt_param_b(param_b)
        m["params_label"] = label
        m["params_detail"] = label
    else:
        m["params_label"] = None
        m["params_detail"] = None

    m["param_b"] = param_b
    m["param_active_b"] = param_active_b
    m["architecture"] = arch
    m["is_moe"] = is_moe


def fetch_hf_model_meta(hf_repo: str, timeout: int = 8) -> dict:
    """Public HF model API — release date + whether the repo is reachable."""
    url = f"{HF_BASE}/api/models/{hf_repo}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "sparkbench.dev/build (hf-meta)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        created = data.get("createdAt") or data.get("lastModified")
        release_date = created[:10] if created else None
        return {"hf_ok": True, "release_date": release_date}
    except urllib.error.HTTPError:
        return {"hf_ok": False, "release_date": None}
    except Exception:
        return {"hf_ok": False, "release_date": None}


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


def iter_profile_bench_points(
    profile_id: str,
    *,
    profile_ctx: dict[str, dict],
    benchmarks: dict[str, dict],
    history: dict[str, dict],
    verification_tok_s: float | None = None,
    verification_ctx: int | None = None,
) -> list[tuple[float, int | None]]:
    """All (tok_s, ctx) measurements recorded for one inference profile."""
    points: list[tuple[float, int | None]] = []
    ctx_base = resolve_profile_ctx(profile_id, profile_ctx)

    bench = benchmarks.get(profile_id) or {}
    if bench.get("tok_s") is not None:
        points.append((float(bench["tok_s"]), ctx_base))

    hist = history.get(profile_id) or {}
    for run in hist.get("runs") or []:
        if run.get("tok_s") is not None:
            points.append((float(run["tok_s"]), ctx_base))

    if verification_tok_s is not None:
        points.append((float(verification_tok_s), verification_ctx or ctx_base))

    return points


def attach_peak_bench(
    m: dict,
    v: dict,
    profile_ctx: dict[str, dict],
    benchmarks: dict[str, dict],
    history: dict[str, dict],
) -> None:
    """Fastest tok/s across benchmark history for this model's profile(s)."""
    profiles = {p for p in (m.get("golden_profile"), v.get("tok_s_profile")) if p}
    if not profiles:
        ctx = resolve_tok_s_ctx(v, profile_ctx)
        m["tok_s_ctx"] = ctx
        m["tok_s_ctx_label"] = format_ctx_label(ctx) if ctx else None
        m["throughput"] = format_throughput(m.get("tok_s"), ctx)
        return

    note = v.get("note") or ""
    golden_ctx = parse_golden_ctx_from_note(note)
    ver_profile = v.get("tok_s_profile")
    ver_ctx = resolve_tok_s_ctx(v, profile_ctx)
    if golden_ctx and ver_profile == m.get("golden_profile"):
        ver_ctx = golden_ctx

    best_tok_s: float | None = None
    best_ctx: int | None = None
    for profile_id in profiles:
        v_tok = v.get("tok_s") if ver_profile == profile_id else None
        v_ctx = ver_ctx if ver_profile == profile_id else None
        for tok_s, ctx in iter_profile_bench_points(
            profile_id,
            profile_ctx=profile_ctx,
            benchmarks=benchmarks,
            history=history,
            verification_tok_s=v_tok,
            verification_ctx=v_ctx,
        ):
            if best_tok_s is None or tok_s > best_tok_s:
                best_tok_s = tok_s
                best_ctx = ctx

    if best_tok_s is not None:
        m["tok_s"] = best_tok_s
        m["tok_s_ctx"] = best_ctx
        m["tok_s_ctx_label"] = format_ctx_label(best_ctx) if best_ctx else None
        m["throughput"] = format_throughput(best_tok_s, best_ctx)


def attach_max_ctx_bench(m: dict, v: dict, profile_ctx: dict, benchmarks: dict) -> None:
    """Throughput at the golden max-fit context (from verification headline when present)."""
    golden_profile = m.get("golden_profile")
    if not golden_profile:
        m["max_ctx_tok_s"] = m.get("tok_s")
        m["max_ctx_ctx"] = m.get("tok_s_ctx")
        m["max_ctx_ctx_label"] = m.get("tok_s_ctx_label")
        m["max_ctx_has_bench"] = bool(m.get("tok_s"))
        m["max_ctx_pending"] = False
        return

    note = v.get("note") or ""
    golden_ctx = parse_golden_ctx_from_note(note)
    is_golden_headline = (
        golden_ctx is not None
        and v.get("tok_s_profile") == golden_profile
        and v.get("tok_s") is not None
    )

    ctx = golden_ctx if is_golden_headline else resolve_profile_ctx(golden_profile, profile_ctx)
    if is_golden_headline:
        tok_s = v.get("tok_s")
    else:
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


def infer_bench_ctx_from_recipe(recipe: dict | None) -> int | None:
    """Runtime -c from recipe args (preferred over context.default max-fit)."""
    if not recipe:
        return None
    for key in ("llamacpp_args", "ds4_args", "eugr_args"):
        args = recipe.get(key) or []
        if not isinstance(args, list):
            continue
        for i, arg in enumerate(args):
            if str(arg) in ("-c", "--ctx-size") and i + 1 < len(args):
                try:
                    return int(str(args[i + 1]).replace("_", ""))
                except ValueError:
                    pass
    default = (recipe.get("context") or {}).get("default")
    return int(default) if default else None


def load_recipes() -> tuple[dict[str, dict], dict[str, dict]]:
    """Golden recipe YAMLs keyed by profile id and inventory path."""
    recipes_dir = os.path.join(DATA_DIR, "recipes")
    by_id: dict[str, dict] = {}
    by_inv: dict[str, dict] = {}
    if not os.path.isdir(recipes_dir):
        return by_id, by_inv
    for fname in os.listdir(recipes_dir):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(recipes_dir, fname)
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        rid = doc.get("id")
        inv = doc.get("inventory_path")
        if rid:
            by_id[str(rid)] = doc
        if inv:
            by_inv[str(inv)] = doc
    return by_id, by_inv


def _recipe_ladder_cell(cell: dict, *, golden: bool = False) -> dict | None:
    if not isinstance(cell, dict):
        return None
    if cell.get("status") == "load_fail":
        return None
    tok_s = cell.get("tok_s")
    ctx = cell.get("ctx") or cell.get("loaded_ctx")
    if tok_s is None or ctx is None:
        return None
    kv = str(cell.get("kv") or "").strip()
    return {
        "ctx": int(ctx),
        "ctx_label": format_ctx_label(int(ctx)),
        "kv": kv,
        "tok_s": round(float(tok_s), 1),
        "golden": golden,
        "peak": False,
        "method": str(cell.get("method") or "").strip(),
    }


def extract_recipe_ladder(recipe: dict) -> list[dict]:
    """Pull every benched ctx/kv/tok_s cell from a golden recipe."""
    raw: list[dict] = []
    bm = recipe.get("bench_matrix") or {}

    golden = _recipe_ladder_cell(bm.get("golden_cell") or {}, golden=True)
    if golden:
        raw.append(golden)

    ctx_ladder = bm.get("ctx_ladder")
    if isinstance(ctx_ladder, dict):
        for rung in ctx_ladder.get("rungs") or []:
            row = _recipe_ladder_cell(rung)
            if row:
                raw.append(row)
    elif isinstance(ctx_ladder, list):
        for rung in ctx_ladder:
            row = _recipe_ladder_cell(rung)
            if row:
                raw.append(row)

    ctx_block = recipe.get("context") or {}
    nested = ctx_block.get("ctx_ladder")
    if isinstance(nested, dict):
        for rung in nested.get("rungs") or []:
            row = _recipe_ladder_cell(rung)
            if row:
                raw.append(row)

    for cell in bm.get("kv_sweep") or []:
        row = _recipe_ladder_cell(cell)
        if row:
            raw.append(row)

    kv_doc = recipe.get("kv_sweep") or {}
    for cell in kv_doc.get("results") or []:
        row = _recipe_ladder_cell(cell)
        if row:
            raw.append(row)

    merged: dict[tuple[int, str], dict] = {}
    for row in raw:
        key = (row["ctx"], row["kv"])
        prev = merged.get(key)
        if prev is None or row["golden"] or (not prev["golden"] and row["tok_s"] > prev["tok_s"]):
            merged[key] = row

    ladder = sorted(merged.values(), key=lambda r: r["ctx"])
    if ladder:
        best = max(ladder, key=lambda r: r["tok_s"])
        for row in ladder:
            row["peak"] = row["tok_s"] == best["tok_s"]
    return ladder


def _merge_ladder_row(ladder: list[dict], row: dict) -> None:
    if not row.get("kv"):
        for i, existing in enumerate(ladder):
            if existing["ctx"] != row["ctx"]:
                continue
            merged = {**existing, **row, "kv": existing["kv"] or row["kv"]}
            if row.get("golden"):
                merged["golden"] = True
            ladder[i] = merged
            return

    key = (row["ctx"], row["kv"])
    for i, existing in enumerate(ladder):
        if (existing["ctx"], existing["kv"]) != key:
            continue
        if row.get("golden"):
            ladder[i] = {**existing, **row, "golden": True}
        elif row["tok_s"] > existing["tok_s"]:
            ladder[i] = {**existing, **row}
        return
    ladder.append(row)


def attach_bench_ladder(
    m: dict,
    v: dict,
    recipe: dict | None,
    profile_ctx: dict[str, dict],
    benchmarks: dict[str, dict],
) -> None:
    """Context ladder rows for model detail pages."""
    profile = m.get("golden_profile") or v.get("tok_s_profile")
    ladder = extract_recipe_ladder(recipe) if recipe else []

    if profile:
        ctx = resolve_profile_ctx(profile, profile_ctx)
        if ctx is None:
            ctx = infer_bench_ctx_from_recipe(recipe)
        bench = benchmarks.get(profile) or {}
        if bench.get("tok_s") is not None and ctx is not None:
            _merge_ladder_row(ladder, {
                "ctx": ctx,
                "ctx_label": format_ctx_label(ctx),
                "kv": "",
                "tok_s": round(float(bench["tok_s"]), 1),
                "golden": False,
                "peak": False,
                "method": str(bench.get("method") or "").strip(),
            })

    note = v.get("note") or ""
    golden_ctx = parse_golden_ctx_from_note(note)
    if golden_ctx is not None and v.get("tok_s") is not None and v.get("tok_s_profile") == profile:
        kv_match = _GOLDEN_NOTE_KV_RE.search(note)
        _merge_ladder_row(ladder, {
            "ctx": golden_ctx,
            "ctx_label": format_ctx_label(golden_ctx),
            "kv": kv_match.group(1) if kv_match else "",
            "tok_s": round(float(v["tok_s"]), 1),
            "golden": True,
            "peak": False,
            "method": "bench-agent-v2",
        })

    ladder.sort(key=lambda r: r["ctx"])
    if m.get("tok_s") and m.get("tok_s_ctx"):
        _merge_ladder_row(ladder, {
            "ctx": int(m["tok_s_ctx"]),
            "ctx_label": m["tok_s_ctx_label"] or format_ctx_label(int(m["tok_s_ctx"])),
            "kv": "",
            "tok_s": round(float(m["tok_s"]), 1),
            "golden": False,
            "peak": False,
            "method": "",
        })
    if m.get("max_ctx_tok_s") and m.get("max_ctx_ctx"):
        _merge_ladder_row(ladder, {
            "ctx": int(m["max_ctx_ctx"]),
            "ctx_label": m["max_ctx_ctx_label"] or format_ctx_label(int(m["max_ctx_ctx"])),
            "kv": "",
            "tok_s": round(float(m["max_ctx_tok_s"]), 1),
            "golden": True,
            "peak": False,
            "method": "",
        })
    ladder.sort(key=lambda r: r["ctx"])
    if ladder:
        best = max(ladder, key=lambda r: r["tok_s"])
        for row in ladder:
            row["peak"] = row["tok_s"] == best["tok_s"]
    m["bench_ladder"] = ladder


def _round_tok_s(val) -> float | None:
    if val is None:
        return None
    return round(float(val), 1)


def _normalize_bench_run(raw: dict, *, profile: str, ctx_label: str | None) -> dict:
    """One inference-benchmark run row for model detail pages."""
    run_id = str(raw.get("id") or raw.get("latest_run_id") or "").strip()
    measured_at = str(raw.get("measured_at") or "").strip()
    run_tok_s = [_round_tok_s(x) for x in (raw.get("run_tok_s") or []) if x is not None]
    tok_s = _round_tok_s(raw.get("tok_s"))
    tok_s_min = _round_tok_s(raw.get("tok_s_min"))
    tok_s_max = _round_tok_s(raw.get("tok_s_max"))
    if tok_s_min is None and run_tok_s:
        tok_s_min = min(run_tok_s)
    if tok_s_max is None and run_tok_s:
        tok_s_max = max(run_tok_s)
    sessions = raw.get("sessions")
    turns = raw.get("turns_per_session")
    session_label = None
    if sessions and turns:
        session_label = f"{sessions}×{turns}"
    elif sessions:
        session_label = str(sessions)
    fill = raw.get("context_fill_target_tokens")
    note = _clean_note(str(raw.get("note") or raw.get("system_note") or ""))
    tool_ok = raw.get("tool_roundtrip_ok")
    return {
        "id": run_id,
        "profile": profile,
        "ctx_label": ctx_label,
        "measured_at": measured_at,
        "date": measured_at[:10] if measured_at else "",
        "method": str(raw.get("method") or "").strip(),
        "engine": str(raw.get("engine") or "").strip(),
        "tok_s": tok_s,
        "tok_s_min": tok_s_min,
        "tok_s_max": tok_s_max,
        "run_tok_s": run_tok_s,
        "session_label": session_label,
        "completion_tokens": raw.get("completion_tokens"),
        "prompt_tokens": raw.get("prompt_tokens"),
        "elapsed_s": raw.get("elapsed_s"),
        "context_fill": int(fill) if fill is not None else None,
        "tool_ok": tool_ok,
        "bench_version": str(raw.get("bench_standard_version") or "").strip(),
        "note": note,
        "latest": False,
    }


def attach_bench_runs(
    m: dict,
    v: dict,
    profile_ctx: dict[str, dict],
    benchmarks: dict[str, dict],
    history: dict[str, dict],
) -> None:
    """All recorded inference benchmark runs for this model's profile(s)."""
    profiles: list[str] = []
    for p in (m.get("golden_profile"), v.get("tok_s_profile")):
        if p and p not in profiles:
            profiles.append(p)
    if not profiles:
        m["bench_runs"] = []
        return

    runs: list[dict] = []
    seen_ids: set[str] = set()

    for profile in profiles:
        ctx = resolve_profile_ctx(profile, profile_ctx)
        ctx_label = format_ctx_label(ctx) if ctx else None

        hist = history.get(profile) or {}
        for run in hist.get("runs") or []:
            norm = _normalize_bench_run(run, profile=profile, ctx_label=ctx_label)
            rid = norm["id"]
            if rid:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
            runs.append(norm)

        latest = benchmarks.get(profile) or {}
        if latest.get("tok_s") is None:
            continue
        rid = str(latest.get("latest_run_id") or "").strip()
        if rid and rid in seen_ids:
            continue
        norm = _normalize_bench_run(
            {**latest, "id": rid},
            profile=profile,
            ctx_label=ctx_label,
        )
        norm["latest"] = True
        runs.append(norm)
        if rid:
            seen_ids.add(rid)

    runs.sort(key=lambda r: r.get("measured_at") or "", reverse=True)
    m["bench_runs"] = runs


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
    leaderboard_exclude = set(golden_raw.get("leaderboard_exclude") or [])

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
    bench_history = load_inference_benchmark_history()
    recipes_by_id, recipes_by_inv = load_recipes()

    models = []
    for inv_path, v in verification.items():
        if inv_path in leaderboard_exclude or v.get("leaderboard_excluded"):
            continue
        if v.get("spark_status") != "works":
            continue
        cat = resolve_catalog(inv_path)
        lab, slug = inv_path.split("/", 1) if "/" in inv_path else ("", inv_path)
        hf_repo = cat.get("hf_repo") or inv_path
        m = {
            "id": inv_path,
            "name": public_model_name(cat.get("name") or slug, inv_path),
            "lab": cat.get("lab") or lab,
            "slug": slug,
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
            "release_date": cat.get("release_date"),
        }
        attach_model_params(m, cat)
        override = use_case_overrides.get(inv_path)
        if override:
            m["use_cases"] = sorted({str(t) for t in override})
        else:
            m["use_cases"] = derive_use_cases(m)
        attach_peak_bench(m, v, profile_ctx, benchmarks, bench_history)
        attach_max_ctx_bench(m, v, profile_ctx, benchmarks)
        recipe = recipes_by_id.get(m.get("golden_profile") or "") or recipes_by_inv.get(inv_path)
        attach_bench_ladder(m, v, recipe, profile_ctx, benchmarks)
        attach_bench_runs(m, v, profile_ctx, benchmarks, bench_history)
        models.append(m)

    models.sort(key=lambda m: m["tok_s"] or 0, reverse=True)
    return models


def compute_stats(models):
    tok_values = [m["tok_s"] for m in models if m["tok_s"]]
    engines_in_data = sorted({m["engine"] for m in models if m["engine"]})
    peak_model = max(models, key=lambda m: m["tok_s"] or 0) if models else None
    editors_pick = next((m for m in models if m["id"] == EDITORS_PICK_ID), None)
    golden_models = [m for m in models if m.get("golden_profile")]
    max_ctx_benched = [m for m in golden_models if m.get("max_ctx_has_bench")]
    return {
        "count": len(models),
        "peak_tok_s": max(tok_values) if tok_values else 0,
        "peak_throughput": peak_model["throughput"] if peak_model and peak_model.get("tok_s") else None,
        "median_tok_s": sorted(tok_values)[len(tok_values) // 2] if tok_values else 0,
        "engines": PRODUCT_ENGINES,
        "engines_in_data": engines_in_data,
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
    """HF metadata per model: public link check + release date from HF API.

    Models with `hf_ok=False` render WITHOUT a HuggingFace link rather than
    shipping a 401/404. Skipped entirely when SKIP_LINK_CHECK=1 (fast local builds).
    """
    if os.environ.get("SKIP_LINK_CHECK") == "1":
        for m in models:
            m["hf_ok"] = True
        print("  link check: skipped (SKIP_LINK_CHECK=1)")
        return
    broken = []
    dated = 0
    for m in models:
        meta = fetch_hf_model_meta(m["hf_repo"])
        if meta.get("release_date") and not m.get("release_date"):
            m["release_date"] = meta["release_date"]
        if m.get("release_date"):
            dated += 1
        ok = meta.get("hf_ok") or check_hf_link(m["hf_url"])
        m["hf_ok"] = ok
        if not ok:
            broken.append(m)
    if broken:
        print(f"  link check: {len(broken)} model(s) without a public HF link:")
        for m in broken:
            print(f"    - {m['id']:55s} → {m['hf_url']}")
    else:
        print(f"  link check: all {len(models)} HF links resolve")
    print(f"  release dates: {dated}/{len(models)} models")


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
