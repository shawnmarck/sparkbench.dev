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
EDITORS_PICK_PROFILE = "qwen36-35b-a3b-mtp-eugr"
# Legacy bench-v2 fill ratio (kept for ladder metadata). Display headline is PBM 4k.
BENCH_FILL_RATIO = 0.75
PBM_DISPLAY_FILL = "4k"
PBM_DISPLAY_LABEL = "4k context fill"
PBM_FILL_KEYS = ("4k", "50k", "100k")
PBM_FILL_TOKENS = {"4k": 4096, "50k": 50000, "100k": 100000}

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


def collect_model_profiles(
    m: dict,
    v: dict,
    profiles_by_inv: dict[str, list[str]],
) -> list[str]:
    """Inference profile ids tied to this inventory path (golden + companions)."""
    profiles: list[str] = []
    for p in (m.get("golden_profile"), v.get("tok_s_profile")):
        if p and p not in profiles:
            profiles.append(p)
    for p in profiles_by_inv.get(m["id"], []):
        if p and p not in profiles:
            profiles.append(p)
    return profiles


def attach_peak_bench(
    m: dict,
    v: dict,
    profile_ctx: dict[str, dict],
    benchmarks: dict[str, dict],
    history: dict[str, dict],
    profiles_by_inv: dict[str, list[str]],
) -> None:
    """Fastest tok/s across benchmark history for this model's profile(s)."""
    profiles = collect_model_profiles(m, v, profiles_by_inv)
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
    best_profile: str | None = None
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
                best_profile = profile_id

    if best_tok_s is not None:
        m["tok_s"] = best_tok_s
        m["tok_s_ctx"] = best_ctx
        m["tok_s_ctx_label"] = format_ctx_label(best_ctx) if best_ctx else None
        m["throughput"] = format_throughput(best_tok_s, best_ctx)
        if best_profile:
            m["peak_profile"] = best_profile


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


_FORMAT_LABELS = {
    "nvfp4": "NVFP4",
    "fp4": "FP4",
    "fp8": "FP8",
    "fp16": "FP16",
    "bf16": "BF16",
    "q4": "Q4",
    "q5": "Q5",
    "q8": "Q8",
    "prismaquant": "PrismaQuant",
    "awq": "AWQ",
    "gptq": "GPTQ",
}

# Not precision — never use these as the Quant column value.
_NON_QUANT_LABELS = frozenset({"GGUF", "HF", "MTP", "DFLASH", "SAFETENSORS"})

_ENGINE_VARIANT_ALIASES = {
    "eugr": {"eugr", "vllm"},
    "llamacpp": {"llamacpp", "llama", "llama.cpp"},
    "ds4": {"ds4"},
}

# Specific llama.cpp / GGUF quants from notes & ids.
_LLAMA_QUANT_RE = re.compile(
    r"\b(IQ\d+(?:_[A-Z0-9]+)?|Q\d+_K(?:_[A-Z]+)?|Q\d+_0|Q\d+)\b",
    re.I,
)

# Precision tokens, longest / most-specific first.
_PRECISION_KEYS = (
    "prismaquant",
    "nvfp4",
    "fp16",
    "bf16",
    "fp8",
    "fp4",
    "awq",
    "gptq",
    "q4",
    "q5",
    "q8",
)

# Strip these from display names once Quant has its own column.
_NAME_QUANT_PAREN_RE = re.compile(
    r"""\s*[\(\[\{]\s*(?:
        NVFP4|FP4|FP8|FP16|BF16|AWQ|GPTQ|PrismaQuant|
        MoQ\s*GGUF|MTP\s*GGUF|GGUF|
        IQ\d+(?:_[A-Z0-9]+)?|Q\d+_K(?:_[A-Z]+)?|Q\d+|
        llama\.cpp|eugr|vLLM
    )\s*[\)\]\}]""",
    re.I | re.X,
)
_NAME_QUANT_TRAIL_RE = re.compile(
    r"""(?:^|[\s·\-_/,])(?:
        NVFP4|FP4|FP8|FP16|BF16|AWQ|GPTQ|PrismaQuant|
        MoQ\s*GGUF|IQ\d+(?:_[A-Z0-9]+)?|Q\d+_K(?:_[A-Z]+)?|Q\d+
    )\s*$""",
    re.I | re.X,
)
_NAME_ENGINE_PAREN_RE = re.compile(
    r"\s*\((?:llama\.cpp|eugr|vLLM)\)\s*",
    re.I,
)


def _format_label(fmt: str | None) -> str | None:
    if not fmt:
        return None
    key = str(fmt).strip().lower()
    return _FORMAT_LABELS.get(key) or key.upper()


def _normalize_quant_label(label: str | None) -> str | None:
    if not label:
        return None
    cleaned = str(label).strip()
    if not cleaned:
        return None
    upper = cleaned.upper().replace(" ", "")
    if upper in _NON_QUANT_LABELS or cleaned.upper() in _NON_QUANT_LABELS:
        return None
    # Canonicalize known keys
    mapped = _FORMAT_LABELS.get(cleaned.lower())
    if mapped:
        return mapped
    # Keep llama-style quants as written (Q4_K_M, IQ4_XS)
    if _LLAMA_QUANT_RE.fullmatch(cleaned):
        return cleaned.upper() if cleaned.upper().startswith("IQ") else cleaned
    return cleaned


def _extract_quant_from_text(*parts: object) -> str | None:
    hay = " ".join(str(p) for p in parts if p)
    if not hay:
        return None
    # Prefer explicit llama / GGUF quant codes
    m = _LLAMA_QUANT_RE.search(hay)
    if m:
        token = m.group(1)
        return token.upper() if token.upper().startswith("IQ") else token
    lower = hay.lower()
    for key in _PRECISION_KEYS:
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(key)}(?:[^a-z0-9]|$)", lower):
            return _format_label(key)
    return None


def derive_quant_label(
    cat: dict | None,
    recipe: dict | None,
    engine: str | None,
) -> str | None:
    """Precision/quant for the benched variant (NVFP4, FP8, Q4_K_M, …).

    Returns None when only a container is known (GGUF / HF weights).
    """
    cat = cat or {}
    recipe = recipe or {}
    engine = (engine or recipe.get("engine") or "").strip().lower()
    aliases = _ENGINE_VARIANT_ALIASES.get(engine, {engine} if engine else set())

    # 1) Catalog variants matching this engine — note often has the real quant.
    variants = cat.get("variants") or []
    if isinstance(variants, list):
        matched = []
        for v in variants:
            if not isinstance(v, dict):
                continue
            v_engine = str(v.get("engine") or "").strip().lower()
            if aliases and v_engine and v_engine not in aliases:
                continue
            matched.append(v)
        picks = matched or [v for v in variants if isinstance(v, dict)]
        for pick in picks:
            note = (pick.get("note") or "").strip()
            from_note = _extract_quant_from_text(note)
            if from_note:
                return _normalize_quant_label(from_note)
            fmt = _normalize_quant_label(_format_label(pick.get("format")))
            if fmt:
                return fmt
            from_repo = _extract_quant_from_text(pick.get("hf_repo"), pick.get("subpath"))
            if from_repo:
                return _normalize_quant_label(from_repo)

    # 2) Recipe id / name, catalog identity, capabilities, HF repo.
    found = _extract_quant_from_text(
        recipe.get("id"),
        recipe.get("name"),
        cat.get("name"),
        cat.get("slug"),
        cat.get("hf_repo"),
        *(cat.get("capabilities") or []),
    )
    return _normalize_quant_label(found)


# Back-compat alias used during the detail-page redesign.
derive_variant_label = derive_quant_label


def strip_quant_from_name(name: str, quant: str | None = None) -> str:
    """Remove quant / engine tokens from a display name once Quant is its own column."""
    cleaned = (name or "").strip()
    if not cleaned:
        return cleaned

    cleaned = _NAME_QUANT_PAREN_RE.sub("", cleaned)
    cleaned = _NAME_ENGINE_PAREN_RE.sub(" ", cleaned)
    cleaned = _NAME_QUANT_TRAIL_RE.sub("", cleaned)

    if quant:
        q = re.escape(str(quant).strip())
        cleaned = re.sub(rf"\s*[(\[{{]\s*{q}\s*[)\]}}]", "", cleaned, flags=re.I)
        cleaned = re.sub(rf"(?:^|[\s·\-_/,]){q}\s*$", "", cleaned, flags=re.I)

    cleaned = re.sub(r"\s*[·\-_/,]+\s*$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned or (name or "").strip()


def attach_recipe_summary(m: dict, recipe: dict | None) -> None:
    """Public recipe fields for the model detail page."""
    profile = m.get("golden_profile") or (recipe or {}).get("id")
    if profile:
        m["run_cmd"] = f"spark inference up {profile}"
    else:
        m["run_cmd"] = None

    # Spec / draft fields even when recipe YAML is missing (tags/capabilities fallback).
    attach_spec_fields(m, recipe)

    if not recipe:
        m["recipe_summary"] = None
        return

    ctx_block = recipe.get("context") or {}
    default_ctx = ctx_block.get("default")
    native_ctx = ctx_block.get("native")
    try:
        default_ctx = int(default_ctx) if default_ctx is not None else None
    except (TypeError, ValueError):
        default_ctx = None
    try:
        native_ctx = int(native_ctx) if native_ctx is not None else None
    except (TypeError, ValueError):
        native_ctx = None

    runtime_ctx = infer_bench_ctx_from_recipe(recipe)
    ctx = runtime_ctx or default_ctx
    kv = ctx_block.get("kv_default") or None
    golden_cell = ((ctx_block.get("bench_matrix") or {}).get("golden_cell") or {})
    if not kv and golden_cell.get("kv"):
        kv = golden_cell.get("kv")

    rid = str(recipe.get("id") or profile or "")
    engine = recipe.get("engine") or m.get("engine") or ""
    m["recipe_summary"] = {
        "profile": rid or None,
        "engine": engine,
        "engine_label": engine_label(engine),
        "ctx": ctx,
        "ctx_label": format_ctx_label(ctx) if ctx else None,
        "native_ctx": native_ctx,
        "native_ctx_label": format_ctx_label(native_ctx) if native_ctx else None,
        "kv": kv,
        "served_name": recipe.get("served_name") or None,
        "yaml_url": f"{TOOL_REPO}/blob/main/recipes/{rid}.yaml" if rid else None,
        "spec_method": m.get("spec_method"),
        "spec_label": m.get("spec_label"),
        "spec_n": m.get("spec_n"),
        "has_draft": m.get("has_draft"),
        "spec_tag": m.get("spec_tag"),
    }


_SPEC_METHOD_LABELS = {
    "dflash": "DFlash",
    "dflash2": "DFlash2",
    "mtp": "MTP",
    "dspark": "DSpark",
    "eagle": "EAGLE",
    "eagle3": "EAGLE-3",
    "draft": "Draft",
    "ngram": "NGram",
}


def attach_spec_fields(m: dict, recipe: dict | None) -> None:
    """Mark when the golden/benched recipe uses a draft / speculative decoder."""
    recipe = recipe or {}
    spec = recipe.get("speculative") or {}
    if not spec and isinstance(recipe.get("mtp"), dict):
        spec = recipe.get("mtp") or {}
    method = str(spec.get("method") or "").strip().lower()
    variant = str(spec.get("variant") or "").strip().lower()
    n_raw = spec.get("num_speculative_tokens")
    try:
        n = int(n_raw) if n_raw is not None else None
    except (TypeError, ValueError):
        n = None

    # Fallback: recipe tags / id / name when speculative block is absent.
    if not method:
        hay = " ".join(
            str(x)
            for x in (
                recipe.get("id"),
                recipe.get("name"),
                *(recipe.get("tags") or []),
                *(m.get("capabilities") or []),
            )
            if x
        ).lower()
        for key in ("dflash2", "dflash", "dspark", "eagle3", "eagle", "mtp"):
            if re.search(rf"(?:^|[^a-z0-9]){re.escape(key)}(?:[^a-z0-9]|$)", hay):
                method = key
                break

    label_key = variant if variant in _SPEC_METHOD_LABELS else method
    label = _SPEC_METHOD_LABELS.get(label_key) if label_key else None
    has_draft = bool(label) or bool(spec.get("sidecar_path") or spec.get("sidecar_inventory"))
    if has_draft and not label:
        label = "Draft"

    m["spec_method"] = method or None
    m["spec_label"] = label
    m["spec_n"] = n
    m["has_draft"] = has_draft
    if has_draft and label:
        m["spec_tag"] = f"{label} · n{n}" if n is not None else label
    else:
        m["spec_tag"] = None


def _ingest_recipe_doc(
    doc: dict,
    *,
    by_id: dict[str, dict],
    by_inv: dict[str, dict],
    profiles_by_inv: dict[str, list[str]],
) -> None:
    rid = doc.get("id")
    inv = doc.get("inventory_path")
    if rid:
        by_id[str(rid)] = doc
    if inv:
        inv = str(inv)
        by_inv[inv] = doc
        if rid:
            profiles_by_inv.setdefault(inv, [])
            if str(rid) not in profiles_by_inv[inv]:
                profiles_by_inv[inv].append(str(rid))


def load_recipes() -> tuple[dict[str, dict], dict[str, dict], dict[str, list[str]]]:
    """Recipe YAMLs keyed by profile id, inventory path, and all profiles per inventory."""
    recipes_dir = os.path.join(DATA_DIR, "recipes")
    by_id: dict[str, dict] = {}
    by_inv: dict[str, dict] = {}
    profiles_by_inv: dict[str, list[str]] = {}
    if not os.path.isdir(recipes_dir):
        return by_id, by_inv, profiles_by_inv

    def scan_dir(directory: str) -> None:
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if not fname.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, fname)
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
            _ingest_recipe_doc(doc, by_id=by_id, by_inv=by_inv, profiles_by_inv=profiles_by_inv)

    scan_dir(recipes_dir)
    scan_dir(os.path.join(recipes_dir, "drafts"))
    return by_id, by_inv, profiles_by_inv


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
    profiles_by_inv: dict[str, list[str]],
) -> None:
    """All recorded inference benchmark runs for this model's profile(s)."""
    profiles = collect_model_profiles(m, v, profiles_by_inv)
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


def load_pbm_profiles() -> dict:
    path = f"{DATA_DIR}/perfbench-metrics.yaml"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("profiles") or {}
    return raw if isinstance(raw, dict) else {}


def pbm_tok_s(pbm: dict, profile_id: str | None, fill: str = PBM_DISPLAY_FILL):
    if not profile_id:
        return None
    entry = pbm.get(profile_id) or {}
    val = entry.get(f"tok_s_{fill}")
    if val is None:
        return None
    try:
        return round(float(val), 1)
    except (TypeError, ValueError):
        return None


def _pbm_curve_points(fills: dict[str, float | None]) -> list[dict]:
    points = []
    for key in PBM_FILL_KEYS:
        tok = fills.get(key)
        if tok is None:
            continue
        points.append({
            "fill": key,
            "tokens": PBM_FILL_TOKENS[key],
            "tok_s": tok,
        })
    return points


def render_pbm_sparkline(points: list[dict], *, w: int = 56, h: int = 22) -> str | None:
    """Tiny inline SVG for leaderboard rows (3-point fill curve)."""
    if len(points) < 2:
        return None
    pad_x, pad_y = 2, 3
    xs = [i for i in range(len(points))]
    ys = [float(p["tok_s"]) for p in points]
    y_min, y_max = min(ys), max(ys)
    y_span = max(y_max - y_min, 0.1)
    x_span = max(len(points) - 1, 1)
    coords = []
    for i, y in enumerate(ys):
        px = pad_x + (i / x_span) * (w - 2 * pad_x)
        py = h - pad_y - ((y - y_min) / y_span) * (h - 2 * pad_y)
        coords.append(f"{px:.1f},{py:.1f}")
    poly = " ".join(coords)
    dots = "".join(
        f'<circle cx="{c.split(",")[0]}" cy="{c.split(",")[1]}" r="1.6" fill="#f5a14a"/>'
        for c in coords
    )
    return (
        f'<svg class="pbm-spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'aria-hidden="true" focusable="false">'
        f'<polyline fill="none" stroke="#f5a14a" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{poly}"/>'
        f"{dots}</svg>"
    )


def render_pbm_chart(points: list[dict], *, w: int = 420, h: int = 200) -> dict | None:
    """Detail-page chart geometry: path, dots, axis labels."""
    if len(points) < 2:
        return None
    left, right, top, bottom = 44, 18, 16, 36
    plot_w = w - left - right
    plot_h = h - top - bottom
    ys = [float(p["tok_s"]) for p in points]
    y_max = max(ys) * 1.12
    y_min = 0.0
    y_span = max(y_max - y_min, 0.1)
    x_span = max(len(points) - 1, 1)

    plotted = []
    for i, p in enumerate(points):
        x = left + (i / x_span) * plot_w
        y = top + plot_h - ((float(p["tok_s"]) - y_min) / y_span) * plot_h
        plotted.append({
            "fill": p["fill"],
            "tok_s": p["tok_s"],
            "x": round(x, 1),
            "y": round(y, 1),
        })

    path = "M " + " L ".join(f'{p["x"]},{p["y"]}' for p in plotted)
    # Soft area under the curve
    area = (
        f'{path} L {plotted[-1]["x"]},{top + plot_h} '
        f'L {plotted[0]["x"]},{top + plot_h} Z'
    )
    y_ticks = []
    for frac in (0.0, 0.5, 1.0):
        yy = top + frac * plot_h
        label_val = y_max * (1.0 - frac)
        y_ticks.append({"y": round(yy, 1), "label": f"{label_val:.0f}"})

    return {
        "w": w,
        "h": h,
        "left": left,
        "top": top,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "path": path,
        "area": area,
        "points": plotted,
        "y_ticks": y_ticks,
        "baseline_y": top + plot_h,
    }


def attach_pbm_metrics(m: dict, v: dict, pbm: dict, *, profile: str | None = None) -> None:
    """Attach 4k/50k/100k PBM fills, retention, and chart geometry."""
    profile = profile or m.get("golden_profile") or v.get("tok_s_profile")
    fills = {key: pbm_tok_s(pbm, profile, key) for key in PBM_FILL_KEYS}
    m["pbm_profile"] = profile
    m["pbm_4k"] = fills["4k"]
    m["pbm_50k"] = fills["50k"]
    m["pbm_100k"] = fills["100k"]
    m["pbm_fills"] = fills

    tok4 = fills["4k"]
    tok100 = fills["100k"]
    if tok4 and tok100 is not None and tok4 > 0:
        m["pbm_retention"] = round(tok100 / tok4 * 100)
    else:
        m["pbm_retention"] = None

    curve = _pbm_curve_points(fills)
    m["pbm_curve"] = curve
    m["pbm_spark_svg"] = render_pbm_sparkline(curve)
    m["pbm_chart"] = render_pbm_chart(curve)
    m["pbm_curve_label"] = curve[0]["fill"] if len(curve) == 1 else None

    # Default leaderboard headline = 4k
    if tok4 is not None:
        m["tok_s"] = tok4
        m["tok_s_ctx"] = PBM_FILL_TOKENS["4k"]
        m["tok_s_ctx_label"] = "4k"
        m["throughput"] = format_throughput(tok4, PBM_FILL_TOKENS["4k"])
        m["tok_s_method"] = "perfbench-metrics"
        m["pbm_fill"] = "4k"


def apply_editors_pick_headline(
    m: dict,
    recipes_by_id: dict[str, dict],
    benchmarks: dict[str, dict],
    profile_ctx: dict[str, dict],
    pbm: dict | None = None,
) -> None:
    """Editor's pick highlights the MTP companion profile when benched."""
    if m["id"] != EDITORS_PICK_ID:
        return
    recipe = recipes_by_id.get(EDITORS_PICK_PROFILE) or {}
    name = (recipe.get("name") or "").strip()
    if name:
        m["name"] = _RECIPE_NAME_PREFIX_RE.sub("", name).strip() or name
    pbm = pbm or {}
    if pbm_tok_s(pbm, EDITORS_PICK_PROFILE) is not None:
        attach_pbm_metrics(m, {}, pbm, profile=EDITORS_PICK_PROFILE)
        m["peak_profile"] = EDITORS_PICK_PROFILE
        return
    bench = benchmarks.get(EDITORS_PICK_PROFILE) or {}
    if bench.get("tok_s") is None:
        return
    ctx = resolve_profile_ctx(EDITORS_PICK_PROFILE, profile_ctx)
    if ctx is None:
        ctx = infer_bench_ctx_from_recipe(recipe)
    m["tok_s"] = round(float(bench["tok_s"]), 1)
    m["tok_s_ctx"] = ctx
    m["tok_s_ctx_label"] = format_ctx_label(ctx) if ctx else None
    m["throughput"] = format_throughput(m["tok_s"], ctx)
    m["peak_profile"] = EDITORS_PICK_PROFILE


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
    currently_testing = str(golden_raw.get("currently_testing") or "").strip()

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
    recipes_by_id, recipes_by_inv, profiles_by_inv = load_recipes()
    pbm = load_pbm_profiles()

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
            "now_testing": bool(currently_testing) and inv_path == currently_testing,
        }
        attach_model_params(m, cat)
        override = use_case_overrides.get(inv_path)
        if override:
            m["use_cases"] = sorted({str(t) for t in override})
        else:
            m["use_cases"] = derive_use_cases(m)
        attach_peak_bench(m, v, profile_ctx, benchmarks, bench_history, profiles_by_inv)
        attach_max_ctx_bench(m, v, profile_ctx, benchmarks)
        recipe = recipes_by_id.get(m.get("golden_profile") or "") or recipes_by_inv.get(inv_path)
        quant = derive_quant_label(cat, recipe, m.get("engine"))
        m["quant_label"] = quant
        m["variant_label"] = quant  # detail-page alias
        m["name"] = strip_quant_from_name(m["name"], quant)
        attach_recipe_summary(m, recipe)
        attach_bench_ladder(m, v, recipe, profile_ctx, benchmarks)
        attach_bench_runs(m, v, profile_ctx, benchmarks, bench_history, profiles_by_inv)
        attach_pbm_metrics(m, v, pbm)
        apply_editors_pick_headline(m, recipes_by_id, benchmarks, profile_ctx, pbm=pbm)
        models.append(m)

    models.sort(key=lambda m: m["tok_s"] or 0, reverse=True)
    return models


def compute_stats(models):
    tok_values = [m["tok_s"] for m in models if m["tok_s"]]
    engines_in_data = sorted({m["engine"] for m in models if m["engine"]})
    peak_model = max(models, key=lambda m: m["tok_s"] or 0) if models else None
    editors_pick = next((m for m in models if m["id"] == EDITORS_PICK_ID), None)
    now_testing = next((m for m in models if m.get("now_testing")), None)
    golden_models = [m for m in models if m.get("golden_profile")]
    max_ctx_benched = [m for m in golden_models if m.get("max_ctx_has_bench")]
    pbm_counts = {
        key: sum(1 for m in models if m.get(f"pbm_{key}") is not None)
        for key in PBM_FILL_KEYS
    }
    data_updated_at = ""
    stamps = [str(m.get("updated_at") or "")[:10] for m in models if m.get("updated_at")]
    if stamps:
        data_updated_at = max(stamps)
    return {
        "count": len(models),
        "peak_tok_s": max(tok_values) if tok_values else 0,
        "peak_throughput": peak_model["throughput"] if peak_model and peak_model.get("tok_s") else None,
        "median_tok_s": sorted(tok_values)[len(tok_values) // 2] if tok_values else 0,
        "engines": PRODUCT_ENGINES,
        "engines_in_data": engines_in_data,
        "editors_pick": editors_pick,
        "now_testing": now_testing,
        "max_ctx_golden_count": len(golden_models),
        "max_ctx_bench_count": len(max_ctx_benched),
        "bench_fill_ratio": BENCH_FILL_RATIO,
        "bench_fill_pct": int(BENCH_FILL_RATIO * 100),
        "pbm_fill": PBM_DISPLAY_FILL,
        "pbm_fill_label": PBM_DISPLAY_LABEL,
        "pbm_fill_keys": list(PBM_FILL_KEYS),
        "pbm_counts": pbm_counts,
        "data_updated_at": data_updated_at,
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

    # Compute bar widths relative to peak for each PBM fill (+ legacy max-ctx)
    peak = stats["peak_tok_s"] or 1
    max_ctx_peak = max((m["max_ctx_tok_s"] or 0) for m in models) or 1
    fill_peaks = {
        key: max((m.get(f"pbm_{key}") or 0) for m in models) or 1
        for key in PBM_FILL_KEYS
    }
    for m in models:
        m["tok_s_pct"] = round((m["tok_s"] or 0) / peak * 100, 1)
        m["max_ctx_tok_s_pct"] = round((m["max_ctx_tok_s"] or 0) / max_ctx_peak * 100, 1)
        for key in PBM_FILL_KEYS:
            val = m.get(f"pbm_{key}") or 0
            m[f"pbm_{key}_pct"] = round(val / fill_peaks[key] * 100, 1)
        m["editors_pick"] = m["id"] == EDITORS_PICK_ID
    stats["pbm_fill_peaks"] = {k: round(v, 1) for k, v in fill_peaks.items()}

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
