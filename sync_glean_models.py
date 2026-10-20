# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ruamel.yaml>=0.18.10,<0.19",
#     "truststore==0.10.4",
# ]
# ///
"""Fetch Glean's model catalog and merge supported models into OMP models.yml.

Required environment variables:
    GLEAN_INSTANCE_ID  The instance prefix, e.g. ``scio-prod``.
    GLEAN_API_KEY      Bearer token for the Glean model catalog endpoint.

By default, updates ``~/.omp/agent/models.yml``. Existing provider settings and
models not present in the fetched catalog are retained. ``--format dsh`` writes
a copy-ready ``cordis.patch.yml`` beside this script without touching ``~/.dsh``.
Run with ``--dry-run`` to preview the YAML diff, or ``--input-json`` to test
against a saved response.

Examples:
    export GLEAN_INSTANCE_ID=scio-prod
    export GLEAN_API_KEY=...  # Prefer loading this from your secret manager.
    uv run sync_glean_models.py --dry-run
    uv run sync_glean_models.py
    uv run sync_glean_models.py --format dsh --dry-run
    uv run sync_glean_models.py --format dsh
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import truststore
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
DEFAULT_OUTPUT = Path.home() / ".omp" / "agent" / "models.yml"
DEFAULT_DSH_OUTPUT = Path(__file__).with_name("cordis.patch.yml")

# Route selection is endpoint-first, shared by OMP and DSH output. The first
# matching endpoint wins, so Anthropic Messages outranks Responses, which
# outranks chat completions.
ENDPOINT_PROVIDER_PRIORITY = (
    ("v1/messages", "glean-anthropic"),
    ("v1/responses", "glean-openai"),
    ("v1/chat/completions", "glean"),
)

PROVIDER_SPECS = {
    "glean": {
        "api": "openai-completions",
        "base_url_path": "/rest/api/v1/openai/v1",
    },
    "glean-openai": {
        "api": "openai-responses",
        "base_url_path": "/rest/api/v1/openai/v1",
        "discovery": "openai-models-list",
    },
    "glean-anthropic": {
        "api": "anthropic-messages",
        "base_url_path": "/rest/api/v1/anthropic",
        "discovery": "openai-models-list",
    },
}
ALLOWED_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}

# Glean's catalog can omit this model's Messages endpoint even though it works
# through the Anthropic route. Keep it included, using the metadata previously
# advertised for it by the Glean catalog.
ANTHROPIC_ENDPOINT_OMISSIONS = {"CLAUDE_5_5_OPUS"}

# This tenant's gateway rejects adaptive thinking for Haiku; don't expose OMP
# reasoning controls for it even when the catalog's supportsReasoning flag is true.
MODELS_WITHOUT_WORKING_REASONING = {"CLAUDE_4_5_HAIKU_20251001"}

# The simple Haiku aliases are not the working model ID on this gateway.
MODEL_ID_OVERRIDES = {
    "CLAUDE_4_5_HAIKU_20251001": "anthropic.claude-haiku-4-5-20251001-v1:0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Glean model metadata and merge it into OMP models.yml."
    )
    parser.add_argument(
        "--instance-id",
        default=os.environ.get("GLEAN_INSTANCE_ID"),
        help="Glean instance prefix (defaults to GLEAN_INSTANCE_ID).",
    )
    parser.add_argument(
        "--format",
        choices=("omp", "dsh"),
        default="omp",
        help="Output format: OMP models.yml or DSH cordis.patch.yml (default: omp).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (format-specific default when omitted).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Use a saved catalog response instead of making an HTTP request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed YAML diff without writing a file.",
    )
    return parser.parse_args()


def validate_instance_id(instance_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", instance_id):
        raise ValueError(
            "GLEAN_INSTANCE_ID must be a hostname prefix containing only "
            "letters, digits, and hyphens."
        )
    return instance_id


def fetch_catalog(instance_id: str, api_key: str, timeout: float) -> dict[str, Any]:
    url = f"https://{instance_id}-be.glean.com/rest/api/v1/models"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "omp-glean-model-sync/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(
            f"Glean catalog request failed with HTTP {exc.code}."
        ) from None
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach the Glean model catalog: {exc.reason}"
        ) from None

    if len(body) > MAX_RESPONSE_BYTES:
        raise RuntimeError(
            f"Catalog response exceeds the {MAX_RESPONSE_BYTES // (1024 * 1024)} MiB limit."
        )
    try:
        payload = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Glean returned invalid JSON: {exc}") from None
    return validate_catalog(payload)


def load_catalog_fixture(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read catalog fixture {path}: {exc}") from None
    return validate_catalog(payload)


def validate_catalog(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise TypeError("Expected the catalog response to contain a 'models' array.")
    if not payload["models"]:
        raise RuntimeError(
            "The catalog returned no models; refusing to modify the config."
        )
    return payload


def aliases_for(model: Mapping[str, Any]) -> list[str]:
    aliases = model.get("modelAliases")
    if not isinstance(aliases, list):
        return []
    return [alias for alias in aliases if isinstance(alias, str) and alias.strip()]


def provider_for_model(model: Mapping[str, Any]) -> str | None:
    """Map by wire capability using Messages > Responses > chat precedence."""
    if model.get("model") in ANTHROPIC_ENDPOINT_OMISSIONS:
        return "glean-anthropic"
    features = model.get("features", {})
    if not isinstance(features, dict):
        return None
    endpoints = features.get("supportedEndpoints", [])
    if not isinstance(endpoints, list):
        return None
    for endpoint, provider in ENDPOINT_PROVIDER_PRIORITY:
        if endpoint in endpoints:
            return provider
    return None


def canonical_model_id(model: Mapping[str, Any]) -> str | None:
    internal_id = model.get("model")
    if isinstance(internal_id, str) and internal_id in MODEL_ID_OVERRIDES:
        return MODEL_ID_OVERRIDES[internal_id]

    aliases = aliases_for(model)
    if not aliases:
        return None

    # Prefer an API-facing alias over cloud-provider-qualified aliases. Choosing
    # the last plain alias prefers dated/version-specific GPT IDs and variants
    # such as gpt-6-sol while keeping Claude's simple public alias.
    plain_aliases = [
        alias
        for alias in aliases
        if not alias.startswith(("anthropic.", "accounts/", "nvidia/", "zai-org/"))
    ]
    return (plain_aliases or aliases)[-1]


def catalog_models(
    payload: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    generated: dict[str, list[dict[str, Any]]] = {name: [] for name in PROVIDER_SPECS}
    aliases_by_provider: dict[str, set[str]] = {name: set() for name in PROVIDER_SPECS}
    found_opus_55 = False

    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        internal_id = model.get("model")
        features = model.get("features")
        if not isinstance(features, dict):
            continue

        supported_inputs = features.get("supportedInputs", {})
        if not isinstance(supported_inputs, dict) or not supported_inputs.get(
            "hasText"
        ):
            continue

        provider = provider_for_model(model)
        if provider is None:
            continue

        entry = make_omp_model(model, provider)
        if entry is None:
            continue
        if internal_id == "CLAUDE_5_5_OPUS":
            found_opus_55 = True
        generated[provider].append(entry)
        aliases_by_provider[provider].update(aliases_for(model))
        aliases_by_provider[provider].add(entry["id"])

    # The catalog may omit Opus 5.5 entirely. Preserve the known working model
    # with its previously advertised limits, costs, and effort levels.
    if not found_opus_55:
        opus_55 = make_opus_55_fallback()
        generated["glean-anthropic"].append(opus_55)
        aliases_by_provider["glean-anthropic"].update(
            {"claude-opus-5-5", "anthropic.claude-opus-5-5"}
        )

    for entries in generated.values():
        entries.sort(key=lambda entry: entry["id"].casefold())

    return generated, aliases_by_provider


def make_omp_model(model: Mapping[str, Any], provider: str) -> dict[str, Any] | None:
    features = model.get("features", {})
    token_info = features.get("tokenInfo", {})
    if not isinstance(token_info, dict):
        token_info = {}

    model_id = canonical_model_id(model)
    if not model_id:
        print(
            f"Skipping {model.get('model', '<unknown>')}: no usable model alias.",
            file=sys.stderr,
        )
        return None

    context_window = token_info.get("maxContextTokens") or token_info.get(
        "maxInputTokens"
    )
    max_tokens = token_info.get("maxOutputTokens")
    if not _positive_integer(context_window) or not _positive_integer(max_tokens):
        print(
            f"Skipping {model_id}: catalog has no positive context/output token limits.",
            file=sys.stderr,
        )
        return None

    supported_inputs = features.get("supportedInputs", {})
    image_extensions = supported_inputs.get("imageExtensions", [])
    model_input = ["text"]
    if isinstance(image_extensions, list) and image_extensions:
        model_input.append("image")

    internal_id = model.get("model")
    reasoning = bool(features.get("supportsReasoning", False))
    if internal_id in MODELS_WITHOUT_WORKING_REASONING:
        reasoning = False

    cost = token_info
    entry: dict[str, Any] = {
        "id": model_id,
        "name": f"Glean {model.get('displayName') or model_id}",
        "reasoning": reasoning,
        "input": model_input,
        "supportsTools": bool(features.get("supportsFunctionCalling", False)),
        "contextWindow": int(context_window),
        "maxTokens": int(max_tokens),
        "cost": {
            "input": _price_per_million(cost.get("pricingPerInputToken")),
            "output": _price_per_million(cost.get("pricingPerOutputToken")),
            "cacheRead": _price_per_million(cost.get("pricingPerCachedInputTokenRead")),
            # Glean's catalog has no cached-write price field.
            "cacheWrite": 0,
        },
    }
    if provider == "glean" and reasoning:
        entry["compat"] = {"supportsReasoningEffort": True}

    levels = features.get("supportedThinkingLevels", [])
    if reasoning and isinstance(levels, list):
        efforts = [level for level in levels if level in ALLOWED_EFFORTS]
        unsupported = [
            level
            for level in levels
            if level != "none" and level not in ALLOWED_EFFORTS
        ]
        if unsupported:
            print(
                f"Ignoring unsupported OMP thinking levels for {model_id}: "
                f"{', '.join(map(str, unsupported))}.",
                file=sys.stderr,
            )
        if efforts:
            mode = "anthropic-adaptive" if provider == "glean-anthropic" else "effort"
            entry["thinking"] = {"mode": mode, "efforts": efforts}

    return entry


def opus_55_fallback_record() -> dict[str, Any]:
    # Glean's prior catalog metadata for Claude Opus 5.5. The current payload can
    # omit the model entirely; the user confirmed it works on the Anthropic route.
    return {
        "model": "CLAUDE_5_5_OPUS",
        "modelAliases": ["claude-opus-5-5", "anthropic.claude-opus-5-5"],
        "creator": "CREATOR_ANTHROPIC",
        "displayName": "Claude Opus 5.5",
        "features": {
            "supportedInputs": {
                "hasText": True,
                "imageExtensions": ["jpg", "jpeg", "png", "gif", "webp"],
            },
            "supportedEndpoints": ["v1/messages"],
            "supportedThinkingLevels": ["low", "medium", "high", "xhigh", "max"],
            "supportsReasoning": True,
            "supportsFunctionCalling": True,
            "tokenInfo": {
                "maxContextTokens": 1_000_000,
                "maxOutputTokens": 128_000,
                "pricingPerInputToken": 0.000004,
                "pricingPerOutputToken": 0.00002,
                "pricingPerCachedInputTokenRead": 0.0000002,
            },
        },
    }


def make_opus_55_fallback() -> dict[str, Any]:
    entry = make_omp_model(opus_55_fallback_record(), "glean-anthropic")
    if entry is None:
        raise RuntimeError("Could not build the Claude Opus 5.5 fallback entry.")
    return entry


DSH_PROVIDER_SPECS = {
    "glean": {
        "displayName": "Glean",
        "apiKeyEnv": "GLEAN_API_KEY",
        "api": "openai-completions",
        "base_url_path": "/rest/api/v1/openai/v1",
    },
    "glean-anthropic": {
        "displayName": "Glean Anthropic",
        "apiKeyEnv": "GLEAN_ANTHROPIC_API_KEY",
        "api": "anthropic-messages",
        "base_url_path": "/rest/api/v1/anthropic",
    },
    "glean-openai": {
        "displayName": "Glean OpenAI",
        "apiKeyEnv": "GLEAN_GPT_API_KEY",
        "api": "openai-responses",
        "base_url_path": "/rest/api/v1/openai/v1/",
    },
}


def catalog_dsh_models(payload: Mapping[str, Any]) -> dict[str, list[CommentedMap]]:
    generated: dict[str, list[CommentedMap]] = {
        provider: [] for provider in DSH_PROVIDER_SPECS
    }
    found_opus_55 = False

    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        provider = dsh_provider_for_model(model)
        if provider is None:
            continue
        entry = make_dsh_model(model, provider)
        if entry is None:
            continue
        if model.get("model") == "CLAUDE_5_5_OPUS":
            found_opus_55 = True
        generated[provider].append(entry)

    if not found_opus_55:
        fallback = make_dsh_model(opus_55_fallback_record(), "glean-anthropic")
        if fallback is None:
            raise RuntimeError(
                "Could not build the Claude Opus 5.5 DSH fallback entry."
            )
        generated["glean-anthropic"].append(fallback)

    for entries in generated.values():
        entries.sort(key=lambda entry: str(entry["id"]).casefold())
    return generated


def dsh_provider_for_model(model: Mapping[str, Any]) -> str | None:
    features = model.get("features", {})
    if not isinstance(features, dict):
        return None
    supported_inputs = features.get("supportedInputs", {})
    if not isinstance(supported_inputs, dict) or not supported_inputs.get("hasText"):
        return None
    return provider_for_model(model)


def make_dsh_model(model: Mapping[str, Any], provider: str) -> CommentedMap | None:
    internal_id = model.get("model")
    if not isinstance(internal_id, str) or not internal_id:
        print("Skipping DSH model without a catalog model id.", file=sys.stderr)
        return None

    features = model.get("features", {})
    if not isinstance(features, dict):
        return None
    token_info = features.get("tokenInfo", {})
    if not isinstance(token_info, dict):
        token_info = {}
    context_window = token_info.get("maxContextTokens") or token_info.get(
        "maxInputTokens"
    )
    max_tokens = token_info.get("maxOutputTokens")
    if not _positive_integer(context_window) or not _positive_integer(max_tokens):
        print(
            f"Skipping DSH model {internal_id}: catalog has no positive "
            "context/output token limits.",
            file=sys.stderr,
        )
        return None

    supported_inputs = features.get("supportedInputs", {})
    image_extensions = (
        supported_inputs.get("imageExtensions", [])
        if isinstance(supported_inputs, dict)
        else []
    )
    modalities = CommentedSeq(["text"])
    if isinstance(image_extensions, list) and image_extensions:
        modalities.append("image")
    modalities.fa.set_flow_style()

    reasoning_efforts = dsh_reasoning_efforts(model)
    entry = CommentedMap(
        {
            "id": internal_id,
            "name": model.get("displayName") or internal_id,
            "contextWindow": int(context_window),
            "maxTokens": int(max_tokens),
            "input": modalities,
            "reasoningEfforts": reasoning_efforts,
        }
    )
    if provider == "glean" and reasoning_efforts is not False:
        entry["compat"] = CommentedMap({"supportsReasoningEffort": True})
    return entry


def dsh_reasoning_efforts(model: Mapping[str, Any]) -> CommentedMap | bool:
    features = model.get("features", {})
    if not isinstance(features, dict) or not features.get("supportsReasoning", False):
        return False
    if model.get("model") in MODELS_WITHOUT_WORKING_REASONING:
        return False

    levels = features.get("supportedThinkingLevels", [])
    if not isinstance(levels, list):
        return False

    efforts = CommentedMap()
    for level in levels:
        if level == "none":
            efforts["off"] = None
        elif level in ALLOWED_EFFORTS:
            efforts[level] = level
    if not efforts or list(efforts) == ["off"]:
        return False
    return efforts


def build_dsh_config(payload: Mapping[str, Any], instance_id: str) -> CommentedSeq:
    generated = catalog_dsh_models(payload)
    providers = CommentedMap()
    host = f"https://{instance_id}-be.glean.com"

    for provider_id, spec in DSH_PROVIDER_SPECS.items():
        profile = CommentedMap(
            {
                "displayName": spec["displayName"],
                "apiKeyEnv": spec["apiKeyEnv"],
                "api": spec["api"],
                "baseURL": f"{host}{spec['base_url_path']}",
                "models": CommentedSeq(generated[provider_id]),
            }
        )
        if provider_id == "glean-anthropic":
            profile["compat"] = CommentedMap(
                {"forceAdaptiveThinking": True, "supportsStrictTools": False}
            )
        providers[provider_id] = profile

    plugin = CommentedMap(
        {
            "id": "llm-pi-ai",
            "name": "@deepseek-ai/dsh-llm-pi-ai",
            "config": CommentedMap({"providers": providers}),
        }
    )
    return CommentedSeq([plugin])


def _positive_integer(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _price_per_million(value: Any) -> int | float:
    if value is None:
        return 0
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Invalid token price type in catalog: {value!r}.")
    price = float(value) * 1_000_000
    if not math.isfinite(price) or price < 0:
        raise ValueError(f"Invalid token price in catalog: {value!r}.")
    rounded = round(price, 8)
    return int(rounded) if rounded.is_integer() else rounded


def load_config(path: Path, yaml: YAML) -> tuple[CommentedMap, str]:
    if not path.exists():
        config = CommentedMap()
        return config, ""
    try:
        original = path.read_text(encoding="utf-8")
        config = yaml.load(original)
    except (OSError, UnicodeError, YAMLError) as exc:
        raise RuntimeError(f"Could not parse OMP config {path}: {exc}") from None
    if config is None:
        config = CommentedMap()
    if not isinstance(config, Mapping):
        raise TypeError(f"OMP config {path} must contain a YAML mapping.")
    return config, original


def migrate_legacy_responses_provider(
    providers: MutableMapping[str, Any],
    aliases_by_provider: Mapping[str, set[str]],
) -> None:
    """Move the former Responses route named `glean` to the new route layout."""
    legacy = providers.get("glean")
    if (
        not isinstance(legacy, MutableMapping)
        or legacy.get("api") != "openai-responses"
    ):
        return

    legacy_models = legacy.get("models", [])
    if not isinstance(legacy_models, Sequence) or isinstance(legacy_models, str):
        raise TypeError("Legacy OMP provider 'glean' models must be a YAML list.")

    target = providers.get("glean-openai")
    if target is None:
        target = copy.deepcopy(legacy)
        providers["glean-openai"] = target
    elif not isinstance(target, MutableMapping):
        raise TypeError("OMP provider 'glean-openai' must be a YAML mapping.")
    elif target.get("api") not in (None, "openai-responses"):
        raise RuntimeError(
            "Cannot migrate the old glean Responses route: glean-openai has "
            f"api={target.get('api')!r}, expected 'openai-responses'."
        )
    else:
        for key, value in legacy.items():
            if key not in {"api", "models", "name"} and key not in target:
                target[key] = copy.deepcopy(value)
        target_models = target.get("models")
        if target_models is None:
            target_models = CommentedSeq()
            target["models"] = target_models
        if not isinstance(target_models, MutableSequence) or isinstance(
            target_models, str
        ):
            raise TypeError("OMP provider 'glean-openai' models must be a YAML list.")
        existing_ids = {
            model.get("id") for model in target_models if isinstance(model, Mapping)
        }
        for model in legacy_models:
            model_id = model.get("id") if isinstance(model, Mapping) else None
            if model_id not in existing_ids:
                target_models.append(copy.deepcopy(model))
                if model_id is not None:
                    existing_ids.add(model_id)

    target_models = target.get("models")
    if target_models is None:
        target_models = CommentedSeq()
        target["models"] = target_models
    if not isinstance(target_models, MutableSequence) or isinstance(target_models, str):
        raise TypeError("OMP provider 'glean-openai' models must be a YAML list.")

    chat_models = CommentedSeq()
    anthro_models: list[Any] = []
    moved_ids: dict[str, set[str]] = {
        "glean": set(),
        "glean-anthropic": set(),
    }
    for model in legacy_models:
        if not isinstance(model, Mapping):
            continue
        model_id = model.get("id")
        destination = next(
            (
                provider_id
                for provider_id in ("glean-anthropic", "glean-openai", "glean")
                if model_id in aliases_by_provider.get(provider_id, set())
            ),
            "glean-openai",
        )
        if destination == "glean-openai":
            continue
        if isinstance(model_id, str):
            moved_ids[destination].add(model_id)
        if destination == "glean":
            chat_models.append(copy.deepcopy(model))
        else:
            anthro_models.append(copy.deepcopy(model))

    if moved_ids["glean"] or moved_ids["glean-anthropic"]:
        for index in range(len(target_models) - 1, -1, -1):
            model = target_models[index]
            if isinstance(model, Mapping) and model.get("id") in (
                moved_ids["glean"] | moved_ids["glean-anthropic"]
            ):
                del target_models[index]

    if anthro_models:
        anthro = providers.get("glean-anthropic")
        if anthro is None:
            anthro = CommentedMap()
            providers["glean-anthropic"] = anthro
        if not isinstance(anthro, MutableMapping):
            raise TypeError("OMP provider 'glean-anthropic' must be a YAML mapping.")
        anthro_list = anthro.get("models")
        if anthro_list is None:
            anthro_list = CommentedSeq()
            anthro["models"] = anthro_list
        if not isinstance(anthro_list, MutableSequence) or isinstance(anthro_list, str):
            raise TypeError(
                "OMP provider 'glean-anthropic' models must be a YAML list."
            )
        anthro_ids = {
            model.get("id") for model in anthro_list if isinstance(model, Mapping)
        }
        for model in anthro_models:
            model_id = model.get("id")
            if model_id not in anthro_ids:
                anthro_list.append(model)
                if model_id is not None:
                    anthro_ids.add(model_id)

    target["api"] = "openai-responses"
    target.pop("name", None)
    legacy["api"] = "openai-completions"
    legacy["models"] = chat_models
    legacy.pop("discovery", None)
    legacy.pop("name", None)


def merge_provider_models(
    config: CommentedMap,
    instance_id: str,
    generated: Mapping[str, list[dict[str, Any]]],
    aliases_by_provider: Mapping[str, set[str]],
) -> None:
    providers = config.get("providers")
    if providers is None:
        providers = CommentedMap()
        config["providers"] = providers
    if not isinstance(providers, MutableMapping):
        raise TypeError("OMP config 'providers' must be a mutable YAML mapping.")
    migrate_legacy_responses_provider(providers, aliases_by_provider)

    for provider_id, spec in PROVIDER_SPECS.items():
        provider = providers.get(provider_id)
        if provider is None:
            provider = CommentedMap()
            providers[provider_id] = provider
        if not isinstance(provider, MutableMapping):
            raise TypeError(
                f"OMP provider {provider_id!r} must be a mutable YAML mapping."
            )

        configured_api = provider.get("api")
        if configured_api and configured_api != spec["api"]:
            raise RuntimeError(
                f"Provider {provider_id!r} uses api={configured_api!r}; expected "
                f"{spec['api']!r}. Refusing to merge models into the wrong transport."
            )
        provider.pop("name", None)
        provider.setdefault(
            "baseUrl", f"https://{instance_id}-be.glean.com{spec['base_url_path']}"
        )
        provider.setdefault("apiKey", "GLEAN_API_KEY")
        provider["api"] = spec["api"]
        if provider_id == "glean-anthropic":
            provider.setdefault("disableStrictTools", True)

        discovery_type = spec.get("discovery")
        if discovery_type is None:
            provider.pop("discovery", None)
        else:
            discovery = provider.get("discovery")
            if discovery is None:
                discovery = CommentedMap()
                provider["discovery"] = discovery
            if not isinstance(discovery, MutableMapping):
                raise TypeError(
                    f"OMP provider {provider_id!r} discovery must be a mapping."
                )
            discovery["type"] = discovery_type

        current_models = provider.get("models")
        if current_models is None:
            current_models = CommentedSeq()
            provider["models"] = current_models
        if not isinstance(current_models, MutableSequence) or isinstance(
            current_models, str
        ):
            raise TypeError(f"OMP provider {provider_id!r} models must be a YAML list.")

        replace_aliases = aliases_by_provider[provider_id]
        for index in range(len(current_models) - 1, -1, -1):
            existing = current_models[index]
            if isinstance(existing, Mapping) and existing.get("id") in replace_aliases:
                del current_models[index]
        current_models.extend(
            commented_model(entry) for entry in generated[provider_id]
        )


def commented_model(entry: Mapping[str, Any]) -> CommentedMap:
    model = CommentedMap()
    for key in ("id", "name", "reasoning"):
        model[key] = entry[key]
    input_values = CommentedSeq(entry["input"])
    input_values.fa.set_flow_style()
    model["input"] = input_values
    model["supportsTools"] = entry["supportsTools"]
    model["contextWindow"] = entry["contextWindow"]
    model["maxTokens"] = entry["maxTokens"]
    cost = CommentedMap(entry["cost"])
    cost.fa.set_flow_style()
    model["cost"] = cost
    if "compat" in entry:
        model["compat"] = CommentedMap(entry["compat"])
    if "thinking" in entry:
        thinking = CommentedMap()
        thinking["mode"] = entry["thinking"]["mode"]
        efforts = CommentedSeq(entry["thinking"]["efforts"])
        efforts.fa.set_flow_style()
        thinking["efforts"] = efforts
        thinking.fa.set_flow_style()
        model["thinking"] = thinking
    return model


def dump_config(config: Any, yaml: YAML) -> str:
    from io import StringIO

    stream = StringIO()
    yaml.dump(config, stream)
    return stream.getvalue()


def write_atomically(path: Path, content: str) -> Path | None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing to replace symlink output {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600

    if path.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
        shutil.copy2(path, backup_path)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, existing_mode)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)
        raise
    return backup_path


def main() -> int:
    truststore.inject_into_ssl()
    args = parse_args()
    try:
        if not args.instance_id:
            raise ValueError("Set GLEAN_INSTANCE_ID or pass --instance-id.")
        instance_id = validate_instance_id(args.instance_id)
        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than zero.")
        output_path = args.output or (
            DEFAULT_DSH_OUTPUT if args.format == "dsh" else DEFAULT_OUTPUT
        )
        if args.input_json:
            payload = load_catalog_fixture(args.input_json)
            print(f"Loaded catalog fixture: {args.input_json}")
        else:
            api_key = os.environ.get("GLEAN_API_KEY")
            if not api_key:
                raise ValueError("GLEAN_API_KEY is required in the environment.")
            payload = fetch_catalog(instance_id, api_key, args.timeout)
            print(
                f"Fetched {len(payload['models'])} catalog records from "
                f"https://{instance_id}-be.glean.com/rest/api/v1/models"
            )

        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.width = 100
        yaml.indent(mapping=2, sequence=4, offset=2)

        if args.format == "omp":
            generated, aliases = catalog_models(payload)
            config, original = load_config(output_path, yaml)
            merge_provider_models(config, instance_id, generated, aliases)
            for provider, entries in generated.items():
                print(f"{provider}: {len(entries)} models")
        else:
            config = build_dsh_config(payload, instance_id)
            try:
                original = output_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                original = ""
            for provider_id, profile in config[0]["config"]["providers"].items():
                print(f"{provider_id}: {len(profile['models'])} models")

        updated = dump_config(config, yaml)
        if args.dry_run:
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=str(output_path),
                tofile=f"{output_path} (proposed)",
            )
            sys.stdout.writelines(diff)
            return 0

        if original == updated:
            print(f"No changes needed in {output_path}.")
            return 0

        backup = write_atomically(output_path, updated)
        print(f"Wrote {output_path}.")
        if backup:
            print(f"Previous file backed up to {backup}.")
        if args.format == "dsh":
            print("This is a generated copy; no ~/.dsh profile was modified.")
        else:
            print("Non-matching model entries and other providers were preserved.")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
