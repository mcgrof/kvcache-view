# SPDX-License-Identifier: MIT
"""Archive public HF counters and compare them with a nearby OpenRouter window."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .client import Client, DataError, completed_day, read_json, utc_now, validate_rankings, write_json

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("downloads", "downloadsAllTime", "likes", "createdAt", "lastModified", "safetensors")


def fetch_model(repo_id):
    if (not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id)
            or any(part in (".", "..") for part in repo_id.split("/"))):
        raise ValueError("Invalid Hugging Face repository ID")
    url = "https://huggingface.co/api/models/" + repo_id + "?" + urllib.parse.urlencode(
        [("expand[]", field) for field in FIELDS])
    result = {"requested_id": repo_id, "repo_id": repo_id, "source_url": url}
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                          "User-Agent": "kvcache-view-adoption/1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                raise ValueError("Unexpected HF metadata")
            safetensors = payload.get("safetensors")
            parameters = safetensors.get("total") if isinstance(safetensors, dict) else None
            result.update(repo_id=payload["id"], downloads=payload.get("downloads"),
                          downloads_all_time=payload.get("downloadsAllTime"), likes=payload.get("likes"),
                          created_at=payload.get("createdAt"), last_modified=payload.get("lastModified"),
                          parameters_total=parameters,
                          response_sha256=hashlib.sha256(raw).hexdigest(), http_status=200,
                          source_fields=payload, error=None)
            break
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            result.update(error=f"HTTP {error.code}", http_status=error.code)
            break
        except (OSError, ValueError):
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            result["error"] = "Metadata unavailable after retries"
    result["retrieved_at"] = utc_now()
    return result


def fetch_snapshot(repo_ids):
    start = utc_now()
    with ThreadPoolExecutor(max_workers=4) as pool:
        data = list(pool.map(fetch_model, sorted(set(repo_ids))))
    if not any(row.get("downloads") is not None for row in data):
        raise DataError("No HF download counters returned; existing files were preserved")
    return {"meta": {"retrieved_at": start, "retrieval_completed_at": utc_now(),
                     "source": "Hugging Face public model metadata API", "fields": list(FIELDS),
                     "download_window": "Trailing 30 days as reported by HF; exact counter boundaries unpublished",
                     "counting_rules": "https://huggingface.co/docs/hub/models-download-stats"}, "data": data}


def main(argv=None):
    from .huggingface import analyze_huggingface, build_hf_mapping
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=completed_day())
    parser.add_argument("--snapshot-date", type=dt.date.fromisoformat,
                        default=dt.datetime.now(dt.timezone.utc).date())
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--replace-snapshot", action="store_true",
                        help="Explicitly replace same-day raw files during an online refresh")
    args = parser.parse_args(argv)
    if args.end > completed_day():
        parser.error("The OpenRouter window must end on a completed UTC day")
    if not args.offline and args.snapshot_date != dt.datetime.now(dt.timezone.utc).date():
        parser.error("A live HF counter must be dated today; use --offline for archived snapshots")
    if args.end != args.snapshot_date - dt.timedelta(days=1):
        parser.error("Compare each HF snapshot with the preceding completed UTC day")
    folder = ROOT / "data/openrouter"
    raw_dir = folder / "huggingface"
    registry_path, catalog_path = folder / "lifetime-registry.json", folder / "lifetime-catalog.json"
    registry, catalog = read_json(registry_path), read_json(catalog_path)
    mapping = build_hf_mapping(registry, catalog)
    hf_path = raw_dir / f"snapshot-{args.snapshot_date}.json"
    rankings_path = raw_dir / f"rankings-{args.end}.json.gz"
    start = args.end - dt.timedelta(days=29)
    if not args.offline:
        if not args.replace_snapshot and (hf_path.exists() or rankings_path.exists()):
            raise DataError("Dated raw files already exist; use --offline or explicitly --replace-snapshot")
        client = Client(os.environ.get("OPEN_ROUTER_API"))
        if not client.key:
            raise DataError("Set OPEN_ROUTER_API, or use --offline with saved snapshots")
        params = {"start_date": str(start), "end_date": str(args.end), "period": "day", "modality": "text"}
        url = "https://openrouter.ai/api/v1/datasets/rankings-daily?" + urllib.parse.urlencode(params)
        rankings = validate_rankings(client.get(url, authenticated=True), start, args.end)
        rankings["meta"]["source_url"] = url
        snapshot = fetch_snapshot(group["repo_id"] for group in mapping["groups"])
        write_json(rankings_path, rankings)
        write_json(hf_path, snapshot)
    rankings, snapshot = read_json(rankings_path), read_json(hf_path)
    validate_rankings(rankings, start, args.end)
    if snapshot.get("meta", {}).get("retrieved_at", "")[:10] != str(args.snapshot_date):
        raise DataError("HF snapshot timestamp does not match the requested snapshot date")
    analysis = analyze_huggingface(rankings["data"], registry, catalog, snapshot, str(args.end))
    lifetime_path = folder / "model-lifetime.json"
    lifetime = read_json(lifetime_path)
    old = {model["id"]: model for model in lifetime["models"]}
    for model in analysis["models"]:
        model["lifetime_status_as_of"] = lifetime["end_date"]
        model["lifetime_status"] = {slug: old[slug]["status"] for slug in model["model_ids"] if slug in old}
    source_paths = (registry_path, catalog_path, hf_path, rankings_path, lifetime_path)
    analysis["provenance"] = {
        "openrouter": rankings["meta"], "huggingface": snapshot["meta"],
        "source_files": [str(path.relative_to(ROOT)) for path in source_paths],
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "openrouter_license": "CC BY 4.0", "huggingface_terms": "https://huggingface.co/terms-of-service",
        "modifications": "Selected public metadata; exact-repository grouping; token bounds and correlations. No prompts or account analytics.",
    }
    write_json(folder / "huggingface-comparison.json", analysis)
    print("Wrote data/openrouter/huggingface-comparison.json")


if __name__ == "__main__":
    try:
        main()
    except (DataError, OSError, ValueError, KeyError) as error:
        raise SystemExit("Hugging Face comparison failed: " + str(error))
