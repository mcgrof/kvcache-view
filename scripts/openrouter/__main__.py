# SPDX-License-Identifier: MIT
"""Run with python3 -m scripts.openrouter; make targets wrap this entry point."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .client import (Client, DataError, FLOOR, atomic_write, completed_day,
                     fetch_catalog, load_rankings, read_json, update_rankings,
                     write_json)

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", "trends", "attention", "context", "cache"], default="all", nargs="?")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=FLOOR)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=completed_day())
    parser.add_argument("--offline", action="store_true", help="Render saved snapshots; never access the network")
    parser.add_argument("--full-refresh", action="store_true", help="Refetch each requested year, including historical corrections")
    args = parser.parse_args(argv)
    if args.start < FLOOR or args.start > args.end or args.end > completed_day():
        parser.error("Use 2025-01-01 <= start <= end <= yesterday in UTC")
    data_dir = ROOT / "data" / "openrouter"
    lock_id = hashlib.sha256(str(ROOT).encode()).hexdigest()[:20]
    with open(Path(tempfile.gettempdir()) / ("kvcache-openrouter-" + lock_id + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args, data_dir)


def run(args, data_dir):
    client = Client(None if args.offline else os.environ.get("OPEN_ROUTER_API"))
    buckets = ("text", "100K", "1M") if args.target in ("all", "context") else (("text",) if args.target in ("trends", "attention") else ())
    catalog_path = data_dir / "catalog.json"
    # Fail before modifying anything when an authenticated refresh has no key.
    if not args.offline and buckets and not client.key:
        raise DataError("Set OPEN_ROUTER_API, or rebuild existing data with OPENROUTER_OFFLINE=1")
    if not args.offline:
        print("Refreshing public OpenRouter catalog")
        catalog = fetch_catalog(client)
        write_json(catalog_path, catalog)
        for bucket in buckets:
            print(f"Refreshing {bucket} rankings through {args.end}")
            update_rankings(client, data_dir, bucket, args.start, args.end, args.full_refresh)
    elif not catalog_path.exists():
        raise DataError("No saved catalog; run an online update first")
    catalog = read_json(catalog_path)
    provenance = {"source_url": "https://openrouter.ai/rankings", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "generated_for_end_date": str(args.end), "catalog": catalog["meta"], "snapshots": []}
    datasets = {}
    for bucket in buckets:
        datasets[bucket], snapshots = load_rankings(data_dir, bucket, args.start, args.end)
        for snap in snapshots:
            snap["dataset"] = bucket
            snap["path"] = str(Path(snap["path"]).relative_to(ROOT))
            provenance["snapshots"].append(snap)
    stamps = [s["meta"]["as_of"] for s in provenance["snapshots"]]
    provenance["as_of"] = max(stamps) if stamps else catalog["meta"]["as_of"]
    # Prepare all outputs before publishing any generated page.
    outputs = {}
    if buckets:
        from .analysis import analyze
        from .render import render_context, render_trends
        registry = read_json(data_dir / "architecture-registry.json")
        analysis = analyze(datasets, registry, catalog["data"], str(args.end))
        analysis["provenance"] = provenance
        serialized = json.dumps(analysis, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.target in ("all", "trends"):
            page_provenance = dict(provenance, analysis_url="data/openrouter/hybrid-trends.json", provenance_url="data/openrouter/hybrid-trends-provenance.json")
            outputs[data_dir / "hybrid-trends.json"] = serialized
            outputs[data_dir / "hybrid-trends-provenance.json"] = json.dumps(page_provenance, sort_keys=True, indent=2) + "\n"
            outputs[ROOT / "hybrid-trends.html"] = render_trends(analysis, page_provenance)
        if args.target in ("all", "context"):
            page_provenance = dict(provenance, analysis_url="data/openrouter/context-demand.json", provenance_url="data/openrouter/context-demand-provenance.json")
            outputs[data_dir / "context-demand.json"] = serialized
            outputs[data_dir / "context-demand-provenance.json"] = json.dumps(page_provenance, sort_keys=True, indent=2) + "\n"
            outputs[ROOT / "context-demand.html"] = render_context(analysis, page_provenance)
        if args.target in ("all", "attention"):
            from .attention import analyze_attention, render_attention
            attention_registry = read_json(data_dir / "attention-registry.json")
            attention = analyze_attention(datasets, registry, catalog["data"], str(args.end), attention_registry)
            page_provenance = dict(provenance, analysis_url="data/openrouter/attention-trends.json", provenance_url="data/openrouter/attention-trends-provenance.json")
            attention["provenance"] = page_provenance
            outputs[data_dir / "attention-trends.json"] = json.dumps(attention, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            outputs[data_dir / "attention-trends-provenance.json"] = json.dumps(page_provenance, sort_keys=True, indent=2) + "\n"
            outputs[ROOT / "attention-trends.html"] = render_attention(attention, page_provenance)
    if args.target in ("all", "cache"):
        from .cache import analyze_cache, render_cache
        cache = analyze_cache(catalog)
        outputs[data_dir / "cache-analysis.json"] = json.dumps(cache, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        cache_provenance = {"kind": "catalog", "source_url": catalog["meta"]["source_url"], "as_of": catalog["meta"]["as_of"], "catalog": catalog["meta"], "analysis_url": "data/openrouter/cache-analysis.json", "provenance_url": "data/openrouter/cache-provenance.json"}
        outputs[data_dir / "cache-provenance.json"] = json.dumps(cache_provenance, sort_keys=True, indent=2) + "\n"
        outputs[ROOT / "cache-telemetry.html"] = render_cache(cache, cache_provenance)
    for path, value in outputs.items():
        atomic_write(path, value)
        print("Wrote " + str(path.relative_to(ROOT)))


if __name__ == "__main__":
    try:
        main()
    except (DataError, OSError, ValueError, KeyError) as error:
        print("OpenRouter update failed: " + str(error), file=sys.stderr)
        sys.exit(1)
