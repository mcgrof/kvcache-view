# SPDX-License-Identifier: MIT
"""Fetch public metadata only. Never write credentials or account analytics."""

import datetime as dt
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://openrouter.ai/api/v1"
FLOOR = dt.date(2025, 1, 1)
BUCKETS = ("text", "100K", "1M")


class DataError(RuntimeError):
    """Actionable error without an upstream body or credential-bearing headers."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def completed_day():
    return dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode()
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temp = f.name
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp, 0o644)
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


def write_json(path, data):
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if str(path).endswith(".gz"):
        raw = gzip.compress(raw, mtime=0)
    atomic_write(path, raw)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward bearer credentials to redirected origins.
        return None


class Client:
    def __init__(self, key=None, interval=2.1):
        # http.client includes an invalid header's value in some ValueErrors.
        # Validate before a bearer header exists, and never echo the key.
        if key is not None and (
            not isinstance(key, str)
            or not key.isascii()
            or any(not char.isprintable() or char.isspace() for char in key)
        ):
            raise DataError("OPEN_ROUTER_API must contain an ASCII token without whitespace or control characters")
        self.key = key
        self.interval = interval
        self.previous = 0.0
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, url, authenticated=False):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "openrouter.ai" or not parsed.path.startswith("/api/v1/"):
            raise DataError("Refusing a non-OpenRouter API URL")
        if authenticated and not self.key:
            raise DataError("Set OPEN_ROUTER_API for rankings updates, or use OPENROUTER_OFFLINE=1 to rebuild saved snapshots")
        headers = {"Accept": "application/json", "User-Agent": "kvcache-view-openrouter/1"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(4):
            time.sleep(max(0, self.previous + self.interval - time.monotonic()))
            self.previous = time.monotonic()
            request = urllib.request.Request(url, headers=headers)
            try:
                with self.opener.open(request, timeout=60) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict) or "error" in payload:
                    raise DataError("OpenRouter returned an unexpected or error payload")
                return payload
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    retry = e.headers.get("Retry-After", "")
                    try:
                        delay = float(retry) if retry else 2 ** (attempt + 1)
                    except ValueError:
                        try:
                            import email.utils
                            retry_at = email.utils.parsedate_to_datetime(retry)
                            delay = (retry_at - dt.datetime.now(dt.timezone.utc)).total_seconds()
                        except (TypeError, ValueError):
                            delay = 2 ** (attempt + 1)
                    if delay > 60:
                        raise DataError("OpenRouter requested a retry after more than 60 seconds; retry the make command later") from None
                    time.sleep(max(0, delay))
                    continue
                hint = " Check OPEN_ROUTER_API." if e.code in (401, 403) else ""
                raise DataError(f"OpenRouter HTTP {e.code}.{hint} Existing generated pages were not rebuilt.") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                if attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise DataError("OpenRouter request failed after retries; existing snapshots remain available") from None
            except ValueError:
                # Header validation and other urllib internals can include
                # request values in their exception text. Do not surface it.
                raise DataError("OpenRouter request validation failed; check OPEN_ROUTER_API and retry") from None


def fetch_catalog(client):
    url = API + "/models?output_modalities=all"
    seen_urls, rows, ids = set(), [], set()
    while url:
        if url in seen_urls:
            raise DataError("Catalog pagination repeated a page")
        seen_urls.add(url)
        payload = client.get(url)
        if not isinstance(payload.get("data"), list):
            raise DataError("Catalog response has no model list")
        for model in payload["data"]:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                raise DataError("Catalog contains an invalid model")
            if model["id"] in ids:
                raise DataError("Catalog pagination repeated a model")
            ids.add(model["id"])
            rows.append(model)
        next_url = (payload.get("links") or {}).get("next")
        url = urllib.parse.urljoin(API + "/models", next_url) if next_url else None
    if not rows:
        raise DataError("Refusing to replace the catalog with an empty response")
    return {"data": rows, "meta": {"as_of": utc_now(), "source_url": API + "/models?output_modalities=all", "pages": sorted(seen_urls)}}


def validate_rankings(payload, start, end):
    if not isinstance(payload.get("data"), list) or not isinstance(payload.get("meta"), dict):
        raise DataError("Rankings response is missing data or meta")
    meta = payload["meta"]
    if meta.get("start_date") != str(start) or meta.get("end_date") != str(end) or not meta.get("as_of"):
        raise DataError("Rankings response does not match the requested date range or lacks its attribution timestamp")
    seen, per_day = set(), {}
    for row in payload["data"]:
        try:
            date = dt.date.fromisoformat(row["date"])
            slug = row["model_permaslug"]
            value = row["total_tokens"]
            if not isinstance(value, (str, int)) or isinstance(value, bool) or str(int(value)) != str(value):
                raise ValueError
            tokens = int(value)
            if not isinstance(slug, str) or not slug or tokens < 0 or not start <= date <= end:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise DataError("Rankings response has an invalid date, model or token count") from None
        pair = (str(date), slug)
        if pair in seen:
            raise DataError("Rankings response contains a duplicate date/model row")
        seen.add(pair)
        per_day[str(date)] = per_day.get(str(date), 0) + 1
    if any(n > 51 for n in per_day.values()):
        raise DataError("Daily rankings exceeded the documented top-50 plus other schema")
    # Entire missing dates remain missing and are reported as coverage gaps.
    return payload


def merge_snapshot(old, new, start, end, bucket, url):
    """Replace refreshed dates while rejecting detectable partial responses.

    Fresh days can legitimately have fewer than 50 named models and no tail,
    so their completeness cannot always be established from the schema alone.
    Existing observations provide additional guards against silent data loss.
    """
    old = old or {"data": [], "meta": {}}
    old_days, new_days = {}, {}
    for row in old["data"]:
        if str(start) <= row["date"] <= str(end):
            old_days.setdefault(row["date"], set()).add(row["model_permaslug"])
    for row in new["data"]:
        new_days.setdefault(row["date"], set()).add(row["model_permaslug"])
    if set(old_days) - set(new_days):
        raise DataError("Refresh omitted previously observed dates; saved rankings were preserved")
    for day, previous in old_days.items():
        current = new_days[day]
        if "other" in previous and "other" not in current:
            raise DataError("Refresh omitted a previously reported daily tail; saved rankings were preserved")
        if len(previous - {"other"}) >= 50 and len(current - {"other"}) < 50:
            raise DataError("Refresh shortened a previously complete top 50; saved rankings were preserved")
    rows = [r for r in old["data"] if not str(start) <= r["date"] <= str(end)] + new["data"]
    rows.sort(key=lambda r: (r["date"], r["model_permaslug"] == "other", -int(r["total_tokens"]), r["model_permaslug"]))
    snap = dict(new["meta"], source_url=url)
    snapshots = list(old["meta"].get("snapshots", []))
    snapshots.append(snap)
    old_start = old["meta"].get("start_date", str(start))
    old_end = old["meta"].get("end_date", str(end))
    return {"data": rows, "meta": {"version": new["meta"].get("version"), "as_of": new["meta"]["as_of"], "start_date": min(old_start, str(start)), "end_date": max(old_end, str(end)), "filters": {"period": "day", "modality": "text", **({"context_bucket": bucket} if bucket != "text" else {})}, "snapshots": snapshots}}


def update_rankings(client, data_dir, bucket, start, end, full=False):
    if bucket not in BUCKETS:
        raise DataError("Unsupported context bucket")
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31))
        path = Path(data_dir) / "rankings" / bucket / f"{year}.json.gz"
        old = read_json(path) if path.exists() else None
        queries = []
        if full or old is None:
            queries.append((lo, hi))
        else:
            first = dt.date.fromisoformat(old["meta"]["start_date"])
            last = dt.date.fromisoformat(old["meta"]["end_date"])
            if lo < first:
                queries.append((lo, min(hi, first - dt.timedelta(days=1))))
            # Refresh a seven-day overlap, including last year's tail at rollover.
            refresh = max(lo, last - dt.timedelta(days=6))
            if last < hi or hi >= end - dt.timedelta(days=6):
                queries.append((refresh, hi))
        for lower, upper in queries:
            if lower > upper:
                continue
            params = {"start_date": str(lower), "end_date": str(upper), "period": "day", "modality": "text"}
            if bucket != "text":
                params["context_bucket"] = bucket
            url = API + "/datasets/rankings-daily?" + urllib.parse.urlencode(params)
            payload = validate_rankings(client.get(url, authenticated=True), lower, upper)
            old = merge_snapshot(old, payload, lower, upper, bucket, url)
            write_json(path, old)


def load_rankings(data_dir, bucket, start, end):
    rows, snapshots = [], []
    for path in sorted((Path(data_dir) / "rankings" / bucket).glob("*.json.gz")):
        payload = read_json(path)
        rows.extend(r for r in payload["data"] if str(start) <= r["date"] <= str(end))
        snapshots.append({"path": str(path), "meta": payload["meta"]})
    if not snapshots:
        raise DataError(f"No saved {bucket} rankings; fetch online before an offline rebuild")
    return rows, snapshots
