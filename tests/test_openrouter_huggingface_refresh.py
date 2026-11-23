# SPDX-License-Identifier: MIT
"""Public metadata collection must preserve missing counters and source identity."""
import hashlib
import io
import json
import unittest
from unittest.mock import patch
import urllib.error

from scripts.openrouter.client import DataError
from scripts.openrouter.huggingface_refresh import fetch_model, fetch_snapshot


class HuggingFaceRefreshTests(unittest.TestCase):
    def test_zero_counter_rename_and_malformed_optional_metadata_preserved(self):
        raw = json.dumps({"id": "new-owner/model", "downloads": 0,
                          "downloadsAllTime": 200, "safetensors": "unavailable"}).encode()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(raw)) as opened:
            result = fetch_model("old-owner/model")
        self.assertEqual(result["requested_id"], "old-owner/model")
        self.assertEqual(result["repo_id"], "new-owner/model")
        self.assertEqual(result["downloads"], 0)
        self.assertEqual(result["downloads_all_time"], 200)
        self.assertIsNone(result["parameters_total"])
        self.assertEqual(result["response_sha256"], hashlib.sha256(raw).hexdigest())
        request = opened.call_args.args[0]
        self.assertIn("downloadsAllTime", request.full_url)
        self.assertIsNone(request.get_header("Authorization"))

    def test_inaccessible_repository_has_no_manufactured_zero_counter(self):
        error = urllib.error.HTTPError("https://huggingface.co/api/models/org/model", 401, "denied", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            result = fetch_model("org/model")
        self.assertEqual(result["http_status"], 401)
        self.assertNotIn("downloads", result)

    def test_invalid_repository_never_becomes_an_arbitrary_request(self):
        with patch("urllib.request.urlopen") as opened:
            for repo in ("https://example.com/private", "../private", "org/model?token=secret"):
                with self.assertRaises(ValueError):
                    fetch_model(repo)
            opened.assert_not_called()

    def test_total_outage_cannot_replace_a_snapshot_but_partial_errors_remain(self):
        def failed(repo):
            return {"repo_id": repo, "error": "HTTP 401"}
        with patch("scripts.openrouter.huggingface_refresh.fetch_model", side_effect=failed):
            with self.assertRaises(DataError):
                fetch_snapshot(["org/a", "org/b"])
        def partial(repo):
            return {"repo_id": repo, "downloads": 0} if repo == "org/a" else failed(repo)
        with patch("scripts.openrouter.huggingface_refresh.fetch_model", side_effect=partial):
            result = fetch_snapshot(["org/b", "org/a", "org/a"])
        self.assertEqual(len(result["data"]), 2)
        self.assertEqual(result["data"][0]["downloads"], 0)
        self.assertEqual(result["data"][1]["error"], "HTTP 401")


if __name__ == "__main__":
    unittest.main()
