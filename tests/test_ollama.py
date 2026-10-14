from __future__ import annotations

import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cegraph.llm import OllamaRepairClient
from cegraph.models import RepairPrompt


def fake_request(self, path, payload=None):
    if path == "/api/tags":
        return {"models": [{"name": "local:8b", "digest": "a" * 64}]}
    if path == "/api/show":
        return {"details": {"parameter_size": "8B"}}
    if path == "/api/version":
        return {"version": "test"}
    assert path == "/api/chat"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 512
    return {
        "done": True,
        "done_reason": "stop",
        "message": {"content": "```diff\n--- a/file.py\n+++ b/file.py\n```"},
        "prompt_eval_count": 11,
        "eval_count": 5,
    }


def test_local_model_keeps_digest_native_counts_and_raw_response(monkeypatch):
    monkeypatch.setattr(OllamaRepairClient, "_request", fake_request)
    result = OllamaRepairClient("local:8b").generate_patch(RepairPrompt("system", "query"))
    assert result.patch_text == "--- a/file.py\n+++ b/file.py"
    assert result.raw_response.startswith("```diff")
    assert result.metadata["model_digest"] == "a" * 64
    assert result.metadata["server_authenticated"] is False
    assert result.metadata["model_identity_source"] == "local_service_self_report"
    assert result.metadata["prompt_eval_count"] == 11
    assert result.metadata["eval_count"] == 5


def test_truncated_local_generation_cannot_be_scored_as_complete(monkeypatch):
    def truncated(self, path, payload=None):
        result = fake_request(self, path, payload)
        if path == "/api/chat":
            result["done_reason"] = "length"
        return result

    monkeypatch.setattr(OllamaRepairClient, "_request", truncated)
    result = OllamaRepairClient("local:8b").generate_patch(RepairPrompt("s", "u"))
    assert result.patch_text == ""
    assert result.raw_response
    assert result.metadata["done_reason"] == "length"


def test_changed_digest_stops_before_generation(monkeypatch):
    monkeypatch.setattr(OllamaRepairClient, "_request", fake_request)
    client = OllamaRepairClient("local:8b")
    monkeypatch.setattr(client, "_model_digest", lambda: "b" * 64)
    with pytest.raises(ValueError, match="digest changed"):
        client.generate_patch(RepairPrompt("s", "u"))


def test_cloud_route_rejected(monkeypatch):
    def cloud(self, path, payload=None):
        if path == "/api/show":
            return {"remote_host": "https://example.invalid"}
        return fake_request(self, path, payload)

    monkeypatch.setattr(OllamaRepairClient, "_request", cloud)
    with pytest.raises(ValueError, match="cloud"):
        OllamaRepairClient("local:8b")


def test_missing_model_never_downloads(monkeypatch):
    monkeypatch.setattr(
        OllamaRepairClient, "_request", lambda self, path, payload=None: {"models": []}
    )
    with pytest.raises(ValueError, match="already be installed"):
        OllamaRepairClient("missing")


def test_native_count_mismatch_and_capacity_fail_closed(monkeypatch):
    monkeypatch.setattr(OllamaRepairClient, "_request", fake_request)
    client = OllamaRepairClient("local:8b", prompt_token_counter=lambda prompt: 12)
    with pytest.raises(ValueError, match="differs"):
        client.generate_patch(RepairPrompt("s", "u"))
    client.prompt_token_counter = lambda prompt: 4096
    with pytest.raises(ValueError, match="capacity"):
        client.generate_patch(RepairPrompt("s", "u"))


def test_http_redirect_cannot_send_context_to_another_endpoint():
    visited = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            visited.append(self.path)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/escaped")
            self.end_headers()

        def do_GET(self):
            visited.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = object.__new__(OllamaRepairClient)
    client.endpoint = f"http://127.0.0.1:{server.server_port}"
    client.timeout = 2
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            client._request("/api/chat", {"messages": ["private fixture context"]})
        assert error.value.code == 302
        assert visited == ["/api/chat"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
