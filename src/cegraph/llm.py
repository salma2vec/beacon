"""LLM adapters for benchmark repair requests."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from openai import OpenAI

from .models import RepairPrompt, RepairResponse


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class RepairClient(Protocol):
    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse: ...


class OllamaResponseError(ValueError):
    """A rejected completion retained for failure-inclusive experiment records."""

    def __init__(self, message: str, response: dict[str, Any]) -> None:
        super().__init__(message)
        self.response = response


class EchoRepairClient:
    """A deterministic fallback client for dry runs and tests."""

    def __init__(self, model: str = "echo") -> None:
        self.model = model

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        message = "# dry-run\n" + prompt.user
        return RepairResponse(patch_text="", raw_response=message, model=self.model)


class OpenAIRepairClient:
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        text = getattr(response, "output_text", "")
        return RepairResponse(patch_text=text, raw_response=text, model=self.model)


class OllamaRepairClient:
    """One-shot requests to an already installed model on the local Ollama server.

    This adapter never downloads a model or falls back to a paid provider. The
    endpoint is fixed to loopback, and each request checks the reported digest.
    The operator must trust that listener: HTTP does not authenticate it, and
    a self-reported digest is continuity metadata, not model attestation.
    Local model output still needs the same untrusted-code evaluation controls.
    """

    endpoint = "http://127.0.0.1:11434"

    def __init__(
        self,
        model: str,
        seed: int = 42,
        timeout: float = 180,
        prompt_token_counter: Callable[[RepairPrompt], int] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        self.model, self.timeout = model, timeout
        self.prompt_token_counter = prompt_token_counter
        self.response_schema = (
            json.loads(json.dumps(response_schema)) if response_schema is not None else None
        )
        self.options = {"seed": seed, "temperature": 0, "num_ctx": 4096, "num_predict": 512}
        self.digest = self._model_digest()
        details = self._request("/api/show", {"model": model})
        if details.get("remote_host") or details.get("remote_model") or "cloud" in model:
            raise ValueError("This adapter requires a local model, not an Ollama cloud model")
        self.runtime_version = self._request("/api/version").get("version")

    def _request(self, path: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.endpoint + path, data=data, headers={"Content-Type": "application/json"}
        )
        # Disable configured proxies: this client's requests stay on loopback.
        # This cannot prevent an untrusted listener from forwarding the prompt.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=self.timeout) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Ollama response exceeds 1 MiB")
        result = json.loads(raw)
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError("Invalid or failed Ollama response")
        return result

    def _model_digest(self) -> str:
        models = self._request("/api/tags").get("models", [])
        for model in models:
            if isinstance(model, dict) and model.get("name") == self.model:
                digest = model.get("digest")
                if isinstance(digest, str) and len(digest) == 64:
                    return digest
        raise ValueError("Requested model must already be installed in local Ollama")

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        if self._model_digest() != self.digest:
            raise ValueError("Installed model digest changed during the experiment")
        expected_count = self.prompt_token_counter(prompt) if self.prompt_token_counter else None
        capacity = self.options["num_ctx"] - self.options["num_predict"]
        if expected_count is not None and expected_count >= capacity:
            raise ValueError("Full native-token prompt exceeds reserved context capacity")
        result = self._request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                "stream": False,
                "options": self.options,
                "keep_alive": "5m",
                **({"format": self.response_schema} if self.response_schema is not None else {}),
            },
        )
        message = result.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if result.get("done") is not True or not isinstance(content, str):
            raise OllamaResponseError("Ollama did not return a complete chat response", result)
        native_count = result.get("prompt_eval_count")
        if type(native_count) is not int or not 0 < native_count < capacity:
            raise OllamaResponseError(
                "Native prompt count is missing or risks context truncation", result
            )
        if expected_count is not None and native_count != expected_count:
            raise OllamaResponseError(
                "Native prompt count differs from full-prompt tokenizer count", result
            )
        complete = result.get("done_reason") == "stop"
        # Strip only a surrounding code fence; never repair or retry a failed diff.
        patch = content.strip("\r\n")
        if self.response_schema is None and patch.startswith("```") and patch.endswith("```"):
            patch = "\n".join(patch.splitlines()[1:-1])
        return RepairResponse(
            patch if complete else "",
            content,
            self.model,
            {
                "provider": "ollama_loopback",
                "server_authenticated": False,
                "model_identity_source": "local_service_self_report",
                "output_format": "json_schema" if self.response_schema is not None else "diff",
                "model_digest": self.digest,
                "runtime_version": self.runtime_version,
                "options": self.options,
                "expected_prompt_tokens": expected_count,
                "native_prompt_verified": expected_count is not None,
                "reserved_prompt_capacity": capacity,
                **{
                    key: result.get(key)
                    for key in (
                        "done_reason",
                        "prompt_eval_count",
                        "eval_count",
                        "total_duration",
                        "prompt_eval_duration",
                        "eval_duration",
                        "load_duration",
                    )
                },
            },
        )
