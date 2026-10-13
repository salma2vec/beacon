"""LLM adapters for benchmark repair requests."""

from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI

from .models import RepairPrompt, RepairResponse


class RepairClient(Protocol):
    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse: ...


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
