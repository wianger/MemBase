from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from ..settings import SimpleMemSettings


class LLMClient:
    """Unified LLM client used by the vendored SimpleMem backend."""

    def __init__(
        self,
        settings: SimpleMemSettings,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        enable_thinking: bool | None = None,
        use_streaming: bool | None = None,
    ) -> None:
        self.settings = settings
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model or settings.llm_model
        self.base_url = base_url if base_url is not None else settings.openai_base_url
        self.enable_thinking = (
            settings.enable_thinking if enable_thinking is None else enable_thinking
        )
        self.use_streaming = (
            settings.use_streaming if use_streaming is None else use_streaming
        )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        is_qwen_api = bool(
            self.base_url and "dashscope.aliyuncs.com" in self.base_url
        )
        if is_qwen_api:
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                kwargs["extra_body"] = {"enable_thinking": False}

        last_exception: Exception | None = None
        for attempt in range(max_retries):
            try:
                if self.use_streaming:
                    kwargs["stream"] = True
                    return self._handle_streaming_response(**kwargs)
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    import time

                    time.sleep(2 ** attempt)
                else:
                    raise
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("LLM request failed without raising a concrete exception.")

    def _handle_streaming_response(self, **kwargs: Any) -> str:
        full_content: list[str] = []
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            content = chunk.choices[0].delta.content
            if content is not None:
                full_content.append(content)
        return "".join(full_content)

    def extract_json(self, text: str) -> Any:
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text.lower():
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        json_str = self._clean_json_string(json_str)
                        return json.loads(json_str)

        if "```" in text:
            start = text.find("```") + 3
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = self._clean_json_string(json_str)
                    return json.loads(json_str)

        for start_char in ["{", "["]:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        raise ValueError(
            f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}..."
        )

    def _clean_json_string(self, json_str: str) -> str:
        import re

        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        end_char = "}" if start_char == "{" else "]"
        start_idx = text.find(start_char)
        if start_idx == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start_idx : idx + 1])
        return None

