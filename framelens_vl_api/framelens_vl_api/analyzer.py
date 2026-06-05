"""High-level orchestration for multi-pass video analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backends import (
    GenerationOptions,
    OpenAIVideoBackend,
    QwenFreeAPIBackend,
    TransformersQwen3VLBackend,
    VideoBackend,
)
from .prompts import FULL_ANALYSIS_TASKS, QUICK_ANALYSIS_PROMPT
from .schemas import AnalyzeRequest, AnalysisResponse
from .utils import as_file_uri, extract_json_object, merge_prompt


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def read_token_file(path: str | os.PathLike[str]) -> str | None:
    """Read a token from a plain text file.

    The file may contain either the token itself or a shell-style assignment:
    QWEN_FREE_API_TOKEN=...
    Blank lines and lines starting with # are ignored.
    """
    return read_first_config_value(path, assignment_name="QWEN_FREE_API_TOKEN")


def read_first_config_value(
    path: str | os.PathLike[str],
    assignment_name: str | None = None,
) -> str | None:
    """Read the first non-comment value from a plain text config file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return None

    for line in config_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if assignment_name and value.startswith(f"{assignment_name}="):
            value = value.split("=", 1)[1].strip()
        return value.strip("'\"")
    return None


def create_backend(name: str | None, model_id: str | None) -> VideoBackend:
    backend_name = name or os.getenv("QWEN_BACKEND", "transformers")
    model = model_id or os.getenv("QWEN_MODEL", DEFAULT_MODEL)

    if backend_name == "transformers":
        return TransformersQwen3VLBackend(model)
    if backend_name == "openai":
        base_url = os.getenv("QWEN_OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("QWEN_OPENAI_API_KEY", "EMPTY")
        return OpenAIVideoBackend(model, base_url=base_url, api_key=api_key)
    if backend_name == "qwen_free":
        token = os.getenv("QWEN_FREE_API_TOKEN")
        token_file = os.getenv("QWEN_FREE_API_TOKEN_FILE")
        if not token and token_file:
            token = read_token_file(token_file)
        if not token:
            raise ValueError(
                "QWEN_FREE_API_TOKEN or QWEN_FREE_API_TOKEN_FILE is required "
                "for backend='qwen_free'."
            )
        model_file = os.getenv("QWEN_FREE_API_MODEL_FILE")
        free_model = model_id or os.getenv("QWEN_FREE_API_MODEL")
        if not free_model and model_file:
            free_model = read_first_config_value(
                model_file,
                assignment_name="QWEN_FREE_API_MODEL",
            )
        free_model = free_model or "qwen3-vl-plus"
        base_url = os.getenv("QWEN_FREE_API_BASE_URL", "https://qwen.aikit.club/v1")
        return QwenFreeAPIBackend(free_model, base_url=base_url, api_token=token)

    raise ValueError(f"Unsupported backend: {backend_name}")


def video_uri_from_request(request: AnalyzeRequest) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if request.video_url:
        return request.video_url, warnings

    assert request.video_path is not None
    path = Path(request.video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video path does not exist: {path}")

    uri = as_file_uri(path)
    if request.backend == "openai" or (
        request.backend is None and os.getenv("QWEN_BACKEND") == "openai"
    ):
        warnings.append(
            "OpenAI-compatible video_url servers usually need an HTTP(S) URL. "
            "file:// works only if that server explicitly supports local file reads."
        )
    return uri, warnings


class VideoAnalyzer:
    def analyze(self, request: AnalyzeRequest) -> AnalysisResponse:
        backend = create_backend(request.backend, request.model)
        video_uri, warnings = video_uri_from_request(request)

        options = GenerationOptions(
            fps=request.fps,
            max_pixels=request.max_pixels,
            total_pixels=request.total_pixels,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        if request.mode == "quick":
            tasks = {"quick": QUICK_ANALYSIS_PROMPT}
        else:
            tasks = FULL_ANALYSIS_TASKS

        analysis: dict[str, Any] = {}
        raw: dict[str, str] = {}
        for task_name, prompt in tasks.items():
            task_prompt = merge_prompt(prompt, request.prompt)
            output = backend.generate(video_uri, task_prompt, options)
            raw[task_name] = output
            parsed = extract_json_object(output)
            analysis[task_name] = parsed if parsed is not None else {"raw_text": output}

        return AnalysisResponse(
            backend=backend.name,
            model=backend.model_id,
            mode=request.mode,
            video_uri=video_uri,
            analysis=analysis,
            raw=raw if request.return_raw else None,
            warnings=warnings,
        )
