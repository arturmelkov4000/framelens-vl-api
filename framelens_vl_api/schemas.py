"""Pydantic schemas for the video analysis API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AnalysisMode = Literal["quick", "full"]
BackendName = Literal["transformers", "openai", "qwen_free"]


class AnalyzeRequest(BaseModel):
    video_path: str | None = Field(
        default=None,
        description="Local path visible to this service, e.g. /Users/me/video.mov.",
    )
    video_url: str | None = Field(
        default=None,
        description="HTTP(S) or file:// URL to a video.",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional extra task or business context to add to every analysis prompt.",
    )
    mode: AnalysisMode = "full"
    backend: BackendName | None = Field(
        default=None,
        description="Override QWEN_BACKEND for this request.",
    )
    model: str | None = Field(
        default=None,
        description="Override QWEN_MODEL for this request.",
    )
    fps: float = Field(default=1.0, gt=0, le=30)
    max_pixels: int | None = Field(default=360 * 420, gt=0)
    total_pixels: int | None = Field(default=20480 * 32 * 32, gt=0)
    max_new_tokens: int = Field(default=768, gt=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    return_raw: bool = True

    @model_validator(mode="after")
    def require_video_source(self) -> "AnalyzeRequest":
        if not self.video_path and not self.video_url:
            raise ValueError("Provide either video_path or video_url.")
        if self.video_path and self.video_url:
            raise ValueError("Provide only one of video_path or video_url.")
        return self


class AnalysisResponse(BaseModel):
    backend: str
    model: str
    mode: AnalysisMode
    video_uri: str
    analysis: dict[str, Any]
    raw: dict[str, str] | None = None
    warnings: list[str] = Field(default_factory=list)
