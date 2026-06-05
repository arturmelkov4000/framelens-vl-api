"""HTTP API entrypoint."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import __version__
from .analyzer import DEFAULT_MODEL, VideoAnalyzer, read_first_config_value
from .schemas import AnalyzeRequest, AnalysisResponse


UPLOAD_DIR = Path(
    os.getenv("FRAMELENS_UPLOAD_DIR")
    or os.getenv("QWEN_VIDEO_UPLOAD_DIR", "data/uploads")
).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="FrameLens VL API",
    version=__version__,
    description="Local API wrapper for structured video-language model analysis.",
)
app.mount("/media", StaticFiles(directory=str(UPLOAD_DIR)), name="media")

analyzer = VideoAnalyzer()


@app.get("/health")
def health() -> dict[str, str]:
    backend = os.getenv("QWEN_BACKEND", "transformers")
    model = os.getenv("QWEN_MODEL", DEFAULT_MODEL)
    if backend == "qwen_free":
        model = os.getenv("QWEN_FREE_API_MODEL")
        model_file = os.getenv("QWEN_FREE_API_MODEL_FILE")
        if not model and model_file:
            model = read_first_config_value(
                model_file,
                assignment_name="QWEN_FREE_API_MODEL",
            )
        model = model or "qwen3-vl-plus"
    return {
        "status": "ok",
        "version": __version__,
        "backend": backend,
        "model": model,
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    try:
        return analyzer.analyze(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze/upload", response_model=AnalysisResponse)
def analyze_upload(
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    mode: str = Form(default="full"),
    backend: str | None = Form(default=None),
    model: str | None = Form(default=None),
    fps: float = Form(default=1.0),
    max_pixels: int | None = Form(default=360 * 420),
    total_pixels: int | None = Form(default=20480 * 32 * 32),
    max_new_tokens: int = Form(default=768),
    temperature: float = Form(default=0.0),
    return_raw: bool = Form(default=True),
) -> AnalysisResponse:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    destination = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    request = AnalyzeRequest(
        video_path=str(destination),
        prompt=prompt,
        mode=mode,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        model=model,
        fps=fps,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_raw=return_raw,
    )
    try:
        return analyzer.analyze(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
