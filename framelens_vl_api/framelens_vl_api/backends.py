"""Inference backends for Qwen3-VL video analysis."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Protocol


@dataclass(frozen=True)
class GenerationOptions:
    fps: float = 1.0
    max_pixels: int | None = 360 * 420
    total_pixels: int | None = 20480 * 32 * 32
    max_new_tokens: int = 768
    temperature: float = 0.0


class VideoBackend(Protocol):
    name: str
    model_id: str

    def generate(self, video_uri: str, prompt: str, options: GenerationOptions) -> str:
        """Generate text for one video prompt."""


class TransformersQwen3VLBackend:
    """Official Transformers + qwen-vl-utils backend.

    This is the backend that passes Qwen3-VL a real `type: video` message and lets
    qwen-vl-utils sample the video with temporal/fps metadata.
    """

    name = "transformers"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = os.getenv("QWEN_TORCH_DTYPE", "auto")
        device_map = os.getenv("QWEN_DEVICE_MAP", "auto")

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=device_map,
            )
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device_map,
            )

        if device_map == "none":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = model
        self._processor = processor
        return model, processor

    def generate(self, video_uri: str, prompt: str, options: GenerationOptions) -> str:
        from qwen_vl_utils import process_vision_info

        model, processor = self._load()

        video_block: dict[str, object] = {
            "type": "video",
            "video": video_uri,
            "fps": options.fps,
        }
        if options.max_pixels is not None:
            video_block["max_pixels"] = options.max_pixels
        if options.total_pixels is not None:
            video_block["total_pixels"] = options.total_pixels

        messages = [
            {
                "role": "user",
                "content": [
                    video_block,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        vision_kwargs = {}
        try:
            image_inputs, video_inputs, vision_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=16,
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)

        processor_kwargs = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        processor_kwargs.update(vision_kwargs or {})

        try:
            inputs = processor(**processor_kwargs)
        except TypeError:
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                do_resize=False,
                return_tensors="pt",
            )

        inputs = inputs.to(model.device)
        generation_kwargs = {
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.temperature > 0,
        }
        if options.temperature > 0:
            generation_kwargs["temperature"] = options.temperature

        generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        decoded = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


class OpenAIVideoBackend:
    """OpenAI-compatible backend for vLLM/SGLang Qwen3-VL servers."""

    name = "openai"

    def __init__(self, model_id: str, base_url: str, api_key: str = "EMPTY"):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self._client = None

    def _load_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=float(os.getenv("QWEN_OPENAI_TIMEOUT", "3600")),
            )
        return self._client

    def generate(self, video_uri: str, prompt: str, options: GenerationOptions) -> str:
        client = self._load_client()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {"url": video_uri},
                            "uuid": video_uri,
                        },
                    ],
                }
            ],
            max_tokens=options.max_new_tokens,
            temperature=options.temperature,
        )
        return response.choices[0].message.content or ""


class QwenFreeAPIBackend:
    """Qwen web-token proxy backend from Staks-sor/qwen_free_api.

    This uses the same OpenAI-compatible proxy endpoint as the cloned repo:
    https://qwen.aikit.club/v1/chat/completions

    It is intentionally configured through environment variables so session
    tokens are not stored in repository files.
    """

    name = "qwen_free"

    def __init__(self, model_id: str, base_url: str, api_token: str):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Origin": "https://chat.qwen.ai",
            "Referer": "https://chat.qwen.ai/",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
        }

    def _to_video_url(self, video_uri: str) -> str:
        if video_uri.startswith("data:") or video_uri.startswith("http://") or video_uri.startswith("https://"):
            return video_uri

        parsed = urlparse(video_uri)
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path))
        else:
            path = Path(video_uri).expanduser()

        path = path.resolve()
        mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def generate(self, video_uri: str, prompt: str, options: GenerationOptions) -> str:
        import requests

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": self._to_video_url(video_uri)},
                            "fps": options.fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": options.temperature,
            "max_tokens": options.max_new_tokens,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=float(os.getenv("QWEN_FREE_API_TIMEOUT", "180")),
        )
        if response.status_code != 200:
            raise RuntimeError(f"Qwen free API error {response.status_code}: {response.text}")

        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content") or ""
