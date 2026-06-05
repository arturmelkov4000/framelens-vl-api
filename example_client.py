"""Example client for framelens_vl_api."""

from __future__ import annotations

import json
import sys

import requests


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python example_client.py /path/to/video.mov")

    video_path = sys.argv[1]
    payload = {
        "video_path": video_path,
        "mode": "full",
        "fps": 2.0,
        "prompt": "Pay special attention to jump cuts and camera dynamics.",
        "max_new_tokens": 768,
    }
    response = requests.post("http://localhost:8765/analyze", json=payload, timeout=3600)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
