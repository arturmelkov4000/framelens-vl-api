# FrameLens VL API

FrameLens VL API is a small FastAPI service for turning video-language model output into structured JSON that other tools can consume.

The first supported model family is Qwen3-VL. The API can run against a local Transformers model, an OpenAI-compatible vLLM/SGLang/DashScope-style server, or an optional Qwen web-token proxy backend.

## Why This Exists

Most video-language model demos are notebooks or one-off scripts. FrameLens VL API focuses on a stable HTTP interface for automation:

- local video analysis without uploading source footage
- structured JSON output for editing, search, QA, routing, and retrieval workflows
- quick single-pass analysis or fuller multi-pass analysis
- explicit camera/editing prompts for cuts, reframing, handheld motion, zooms, pans, and timelines

## Install

Use Python 3.11 or 3.12 for the native ML backend.

```bash
git clone https://github.com/arturmelkov4000/framelens-vl-api.git
cd framelens-vl-api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a Mac with limited RAM, start with `Qwen/Qwen3-VL-4B-Instruct`. For better quality use `Qwen/Qwen3-VL-8B-Instruct` or a remote vLLM/SGLang server.

## Run

### Local Transformers Backend

```bash
source .venv/bin/activate
export QWEN_BACKEND=transformers
export QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
uvicorn framelens_vl_api.app:app --host 127.0.0.1 --port 8765
```

Analyze a local video:

```bash
curl -X POST http://127.0.0.1:8765/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/path/to/video.mov",
    "mode": "full",
    "fps": 2.0,
    "prompt": "Pay special attention to camera dynamics and jump cuts."
  }'
```

Upload a video:

```bash
curl -X POST http://127.0.0.1:8765/analyze/upload \
  -F "file=@/path/to/video.mov" \
  -F "mode=full" \
  -F "fps=2.0"
```

### OpenAI-Compatible Backend

Run Qwen3-VL in vLLM, SGLang, DashScope, or another compatible server, then point FrameLens at it:

```bash
export QWEN_BACKEND=openai
export QWEN_OPENAI_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
uvicorn framelens_vl_api.app:app --host 127.0.0.1 --port 8765
```

For this backend, prefer `video_url` with an HTTP(S) URL:

```bash
curl -X POST http://127.0.0.1:8765/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "video_url": "http://127.0.0.1:8765/media/example.mov",
    "mode": "full",
    "backend": "openai"
  }'
```

### Qwen Web-Token Proxy Backend

This backend is optional and intended for local experiments. It accepts a Qwen web-session JWT token, not an official `sk-...` API key. Do not commit tokens or credential files.

```bash
source .venv/bin/activate
export QWEN_BACKEND=qwen_free
export QWEN_FREE_API_MODEL=qwen3-vl-plus
export QWEN_FREE_API_TOKEN='paste-token-here'
uvicorn framelens_vl_api.app:app --host 127.0.0.1 --port 8765
```

You can also keep the token in a local untracked file:

```bash
export QWEN_FREE_API_TOKEN_FILE=/path/to/QWEN_TOKEN.txt
```

## API

`POST /analyze`

```json
{
  "video_path": "/path/to/video.mov",
  "video_url": null,
  "mode": "full",
  "backend": "transformers",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "fps": 2.0,
  "max_pixels": 151200,
  "total_pixels": 20971520,
  "max_new_tokens": 768,
  "temperature": 0,
  "prompt": "Optional extra instruction"
}
```

Modes:

- `quick`: one broad prompt, cheaper and faster.
- `full`: five focused passes: overview, timeline, camera/editing, visual details, dense tags.

Response shape:

```json
{
  "backend": "transformers",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "mode": "full",
  "video_uri": "file:///path/to/video.mov",
  "analysis": {
    "overview": {},
    "timeline": {},
    "camera_editing": {},
    "visual_details": {},
    "dense_tags": {}
  },
  "raw": {},
  "warnings": []
}
```

## Configuration

Useful environment variables:

- `QWEN_BACKEND`: `transformers`, `openai`, or `qwen_free`
- `QWEN_MODEL`: model id for the Transformers or OpenAI-compatible backend
- `QWEN_OPENAI_BASE_URL`: base URL for vLLM/SGLang/DashScope-style servers
- `QWEN_OPENAI_API_KEY`: API key for OpenAI-compatible servers, defaults to `EMPTY`
- `QWEN_FREE_API_TOKEN` or `QWEN_FREE_API_TOKEN_FILE`: token for the optional proxy backend
- `FRAMELENS_UPLOAD_DIR`: upload directory for `/analyze/upload`, defaults to `data/uploads`

## Development

```bash
python -m unittest discover -s tests
```

## Security

Never commit tokens, local video files, generated analysis reports, or upload directories. See [SECURITY.md](SECURITY.md).

## License

MIT
