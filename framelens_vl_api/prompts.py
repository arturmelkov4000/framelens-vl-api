"""Prompt presets for structured Qwen3-VL video analysis."""

BASE_JSON_RULES = """
Return valid JSON only. Do not wrap it in markdown.
If evidence is weak, use null and explain uncertainty in an "uncertainty" field.
Separate observed facts from inference. Do not invent audio content unless a transcript is provided.
"""

QUICK_ANALYSIS_PROMPT = f"""
Analyze this video as thoroughly as possible from the visual stream.

Include:
- concise summary
- scene and setting
- subjects and actions
- timeline with approximate moments
- camera/editing dynamics: cuts, jump cuts, pan, tilt, zoom/dolly, handheld shake, reframing
- visible text/OCR
- notable objects, brands, tools, UI, or environment details
- uncertainties and limitations

{BASE_JSON_RULES}

Schema:
{{
  "summary": "...",
  "scene": "...",
  "subjects": [],
  "timeline": [
    {{
      "start_s": 0.0,
      "end_s": 1.0,
      "action": "...",
      "camera_or_editing": "...",
      "confidence": 0.0
    }}
  ],
  "camera_dynamics": {{
    "overall": "...",
    "cuts": [],
    "continuous_camera_motion": [],
    "reframing": [],
    "confidence": 0.0
  }},
  "visible_text": [],
  "objects": [],
  "uncertainty": []
}}
"""

FULL_ANALYSIS_TASKS = {
    "overview": f"""
Analyze the video visually and produce a factual overview.

Focus on:
- setting and scene type
- people/subjects
- main visible activity
- important objects/tools/products/text
- likely purpose of the clip

{BASE_JSON_RULES}

Schema:
{{
  "summary": "...",
  "setting": "...",
  "subjects": [
    {{
      "label": "...",
      "appearance": "...",
      "role_or_inference": "...",
      "confidence": 0.0
    }}
  ],
  "main_actions": [],
  "notable_objects": [],
  "visible_text": [],
  "purpose_inference": {{
    "value": "...",
    "confidence": 0.0
  }},
  "uncertainty": []
}}
""",
    "timeline": f"""
Create a time-based timeline of visible events in the video.

Use approximate timestamps. Prefer more short segments over one vague segment.
Describe what visibly changes over time.

{BASE_JSON_RULES}

Schema:
{{
  "timeline": [
    {{
      "start_s": 0.0,
      "end_s": 0.0,
      "shot_description": "...",
      "visible_action": "...",
      "subject_motion": "...",
      "important_objects": [],
      "confidence": 0.0
    }}
  ],
  "key_moments": [
    {{
      "time_s": 0.0,
      "event": "...",
      "why_it_matters": "...",
      "confidence": 0.0
    }}
  ],
  "uncertainty": []
}}
""",
    "camera_editing": f"""
Analyze ONLY camera, framing, and editing dynamics.

Separate camera movement from subject/object movement.
Identify:
- hard cuts and jump cuts
- shot size changes
- reframing
- pan, tilt, zoom, dolly/truck if visible
- handheld shake or stabilization
- whether motion can only be inferred from sampled video frames

{BASE_JSON_RULES}

Schema:
{{
  "overall_camera_style": "...",
  "editing_pattern": "...",
  "cuts": [
    {{
      "approx_time_s": 0.0,
      "from": "...",
      "to": "...",
      "type": "hard_cut|jump_cut|match_cut|uncertain",
      "confidence": 0.0
    }}
  ],
  "continuous_camera_motion": [
    {{
      "start_s": 0.0,
      "end_s": 0.0,
      "motion_type": "pan|tilt|zoom|dolly|truck|handheld_shake|static|uncertain",
      "description": "...",
      "confidence": 0.0
    }}
  ],
  "reframing_and_shot_sizes": [],
  "limitations": []
}}
""",
    "visual_details": f"""
Extract detailed visual information from the video.

Focus on details that are useful for downstream programs:
- objects and tools
- readable text/OCR
- brands/logos
- clothing, environment, materials
- composition, focus, blur, lighting, occlusion
- safety-relevant or workflow-relevant observations if present

{BASE_JSON_RULES}

Schema:
{{
  "objects": [
    {{
      "name": "...",
      "description": "...",
      "approx_times_s": [],
      "confidence": 0.0
    }}
  ],
  "readable_text": [
    {{
      "text": "...",
      "where": "...",
      "approx_time_s": 0.0,
      "confidence": 0.0
    }}
  ],
  "environment_details": [],
  "technical_quality": {{
    "focus": "...",
    "motion_blur": "...",
    "lighting": "...",
    "occlusions": "..."
  }},
  "uncertainty": []
}}
""",
    "dense_tags": f"""
Produce compact machine-friendly tags for the video.

Create tags that another program can use for search, routing, editing, or retrieval.

{BASE_JSON_RULES}

Schema:
{{
  "content_tags": [],
  "action_tags": [],
  "object_tags": [],
  "scene_tags": [],
  "camera_tags": [],
  "editing_tags": [],
  "risk_or_quality_tags": [],
  "search_queries": [],
  "one_line_caption": "..."
}}
""",
}
