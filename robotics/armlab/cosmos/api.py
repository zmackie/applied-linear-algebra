"""Egocentric reasoning API: POST a video, get back what is happening and what to do next.

  ARMLAB_REASON_VLM=cosmos uvicorn armlab.cosmos.api:app --port 8080

  curl -F video=@clip.mp4 -F task="put the cup in the sink" localhost:8080/v1/analyze
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .reason import VideoReasoner, make_reasoner, video_bytes_to_tmp

EGO_QUESTION = """This is an egocentric video (first-person, from a robot's or person's camera).
{task_line}Describe what is happening, the current state of the relevant objects, and the most sensible next action.
Return JSON with exactly these keys:
{{"what_is_happening": "<one or two sentences>",
  "objects": ["<relevant object and its state>"],
  "next_action": "<the single next action, imperative, concrete>",
  "next_action_steps": ["<short sub-step>"],
  "hazards": ["<anything unsafe, or empty>"],
  "task_progress": "<not started|in progress|done|unknown>"}}"""


class Analysis(BaseModel):
    what_is_happening: str
    objects: list[str] = []
    next_action: str
    next_action_steps: list[str] = []
    hazards: list[str] = []
    task_progress: str = "unknown"
    reasoning: str = ""
    model: str = ""
    latency_s: float = 0.0
    raw_answer: str = ""


@lru_cache(maxsize=1)
def get_reasoner() -> VideoReasoner:
    return make_reasoner(os.environ.get("ARMLAB_REASON_VLM", "cosmos"),
                         mode=os.environ.get("ARMLAB_REASON_MODE") or None)


app = FastAPI(title="armlab egocentric reasoning API", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "model": os.environ.get("ARMLAB_REASON_VLM", "cosmos")}


@app.post("/v1/analyze", response_model=Analysis)
async def analyze(video: UploadFile = File(...), task: str = Form("")):
    data = await video.read()
    if not data:
        raise HTTPException(400, "empty video")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "video larger than 50 MB")
    path = video_bytes_to_tmp(data, Path(video.filename or "clip.mp4").suffix or ".mp4")
    try:
        reasoner = get_reasoner()
        t0 = time.time()
        task_line = f"The goal is: {task.strip()}\n" if task.strip() else ""
        out = reasoner.ask(path, EGO_QUESTION.format(task_line=task_line))
        j = out["json"] or {}
        return Analysis(
            what_is_happening=str(j.get("what_is_happening") or out["answer"][:500]),
            objects=[str(x) for x in j.get("objects") or []],
            next_action=str(j.get("next_action") or ""),
            next_action_steps=[str(x) for x in j.get("next_action_steps") or []],
            hazards=[str(x) for x in j.get("hazards") or []],
            task_progress=str(j.get("task_progress") or "unknown"),
            reasoning=out["reasoning"],
            model=getattr(reasoner.vlm, "name", ""),
            latency_s=round(time.time() - t0, 2),
            raw_answer=out["answer"] if not j else "",
        )
    finally:
        path.unlink(missing_ok=True)
