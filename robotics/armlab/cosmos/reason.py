"""Ask a video-capable VLM (self-hosted Cosmos 3 Nano by default) questions about a clip.

Two input modes:
  native  - send the mp4 itself (`video_url`), the way Cosmos Reason is meant to be served (vLLM / NIM).
  frames  - sample N frames and send them as timestamped images (works with any VLM, incl. Claude).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ..policy.vlm import VLM, OpenAICompatVLM, VideoPart, extract_json

# From nvidia-cosmos/cosmos-reason2 (cosmos_reason2_utils/text.py); kept for Cosmos 3, which also emits <think> blocks.
COSMOS_SYSTEM_PROMPT = "You are a helpful assistant."
COSMOS_REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""


@dataclass
class VideoInfo:
    frames: list[np.ndarray]
    fps: float
    duration_s: float


def read_video(path: str | Path, max_frames: int | None = None) -> VideoInfo:
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))
    frames = [f for f in reader]
    reader.close()
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return VideoInfo(frames, fps, len(frames) / fps if fps else 0.0)


def sample_frames(path: str | Path, n: int = 8) -> list[tuple[float, np.ndarray]]:
    reader = imageio.get_reader(str(path))
    fps = float(reader.get_meta_data().get("fps", 25.0))
    frames = [f for f in reader]
    reader.close()
    if not frames:
        raise ValueError(f"no frames in {path}")
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [(i / fps, frames[i]) for i in idx]


def write_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16,
                            ffmpeg_log_level="error") as w:
        for f in frames:
            w.append_data(f)
    return path


def split_answer(text: str) -> tuple[str, str]:
    """Separate a Cosmos-style '<think>...</think> answer' reply into (reasoning, answer)."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text.strip()


class VideoReasoner:
    def __init__(self, vlm: VLM, mode: str | None = None, num_frames: int = 8, fps: float = 4.0,
                 reasoning: bool = True):
        self.vlm = vlm
        is_cosmos = isinstance(vlm, OpenAICompatVLM) and "cosmos" in vlm.model.lower()
        self.mode = mode or ("native" if is_cosmos else "frames")
        self.cosmos = is_cosmos
        self.num_frames = num_frames
        self.fps = fps
        self.reasoning = reasoning
        if self.mode == "native" and isinstance(vlm, OpenAICompatVLM):
            # vLLM honours per-request video sampling (server started with --media-io-kwargs num_frames=-1).
            vlm.extra_body = {**vlm.extra_body, "mm_processor_kwargs": {"fps": fps, "do_sample_frames": True}}
            vlm.json_mode = False

    def ask(self, video: str | Path, question: str, want_json: bool = True, max_tokens: int = 4096) -> dict:
        """Returns {"answer": str, "json": dict|None, "reasoning": str, "usage": Usage}."""
        prompt = question
        if want_json:
            prompt += "\n\nGive the final answer as a single JSON object."
        if self.cosmos and self.reasoning:
            prompt += "\n\n" + COSMOS_REASONING_PROMPT
        if self.mode == "native":
            parts = [VideoPart(Path(video).read_bytes()), prompt]
        else:
            parts = []
            for t, frame in sample_frames(video, self.num_frames):
                parts += [f"Frame at t={t:.1f} s:", frame]
            parts.append(prompt)
        system = COSMOS_SYSTEM_PROMPT if self.cosmos else "You analyze video clips carefully and answer precisely."
        text, usage = self.vlm.complete(system, parts, schema=None, max_tokens=max_tokens)
        reasoning, answer = split_answer(text)
        parsed = None
        if want_json:
            try:
                parsed = extract_json(answer)
            except Exception:
                parsed = None
        return {"answer": answer, "json": parsed, "reasoning": reasoning, "usage": usage}


def make_reasoner(spec: str, **kw) -> VideoReasoner:
    from ..policy.vlm import make_vlm

    vlm_kw = {}
    if spec.split(":")[0].split("@")[0] in ("cosmos", "vllm"):
        vlm_kw["json_mode"] = False
    return VideoReasoner(make_vlm(spec, **vlm_kw), **kw)


def video_bytes_to_tmp(data: bytes, suffix: str = ".mp4") -> Path:
    import tempfile

    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.close()
    return Path(f.name)

