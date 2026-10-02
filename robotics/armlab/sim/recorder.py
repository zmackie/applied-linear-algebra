"""Episode video recorder with a decision overlay (styled after the report's rollout viewer)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import CONTROL_DT, ArmEnv, FrameInfo

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
BADGE_COLORS = {
    "system1": (86, 214, 196),
    "oracle": (86, 214, 196),
    "llm": (120, 160, 255),
    "llm-correction": (255, 170, 60),
    "think": (230, 90, 90),
    "paused": (150, 150, 160),
}


def _font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


class Recorder:
    """Attach to an ArmEnv; renders one frame every `every` control steps."""

    def __init__(self, env: ArmEnv, path: str | Path, title: str, every: int = 2, main_size=(480, 640), inset_size=(240, 320)):
        self.env = env
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.every = every
        self.main_size, self.inset_size = main_size, inset_size
        self.fps = round(1.0 / (CONTROL_DT * every))
        self.writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264", quality=7, macro_block_size=16,
                                         ffmpeg_log_level="error")
        self.f_title, self.f_body, self.f_small = _font(17), _font(15), _font(13)
        self.state = {"source": "", "decision": 0, "rationale": "", "command": "", "status": ""}
        self.n_frames = 0
        env.frame_callbacks.append(self._on_frame)

    def set_decision(self, index: int, source: str, rationale: str, command: str):
        self.state.update(decision=index, source=source, rationale=rationale, command=command)

    def _on_frame(self, env: ArmEnv, fi: FrameInfo):
        if fi.step % self.every:
            return
        self.write_frame(phase=fi.phase)

    def write_frame(self, phase: str = "move", banner: str | None = None, repeat: int = 1):
        env = self.env
        mh, mw = self.main_size
        ih, iw = self.inset_size
        main = env.render("front", mh, mw, shadows=False)
        head = env.render("head", ih, iw, shadows=False)
        wrist = env.render("wrist", ih, iw, shadows=False)
        top = np.concatenate([main, np.concatenate([head, wrist], 0)], 1)
        panel_h = 150
        img = Image.new("RGB", (top.shape[1], top.shape[0] + panel_h + 34), (16, 24, 38))
        img.paste(Image.fromarray(top), (0, 34))
        d = ImageDraw.Draw(img)
        d.text((10, 8), textwrap.shorten(self.title, 120), font=self.f_title, fill=(235, 240, 250))
        d.text((mw + 6, 38), "HEAD", font=self.f_small, fill=(255, 255, 255))
        d.text((mw + 6, 38 + ih), "WRIST", font=self.f_small, fill=(255, 255, 255))

        y0 = 34 + top.shape[0] + 8
        source = "think" if phase == "think" else self.state["source"]
        label = {"think": "MODEL THINKING - world keeps moving", "paused": "SIM PAUSED - model thinking",
                 "system1": "SYSTEM-1 ACTION", "oracle": "SCRIPTED ORACLE", "llm": "LLM DIRECT (EEF)",
                 "llm-correction": "LLM CORRECTION"}.get(source, source.upper())
        if banner:
            source, label = "paused", banner
        color = BADGE_COLORS.get(source, (200, 200, 200))
        d.rounded_rectangle((10, y0, 330, y0 + 26), 6, fill=color)
        d.text((18, y0 + 4), label, font=self.f_body, fill=(10, 16, 28))
        st = env.status()
        meta = f"Decision {self.state['decision']}  |  step {env.step_count}/{env.task.max_steps}  |  {env.sim_time:5.2f} s  |  score {st.score:.0f}{'  |  SUCCESS' if st.success else ''}"
        d.text((345, y0 + 5), meta, font=self.f_body, fill=(200, 210, 225))
        for i, line in enumerate(textwrap.wrap(self.state["rationale"], 70)[:4]):
            d.text((10, y0 + 36 + 19 * i), line, font=self.f_body, fill=(235, 240, 250))
        for i, line in enumerate(textwrap.wrap(self.state["command"], 46)[:6]):
            d.text((mw + 10, y0 + 36 + 17 * i), line, font=self.f_small, fill=(170, 185, 205))
        frame = np.asarray(img)
        for _ in range(repeat):
            self.writer.append_data(frame)
            self.n_frames += 1

    def pause_card(self, seconds_thinking: float, hold_s: float = 0.6):
        """In paused mode, show a short freeze so viewers can see where the model was deliberating."""
        self.write_frame(banner=f"SIM PAUSED - model thought {seconds_thinking:.1f} s", repeat=max(1, round(hold_s * self.fps)))

    def close(self):
        if self._on_frame in self.env.frame_callbacks:
            self.env.frame_callbacks.remove(self._on_frame)
        self.writer.close()
