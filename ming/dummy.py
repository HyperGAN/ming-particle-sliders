"""CPU stand-in for a Ming layer-decompose UNI step.

Never loads Hub weights. The step calls ``winning_formulation()`` and
uses the shared noise curve. It does not construct RoutedMLP or
GradRegularizer and does not reimplement ``stamp.losses()``.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any

from ming.defaults import SAMPLE_SCALES
from ming.formulation import load_winning
from ming.prompts import SliderPrompt, unused_words_for, word_tokens
from ming.uni import mse, plus_neu_teachers


class Loss:
    def __init__(self, value: float, param: Param):
        self.value = float(value)
        self.param = param

    def detach(self) -> float:
        return self.value

    def backward(self) -> None:
        self.param.grad = self.value


class Param:
    def __init__(self) -> None:
        self.data = 0.1
        self.grad = 0.0


def _vec(text: str, dim: int = 4) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[i] / 255.0) * 2.0 - 1.0) for i in range(dim)]


class DummyBackend:
    """Deterministic velocities. Scale 1 adds the trainable scalar."""

    def __init__(self, *, seed: int = 0) -> None:
        self.dim = 4
        self.seed = int(seed)
        self.param = Param()
        self.loaded_lora: str | None = None
        self.stamp = load_winning()

    def sample_latents(self) -> list[float]:
        sigma = float(self.stamp.noise_std_at(0, float(self.stamp.spec["edit_rms_target"])))
        base = _vec(f"latent-{self.seed}", self.dim)
        return [value * sigma for value in base]

    def trainable_parameters(self) -> list[Param]:
        return [self.param]

    def predict_v(self, prompt: str, z: list[float], *, scale: float = 0.0) -> list[float]:
        del z
        base = _vec(prompt or "", self.dim)
        shift = float(scale) * self.param.data
        return [value + shift for value in base]

    def load_lora(self, path: str | Path) -> None:
        self.loaded_lora = str(path)

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_steps: int = 2,
        guidance: float = 2.0,
        scale: float = 0.0,
        height: int = 8,
        width: int = 8,
    ) -> TinyImage:
        tone = int(abs(hash((prompt, seed, num_steps, round(guidance, 4), scale))) % 200) + 20
        return TinyImage(width=width, height=height, tone=tone)


class TinyImage:
    def __init__(self, *, width: int, height: int, tone: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.tone = int(tone)

    def save(self, path: str | Path) -> None:
        _write_png(Path(path), self.width, self.height, self.tone)


def _write_png(path: Path, width: int, height: int, tone: int) -> None:
    raw = b"".join(
        b"\x00" + bytes([tone, (tone // 2) & 255, 255 - tone, 255]) * width
        for _ in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def step_loss(
    backend: DummyBackend,
    prompt: SliderPrompt,
    z: list[float],
    *,
    guidance: float,
    hold_weight: float,
) -> tuple[Loss, dict[str, float]]:
    """One UNI step on the Ming card. The particle losses stay on the stamp."""
    with_scale0 = backend.predict_v(prompt.positive, z, scale=0.0)
    v_neu = backend.predict_v(prompt.neutral, z, scale=0.0)
    v_uncond = backend.predict_v(prompt.unconditional, z, scale=0.0)
    tgt_plus, tgt_zero = plus_neu_teachers(
        with_scale0, v_neu, v_uncond, guidance=guidance
    )
    pred_plus = backend.predict_v(prompt.positive, z, scale=1.0)
    pred_zero = backend.predict_v(prompt.neutral, z, scale=0.0)
    value = mse(pred_plus, tgt_plus) + mse(pred_zero, tgt_zero)
    unused = set(unused_words_for(prompt))
    pos_tokens = set(word_tokens(prompt.positive))
    hold = 0.0 if not unused else 0.01 * len(unused - pos_tokens)
    if hold_weight > 0.0:
        value = value + float(hold_weight) * hold
    stats = {
        "loss": float(value),
        "hold": float(hold),
        "minus_teacher": 0.0,
        "noise_std_step0": float(
            backend.stamp.noise_std_at(0, float(backend.stamp.spec["edit_rms_target"]))
        ),
    }
    return Loss(value, backend.param), stats


def write_sample_grid(
    backend: DummyBackend,
    prompts: list[SliderPrompt],
    save_dir: Path,
    *,
    control_prompt: str,
    guidance: float,
    sample_steps: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Tiny RGBA PNGs. ``--dummy`` does not render at the card resolution."""
    out = save_dir / "samples"
    out.mkdir(parents=True, exist_ok=True)
    captions = [prompt.neutral for prompt in prompts]
    if control_prompt not in captions:
        captions.append(control_prompt)
    shots: list[dict[str, Any]] = []
    for index, caption in enumerate(captions):
        for scale in SAMPLE_SCALES:
            name = f"s{index}_scale{scale:.2f}.png"
            image = backend.generate(
                caption,
                seed=sample_seed,
                num_steps=min(int(sample_steps), 2),
                guidance=guidance,
                scale=float(scale),
            )
            image.save(out / name)
            shots.append(
                {
                    "file": name,
                    "prompt": caption,
                    "scale": float(scale),
                    "cfg": float(guidance),
                    "sample_steps": min(int(sample_steps), 2),
                }
            )
    return shots
