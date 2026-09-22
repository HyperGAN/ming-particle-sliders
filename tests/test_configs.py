"""Parse the Ming card. No Hub access."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from ming.defaults import (
    ATTRIBUTION,
    COMPANION_REPO,
    DTYPE,
    GUIDANCE,
    HUB_SUBFOLDERS,
    LICENSE_NAME,
    MODEL_ID,
    NUM_LAYERS_EXAMPLE,
    PIPELINE_TAG,
    RESOLUTION,
    RESOLUTION_FAST,
    RESOLUTION_NOTE,
    SAMPLE_STEPS,
    SCHEDULER_SHIFT,
    TASK,
    VRAM_GIB,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "ming"
BANNED_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_model_lock_matches_defaults():
    lock = json.loads((CONFIGS / "model.lock.json").read_text(encoding="utf-8"))
    assert lock["model_id"] == MODEL_ID
    assert lock["task"] == TASK
    assert lock["pipeline_tag"] == PIPELINE_TAG
    assert lock["license"] == LICENSE_NAME
    assert lock["attribution"] == ATTRIBUTION
    assert lock["hub_subfolders"] == list(HUB_SUBFOLDERS)
    assert lock["sample_steps"] == SAMPLE_STEPS == 12
    assert lock["guidance_scale"] == GUIDANCE == 2.0
    assert lock["resolution"] == RESOLUTION == 1024
    assert lock["resolution_fast"] == RESOLUTION_FAST == 512
    assert lock["resolution_note"] == RESOLUTION_NOTE
    assert "1024" in lock["resolution_note"]
    assert "512" in lock["resolution_note"]
    assert lock["dtype"] == DTYPE == "bfloat16"
    assert lock["vram_gib"] == VRAM_GIB == 80
    assert lock["num_layers_example"] == NUM_LAYERS_EXAMPLE == 6
    assert lock["scheduler_shift"] == SCHEDULER_SHIFT
    assert lock["companion_repo"] == COMPANION_REPO
    assert lock["weights_in_repo"] is False
    assert lock["transformer_class"] == "DiffusionTransformer"
    assert lock["vae_class"] == "AutoencoderKLQwenImage"


def test_sample_yaml_locks_the_card():
    config = _load(CONFIGS / "config-layer.yaml")
    prompts = _load(CONFIGS / "prompts-layer.yaml")
    model = config["pretrained_model"]
    assert model["name_or_path"] == MODEL_ID
    assert model["task"] == TASK
    assert model["attribution"] == ATTRIBUTION
    assert model["license"] == LICENSE_NAME
    assert model["subfolders"] == list(HUB_SUBFOLDERS)
    assert config["prompts_file"] == "configs/ming/prompts-layer.yaml"
    assert (ROOT / config["prompts_file"]).is_file()
    assert config["architecture"] == "gmix"
    assert config["formulation"] == "winning_formulation"
    assert config["network"]["training_method"] == "lora"
    assert config["network"]["rank_source"] == "winning_formulation"
    sample = config["sample"]
    assert sample["steps"] == 12
    assert sample["guidance"] == 2.0
    assert sample["resolution"] == 1024
    assert sample["resolution_fast"] == 512
    assert "resolution_note" in sample
    assert "1024" in sample["resolution_note"]
    assert "aspect ratio" in sample["resolution_note"]
    assert sample["dtype"] == "bfloat16"
    assert sample["num_layers"] == 6
    assert prompts["bare_captions"] is True
    assert prompts["control_prompt"] == "a bowl of fruit on a table"
    rows = prompts["rows"]
    assert len(rows) >= 2
    assert rows[0]["num_layers"] == 6
    assert rows[0]["guidance_scale"] == 2.0
    assert rows[0]["resolution"] == 1024
    assert "RGBA" in rows[0]["positive"]
    for row in rows:
        assert row["action"] == "enhance"
        assert row["positive"] != row["neutral"]
        assert row["negative"]
        assert row["target"] == row["neutral"]


def test_repo_does_not_vendor_weights():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    found = [path for path in tracked if Path(path).suffix in BANNED_SUFFIXES]
    assert found == []


def test_requirements_pin_shared_core():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert (
        "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@"
        "b3a100e5184d70a94f1a7562b71c10b81f009591"
        "#subdirectory=packages/particle-sliders-core"
    ) in text
    assert "cursor/particle-sliders-shared-core-030c" in text
    assert "PyYAML" in text
    assert "pytest" in text
    assert "concept-slider-core" not in text.replace(
        "Do not pin packages/concept-slider-core", ""
    )
    assert "mikkel/sliders-conceptmod" not in text.replace(
        "Do not pin packages/concept-slider-core or mikkel/sliders-conceptmod.", ""
    )
