"""CPU check that this product imports the pinned winning-formulation stamp."""

import subprocess
import sys
from pathlib import Path

from particle_sliders import winning_formulation

from comfy_ming import MingParticleSlider
from ming.surfaces import (
    BASE_MODEL_ID,
    HUB_MODEL_ID,
    HUB_PUBLISHED,
    stamp,
)

ROOT = Path(__file__).resolve().parents[1]
PIN = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git"
    "@a119ca1ecd3d5d6c437065839d22739b04f2f4d8"
    "#subdirectory=packages/particle-sliders-core"
)


def test_requirements_pin_the_shared_core():
    text = (ROOT / "requirements.txt").read_text()
    assert PIN in text
    assert "ParticleGAN.git" not in text
    assert "mikkel/sliders-conceptmod" not in text


def test_import_stamp_from_the_installed_package():
    loaded = winning_formulation()
    assert loaded is stamp()
    assert loaded.architecture_id == "gmix"
    assert loaded.formulation_id == "particle-gmix-1600-v2"
    assert loaded.formulation_provisional is True
    assert loaded.spec["critic"] == "gmix"
    loaded.require(loaded.as_dict())
    origin = Path(winning_formulation.__code__.co_filename).resolve()
    assert ROOT not in origin.parents
    assert "particle_sliders" in origin.parts


def test_hub_placeholder_is_unpublished():
    assert BASE_MODEL_ID == "inclusionAI/Ming-Image-0.1-Design-Layer"
    assert HUB_MODEL_ID == "ntc-ai/ming-particle-sliders"
    assert HUB_PUBLISHED is False
    readme = (ROOT / "README.md").read_text()
    formulation = (ROOT / "FORMULATION.md").read_text()
    assert HUB_MODEL_ID in readme
    assert "not published" in readme.lower()
    assert "winning_formulation()" in formulation


def test_comfy_node_refuses_unpublished_weights():
    node = MingParticleSlider()
    try:
        node.load(model=object(), adapter_name="missing", strength=1.0)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("unpublished Hub weights must not load")
    assert HUB_MODEL_ID in message
    assert "gmix" in message


def test_train_and_infer_dry_run_import_the_stamp():
    for script in ("scripts/train_ming.py", "scripts/infer_ming.py"):
        completed = subprocess.run(
            [sys.executable, script, "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "architecture_id=gmix" in completed.stdout
        assert "formulation_id=particle-gmix-1600-v2" in completed.stdout
        assert f"hub={HUB_MODEL_ID}" in completed.stdout
        assert "hub_published=False" in completed.stdout
