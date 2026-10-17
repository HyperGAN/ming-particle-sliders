"""CPU check that this product imports the pinned winning-formulation stamp."""

import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

from particle_sliders import winning_formulation

from comfy_ming import MingParticleSlider
from ming.surfaces import (
    BASE_MODEL_ID,
    HUB_MODEL_ID,
    HUB_PUBLISHED,
    stamp,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_LOCK = json.loads((ROOT / "core.lock.json").read_text())
PIN = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git"
    f"@{CORE_LOCK['commit']}"
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
    installed = distribution("particle-sliders-core")
    assert origin == Path(installed.locate_file("particle_sliders/formulation.py")).resolve()
    assert "site-packages" in origin.parts
    assert installed.version == CORE_LOCK["version"]
    assert distribution("particlegan").version == CORE_LOCK["particlegan_version"]


def test_hub_placeholder_is_unpublished():
    assert BASE_MODEL_ID == "inclusionAI/Ming-Image-0.1-Design-Layer"
    assert HUB_MODEL_ID == "ntc-ai/ming-particle-sliders"
    assert HUB_PUBLISHED is False
    readme = (ROOT / "README.md").read_text()
    formulation = (ROOT / "FORMULATION.md").read_text()
    assert HUB_MODEL_ID in readme
    assert "not published" in readme.lower()
    assert "winning_formulation()" in formulation


def test_installed_core_matches_locked_source_identity():
    # Multiple candidate wheels share version 0.6.0. Version checks alone
    # must not qualify a superseded wheel or an editable source checkout.
    installed = distribution("particle-sliders-core")
    direct_url = json.loads(installed.read_text("direct_url.json"))
    if "vcs_info" in direct_url:
        assert direct_url["vcs_info"]["commit_id"] == CORE_LOCK["commit"]
    else:
        assert direct_url["archive_info"]["hashes"]["sha256"] == CORE_LOCK["shared_wheel_sha256"]


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
        assert "execution=scaffold-only" in completed.stdout


@pytest.mark.parametrize("module", ["ming.train", "ming.infer"])
def test_module_entrypoints_check_metadata_and_refuse_live_execution(module):
    checked = subprocess.run(
        [sys.executable, "-m", module, "--dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert "execution=scaffold-only" in checked.stdout
    refused = subprocess.run(
        [sys.executable, "-m", module], cwd=ROOT, capture_output=True, text=True,
    )
    assert refused.returncode != 0
    assert "is not wired" in refused.stderr


def test_all_surfaces_validate_the_shared_stamp(monkeypatch, capsys):
    from ming import infer, surfaces, train

    loaded = winning_formulation()
    validated = []

    class ObservedStamp:
        def __getattr__(self, name):
            return getattr(loaded, name)

        def require(self, recipe):
            validated.append(recipe)
            return loaded.require(recipe)

    monkeypatch.setattr(surfaces, "winning_formulation", ObservedStamp)
    train.main(["--dry-run"])
    infer.main(["--dry-run"])
    with pytest.raises(RuntimeError, match="not published"):
        MingParticleSlider().load(object(), "missing", 1.0)
    assert validated == [loaded.as_dict()] * 3


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -0.1, 1.1])
def test_comfy_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="outside"):
        MingParticleSlider().load(object(), "missing", strength)


def test_hub_flag_alone_cannot_enable_unimplemented_comfy_execution(monkeypatch):
    import comfy_ming

    monkeypatch.setattr(comfy_ming, "HUB_PUBLISHED", True)
    with pytest.raises(RuntimeError, match="execution is not wired"):
        MingParticleSlider().load(object(), "missing", 1.0)


def test_comfy_loads_as_a_custom_node_package_from_outside_checkout(tmp_path):
    # Comfy loads custom-node directories as packages without adding each
    # directory to sys.path. A top-level `from ming` breaks that mode.
    probe = """
import importlib.util
import sys
from pathlib import Path
root = Path(sys.argv[1])
assert str(root) not in sys.path
spec = importlib.util.spec_from_file_location(
    'ming_custom_node', root / '__init__.py',
    submodule_search_locations=[str(root)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
node = module.NODE_CLASS_MAPPINGS['MingParticleSlider']()
assert node.__class__.__module__ == 'ming_custom_node.comfy_ming'
try:
    node.load(object(), 'missing', 1.0)
except RuntimeError as exc:
    assert 'not published' in str(exc)
else:
    raise AssertionError('scaffold must refuse live adapter execution')
"""
    checked = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(ROOT)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert checked.returncode == 0, checked.stderr
