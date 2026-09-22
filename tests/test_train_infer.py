"""In-repo train/infer wiring. No Hub download."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ming.defaults import (
    GUIDANCE,
    HUB_SUBFOLDERS,
    MODEL_ID,
    RESOLUTION_NOTE,
    SAMPLE_STEPS,
)
from ming.formulation import formulation_record, load_winning
from ming.live import load_ming_pipeline, load_plan
from ming.prompts import load_prompts, unused_words_for
from ming.train import assert_ming_only, parse_args, train
from ming.uni import plus_neu_teachers
from particle_sliders import winning_formulation

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "configs" / "ming" / "prompts-layer.yaml"


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_names_the_card_and_the_shared_core():
    for script in ("train_ming.py", "infer_ming.py"):
        proc = _run(script, ["--help"])
        assert proc.returncode == 0, proc.stderr
        text = proc.stdout
        assert MODEL_ID in text
        assert "--dummy" in text
        assert "winning_formulation" in text
        assert "gmix" in text
        assert "12" in text
    infer = _run("infer_ming.py", ["--dummy"])
    assert infer.returncode != 0
    assert "load_lora" in infer.stderr


def test_print_card_binds_winning_formulation():
    card = train(parse_args(["--print_card"]))
    stamp = winning_formulation()
    assert card["backend"] == "ming_design_layer"
    assert card["model_id"] == MODEL_ID
    assert card["task"] == "layer-decompose"
    assert card["hub_subfolders"] == list(HUB_SUBFOLDERS)
    assert card["sample_steps"] == 12
    assert card["sample_guidance"] == 2.0
    assert card["resolution"] == 1024
    assert card["formulation"]["entry"] == "winning_formulation"
    assert card["formulation"]["architecture"] == "gmix"
    assert card["formulation"]["critic"] == "gmix"
    assert card["formulation"]["id"] == stamp.id
    assert card["formulation"]["provisional"] is stamp.provisional
    assert card["weights_in_repo"] is False


def test_cfg_teacher_uses_the_card_scale():
    plus, zero = plus_neu_teachers([2.0, 0.0], [0.0, 1.0], [0.5, 0.5], guidance=2.0)
    assert plus == pytest.approx([5.0, -1.0])
    assert zero == [0.0, 1.0]
    bare, _zero = plus_neu_teachers([2.0, 0.0], [0.0, 1.0], [0.5, 0.5], guidance=0.0)
    assert bare == [2.0, 0.0]


def test_prompt_card_keeps_bare_captions():
    rows, meta = load_prompts(PROMPTS)
    assert meta.bare_captions is True
    assert meta.control_prompt == "a bowl of fruit on a table"
    assert len(rows) == 2
    assert rows[0].guidance_scale == GUIDANCE
    assert rows[0].num_layers == 6
    assert rows[0].resolution == 1024
    unused = unused_words_for(rows[0])
    assert "print" in unused and "screen" in unused
    assert "print" not in rows[0].positive


def test_refuses_foreign_ids_and_live_download():
    with pytest.raises(ValueError, match="Design-Layer only"):
        assert_ming_only("krea/Krea-2-Raw")
    assert_ming_only(MODEL_ID)
    with pytest.raises(ValueError, match="Design-Layer only"):
        train(parse_args(["--dummy", "--model_id", "inclusionAI/Ming-Lite-Omni", "--steps", "1"]))
    with pytest.raises(NotImplementedError, match="does not download"):
        train(parse_args(["--live"]))
    with pytest.raises(NotImplementedError, match="does not download"):
        load_ming_pipeline(MODEL_ID)
    plan = load_plan()
    assert plan["downloads"] is False
    assert plan["stub"] is True
    assert plan["subfolders"] == list(HUB_SUBFOLDERS)
    assert plan["sample_steps"] == SAMPLE_STEPS


def test_dummy_train_and_infer_do_not_download(tmp_path: Path):
    for name in ("diffusers", "huggingface_hub"):
        sys.modules.pop(name, None)
    sidecar_path = train(
        parse_args(
            [
                "--dummy",
                "--name",
                "layer-ming-dummy",
                "--prompts_file",
                str(PROMPTS),
                "--save_dir",
                str(tmp_path / "train"),
                "--steps",
                "8",
                "--seed",
                "7",
            ]
        )
    )
    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    stamp = load_winning()
    record = formulation_record(stamp)
    assert payload["kind"] == "ming_design_layer"
    assert payload["model_id"] == MODEL_ID
    assert payload["sample_steps"] == 12
    assert payload["sample_guidance"] == 2.0
    assert payload["train_guidance"] == 2.0
    assert payload["resolution"] == 1024
    assert payload["resolution_note"] == RESOLUTION_NOTE
    assert payload["dummy"] is True
    assert payload["allow_hub"] is False
    assert payload["weights_downloaded"] is False
    assert payload["rank"] == int(stamp.spec["adapter_rank"])
    assert payload["rank_source"] == "winning_formulation"
    assert payload["formulation"] == record
    assert payload["formulation"]["architecture"] == "gmix"
    assert payload["formulation"]["critic"] == stamp.spec["critic"] == "gmix"
    assert set(payload["formulation"]["loss_modules"]) == {"particle_sliders.reference"}
    assert payload["load_plan"]["downloads"] is False
    lines = [
        json.loads(line)
        for line in (tmp_path / "train" / "layer-ming-dummy_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(lines) == 2
    assert lines[-1]["minus_teacher"] == 0.0
    assert lines[-1]["noise_std_step0"] == pytest.approx(record["noise_std_step0"])
    meta = json.loads((tmp_path / "train" / "samples" / "final_meta.json").read_text(encoding="utf-8"))
    assert meta["gate"] == "layer-plan"
    assert "a bowl of fruit on a table" in meta["prompts"]
    assert all(shot["cfg"] == 2.0 for shot in meta["samples"])
    assert all(shot["sample_steps"] == 2 for shot in meta["samples"])
    pngs = list((tmp_path / "train" / "samples").glob("*.png"))
    assert len(pngs) == 12
    assert pngs[0].read_bytes().startswith(b"\x89PNG")
    assert "diffusers" not in sys.modules
    assert "huggingface_hub" not in sys.modules

    infer_path = train(
        parse_args(
            [
                "--dummy",
                "--load_lora",
                "models/layer-ming-design_lora",
                "--save_dir",
                str(tmp_path / "infer"),
                "--steps",
                "4",
            ]
        )
    )
    infer = json.loads(Path(infer_path).read_text(encoding="utf-8"))
    assert infer["skipped_train"] is True
    assert infer["load_lora"] == "models/layer-ming-design_lora"
    assert infer["weights_downloaded"] is False
    assert not (tmp_path / "infer" / "layer-ming-design_train.jsonl").exists()


def test_explicit_guidance_override(tmp_path: Path):
    payload = json.loads(
        Path(
            train(
                parse_args(
                    [
                        "--dummy",
                        "--sample_guidance",
                        "1.5",
                        "--sample_steps",
                        "6",
                        "--resolution",
                        "512",
                        "--save_dir",
                        str(tmp_path),
                        "--steps",
                        "1",
                    ]
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert payload["sample_guidance"] == pytest.approx(1.5)
    assert payload["sample_steps"] == 6
    assert payload["resolution"] == 512
    assert payload["train_guidance"] == pytest.approx(1.5)


def test_python_tree_does_not_vendor_the_game():
    banned = ("class RoutedMLP", "class GradRegularizer", "class GlobalMixErrorCritic")
    roots = [ROOT / "ming", ROOT / "scripts", ROOT / "comfy_ming.py", ROOT / "__init__.py"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{needle} vendored in {path}"
        assert "concept_slider_core" not in text
