"""Docs name this repo as the Ming product on the shared gmix stamp."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_is_the_product():
    readme = _text("README.md")
    for needle in (
        "ming-particle-sliders",
        "krea2-particle-sliders",
        "anima-particle-sliders",
        "inclusionAI/Ming-Image-0.1-Design-Layer",
        "particle-sliders-core",
        "winning_formulation",
        "gmix",
        "layer-decompose",
        "12",
        "2.0",
        "1024",
        "512",
        "bfloat16",
        "80 GiB",
        "inclusionAI",
        "scripts/train_ming.py",
        "scripts/infer_ming.py",
        "NTCMingDesignLayerLora",
        "does not ship slider weights",
        "concept-slider-core",
        "cursor/particle-sliders-shared-core-030c",
    ):
        assert needle in readme, needle


def test_formulation_doc():
    text = _text("FORMULATION.md")
    for needle in (
        "particle-sliders-core",
        "winning_formulation",
        "gmix",
        "particle-gmix-1600-v2",
        "provisional",
        "ParticleGAN",
        "pull/38",
        "RoutedMLP",
        "GradRegularizer",
        "concept-slider-core",
        "inclusionAI/Ming-Image-0.1-Design-Layer",
        "b3a100e5184d70a94f1a7562b71c10b81f009591",
    ):
        assert needle in text, needle


def test_reproduce_and_notice():
    reproduce = _text("REPRODUCE.md")
    for needle in (
        "inclusionAI/Ming-Image-0.1-Design-Layer",
        "--dummy",
        "scripts/train_ming.py",
        "scripts/infer_ming.py",
        "particle-sliders-core",
        "winning_formulation",
        "gmix",
        "12",
        "2.0",
        "1024",
        "finished Ming slider",
    ):
        assert needle in reproduce, needle
    notice = _text("NOTICE")
    assert "inclusionAI" in notice
    assert "MIT" in notice
    assert "Ming-Image-0.1-Design-Layer" in notice
    assert "does not contain Ming-Image weights" in notice


def test_comfy_doc():
    text = _text("COMFYUI.md")
    for needle in (
        "MingDesignLayerLora",
        "NTCMingDesignLayerLora",
        "NTC/Ming",
        "comfy_ming.py",
        "12 steps",
        "CFG 2.0",
        "1024",
        "512",
        "inclusionAI",
        "does not vendor",
    ):
        assert needle in text, needle
