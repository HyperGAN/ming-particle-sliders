"""Exercise the installed gate against Ming's locked cached-reference pipeline."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
from runpy import run_path
import sys
from threading import Event
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from ming import scheduled_render
from ming.image_bindings import image_lora_parameters
from particle_sliders.native_lora import LoRANetwork


@pytest.fixture(scope="module")
def core_wheel():
    path = os.environ.get("MING_STEERING_CORE_WHEEL")
    if not path:
        pytest.skip("set MING_STEERING_CORE_WHEEL to the shared core PR #269 wheel")
    return Path(path)


@pytest.fixture(scope="module")
def native_pipeline():
    path = os.environ.get("MING_NATIVE_PIPELINE_SOURCE")
    if not path:
        pytest.skip("set MING_NATIVE_PIPELINE_SOURCE to the archived cached-reference consumer")
    assert sha256(Path(path).read_bytes()).hexdigest() == "34ce578c77923d50067a6fafa093614472c9683598f549079ba5875f67270805"
    spec = importlib.util.spec_from_file_location("diffusion._ming_gate_cpu_fixture", path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    try:
        spec.loader.exec_module(native)
        yield native.ImageGenerationPipeline
    finally:
        del sys.modules[spec.name]


class TinyMingDenoiser(torch.nn.Module):
    dtype = torch.float32
    in_channels = 16
    config = SimpleNamespace(multi_frame_output=True)

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(self.in_channels) * 0.1)
        self.requires_grad_(False)
        self.inputs = []

    def forward(self, latents, time, embeds, *, ref_hidden_states, return_dict, encoder_hidden_states_2):
        assert len(latents) == len(embeds) == len(encoder_hidden_states_2) == len(ref_hidden_states)
        assert all(latent.shape == (16, 2, 64, 64) for latent in latents)
        assert all(ref.shape == (16, 1, 64, 64) for ref in ref_hidden_states)
        self.inputs.append(latents[0].mean(dim=(-2, -1)))
        return ([self.proj(latent.movedim(0, -1)).movedim(-1, 0) + embeds[index].mean() * 0.01
                 for index, latent in enumerate(latents)],)


class VaeConfig(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class FixtureVAE:
    dtype = torch.float32
    config = VaeConfig(temperal_downsample=[True, True, True], latent_channels=16,
                       shift_factor=0.0, scaling_factor=1.0)


@pytest.fixture
def owner(native_pipeline):
    from diffusers import FlowMatchEulerDiscreteScheduler

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79)
        model = TinyMingDenoiser()
        lora = LoRANetwork(model, rank=2, alpha=2, target_plan=[("fixture_proj", model.proj)])
    for layer in lora.unet_loras:
        with torch.no_grad():
            layer.lora_down.weight.fill_(0.02)
            layer.lora_up.weight.fill_(0.03)
    raw = image_lora_parameters(model, lora)
    pipe = native_pipeline(
        vae=FixtureVAE(),
        transformer=model, text_encoder=None, tokenizer=None,
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
    )
    pipe.set_progress_bar_config(disable=True)
    prompt = [torch.ones(3, 8)]
    negative = [torch.zeros_like(prompt[0])]
    latents = torch.ones(2, 16, 64, 64)
    latents[0] *= 0.3
    latents[1] *= -0.7
    inputs = dict(device=torch.device("cpu"), num_frames_per_prompt=2, height=512, width=512,
                  num_inference_steps=4, output_type="latent", guidance_scale=1.0,
                  latents=latents, prompt_embeds=prompt, negative_prompt_embeds=negative,
                  prompt_embeds_2=prompt, negative_prompt_embeds_2=negative,
                  cached_reference_latents=torch.zeros(1, 16, 1, 64, 64, dtype=torch.bfloat16))
    return pipe, model, raw, inputs


def make_plan(module, core_wheel, *, method="random", parameter=0.0, seed=7, **settings):
    return module.LatentGatePlan(core_wheel=core_wheel, method=method, parameter=parameter,
                                 seed=seed, calibration_sha256="1" * 64, **settings)


def test_ming_rejects_non_core_gate_source(tmp_path):
    path = tmp_path / "changed.whl"
    with ZipFile(path, "w") as archive:
        archive.writestr(scheduled_render.CORE_MODULE, "raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="exact shared steering module"):
        make_plan(scheduled_render, path)


def test_independently_loaded_native_helpers_only_see_initialized_core(core_wheel, monkeypatch):
    first = run_path(scheduled_render.__file__)["LatentGatePlan"]
    second = run_path(scheduled_render.__file__)["LatentGatePlan"]
    monkeypatch.delitem(sys.modules, scheduled_render._MODULE_NAME, raising=False)
    started, release = Event(), Event()

    def pause_compile(*args):
        started.set()
        assert release.wait(10)
        return compile(*args)

    monkeypatch.setitem(first.__init__.__globals__, "compile", pause_compile)
    settings = dict(core_wheel=core_wheel, method="random", parameter=0.0,
                    seed=7, calibration_sha256="1" * 64)
    with ThreadPoolExecutor(max_workers=2) as workers:
        initiating = workers.submit(first, **settings)
        try:
            assert started.wait(10)
            concurrent = workers.submit(second, **settings).result(timeout=10)
        finally:
            release.set()
        original = initiating.result(timeout=10)
    assert original._steering is concurrent._steering


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_native_off_and_constant_replay_keeps_order_and_terminal_budget(owner, core_wheel, probability):
    pipe, model, raw, inputs = owner
    strength = 0.5
    with raw.scaled(strength * probability):
        ordinary = pipe(**inputs).images
    count = len(model.inputs)
    plan = make_plan(scheduled_render, core_wheel, parameter=probability)
    gated, receipt = scheduled_render.render(pipe, scale_adapter=raw.scaled, gate_plan=plan,
                                              strength=strength, **inputs)
    assert torch.equal(gated.images, ordinary)
    assert gated.images.shape == (2, 16, 64, 64)
    assert len(model.inputs) - count == 4
    assert receipt["native_timesteps"] == pipe.scheduler.timesteps.tolist()
    assert receipt["sigmas"][-2:] == [0.0, 0.0]
    assert receipt["eligible"] == 3 and receipt["fired"] == 3 * probability
    assert receipt["gate_rate"] == probability
    assert receipt["sigma_weighted_strength"] == strength * probability
    assert receipt["effective_strengths"] == [strength * probability] * 3 + [0.0]
    assert receipt["decisions"][-1] == dict(interval=3, sigma_start=0.0, sigma_end=0.0,
                                            eligible=False, projection=None, probability=None,
                                            fired=False, effective_strength=0.0)
    assert raw.scale == 1.0


@pytest.mark.parametrize("record_native_grid", [False, True])
def test_cfp_uses_live_ordered_readout_and_replays_through_batched_cfg(owner, core_wheel, record_native_grid):
    pipe, model, raw, inputs = owner
    inputs["guidance_scale"] = 2.0
    # This test exercises the documented standalone route in the same installed wheel.
    exports = run_path(scheduled_render.__file__)
    strength = 0.5
    direction = torch.zeros(16, 2)
    direction[0, 1] = 1
    plan = exports["LatentGatePlan"](core_wheel=core_wheel, method="cfp", parameter=-0.71,
                                     calibration_sha256="2" * 64, direction=direction)
    gated, receipt = exports["render"](pipe, scale_adapter=raw.scaled, gate_plan=plan,
                                        strength=strength, **inputs)
    decisions = receipt["decisions"][:-1]
    assert [row["projection"] for row in decisions] == [float(latent[0, 1]) for latent in model.inputs[:-1]]
    assert len({row["projection"] for row in decisions}) == 3
    assert {row["fired"] for row in decisions} == {False, True}
    replay, replay_receipt = exports["render"](
        pipe, scale_adapter=raw.scaled, strength=strength,
        step_gains=[value / strength for value in receipt["effective_strengths"]],
        record_native_grid=record_native_grid, **inputs,
    )
    assert torch.equal(gated.images, replay.images)
    fields = ("native_timesteps", "effective_strengths", "sigmas") if record_native_grid else (
        "native_timesteps", "effective_strengths",
    )
    assert replay_receipt == {name: receipt[name] for name in fields}
    assert raw.scale == 1.0


def test_fixed_native_grid_refuses_a_changed_solver_interval(owner):
    pipe, model, raw, inputs = owner

    def change_next_sigma(owner, index, timestep, values):
        if index == 0:
            owner.scheduler.sigmas[2] = (owner.scheduler.sigmas[1] + owner.scheduler.sigmas[2]) / 2
        return values

    with pytest.raises(RuntimeError, match="sigma grid changed"):
        scheduled_render.render(
            pipe, scale_adapter=raw.scaled, step_gains=[1.0, 1.0, 1.0, 0.0],
            record_native_grid=True, callback_on_step_end=change_next_sigma, **inputs,
        )
    assert raw.scale == 1.0 and not model._forward_pre_hooks and not model._forward_hooks


def test_clabs_restarts_each_request_and_closed_scope_survives_error(owner, core_wheel):
    pipe, model, raw, inputs = owner
    direction = torch.zeros(16, 2)
    direction[0, 0] = 1
    plan = make_plan(scheduled_render, core_wheel, method="clabs", parameter=0.7, direction=direction,
                     clabs=dict(mean=0.0, std=1.0, epsilon=0.001, setpoint=0.0,
                                proportional_gain=0.5, integral_gain=0.3, beta=0.45))

    def stop(*args):
        raise RuntimeError("fixture callback stopped")

    with pytest.raises(RuntimeError, match="fixture callback stopped"):
        scheduled_render.render(pipe, scale_adapter=raw.scaled, gate_plan=plan, strength=0.5,
                                callback_on_step_end=stop, **inputs)
    assert raw.scale == 1.0 and not model._forward_pre_hooks and not model._forward_hooks
    first, a = scheduled_render.render(pipe, scale_adapter=raw.scaled, gate_plan=plan, strength=0.5, **inputs)
    second, b = scheduled_render.render(pipe, scale_adapter=raw.scaled, gate_plan=plan, strength=0.5, **inputs)
    assert a == b and torch.equal(first.images, second.images)
    assert len({x["probability"] for x in a["decisions"] if x["eligible"]}) == 3
    assert raw.scale == 1.0


def test_runtime_rejects_duplicate_solver_interval_before_extra_adapter_scope(owner, core_wheel):
    pipe, model, raw, inputs = owner
    plan = make_plan(scheduled_render, core_wheel)

    def repeat(owner, index, timestep, values):
        owner.scheduler._step_index = index
        return values

    with pytest.raises(RuntimeError, match="before each original native solver interval"):
        scheduled_render.render(pipe, scale_adapter=raw.scaled, gate_plan=plan, strength=0.5,
                                callback_on_step_end=repeat, **inputs)
    assert len(model.inputs) == 1
    assert raw.scale == 1.0 and not model._forward_pre_hooks and not model._forward_hooks
