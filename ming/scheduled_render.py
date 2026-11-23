"""Per-step strength on a caller-owned native denoising pipeline."""

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from itertools import pairwise
from math import isfinite
import sys
from sys import float_info
from threading import Lock
from types import ModuleType
from zipfile import ZipFile


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MODULE = "particle_sliders/adaptive_steering.py"
CORE_MODULE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
_MODULE_NAME = f"_ming_shared_steering_{CORE_REVISION}_ready"
_LOAD_LOCK = Lock()
_FIXED_DEFAULT = object()


def _shared_steering(core_wheel):
    # Keep the installed core (and the source identity of archived checkpoints).
    try:
        with ZipFile(core_wheel) as archive:
            source = archive.read(CORE_MODULE)
    except KeyError as error:
        raise ValueError("Ming requires the shared steering module from core PR #269") from error
    if sha256(source).hexdigest() != CORE_MODULE_SHA256:
        raise ValueError("Ming requires the exact shared steering module from core PR #269")
    with _LOAD_LOCK:
        ready = sys.modules.get(_MODULE_NAME)
        if ready is not None:
            return ready
        module = ModuleType(_MODULE_NAME)
        loading_name = f"{_MODULE_NAME}_{id(module):x}"
        module.__name__ = loading_name
        # Dataclasses need their defining module during execution. Standalone
        # loads have independent locks, so publish only the completed module.
        sys.modules[loading_name] = module
        try:
            exec(compile(source, f"{core_wheel}/{CORE_MODULE}", "exec"), module.__dict__)
        except BaseException:
            del sys.modules[loading_name]
            raise
        ready = sys.modules.setdefault(_MODULE_NAME, module)
        if ready is not module:
            del sys.modules[loading_name]
        return ready


class LatentGatePlan:
    """Explicit calibration for one ordered pair of native Ming target frames."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        import torch

        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(character not in "0123456789abcdef" for character in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the Ming gate calibration record")
        self._steering = _shared_steering(core_wheel)
        self._clabs = self._steering.CLABSParameters(**clabs) if clabs is not None else None
        self._direction = None
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random gate does not use a Ming direction")
        else:
            if direction is None:
                raise ValueError("Ming CFP and CLABS require a calibrated ordered-frame direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (probe.ndim != 2 or probe.shape[0] < 1 or probe.shape[1] != 2
                    or not torch.isfinite(probe).all()
                    or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
                raise ValueError("Ming requires a finite unit direction of shape [channels, 2]")
            self._direction = probe
        if type(parameter) not in (float, int):
            raise ValueError("The calibrated gate parameter must be a number")
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        self._new_gate()

    def _new_gate(self):
        return self._steering.ActivationGate(self._method, self._parameter, seed=self._seed, clabs=self._clabs)

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "signal": "pre-edit spatial mean in FP32 with ordered [channel, frame] coordinates",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }

    def _projection(self, latent, strength):
        import torch

        if self._direction is None:
            return None
        if self._direction.shape != latent.shape[:2]:
            raise ValueError("Direction must match the opened Ming channels and ordered frames")
        pooled = latent.mean(dim=(-2, -1), dtype=torch.float32)
        return float(self._steering.activation_projection(
            pooled.flatten(), self._direction.flatten(), strength=strength,
        ))


def render(pipeline, *, scale_adapter, step_gains=None, gate_plan=None, strength=_FIXED_DEFAULT,
           callback_on_step_end=None, record_native_grid=False, **inputs):
    """Sample with an attached RAW adapter and return the output and step trace.

    Set record_native_grid to include the validated live sigma grid on fixed runs.
    Gated runs include their sigma grid by default.
    """
    if strength is _FIXED_DEFAULT:
        if gate_plan is not None:
            raise ValueError("Gated Ming requires the explicit nominal strength used in calibration")
        strength = 1.0
    if type(strength) not in (int, float) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("nominal strength must be finite")
    steps = inputs.get("num_inference_steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("one gain per requested native denoising step is required")
    if gate_plan is None:
        if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
            raise ValueError("one gain per requested native denoising step is required")
        if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
            raise ValueError("step gains must be finite and nonnegative")
        strengths = [float(strength * gain) for gain in step_gains]
        if any(not isfinite(value) for value in strengths):
            raise ValueError("effective strengths must be finite")
    else:
        if step_gains is not None or not isinstance(gate_plan, LatentGatePlan):
            raise ValueError("Choose fixed Ming gains or one calibrated LatentGatePlan")
        prompt, embeds = inputs.get("prompt"), inputs.get("prompt_embeds")
        if (not strength or inputs.get("num_frames_per_prompt") != 2
                or inputs.get("num_images_per_prompt", 1) != 1
                or (isinstance(prompt, (list, tuple)) and len(prompt) != 1)
                or (embeds is not None and len(embeds) != 1)):
            raise ValueError("Gated Ming requires nonzero nominal strength and one ordered two-frame pair")
        import torch
        gate = gate_plan._new_gate()
        strengths = []

    active = ExitStack()
    capture_grid = gate_plan is not None or record_native_grid
    native_timesteps = []
    evaluations = 0
    scheduler = None
    sigmas = None
    widths = None
    decisions = []

    def check_native_interval(index):
        nonlocal scheduler, sigmas, widths
        if scheduler is None:
            scheduler = pipeline.scheduler
            if scheduler.order != 1 or scheduler.config.get("stochastic_sampling", False):
                raise RuntimeError("Ming grid capture requires its deterministic order-one native scheduler")
            sigmas = scheduler.sigmas.detach().cpu().tolist()
            widths = [start - end for start, end in pairwise(sigmas)]
            if (len(widths) != steps or len(scheduler.timesteps) != steps
                    or any(not isfinite(sigma) or sigma < 0 for sigma in sigmas)
                    or any(width <= 0 for width in widths[:-1]) or widths[-1] < 0 or not any(widths)):
                raise RuntimeError("Ming grid capture needs a descending sigma grid with an optional terminal repeat")
        if (pipeline.scheduler is not scheduler or scheduler.begin_index not in (None, 0)
                or scheduler.step_index != (index if index else None)):
            raise RuntimeError("Ming schedule must run before each original native solver interval")

    def native_gate(args, index):
        if len(args) < 2:
            raise RuntimeError("Expected Ming's positional pre-edit latents and native timestep")
        latents, timestep = args[:2]
        if (not isinstance(latents, list) or len(latents) not in (1, 2)
                or (len(latents) == 2 and not pipeline.do_classifier_free_guidance)
                or any(not isinstance(value, torch.Tensor) or value.ndim != 4
                       or value.shape[1] != 2 or min(value.shape) < 1 for value in latents)
                or (len(latents) == 2 and not torch.equal(latents[0], latents[1]))):
            raise RuntimeError("Expected one ordered Ming latent and its native CFG copy when active")
        native_time = (1000 - scheduler.timesteps[index]) / 1000
        if (not isinstance(timestep, torch.Tensor) or timestep.shape != (len(latents),)
                or not (timestep == native_time.to(timestep.device)).all()):
            raise RuntimeError("Ming denoiser timestep differs from the original solver interval")
        eligible = widths[index] > 0
        projection, probability, fired = None, None, False
        if eligible:
            projection = gate_plan._projection(latents[0], strength)
            decision = gate.step(projection)
            fired, probability = decision.fired, decision.probability
        value = float(strength) if fired else 0.0
        strengths.append(value)
        decisions.append(dict(interval=index, sigma_start=sigmas[index], sigma_end=sigmas[index + 1],
                              eligible=eligible, projection=projection, probability=probability,
                              fired=fired, effective_strength=value))

    def before(_module, args):
        nonlocal evaluations
        if evaluations != len(native_timesteps) or evaluations >= steps:
            raise RuntimeError("native denoiser order differs from the requested schedule")
        if capture_grid:
            check_native_interval(evaluations)
        if gate_plan is not None:
            native_gate(args, evaluations)
        active.enter_context(scale_adapter(strengths[evaluations]))
        evaluations += 1

    def after(_module, _args, _output):
        active.close()

    def step_end(owner, index, timestep, values):
        if owner is not pipeline or index != len(native_timesteps) or evaluations != index + 1:
            raise RuntimeError("native solver order differs from the requested schedule")
        native_timestep = float(timestep)
        if capture_grid and (pipeline.scheduler is not scheduler or scheduler.step_index != index + 1
                             or not isfinite(native_timestep) or native_timestep != float(scheduler.timesteps[index])):
            raise RuntimeError("Ming scheduler did not complete the original native interval")
        if record_native_grid and scheduler.sigmas[index:index + 2].detach().cpu().tolist() != sigmas[index:index + 2]:
            raise RuntimeError("Ming sigma grid changed during the native schedule")
        native_timesteps.append(native_timestep)
        if callback_on_step_end is not None:
            return callback_on_step_end(owner, index, timestep, values)
        return values

    with ExitStack() as request:
        request.callback(pipeline.transformer.register_forward_pre_hook(before).remove)
        request.callback(pipeline.transformer.register_forward_hook(after, always_call=True).remove)
        request.callback(active.close)
        result = pipeline(callback_on_step_end=step_end, **inputs)
    if len(native_timesteps) != steps or evaluations != steps:
        raise RuntimeError("native denoising did not complete the requested schedule")
    trace = {"native_timesteps": native_timesteps, "effective_strengths": strengths}
    if gate_plan is not None:
        usage = gate.usage
        trace.update(activation_gate=gate_plan._receipt(), decisions=decisions,
                     nominal_strength=float(strength), sigmas=sigmas, fired=usage.fired,
                     eligible=usage.eligible, gate_rate=usage.rate,
                     sigma_weighted_strength=sum(abs(value) * width for value, width in zip(strengths, widths, strict=True)))
    elif record_native_grid:
        trace["sigmas"] = sigmas
    return result, trace
