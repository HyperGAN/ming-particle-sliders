"""Per-step strength on a caller-owned native denoising pipeline."""

from contextlib import ExitStack
from math import isfinite
from sys import float_info


def render(pipeline, *, scale_adapter, step_gains, strength=1.0,
           callback_on_step_end=None, **inputs):
    """Sample with an attached RAW adapter and return the output and step trace."""
    if type(strength) not in (int, float) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("nominal strength must be finite")
    steps = inputs.get("num_inference_steps")
    if (type(steps) is not int or steps < 1 or not isinstance(step_gains, (list, tuple))
            or len(step_gains) != steps):
        raise ValueError("one gain per requested native denoising step is required")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("step gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not isfinite(value) for value in strengths):
        raise ValueError("effective strengths must be finite")

    active = ExitStack()
    native_timesteps = []
    evaluations = 0

    def before(_module, _args):
        nonlocal evaluations
        if evaluations != len(native_timesteps) or evaluations >= steps:
            raise RuntimeError("native denoiser order differs from the requested schedule")
        active.enter_context(scale_adapter(strengths[evaluations]))
        evaluations += 1

    def after(_module, _args, _output):
        active.close()

    def step_end(owner, index, timestep, values):
        if owner is not pipeline or index != len(native_timesteps) or evaluations != index + 1:
            raise RuntimeError("native solver order differs from the requested schedule")
        native_timesteps.append(float(timestep))
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
    return result, {"native_timesteps": native_timesteps, "effective_strengths": strengths}
