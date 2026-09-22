"""In-repo train entry for Ming design-layer sliders.

``python scripts/train_ming.py`` calls ``main`` here. ``--dummy`` never
downloads Hub weights. ``--live`` hits the stub in ``ming.live`` and
still does not download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ming.defaults import (
    CONTROL_PROMPT,
    DEFAULT_CONFIG,
    DEFAULT_PROMPTS,
    DTYPE,
    GUIDANCE,
    HOLD_WEIGHT,
    HUB_SUBFOLDERS,
    MODEL_ID,
    NUM_LAYERS_EXAMPLE,
    RESOLUTION,
    RESOLUTION_FAST,
    RESOLUTION_NOTE,
    SAMPLE_SCALES,
    SAMPLE_STEPS,
    TASK,
    VRAM_GIB,
)
from ming.dummy import DummyBackend, step_loss, write_sample_grid
from ming.formulation import formulation_record, load_winning
from ming.live import load_plan
from ming.prompts import load_prompts

REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_ming_only(model_id: str) -> None:
    if str(model_id) != MODEL_ID:
        raise ValueError(
            "Ming-Image-0.1-Design-Layer only. "
            f"Expected {MODEL_ID}, got {model_id!r}."
        )


def _formatter(prog: str) -> argparse.ArgumentDefaultsHelpFormatter:
    return argparse.ArgumentDefaultsHelpFormatter(prog, width=120, max_help_position=32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=_formatter,
        description=(
            "Train a particle slider on inclusionAI/Ming-Image-0.1-Design-Layer "
            f"(task {TASK}). Card: {SAMPLE_STEPS} steps, guidance {GUIDANCE:g}, "
            f"resolution {RESOLUTION} (or {RESOLUTION_FAST}). "
            "Architecture is gmix via particle_sliders.winning_formulation(). "
            "--dummy never downloads Hub weights."
        ),
    )
    parser.add_argument("--name", type=str, default="layer-ming-design")
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help="Hub repo (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TASK,
        help="Ming task (default: %(default)s)",
    )
    parser.add_argument("--prompts_file", type=str, default=DEFAULT_PROMPTS)
    parser.add_argument("--config_file", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--save_dir", type=str, default="models/ming-design-slider")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        help=f"working-resolution bucket (default: %(default)s; fast bucket {RESOLUTION_FAST})",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=SAMPLE_STEPS,
        help="card sampling steps (default: %(default)s)",
    )
    parser.add_argument(
        "--sample_guidance",
        type=float,
        default=GUIDANCE,
        help="card CFG / guidance_scale (default: %(default)s)",
    )
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS_EXAMPLE)
    parser.add_argument("--dtype", type=str, default=DTYPE)
    parser.add_argument("--hold_weight", type=float, default=HOLD_WEIGHT)
    parser.add_argument("--dummy", action="store_true", help="CPU backend, 2 steps, no Hub weights")
    parser.add_argument(
        "--live",
        action="store_true",
        help="request the live loader (stubbed; still does not download weights)",
    )
    parser.add_argument(
        "--allow_hub",
        action="store_true",
        help="recorded on the sidecar; this scaffold still does not download",
    )
    parser.add_argument("--control_prompt", type=str, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--load_lora",
        type=str,
        default=None,
        help="skip the train loop and write the sample grid for this adapter",
    )
    parser.add_argument("--print_card", action="store_true", help="print the Ming card and exit")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _prompts_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.exists() else path


def live_train_card(args: argparse.Namespace, stamp_record: dict) -> dict:
    return {
        "backend": "ming_design_layer",
        "model_id": args.model_id,
        "task": args.task,
        "hub_subfolders": list(HUB_SUBFOLDERS),
        "sample_steps": int(args.sample_steps),
        "sample_guidance": float(args.sample_guidance),
        "resolution": int(args.resolution),
        "resolution_fast": RESOLUTION_FAST,
        "resolution_note": RESOLUTION_NOTE,
        "dtype": args.dtype,
        "vram_gib": VRAM_GIB,
        "num_layers": int(args.num_layers),
        "formulation": stamp_record,
        "weights_in_repo": False,
        "non_goals": [
            "no Hub weight download in this scaffold",
            "no vendored RoutedMLP or GradRegularizer",
            "concept-slider-core is not the product base",
        ],
    }


def train(args: argparse.Namespace) -> dict | Path:
    assert_ming_only(args.model_id)
    if args.task != TASK:
        raise ValueError(f"task must be {TASK}, got {args.task!r}")
    stamp = load_winning()
    record = formulation_record(stamp)
    if int(record["adapter_rank"]) != int(stamp.spec["adapter_rank"]):
        raise RuntimeError("formulation record drifted from winning_formulation()")
    if args.print_card:
        card = live_train_card(args, record)
        print(json.dumps(card, indent=2))
        return card
    if args.live and args.dummy:
        raise ValueError("Choose either --live or --dummy")
    if args.live:
        from ming.live import load_ming_pipeline

        load_plan(args.model_id)
        load_ming_pipeline(args.model_id)
        raise RuntimeError("live loader returned without raising")
    if not args.dummy:
        raise RuntimeError(
            "This entry runs the in-repo CPU UNI loop with --dummy. "
            "The live loader is ming.live.load_ming_pipeline "
            f"({MODEL_ID}, task {TASK}, subfolders {', '.join(HUB_SUBFOLDERS)}, "
            f"{SAMPLE_STEPS} steps, guidance {GUIDANCE:g}, {RESOLUTION_NOTE}) "
            "A non-dummy run is refused so this process does not download Hub weights."
        )

    prompts, meta = load_prompts(_prompts_path(Path(args.prompts_file)))
    steps = min(int(args.steps), 2)
    backend = DummyBackend(seed=int(args.seed))
    skip_train = bool(args.load_lora)
    if skip_train:
        backend.load_lora(args.load_lora)
        steps = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.name}_train.jsonl"
    guidance = float(args.sample_guidance)
    control_prompt = str(args.control_prompt or meta.control_prompt or CONTROL_PROMPT)
    last_stats: dict[str, float] = {}
    params = backend.trainable_parameters()
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        z = backend.sample_latents()
        loss, stats = step_loss(
            backend,
            prompt,
            z,
            guidance=guidance,
            hold_weight=float(args.hold_weight),
        )
        loss.backward()
        lr = float(stamp.spec["g_lr"])
        for param in params:
            param.data -= lr * float(param.grad)
            param.grad = 0.0
        last_stats = stats
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **stats}) + "\n")

    shots = write_sample_grid(
        backend,
        prompts,
        save_dir,
        control_prompt=control_prompt,
        guidance=guidance,
        sample_steps=int(args.sample_steps),
        sample_seed=int(args.sample_seed),
    )
    meta_path = save_dir / "samples" / "final_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "gate": "layer-plan",
                "prompts": [prompt.neutral for prompt in prompts] + [control_prompt],
                "samples": shots,
                "control_prompt": control_prompt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plan = load_plan(args.model_id)
    sidecar = {
        "kind": "ming_design_layer",
        "name": args.name,
        "model_id": args.model_id,
        "task": args.task,
        "hub_subfolders": list(HUB_SUBFOLDERS),
        "rank": int(stamp.spec["adapter_rank"]),
        "rank_source": "winning_formulation",
        "resolution": int(args.resolution),
        "resolution_fast": RESOLUTION_FAST,
        "resolution_note": RESOLUTION_NOTE,
        "dtype": str(args.dtype),
        "vram_gib": VRAM_GIB,
        "num_layers": int(args.num_layers),
        "sample_steps": int(args.sample_steps),
        "sample_guidance": guidance,
        "train_guidance": guidance,
        "hold_weight": float(args.hold_weight),
        "dummy": True,
        "allow_hub": bool(args.allow_hub),
        "weights_downloaded": False,
        "weights_in_repo": False,
        "load_plan": plan,
        "formulation": record,
        "plus_label": meta.plus_label,
        "minus_label": meta.minus_label,
        "concept_words": meta.concept_words,
        "control_prompt": control_prompt,
        "bare_captions": bool(meta.bare_captions),
        "config_file": args.config_file,
        "prompts_file": args.prompts_file,
        "load_lora": args.load_lora,
        "skipped_train": skip_train,
        "minus_teacher": False,
        "last": last_stats,
        "sample_grid": {
            "scales": list(SAMPLE_SCALES),
            "count": len(shots),
            "dir": "samples",
        },
    }
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar_path


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
