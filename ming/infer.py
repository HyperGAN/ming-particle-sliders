"""Sample-only entry. Same trainer, with ``--load_lora`` required."""

from __future__ import annotations

from ming.train import build_parser, train


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.description = (
        "Sample a Ming design-layer slider from this repo. "
        "Uses the train entrypoint with --load_lora, which skips the "
        "train loop and writes the layer-plan grid. Architecture is gmix "
        "via particle_sliders.winning_formulation(). This scaffold does "
        "not download Hub weights. Pass --dummy for the CPU stand-in."
    )
    args = parser.parse_args(argv)
    if not args.load_lora and not args.print_card:
        parser.error(
            "--load_lora PATH is required. It skips training and writes "
            "the sample grid. Pass --dummy for the CPU stand-in."
        )
    if not args.dummy and not args.print_card:
        parser.error(
            "CPU infer needs --dummy. The live loader is ming.live and is "
            "stubbed, so Hub weights are not downloaded."
        )
    train(args)


if __name__ == "__main__":
    main()
