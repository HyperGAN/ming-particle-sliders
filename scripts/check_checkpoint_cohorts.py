"""Produce a real PG0.9.3 CPU fixture packet and reject it in the native engine.

Run produce and consume in separate selected environments. Only receipts belong
in source control; the tiny numerical packet is a local test artifact.
"""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('produce', 'consume'))
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--producer-receipt', type=Path)
    args = parser.parse_args(argv)
    import torch
    torch.default_generator.manual_seed(319)
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_native import fixture, row, equal, COHORT, NATIVE
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.field_critic import FieldCritic
    identity = COHORT.product_identity()
    lock = COHORT.require_cohort()
    _, _, adapter = fixture()
    try:
        rows = [adapter.prepare(row(value)) for value in (0., .3)]
        conditions = torch.tensor([[0., .25], [.3, .25]])
        receipt = dict(schema='ming-checkpoint-cross-cohort-evidence-v1', action=args.action,
                       evidence_kind='fresh tiny CPU fixture using actual released engines in separate processes; no historical user/pretrained checkpoint',
                       product=identity['distribution'], product_version=identity['version'],
                       core_version=distribution('particle-sliders-core').version,
                       core_commit=lock['commit'], pg_version=distribution('particlegan').version,
                       torch=str(torch.__version__), python=sys.version.split()[0], fixture_seed=319,
                       reproducer_sha256=sha(Path(__file__)), fixture_source_sha256=sha(ROOT / 'tests/test_native.py'))
        if args.action == 'produce':
            if identity['profile'] != 'public-0.11' or receipt['pg_version'] != '0.9.3':
                raise RuntimeError('produce requires the actual public core0.11/PG0.9.3 environment')
            game = ConditionalPG09(adapter, rows, FieldCritic(2, 2, width=4), conditions=conditions,
                                   total_steps=4, response_domain='continuous_field')
            game.step()
            packet = game.state_dict()
            args.packet.parent.mkdir(parents=True, exist_ok=True)
            torch.save(packet, args.packet)
            receipt.update(packet_sha256=sha(args.packet), completed_steps=packet['completed_steps'],
                           packet_pg_version=packet['config']['pg_version'],
                           packet_pg_source_profile=packet['config']['pg_source_profile'],
                           packet_pg_sources_sha256=packet['config']['pg_sources_sha256'])
        else:
            if identity['profile'] != 'native-candidate-0.12' or args.producer_receipt is None:
                raise RuntimeError('consume requires native product and --producer-receipt')
            producer = json.loads(args.producer_receipt.read_text())
            if (producer['action'] != 'produce' or producer['pg_version'] != '0.9.3' or
                    producer['packet_sha256'] != sha(args.packet)):
                raise RuntimeError('external packet bytes do not match the producer receipt')
            packet = torch.load(args.packet, weights_only=True)
            if packet['config']['pg_version'] != producer['packet_pg_version']:
                raise RuntimeError('external packet header contradicts its real producer')
            game = NATIVE.build_hndl_game(adapter, rows, conditions, total_steps=4, critic_width=4)
            before = deepcopy(game.state_dict())
            try:
                game.load_state_dict(packet)
            except ValueError as exc:
                if 'original pinned' not in str(exc):
                    raise
                equal(before, game.state_dict())
                receipt.update(packet_sha256=sha(args.packet), producer_receipt_sha256=sha(args.producer_receipt),
                               producer_pg_version=producer['pg_version'], rejection=str(exc),
                               full_game_state_unchanged=True)
            else:
                raise AssertionError('real external PG0.9.3 packet must refuse')
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))
    finally:
        adapter.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
