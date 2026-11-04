"""Trace real old-preview packets and reject them in the current PG094 candidate."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
NEW = 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('produce-preview', 'consume'))
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--producer-receipt', type=Path)
    args = parser.parse_args(argv)
    import torch
    torch.default_generator.manual_seed(319)
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_native import fixture, row, equal, COHORT, NATIVE
    identity = COHORT.product_identity()
    lock = COHORT.require_cohort()
    _, _, adapter = fixture()
    try:
        rows = [adapter.prepare(row(value)) for value in (0., .3)]
        game = NATIVE.build_hndl_game(adapter, rows, torch.tensor([[0., .25], [.3, .25]]), total_steps=4, critic_width=4)
        receipt = dict(schema='ming-modern-cross-cohort-v1', action=args.action,
                       evidence_kind='New tiny CPU fixture from separate installed cohorts; not a user or pretrained checkpoint.',
                       product=identity['distribution'], product_version=identity['version'], profile=identity['profile'],
                       core_commit=lock['commit'], core_version=distribution('particle-sliders-core').version,
                       pg_version=distribution('particlegan').version, torch=str(torch.__version__),
                       reproducer_sha256=sha(Path(__file__)), fixture_source_sha256=sha(ROOT / 'tests/test_native.py'))
        if args.action == 'produce-preview':
            if lock['commit'] != OLD or identity['version'] != '0.1.1.dev12':
                raise RuntimeError('Preview production requires the original dev12 environment with core at 4abc0eb.')
            game.step()
            packet = game.state_dict()
            args.packet.parent.mkdir(parents=True, exist_ok=True)
            torch.save(packet, args.packet)
            receipt.update(packet_sha256=sha(args.packet), packet_pg_version=packet['config']['pg_version'], completed_steps=packet['completed_steps'])
        else:
            if lock['commit'] != NEW or identity['version'] != '0.2.0.dev1' or args.producer_receipt is None:
                raise RuntimeError('consumption requires the current ddb41423 candidate and producer receipt')
            producer = json.loads(args.producer_receipt.read_text())
            if producer['core_commit'] not in (OLD, '3c25f785d14073e6f627c0fa9d2297a041c79913'):
                raise RuntimeError('unrecognized historical producer cohort')
            if producer['packet_sha256'] != sha(args.packet):
                raise RuntimeError('packet bytes differ from producer receipt')
            packet = torch.load(args.packet, weights_only=True)
            if packet['config']['pg_version'] != producer['packet_pg_version']:
                raise RuntimeError('packet header differs from its real producer')
            before = deepcopy(game.state_dict())
            try:
                game.load_state_dict(packet)
            except ValueError as exc:
                if 'original pinned' not in str(exc): raise
                equal(before, game.state_dict())
                receipt.update(packet_sha256=sha(args.packet), producer_receipt_sha256=sha(args.producer_receipt),
                               producer_core_commit=producer['core_commit'], producer_product_version=producer['product_version'],
                               rejection=str(exc), full_game_state_unchanged=True, packet_relabelled=False)
            else:
                raise AssertionError('old cohort packet must fail closed')
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))
    finally:
        adapter.close()
    return 0


if __name__ == '__main__': raise SystemExit(main())
