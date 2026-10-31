"""Build a coherent opt-in wheel for the exact reviewed core draft candidate.

This builds only; it never installs dependencies, downloads models or publishes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-wheel', type=Path, help='Optionally verify the original archived qualification wheel')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    lock = json.loads((ROOT / 'ming/native-candidate.lock.json').read_text())
    if args.core_wheel is not None and hashlib.sha256(args.core_wheel.read_bytes()).hexdigest() != lock['shared_wheel_sha256']:
        raise SystemExit('core wheel does not match the exact native candidate artifact')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ming-native-build-') as directory:
        staging = Path(directory)
        shutil.copytree(ROOT / 'ming', staging / 'ming', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('comfy_ming.py', 'LICENSE', 'NOTICE', 'README.md'):
            shutil.copy2(ROOT / name, staging / name)
        text = (ROOT / 'pyproject.toml').read_text()
        current = tomllib.loads(text)['project']['version']
        text = text.replace(f'version = "{current}"', f'version = "{current}+native012"', 1)
        (staging / 'pyproject.toml').write_text(text)
        (staging / 'requirements.txt').write_text(
            f'particle-sliders-core @ git+{lock["repository"]}.git@{lock["commit"]}#subdirectory={lock["subdirectory"]}\n'
            'particlegan==0.9.4\n'
            f'hndl=={lock["hndl_version"]}\n'
            f'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@{lock["hypergan_commit"]}\n'
            'packaging>=23,<27\n')
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                        '--outdir', str(output), str(staging)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
