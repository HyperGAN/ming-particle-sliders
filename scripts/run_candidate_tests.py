"""Test installed ming_native without collecting the stable Comfy root package."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='ming-native-tests-') as directory:
        target = Path(directory) / 'tests'
        target.mkdir()
        for name in ('test_native.py', 'test_hndl_candidate.py', 'test_ordered_frames.py', 'test_artifacts.py'):
            original = ROOT / 'tests' / name
            copied = target / name
            shutil.copy2(original, copied)
            if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(copied.read_bytes()).digest():
                raise RuntimeError('candidate test copy changed bytes: ' + name)
        env = dict(os.environ, MING_SOURCE_ROOT=str(ROOT), HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        return subprocess.run([sys.executable, '-I', '-m', 'pytest', '-q', '-c', '/dev/null',
                               '--rootdir', directory, '--import-mode=importlib', str(target)], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
