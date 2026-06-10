# packages/ragcore/tests/test_config.py
import os
import subprocess
import sys


def test_config_import_does_not_pull_torch():
    """Importing ragcore.config in a fresh interpreter must not import torch.

    Load-bearing invariant: rag-api is CPU-only and must never drag in torch
    via the shared config. Run in a subprocess so it can't be fooled by torch
    already being cached in this test process's sys.modules.
    """
    code = (
        "import ragcore.config, sys; "
        "assert 'torch' not in sys.modules, 'torch was imported by ragcore.config'; "
        "print('OK')"
    )
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "packages")}
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK" in r.stdout


def test_embedding_model_resolver_defaults():
    from ragcore import config
    assert config.embedding_model_id()  # non-empty string
