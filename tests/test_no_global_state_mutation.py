"""Regression guard: pybvh-ml must not touch process-wide state.

Precision, determinism, threading and seeding are the *application's*
policy (charter design principle 6).  A data library that sets them
makes ``import pybvh_ml`` change a model's numbers — invisible at every
call site, and surviving every abstraction the application puts between
itself and us.

Two independent checks, because either alone has a blind spot:

* :func:`test_no_global_state_calls` scans the package's AST, so a
  mutation buried in a rarely-taken branch is caught without executing
  it.  It reads calls and assignment targets rather than source text, so
  documenting these names in a docstring does not trip it.
* :func:`test_import_changes_no_global_state` imports the package in a
  fresh interpreter and diffs the state it can observe, which catches a
  mutation made *through* something the scan doesn't know about.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = REPO_ROOT / "pybvh_ml"


# Function names that reconfigure the process rather than the caller's
# data.  Keyed by name alone (not the full dotted path) so an aliased
# import — ``from torch import manual_seed`` — is caught too.
BANNED_CALLS = {
    "set_float32_matmul_precision": "torch matmul precision is app policy",
    "use_deterministic_algorithms": "torch determinism is app policy",
    "set_deterministic_debug_mode": "torch determinism is app policy",
    "set_default_dtype": "the app owns torch's default dtype",
    "set_default_device": "the app owns torch's default device",
    "set_default_tensor_type": "the app owns torch's default tensor type",
    "set_num_threads": "threading is app policy",
    "set_num_interop_threads": "threading is app policy",
    "manual_seed": "seeds are passed in (seed=/rng=), never installed",
    "manual_seed_all": "seeds are passed in (seed=/rng=), never installed",
    "seed": "seeds are passed in (seed=/rng=), never installed",
    "set_grad_enabled": "autograd mode is the app's to set",
    "set_detect_anomaly": "autograd debug mode is the app's to set",
    "seterr": "numpy error policy is app-wide",
    "set_printoptions": "print options are app-wide",
    "simplefilter": "warning filters are app-wide",
    "filterwarnings": "warning filters are app-wide",
    "basicConfig": "logging configuration belongs to the app",
    "putenv": "the environment belongs to the app",
    "setrecursionlimit": "interpreter limits belong to the app",
    "setlocale": "locale is app-wide",
    "setdefaulttimeout": "socket defaults are app-wide",
}

# State that is *assigned to* rather than called, e.g.
# ``torch.backends.cudnn.benchmark = True`` or ``os.environ["OMP..."] =``.
# Matched per path segment rather than by dotted prefix, so an aliased
# import (``import torch as _torch``) cannot slip past.
BANNED_ASSIGN_SEGMENTS = {
    "backends": "global backend flags are app policy",
    "rcParams": "matplotlib rc state is app-wide",
    "environ": "the environment belongs to the app",
}


def _dotted(node: ast.AST) -> str:
    """Render an attribute/name chain as a dotted string, else ``""``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _package_sources() -> list[Path]:
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert files, f"no sources found under {PACKAGE_ROOT}"
    return files


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", ""))
            reason = BANNED_CALLS.get(name)
            if reason is not None:
                dotted = _dotted(node.func) or name
                found.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} calls "
                    f"{dotted}() — {reason}")
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            # ``a.b.c = x`` and ``a.b["c"] = x`` both reconfigure ``a.b``.
            dotted = _dotted(target.value if isinstance(target, ast.Subscript)
                             else target)
            for segment in dotted.split("."):
                reason = BANNED_ASSIGN_SEGMENTS.get(segment)
                if reason is not None:
                    found.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} assigns "
                        f"into {dotted} — {reason}")
    return found


def test_no_global_state_calls():
    """No source in the package reconfigures the process.

    ``rng.seed``-style method calls on an object the caller owns would
    also match ``seed``; none exist, and if one ever legitimately does,
    the fix is a narrow allowlist entry naming the receiver — not
    deleting the check.
    """
    violations = [v for path in _package_sources() for v in _violations(path)]
    assert not violations, (
        "pybvh-ml must not mutate process-wide state (charter design "
        "principle 6):\n  " + "\n  ".join(violations))


# Probe run in a fresh interpreter: snapshot the observable global state
# *after* importing numpy/torch (so their own import effects are
# excluded), then import pybvh-ml and diff.
_PROBE = r"""
import json, os, sys, warnings
import numpy as np

def numpy_state():
    return {
        "geterr": dict(np.geterr()),
        "printoptions_precision": np.get_printoptions()["precision"],
        # A library that seeded or drew from the legacy global RNG at
        # import time would move this.
        "global_rng": np.random.get_state()[1][:8].tolist(),
    }

def torch_state(torch):
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "num_threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "grad_enabled": torch.is_grad_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "initial_seed": torch.initial_seed(),
    }

def env_state():
    return {"environ": dict(os.environ),
            "warning_filters": [str(f) for f in warnings.filters]}

try:
    import torch
except ImportError:
    torch = None

before = {"numpy": numpy_state(), "env": env_state()}
if torch is not None:
    before["torch"] = torch_state(torch)

import pybvh_ml
if torch is not None:
    import pybvh_ml.torch

after = {"numpy": numpy_state(), "env": env_state()}
if torch is not None:
    after["torch"] = torch_state(torch)

print(json.dumps({"before": before, "after": after,
                  "had_torch": torch is not None}))
"""


def test_import_changes_no_global_state():
    """Importing the package (and its torch subpackage) changes nothing."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    before, after = result["before"], result["after"]
    differing = sorted(key for key in after if before[key] != after[key])
    detail = "\n".join(
        f"  {key}: {before[key]} -> {after[key]}" for key in differing)
    assert not differing, (
        "importing pybvh_ml changed process-wide state (charter design "
        "principle 6):\n" + detail)
    if not result["had_torch"]:
        pytest.skip("torch not installed — numpy/env halves checked only")
