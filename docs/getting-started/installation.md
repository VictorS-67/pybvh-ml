# Installation

## From PyPI

```bash
pip install pybvh-ml
```

This pulls in [pybvh](https://victors-67.github.io/pybvh/) `>= 0.8.1, < 0.9` automatically — pybvh-ml 0.5 tracks pybvh 0.8's API (short representation tokens, radians-first parameters).

## Optional extras

```bash
pip install "pybvh-ml[torch]"    # PyTorch Dataset classes
pip install "pybvh-ml[hdf5]"     # HDF5 output support
```

PyTorch is never required: all core functions take and return NumPy arrays, and `pybvh_ml.torch` is imported only if PyTorch is installed. HDF5 support is likewise lazy — `.npz` output works out of the box.

## Requirements

- Python >= 3.9
- [pybvh](https://github.com/VictorS-67/pybvh) >= 0.8.1, < 0.9
- NumPy >= 1.21

Optional: PyTorch >= 2.0, h5py >= 3.0.

!!! info "See also"
    [Quick Start](quickstart.md) — from a BVH directory to a training batch · [Find a function](../api/index.md) — the capability map
