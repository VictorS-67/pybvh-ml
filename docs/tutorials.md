# Tutorials

Runnable end-to-end Jupyter notebooks, from first preprocess to a trained classifier.

## Available tutorials

1. **[End-to-end pipeline](https://github.com/VictorS-67/pybvh-ml/blob/main/tutorials/01_end_to_end_pipeline.ipynb)** — BVH directory → `preprocess_directory` → `MotionDataset` with augmentation → tiny MLP classifier, training loop included.
2. **[Augmentation visualized](https://github.com/VictorS-67/pybvh-ml/blob/main/tutorials/02_augmentation_visualized.ipynb)** — every array-level augmentation (`rotate_vertical`, `mirror`, `speed_perturbation_arrays`, `dropout_arrays`, `add_joint_noise`) shown before/after on a real skeleton, plus pipeline composition and `set_epoch` reproducibility.
3. **[Heterogeneous preprocessing](https://github.com/VictorS-67/pybvh-ml/blob/main/tutorials/03_heterogeneous_preprocessing.ipynb)** — mixing skeletons, frame rates, and up-axes: `harmonize=True` + `skip_errors` + the representation-aware compatibility check as a robust ingest recipe.

A reader who finishes all three has walked the full library surface: preprocessing, packing, augmentation, and the PyTorch layer.

## Running locally

```bash
pip install "pybvh-ml[dev]"
cd tutorials/
jupyter notebook
```

Notebooks execute in CI via `pytest --nbmake tutorials/`, so they can't silently rot.

## Editing the tutorials (for contributors)

Each tutorial is a [Jupytext](https://jupytext.readthedocs.io/)-paired pair: a `.ipynb` file (the canonical rendered artifact, with outputs and plots) and a `.py` file in the [Percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format) (the plain-text source, git-friendly, reviewable as Python). Both files are committed.

To edit, change either side and sync:

```bash
jupytext --sync tutorials/*.ipynb
```

Jupytext picks the newer file by mtime and updates the other side. After editing code cells, re-execute so the committed outputs stay in sync with the source:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/*.ipynb
```

pybvh's tutorials page has the [full jupytext workflow write-up](https://victors-67.github.io/pybvh/tutorials/#editing-the-tutorials-for-contributors) (VS Code / Jupyter Lab editing modes); the same conventions apply here.
