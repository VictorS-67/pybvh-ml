# pybvh-ml — Project Charter

## What pybvh-ml is

pybvh-ml is the machine learning bridge layer for pybvh. It owns the journey from "I have pybvh arrays" to "I have a training-ready batch." It provides the opinionated, ML-specific utilities that don't belong in a general-purpose BVH library.

## Core mission

**Turn motion capture data into training-ready inputs for skeleton-based ML models.**

pybvh-ml sits between pybvh (which parses and transforms BVH data) and the user's model (which consumes tensors). It handles tensor layout, preprocessing pipelines, on-the-fly augmentation, skeleton graph metadata, and dataset construction.

## Design principles

1. **numpy core, PyTorch extras.** All base functions take and return NumPy arrays. PyTorch-specific utilities (Dataset classes, tensor helpers, GPU augmentation) live in an optional submodule, imported only if PyTorch is installed. PyTorch is an optional dependency, never required.
2. **pybvh is the foundation.** pybvh-ml depends on pybvh for all BVH parsing, rotation math, and spatial transforms. It never reimplements what pybvh already provides. If pybvh-ml needs a primitive that doesn't exist in pybvh, the right answer may be to add it to pybvh first.
3. **Opinionated but composable.** pybvh-ml makes choices (CTV layout, specific augmentation pipelines) that serve the common case. But every component works standalone — users can use the packer without the Dataset, the augmentor without the preprocessor.
4. **Preprocessing and runtime are separate.** The "run once" preprocessing step (batch convert, normalize, export to disk) and the "every epoch" runtime step (on-the-fly augmentation in the data loader) are distinct modules with clean interfaces.
5. **No model-level constructs.** pybvh-ml provides the *data* that models need (edge lists, joint indices, feature arrays), not the model components themselves (graph convolution layers, attention masks). It stops at the model boundary.

## Code & API quality

Non-negotiable across every change to the codebase:

- **Intuitive API.** The public surface should be discoverable and obvious. Method names match what they do; signatures match how users will call them. If a user needs to read source code to figure out how to use something, the API itself needs work — not a docstring patch. When in doubt about a name or signature, prefer the form that reads naturally at the call site over the form that's easiest to implement.
- **Clear logic, clear code.** Reads top-to-bottom. Named intermediate variables over clever one-liners. Functions that do one thing. Comments only for the *why* (non-obvious constraints, subtle invariants, workarounds for specific bugs) — never the *what*, which well-named code already says.
- **Root-cause fixes, not band-aids.** When a bug surfaces, find the underlying cause and fix it there, even if the fix touches more files than the symptom. Avoid quick patches — special-case branches, suppressed warnings, `if this weird input then ...` guards — that mask the real problem and accumulate as scar tissue. If the proper fix is genuinely too large for the current change, document the trade-off explicitly in the commit message or a `# TODO:` rather than papering over it silently.
- **Name every convention choice, in the docstring.** Where an implementation picks one defensible option among several — a normalizer, a channel order, a padding value, a canonical layout, a sign convention, a fallback — say so where the *user* reads it, not in a code comment. Name what was chosen, name the alternative it was chosen over, and say when the two diverge. "We use X" is not enough: "we use X, the alternatives are Y and Z, and they differ when W" is what lets someone reconcile our arrays against the pipeline a paper describes, and tells them a mismatch is a convention difference rather than a bug. Whether the choice *also* needs a parameter:
    - **A published or widely-used alternative a consumer could reasonably need** → expose it (`standardize_length(method=)`, `preprocess_directory(center_root=)`, `pack_to_ctv` alongside `pack_to_tvc`).
    - **Forced by pybvh semantics or by internal consistency** → docstring only, and say why it is forced (root position first in every `(root_pos, joint_data)` pair, radians throughout, keyword-only augmentation arguments).
    - **A heuristic or fallback standing in for real skeleton information** → the caller must be able to tell which one they got. A return value that cannot distinguish "read from your skeleton" from "guessed from joint names" is the failure, and no docstring wording fixes it.

  `standardize_length` is the reference example of the first two: it names each `method`, and states outright that `resample_linear` is wrong for rotation arrays and what to use instead. `compute_normalization_stats` is the reference for the third: `constant_channels` in the returned dict is what tells the caller which channels got the `std → 1.0` guard instead of a measured std.


## What pybvh-ml owns

- **Tensor packing**: Converting pybvh's structured arrays (root_pos, joint_angles) into model-ready layouts (C,T,V), (T,V,C), flat (T,D), and back
- **Preprocessing pipelines**: Batch conversion of BVH directories to on-disk training datasets (HDF5, npz), with representation conversion and normalization
- **Runtime augmentation**: Fast, array-level augmentation functions designed for on-the-fly use inside data loaders (rotation, mirroring, speed perturbation, noise, dropout — operating on pre-extracted arrays without reconstructing Bvh objects)
- **Skeleton graph metadata**: Edge lists, body-part partitions, joint group indices — the topology data that GCN and Transformer models consume
- **PyTorch integration** (optional): Dataset / IterableDataset classes, collate functions for variable-length sequences, tensor conversion utilities
- **Feature metadata**: Column descriptors that tell users which channels correspond to which features in a packed array

## What pybvh-ml does NOT own

- **BVH parsing or writing** — that's pybvh
- **Rotation math** — that's pybvh
- **Forward kinematics** — that's pybvh
- **Motion analysis** (velocities, foot contacts, etc.) — that's pybvh
- **Model architectures** — that's the user's code
- **Training loops, optimizers, loss functions** — that's the user's code

## The boundary

pybvh-ml understands *how ML models consume skeleton data*. It does not understand *specific models or tasks*. An emotion recognition pipeline and a motion generation pipeline use the same pybvh-ml — the library provides the data plumbing, not the task logic.

## Dependency direction

```
User's model code
       │
       ▼
   pybvh-ml  (ML bridge layer)
       │
       ▼
    pybvh    (BVH foundation)
       │
       ▼
    NumPy
```

pybvh never imports or knows about pybvh-ml. pybvh-ml never imports or knows about the user's model.

## Release records: CHANGELOG vs internal session logs

Two records with different audiences, kept deliberately different (same convention as pybvh):

- **CHANGELOG.md is public-facing and shows only the net change per version.** Every entry describes the migration from the *previous shipped release* to this one. While a version is still unreleased, entries in its dated section are **rewritten in place** as the code evolves — never append churn: if a thing added during the version is later renamed, revised, or removed before shipping, the CHANGELOG shows only the final state, phrased so "previously" always refers to the last shipped release (verify against `git show v<prev>:...` when unsure). Dated sections of *shipped* versions are immutable and period-accurate.
- **`docs/internal_logs/<version>/` (gitignored) is the internal development history.** It records all substantive changes made during the version — including intermediate states that were overwritten before release — each with the *reason* for the change and for its supersession. When you rewrite a CHANGELOG entry per the rule above, the superseded state moves here (see the `NN-superseded-*.md` ledger pattern in `docs/internal_logs/v0.5.0/`). Update these logs as part of landing significant work, not retroactively at release time.
