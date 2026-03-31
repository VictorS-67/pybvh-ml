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
