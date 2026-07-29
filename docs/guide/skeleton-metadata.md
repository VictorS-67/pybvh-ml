# Skeleton Graph Metadata

GCNs need an adjacency structure; Transformers need joint groupings; mirroring needs left/right pairs. pybvh-ml extracts this topology data from a `Bvh` object — the *data* models consume, never the model components themselves.

## The extractors

```python
import pybvh_ml

edges = pybvh_ml.get_edge_list(bvh)            # [(child, parent), ...]
lr_pairs = pybvh_ml.get_lr_pairs(bvh)          # [(left, right), ...] index pairs
partitions = pybvh_ml.get_body_partitions(bvh) # {"torso": [...], "left_arm": [...], ...}
```

- **`get_edge_list`** — parent/child joint index pairs, the input to any GCN adjacency matrix.
- **`get_lr_pairs`** — left/right joint index pairs, detected from joint names via pybvh's L/R heuristics (`LeftArm`/`RightArm`, `LArm`/`RArm`, `arm_l`/`arm_r`, …). This is what [`mirror`](augmentation.md#the-six-functions) consumes.
- **`get_body_partitions`** — named body-part groups (torso, arms, legs, head) as joint index lists, for part-based pooling or attention masks.

![A skeleton with joints color-coded by body-part partition: torso, head, left and right arms and legs, each with its joint count in the legend](../gallery/img/skeleton-partitions.png)

*`get_body_partitions` on a fixture skeleton — each named group colored. ([Gallery](../gallery/index.md) for the edge/L-R-pair figure too.)*

## All-in-one: `get_skeleton_info`

```python
info = pybvh_ml.get_skeleton_info(bvh)
# {"edges", "lr_pairs", "lr_mapping", "joint_names", "euler_orders",
#  "num_joints", "world_up", "rest_forward", "rest_up"}

info = pybvh_ml.get_skeleton_info(bvh, include_partitions=True)  # + "body_partitions"
```

The dict is JSON-serializable and carries everything downstream code needs:

- `edges`, `lr_pairs`, `lr_mapping`, `body_partitions` — graph topology.
- `joint_names`, `euler_orders`, `num_joints` — array bookkeeping (e.g. `euler_orders` for euler-representation [augmentation](augmentation.md) or [conversion](../api/convert.md)).
- `world_up`, `rest_forward`, `rest_up` — the axis strings that configure augmentation (`up_axis=` for `rotate_vertical` / `AugmentationPipeline.standard`). `rest_up` is `None` for degenerate rigs.

`preprocess_directory` stores exactly this dict in the dataset file, which is why a preprocessed dataset is self-sufficient: `AugmentationPipeline.standard(data["skeleton_info"], ...)` works without reopening any source BVH.

## Where the boundary is

pybvh-ml hands you index lists — it does not build adjacency matrices, graph convolution layers, or attention masks. Those are two lines of your model code, and their conventions (self-loops? normalized? partitioned adjacency à la ST-GCN?) belong to the model, not the data layer:

```python
import numpy as np

A = np.zeros((info["num_joints"] + 1,) * 2)   # +1: root is vertex 0 in packed layouts
for child, parent in info["edges"]:
    A[child + 1, parent + 1] = A[parent + 1, child + 1] = 1
```

Mind the off-by-one: `edges` uses joint indices (`0..J-1`), while [packed layouts](tensor-layouts.md#the-three-layouts) put the root at vertex 0 and joints at `1..J`.

## See also

- [Skeleton Metadata API](../api/skeleton.md) — full signatures
- [Tensor Layouts & Packing](tensor-layouts.md) — the vertex indexing the graph must match
