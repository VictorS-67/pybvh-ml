# Skeleton Graph Metadata

GCNs need an adjacency structure; Transformers need joint groupings; mirroring needs left/right pairs. pybvh-ml extracts this topology data from a `Bvh` object — the *data* models consume, never the model components themselves.

## The extractors

```python
import pybvh_ml

edges = pybvh_ml.get_edge_list(bvh)            # [(child, parent), ...]
lr_pairs = pybvh_ml.get_lr_pairs(bvh)          # [(left, right), ...] index pairs
partitions = pybvh_ml.get_body_partitions(bvh) # {"torso": [...], "left_arm": [...], ...}

node_edges = pybvh_ml.get_edge_list(bvh, include_end_sites=True)
node_pairs = pybvh_ml.get_node_lr_pairs(bvh)   # node-space L/R, end sites included
```

- **`get_edge_list`** — parent/child joint index pairs, the input to any GCN adjacency matrix. `include_end_sites=True` switches to node space.
- **`get_lr_pairs`** — left/right joint index pairs, detected from joint names via pybvh's L/R heuristics (`LeftArm`/`RightArm`, `LArm`/`RArm`, `arm_l`/`arm_r`, …). This is what [`mirror`](augmentation.md#the-eight-functions) consumes for `joint_rot` and `joint_pos`.
- **`get_node_lr_pairs`** — the same pairing in node space, covering joints *and* their end sites. What `mirror` needs for a `node_pos` stream, so the fingertips swap sides too.
- **`get_body_partitions`** — named body-part groups (torso, arms, legs, head) as joint index lists, for part-based pooling or attention masks.

## Two index spaces, and they are not interchangeable

*Joint* space (`bvh.joint_angles` order, `J` entries) excludes end sites; *node* space (`bvh.nodes` order, `N >= J` entries) includes them. Node indices diverge from joint indices as soon as any end site precedes a paired joint in file order, so a list from one space silently addresses the wrong vertices in the other — no shape error, just wrong answers. Every key says which space it is in, and `fk_topology["joint_idx"]` is the map between them:

```python
joint_idx = np.asarray(info["fk_topology"]["joint_idx"])
joint_idx >= 0                               # node → is it a joint?
node_positions[:, joint_idx >= 0]            # exactly bvh.joint_positions()
```

![A skeleton with joints color-coded by body-part partition: torso, head, left and right arms and legs, each with its joint count in the legend](../gallery/img/skeleton-partitions.png)

*`get_body_partitions` on a fixture skeleton — each named group colored. ([Gallery](../gallery/index.md) for the edge/L-R-pair figure too.)*

## All-in-one: `get_skeleton_info`

```python
info = pybvh_ml.get_skeleton_info(bvh)
# joint space: {"edges", "lr_pairs", "lr_mapping", "joint_names",
#               "euler_orders", "num_joints"}
# node space:  {"num_nodes", "node_names", "node_edges", "node_lr_pairs",
#               "end_site_indices"}
# plus:        {"fk_topology", "mismatched_end_site_pairs",
#               "world_up", "rest_forward", "rest_up"}

info = pybvh_ml.get_skeleton_info(bvh, include_partitions=True)  # + "body_partitions"
```

The dict is JSON-serializable and carries everything downstream code needs:

- `edges`, `lr_pairs`, `lr_mapping`, `body_partitions` — joint-space graph topology.
- `num_nodes`, `node_names`, `node_edges`, `node_lr_pairs`, `end_site_indices` — the same in node space, for a `node_pos` stream.
- `joint_names`, `euler_orders`, `num_joints` — array bookkeeping (e.g. `euler_orders` for euler-representation [augmentation](augmentation.md) or [conversion](../api/convert.md)).
- `world_up`, `rest_forward`, `rest_up` — the axis strings that configure augmentation (`up_axis=` for `rotate_vertical` / `AugmentationPipeline.standard`). `rest_up` is `None` for degenerate rigs.
- `fk_topology` — bone offsets, parent indices, joint-column indices and Euler orders: everything forward kinematics reads, as plain lists.
- `mismatched_end_site_pairs` — see below.

`preprocess_directory` stores exactly this dict in the dataset file, which is why a preprocessed dataset is self-sufficient: `AugmentationPipeline.standard(data["skeleton_info"], ...)` works without reopening any source BVH.

## Running forward kinematics at train time

`fk_topology` is what makes the [FK refresh](augmentation.md#positions-and-rotations-together-the-fk-refresh) possible with the source `.bvh` long closed. Rebuild it once per dataset — pybvh's constructor validates, which is not free — and hand it to the augmentation step:

```python
topology = pybvh_ml.build_fk_topology(data["skeleton_info"])
pipeline = AugmentationPipeline.standard(data["skeleton_info"])  # wires it for you
```

Datasets written before 0.6.0 carry no `fk_topology` and it is not recoverable from the other keys — the bone offsets are stored nowhere else — so `build_fk_topology` raises rather than guessing. Re-run `preprocess_directory`.

## The end-site pairing check

`node_lr_pairs` **drops** a pair's end sites when the two sides carry different numbers of them: pybvh has no well-defined tip correspondence there, and its property filters rather than raises, matching `lr_pairs`. That policy is right for pybvh and wrong for us if left unchecked — we persist the pair list and mirror at *train* time, far from any `Bvh`, so a dropped tip silently produces exactly the half-swapped skeleton `pybvh.transforms.mirror` refuses to emit.

So `get_skeleton_info` records the offending pairs while the `Bvh` is still open:

```python
info["mismatched_end_site_pairs"]   # [] on a well-formed rig; node-space pairs otherwise
pybvh_ml.find_mismatched_end_site_pairs(bvh)   # the same check, standalone
```

A non-empty list means a node-space mirror would swap those paired joints but leave their end sites on the original side. Fix the rig, or drop to joint space.

## Where the boundary is

pybvh-ml hands you index lists — it does not build adjacency matrices, graph convolution layers, or attention masks. Those are two lines of your model code, and their conventions (self-loops? normalized? partitioned adjacency à la ST-GCN?) belong to the model, not the data layer:

```python
import numpy as np

A = np.zeros((info["num_joints"] + 1,) * 2)   # +1: root is vertex 0 in packed layouts
for child, parent in info["edges"]:
    A[child + 1, parent + 1] = A[parent + 1, child + 1] = 1
```

### Which key indexes which packing

The `+ 1` above is not universal — it depends on what you packed:

| packing | `V` | edges that index it directly |
|---|---|---|
| `streams=("joint_pos",)` | `J` | `edges` — one to one, no shift |
| `streams=("node_pos",)` | `N` | `node_edges` — one to one, no shift |
| anything including `"root_pos"` | `1 + J` | `edges`, shifted: `(child + 1, parent + 1)` |

The default `("root_pos", "joint_rot")` is the third row, which is where the off-by-one comes from: `edges` uses joint indices `0..J-1` while the packed layout puts the root at vertex 0. Pack `("joint_pos",)` and there is no off-by-one to mind. The same applies to the L/R pair lists.

## See also

- [Skeleton Metadata API](../api/skeleton.md) — full signatures
- [Tensor Layouts & Packing](tensor-layouts.md) — the vertex indexing the graph must match
