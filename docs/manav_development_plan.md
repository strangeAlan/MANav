# MANav Development Plan

This document records the planned code changes for developing MANav from the
current UniGoal codebase. It is a planning document only. It should be updated
before large implementation steps, and implementation changes should be kept in
separate git branches.

## Current Starting Point

The base code is the public UniGoal repository:

- Upstream remote: `origin -> https://github.com/bagh2178/UniGoal.git`
- Project remote: `manav -> git@github.com:strangeAlan/MANav.git`
- Current public UniGoal release supports HM3D instance-image-goal navigation
  and text-goal navigation.
- The public code does not currently expose a complete ObjectNav-MP3D command
  matching the UniGoal paper table result `ObjNav-MP3D SR 41.0 / SPL 16.4`.

Local runtime support is included for using a local Qwen service:

- Qwen server: `http://127.0.0.1:18080`
- UniGoal local config: `configs/config_local_qwen.yaml`
- Start script: `script/start_local_vlm.sh`
- Smoke script: `script/run_local_qwen_smoke.sh`
- Server script: `script/qwen_backend_server.py`

The Qwen runtime changes are infrastructure only. They should be committed on a
small branch before ObjectNav or RGB-only work begins. Deployment details are in
`docs/local_qwen_deployment.md`.

## Research Goal

The project goal is not to exactly reproduce the UniGoal paper numbers first.
The goal is to build a multi-agent navigation framework that keeps UniGoal's
stronger navigation and graph structure while replacing online RGB-D perception
with RGB-only reconstruction.

Target direction:

- Online observations should be RGB-only.
- VGGT or an equivalent RGB-only reconstruction module should provide
  pseudo-depth, camera geometry, and object-level 3D grounding.
- UniGoal's scene graph should remain the main memory representation.
- The first functional target is a single-agent ObjectNav baseline.
- The two-agent version should be developed only after the single-agent
  ObjectNav and RGB-only paths are stable.

## Version Control Plan

Keep the stages separated with branches:

```text
main
  Upstream UniGoal baseline.

local/qwen-runtime
  Local Qwen server integration, local config, startup/smoke scripts, and a
  self-contained Qwen HTTP server wrapper.

baseline/objectnav-rgbd
  ObjectNav-HM3D and ObjectNav-MP3D RGB-D baseline support.

feature/rgb-only-vggt
  Replace online RGB-D perception with RGB-only VGGT reconstruction.

feature/teamgraph-two-agent
  Add two-agent shared Graph/TeamGraph memory and coordinated exploration.
```

Current branch progression:

- `local/qwen-runtime` contains the self-contained local Qwen runtime.
- `baseline/objectnav-rgbd` adds the first RGB-D ObjectNav baseline path.

## Evaluation Target

The project should eventually support formal evaluation on both:

- ObjectNav-HM3D
- ObjectNav-MP3D

Local datasets appear to exist under:

```text
/home/hsy/datasets/objectnav/hm3d/v2/
/home/hsy/datasets/objectnav/mp3d/v1/
/home/hsy/datasets/hm3d or /home/hsy/datasets/scene_datasets/hm3d_v0.2/
/home/hsy/datasets/mp3d/
```

The first evaluation milestone should be `val_mini`, not full `val`, because
the perception pipeline is slow and failures need to be diagnosed quickly.

Evaluation order:

1. HM3D ObjectNav `val_mini`, single-agent RGB-D.
2. MP3D ObjectNav `val_mini`, single-agent RGB-D.
3. HM3D ObjectNav `val`, single-agent RGB-D.
4. MP3D ObjectNav `val`, single-agent RGB-D.
5. Repeat the same sequence for RGB-only VGGT.
6. Repeat selected `val_mini` episodes for two-agent comparison.

Metrics to record:

- SR
- SPL
- soft SPL if available
- distance to goal
- episode count
- timeout/failure count
- average wall-clock time per episode

Experiment logs, videos, model files, and datasets must not be committed.

## Current Local Qwen Smoke Command

Use this only to test the current UniGoal release path. It evaluates
InstanceImageNav, not ObjectNav.

```bash
cd /home/hsy/UniGoal

start_local_vlm.sh

TIMEOUT_SECONDS=420 \
NAV_GPU=0 \
EPISODE_ID=0 \
GOAL_TYPE=ins-image \
run_unigoal_local_qwen_smoke.sh
```

For a longer run over the configured evaluation episodes:

```bash
cd /home/hsy/UniGoal

TIMEOUT_SECONDS=999999 \
NAV_GPU=0 \
CONFIG_FILE=configs/config_local_qwen.yaml \
GOAL_TYPE=ins-image \
EPISODE_ID=-1 \
run_unigoal_local_qwen_smoke.sh
```

The result summary is written under:

```text
outputs/experiments/local_qwen_smoke/log/total.json
```

## Baseline Phase: ObjectNav RGB-D

Purpose:

Build a clean single-agent ObjectNav baseline before changing the perception
modality.

Scope:

- Add Habitat ObjectNav task configs for HM3D and MP3D.
- Add `goal_type=object` support if missing.
- Connect ObjectNav dataset splits:
  - `objectnav/hm3d/v2/val_mini`
  - `objectnav/hm3d/v2/val`
  - `objectnav/mp3d/v1/val_mini`
  - `objectnav/mp3d/v1/val`
- Map Habitat object category goals to UniGoal goal graph inputs.
- Keep UniGoal's current RGB-D sensors and BEV mapping.
- Keep UniGoal's current low-level execution.
- Keep local Qwen optional through config, but do not bake Qwen-specific logic
  into ObjectNav task logic.

Important code anchors:

- `main.py`
- `configs/config_habitat.yaml`
- `configs/tasks/instance_imagenav.yaml`
- `src/envs/__init__.py`
- `src/envs/instanceimagegoal_env.py`
- `src/agent/unigoal/agent.py`
- `src/map/bev_mapping.py`
- `src/graph/graph.py`

Expected output of this phase:

- A reproducible command for HM3D ObjectNav `val_mini`.
- A reproducible command for MP3D ObjectNav `val_mini`.
- A committed branch `baseline/objectnav-rgbd`.
- A short result summary committed as documentation, not raw logs.

Current commands:

```bash
cd /home/hsy/UniGoal

# Full ObjectNav-HM3D val_mini, 30 episodes.
run_hm3d.sh

# One HM3D episode smoke.
EPISODE_ID=0 TIMEOUT_SECONDS=180 run_hm3d.sh

# Full ObjectNav-MP3D val_mini, 30 episodes.
run_mp3d.sh

# One MP3D episode smoke.
EPISODE_ID=0 TIMEOUT_SECONDS=180 run_mp3d.sh
```

Current caveat:

- HM3D `val_mini` uses the 6 ObjectNav categories that mostly overlap the
  current Mask R-CNN semantic prediction head.
- MP3D `val_mini` contains categories such as `towel`, `counter`, `cabinet`,
  and `seating`. The first RGB-D baseline currently maps these to the nearest
  available semantic channel only to keep evaluation running. MP3D numbers from
  this baseline should therefore be treated as a runnable baseline, not as a
  faithful UniGoal paper reproduction.

## RGB-Only Phase: VGGT Perception

Purpose:

Replace online true depth with RGB-only pseudo-depth and reconstruction while
preserving UniGoal's graph memory and downstream decision logic.

Reference implementation:

```text
/home/hsy/multi-agent-nav/src/map/spacev6.py
/home/hsy/multi-agent-nav/src/map/vggt/
/home/hsy/model/mvp-nav/vggt
```

Target perception flow:

```text
RGB frame buffer
  -> VGGT pseudo-depth and camera geometry
  -> GroundingDINO/SAM object masks
  -> pseudo-depth object point clouds
  -> UniGoal Graph.objects / Graph.nodes
  -> UniGoal exploration and low-level execution
```

Do not rewrite the entire Graph first. The preferred minimal interface is:

- Add a depth provider abstraction for `Graph.set_observations`.
- Add a depth provider abstraction for `BEV_Map.mapping`.
- Keep the existing `Graph.mapping3d` and `create_object_pcd` logic initially.
- Replace `observations['depth']` with VGGT pseudo-depth only when an
  `rgb_only` config flag is enabled.

Important code anchors:

- `src/graph/graph.py`
  - `set_observations`
  - `mapping3d`
  - `update_node`
  - `get_scenegraph`
- `src/graph/utils/utils.py`
  - `create_object_pcd`
  - `gobs_to_detection_list`
- `src/map/bev_mapping.py`
  - `mapping`
- MVP reference:
  - `spacev6.py: build_pcd`
  - VGGT model loading and pseudo-depth confidence handling

Expected output of this phase:

- Single-agent RGB-only ObjectNav-HM3D `val_mini` run.
- Single-agent RGB-only ObjectNav-MP3D `val_mini` run if MP3D data path is
  stable.
- A comparison table against `baseline/objectnav-rgbd`.

## Memory Direction: UniGoal Graph, Not Text GSSL

Use UniGoal's graph memory as the base shared representation.

Preferred shared entity fields:

- stable entity id
- caption/category
- 3D center
- OBB or bbox
- confidence
- number of observations
- source agent
- first seen step
- last seen step
- status
- optional relation edges

Avoid using long natural-language GSSL strings as the main memory. They grow too
quickly and make cross-agent deduplication difficult.

The cross-agent merge rule should start conservatively:

- same or compatible category
- close 3D center after coordinate alignment
- overlapping or nearby OBB
- sufficient observation confidence

## Two-Agent Phase

The first two-agent version should be simple and controlled.

Initial assumptions:

- Start from the single-agent RGB-only ObjectNav stack.
- Use two environment instances or two synchronized runners first.
- Share object-centric Graph/TeamGraph memory.
- Focus on reducing duplicate exploration and improving target discovery.
- Do not prioritize physical robot-robot collision handling in the first
  version.

Two-agent coordination should initially operate at the long-term goal level:

- each agent updates local graph
- local objects are merged into TeamGraph
- TeamGraph selects or biases frontier/goal assignment
- agents avoid selecting the same explored region when alternatives exist

The two-agent branch should not be started until single-agent RGB-only runs are
stable enough to compare.

## Known Risks

- The current UniGoal public release does not directly support ObjectNav in the
  exposed README command path.
- VGGT inference is likely to be slow and memory-heavy.
- GroundingDINO/SAM plus graph update is already expensive in RGB-D mode.
- HM3D and MP3D ObjectNav category definitions may not align exactly with
  current `configs/categories.py`.
- MP3D semantic assets and Habitat 0.2.3 compatibility must be verified before
  relying on MP3D full evaluation.
- Replacing depth in Graph alone is insufficient; BEV mapping also depends on
  depth and must be handled.

## Immediate Next Tasks

1. Commit the local Qwen runtime and this planning document on
   `local/qwen-runtime`.
2. Create `baseline/objectnav-rgbd`.
3. Audit Habitat ObjectNav configs available in `third_party/habitat-lab`.
4. Add a minimal ObjectNav-HM3D `val_mini` config.
5. Make `goal_type=object` run through UniGoal's goal graph path.
6. Run one HM3D ObjectNav episode and inspect metrics.
7. Only after this baseline works, begin RGB-only VGGT design.
