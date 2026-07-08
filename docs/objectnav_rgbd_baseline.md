# ObjectNav RGB-D Baseline

This branch adds a minimal ObjectNav RGB-D baseline on top of the local Qwen
runtime branch.

## Supported Splits

- HM3D ObjectNav v2 `val_mini`
- MP3D ObjectNav v1 `val_mini`

The scripts are configured for `val_mini` by default. The dataset and scene
files are referenced through ignored symlinks under `data/`.

Required local symlinks on this server:

```text
data/datasets/objectnav -> /home/hsy/datasets/objectnav
data/scene_datasets/hm3d_v0.2/minival -> val
data/scene_datasets/mp3d -> /home/hsy/datasets/mp3d/v1/tasks/mp3d
```

## Commands

Start local Qwen if needed:

```bash
start_local_vlm.sh
```

Run one HM3D episode:

```bash
cd /home/hsy/UniGoal
EPISODE_ID=0 TIMEOUT_SECONDS=180 run_hm3d.sh
```

Run full HM3D `val_mini`:

```bash
cd /home/hsy/UniGoal
TIMEOUT_SECONDS=999999 run_hm3d.sh
```

Run one MP3D episode:

```bash
cd /home/hsy/UniGoal
EPISODE_ID=0 TIMEOUT_SECONDS=180 run_mp3d.sh
```

Run full MP3D `val_mini`:

```bash
cd /home/hsy/UniGoal
TIMEOUT_SECONDS=999999 run_mp3d.sh
```

Override episode count:

```bash
NUM_EVAL_EPISODES=5 TIMEOUT_SECONDS=999999 run_hm3d.sh
NUM_EVAL_EPISODES=5 TIMEOUT_SECONDS=999999 run_mp3d.sh
```

## Outputs

HM3D:

```text
outputs/experiments/objectnav_hm3d_local_qwen/log/total.json
```

MP3D:

```text
outputs/experiments/objectnav_mp3d_local_qwen/log/total.json
```

Raw `outputs/` and `logs/` directories are ignored and should not be committed.

## Validation So Far

Short smoke tests were run with timeouts:

- `EPISODE_ID=0 TIMEOUT_SECONDS=160 run_hm3d.sh`
  - Loaded ObjectNav-HM3D.
  - Entered episode 0 with goal `bed`.
  - Reached the UniGoal scene graph update loop and local Qwen relation calls.
  - Ended by external timeout, not by traceback.

- `EPISODE_ID=0 TIMEOUT_SECONDS=100 run_mp3d.sh`
  - Loaded ObjectNav-MP3D.
  - Loaded MP3D `.house` semantic descriptor and semantic mesh.
  - Entered episode 0 with goal `towel`.
  - Reached the UniGoal scene graph update loop.
  - Ended by external timeout, not by traceback.

## Known Limitations

- This is a runnable RGB-D baseline, not a reproduction of the UniGoal paper's
  ObjectNav-MP3D result.
- The current semantic prediction head is COCO/Mask R-CNN based and only covers
  a subset of ObjectNav categories.
- MP3D categories outside the current semantic head, such as `towel`, are not
  forced into incorrect COCO aliases. They are added to the graph's
  GroundingDINO/SAM detection vocabulary and should be judged from graph
  evidence rather than from the Mask R-CNN stop channel.
- HM3D semantic annotations are not present in the local HM3D scene package, but
  this baseline relies mainly on RGB-D plus learned prediction for online
  behavior.
