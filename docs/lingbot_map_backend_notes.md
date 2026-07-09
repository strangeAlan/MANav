# LingBot-Map Backend Notes

LingBot-Map lives outside this repository at:

```text
/home/hsy/lingbot-map
```

It is being evaluated as the preferred RGB-only reconstruction backend for MANav.
The goal is to use its streaming geometry while keeping UniGoal's navigation,
graph memory, and Habitat-facing runtime isolated.

## Why LingBot-Map

LingBot-Map is designed for streaming 3D reconstruction:

- frame-by-frame causal inference
- KV cache
- keyframe interval control
- windowed mode for long sequences
- `depth` and `depth_conf` outputs
- predicted camera trajectory
- optional `world_points` / `world_points_conf`

This is a better online-navigation fit than repeatedly running heavy multi-frame
offline reconstruction windows.

## Environment Boundary

Use a dedicated environment:

```text
/home/hsy/miniconda3/envs/lingbot-map
```

Do not install LingBot-Map packages into:

```text
/home/hsy/miniconda3/envs/unigoal-habitat
```

UniGoal/Habitat should call LingBot-Map through a small adapter, cache, or local
service. This avoids mixing Habitat's older Python/PyTorch constraints with
LingBot-Map's newer streaming reconstruction stack.

Current setup commands:

```bash
cd /home/hsy/lingbot-map

/home/hsy/miniconda3/bin/conda create -n lingbot-map python=3.10 -y

/home/hsy/miniconda3/envs/lingbot-map/bin/pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

/home/hsy/miniconda3/envs/lingbot-map/bin/pip install -e .

/home/hsy/miniconda3/envs/lingbot-map/bin/pip install \
  --index-url https://pypi.org/simple flashinfer-python
```

If FlashInfer installation is slow or unavailable, the first sanity checks can
use LingBot-Map's SDPA fallback with `--use_sdpa`.

The conda environment has been created. The PyTorch installation was deliberately
left for a manual run because the CUDA wheels are large.

## First Adapter Contract

The first MANav adapter should expose a small provider interface:

```text
input:
  RGB frame
  optional UniGoal/Habitat pose hint
  step id

output:
  depth
  depth_conf
  predicted_pose
  intrinsic
  optional world_points
  optional world_points_conf
```

The minimum UniGoal integration only replaces online `obs['depth']`. The
advanced integration uses object masks plus depth/world points for object-level
grounding.

## Current Prototype

Branch:

```text
feature/rgb-only-lingbot-stream
```

Implemented pieces:

- `script/lingbot_depth_server.py`
- `script/start_lingbot_depth_server.sh`
- `script/stop_lingbot_depth_server.sh`
- `src/perception/lingbot_depth_client.py`
- `configs/config_iin_lingbot_local_qwen.yaml`
- `script/run_iin_lingbot.sh`

Start the depth server manually:

```bash
cd /home/hsy/UniGoal
RUN_IN_BACKGROUND=1 LINGBOT_GPU=0 ./script/start_lingbot_depth_server.sh
```

Stop it:

```bash
cd /home/hsy/UniGoal
./script/stop_lingbot_depth_server.sh
```

Run a one-episode RGB-only IIN smoke:

```bash
cd /home/hsy/UniGoal
NAV_GPU=0 EPISODE_ID=0 TIMEOUT_SECONDS=900 ./script/run_iin_lingbot.sh
```

The wrapper follows "who starts it stops it" for LingBot-Map. If it has to
start the depth server, it stops it on exit. If the server was already running,
it leaves it running.

The current server is intentionally conservative: it keeps a short recent RGB
history and reruns LingBot-Map streaming inference over that history for each
request, returning the latest frame's depth and confidence. This is slower than
a true persistent KV-cache server, but it is simpler for validating the UniGoal
integration path first.

## Important Constraints

- Do not use Habitat true depth when `rgb_only` mode is enabled.
- Do not let LingBot-Map take over navigation pose/localization initially.
- Align LingBot-Map predicted trajectory to UniGoal/Habitat pose before using
  metric distances.
- Use confidence masks before BEV mapping and object point cloud extraction.
- Cache outputs for repeatable debugging, but keep strict online mode free of
  future-frame leakage.

## Risks To Verify

- Coordinate convention and c2w/w2c interpretation.
- Metric scale after trajectory alignment.
- Depth confidence thresholds for BEV safety.
- Stability of object centers/OBBs from mask-filtered points.
- Runtime and GPU memory with local Qwen also running.

## Validation So Far

- LingBot-Map environment imports passed with:
  - `torch 2.8.0+cu128`
  - `torchvision 0.23.0+cu128`
  - `flashinfer 0.6.14`
  - `lingbot_map`
- SDPA streaming sanity check passed on 8 frames from
  `/home/hsy/lingbot-map/example/loop`.
- FlashInfer backend failed because no `nvcc`/CUDA toolkit path was available;
  use SDPA for the first integration pass.
- HTTP depth server returned 480x640 metric depth and confidence maps to the
  UniGoal environment.
