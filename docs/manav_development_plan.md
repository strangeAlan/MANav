# MANav Development Plan

This project starts from the public UniGoal codebase and develops MANav as an
RGB-only navigation stack. The current baseline branch intentionally stays on
UniGoal's supported RGB-D tasks before any perception replacement.

## Current Baseline

Branch:

```text
baseline/iin-tn-rgbd
```

Base branch:

```text
local/qwen-runtime
```

Purpose:

- Keep original UniGoal RGB-D navigation behavior.
- Use local Qwen for LLM/VLM calls through the in-repo runtime.
- Evaluate the two public UniGoal tasks:
  - instance-image-goal navigation, `goal_type=ins-image`
  - text-goal navigation, `goal_type=text`

ObjectNav-HM3D/MP3D is not the current baseline target. It was explored, but the
released UniGoal code does not expose a clean, reproducible ObjectNav-MP3D path
that matches the reported paper result.

## Baseline Commands

One episode:

```bash
cd /home/hsy/UniGoal
NAV_GPU=0 EPISODE_ID=0 TIMEOUT_SECONDS=600 ./script/run_iin.sh
NAV_GPU=0 EPISODE_ID=0 TIMEOUT_SECONDS=600 ./script/run_tn.sh
```

Mini-eval:

```bash
cd /home/hsy/UniGoal
NAV_GPU=0 EPISODE_ID=-1 NUM_EVAL_EPISODES=10 TIMEOUT_SECONDS=7200 ./script/run_iin.sh
NAV_GPU=0 EPISODE_ID=-1 NUM_EVAL_EPISODES=10 TIMEOUT_SECONDS=7200 ./script/run_tn.sh
```

Runtime details are in:

```text
docs/local_qwen_deployment.md
docs/iin_tn_rgbd_baseline.md
```

## Next RGB-Only Branch

Create a new branch from `baseline/iin-tn-rgbd` after the RGB-D baseline is
checked:

```bash
git switch baseline/iin-tn-rgbd
git switch -c feature/rgb-only-lingbot-stream
```

Target changes:

- Replace online Habitat depth with LingBot-Map streaming reconstruction output.
- Preserve RGB-only online policy inputs.
- Keep UniGoal's graph memory and downstream navigation logic first.
- Validate against the same IIN/TN episode set used by the RGB-D baseline.

The first RGB-only pass should be single-agent only. Multi-agent shared graph
memory should come after the single-agent RGB-only path is stable.

## LingBot-Map Direction

LingBot-Map is the preferred RGB-only reconstruction backend for the next
RGB-only branch. It is a streaming 3D reconstruction model with KV cache,
keyframe intervals, depth confidence, camera predictions, and optional world
point outputs. This is a better fit for online navigation than repeatedly
running heavy windowed reconstruction.

Keep LingBot-Map isolated from UniGoal:

```text
UniGoal env:      /home/hsy/miniconda3/envs/unigoal-habitat
LingBot-Map env:  /home/hsy/miniconda3/envs/lingbot-map
LingBot repo:     /home/hsy/lingbot-map
```

Do not install LingBot-Map into the Habitat/UniGoal environment. Its recommended
runtime is Python 3.10 with a newer PyTorch/FlashInfer stack, while UniGoal uses
the older Habitat-compatible environment.

Initial integration shape:

```text
RGB frames from UniGoal
  -> LingBot-Map streaming reconstruction service or adapter
  -> depth + depth_conf + predicted trajectory + optional world_points
  -> scale/coordinate alignment against UniGoal/Habitat agent pose trajectory
  -> UniGoal-compatible depth provider
  -> later: object mask + world_points/depth_conf object grounding
```

Rules for the first RGB-only branch:

- Do not let LingBot-Map take over agent localization.
- Use UniGoal/Habitat pose as the navigation frame reference first.
- Treat LingBot-Map pose as a reconstruction signal for scale/coordinate
  alignment only.
- Use confidence to filter noisy depth before BEV mapping.
- Prefer object-level stable memory before large prompt-facing summaries.
- Keep `habitat_depth` as a config fallback so the RGB-D baseline remains easy
  to compare.

Validation before code integration:

- Run LingBot-Map on a short RGB sequence.
- Confirm streaming inference returns `depth` and `depth_conf`.
- Compare predicted camera trajectory with UniGoal/Habitat pose on the same
  sequence.
- Check whether depth has usable metric scale after alignment.
- Inspect whether object-mask points can produce stable centers/OBBs.
