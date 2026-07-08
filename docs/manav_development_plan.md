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
git switch -c feature/rgb-only-vggt
```

Target changes:

- Replace online Habitat depth with VGGT pseudo-depth or reconstructed depth.
- Preserve RGB-only online policy inputs.
- Keep UniGoal's graph memory and downstream navigation logic first.
- Validate against the same IIN/TN episode set used by the RGB-D baseline.

The first RGB-only pass should be single-agent only. Multi-agent shared graph
memory should come after the single-agent RGB-only path is stable.
