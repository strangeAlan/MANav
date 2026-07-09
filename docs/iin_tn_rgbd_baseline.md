# IIN/TN RGB-D Baseline

This branch keeps UniGoal on its public RGB-D task path and only adds local
Qwen runtime wrappers. It is intended as the baseline before starting RGB-only
perception work.

## Branch

```text
baseline/iin-tn-rgbd
```

Base branch:

```text
local/qwen-runtime
```

## Tasks

- Instance-image-goal navigation: `--goal_type ins-image`
- Text-goal navigation: `--goal_type text`

Both tasks use the existing Habitat InstanceImageNav HM3D v3 config:

```text
configs/config_local_qwen.yaml
configs/tasks/instance_imagenav.yaml
```

## Local Qwen

Start Qwen manually if desired:

```bash
cd /home/hsy/UniGoal
QWEN_GPU=1 start_local_vlm.sh
```

The run scripts also check `http://127.0.0.1:18080/health` and start Qwen in
background mode if it is not already healthy. If a run script starts Qwen, it
also stops Qwen when the run exits. If Qwen was already running before the run
script started, the script leaves it running.

Stop Qwen manually:

```bash
cd /home/hsy/UniGoal
./script/stop_local_vlm.sh
```

## One-Episode Checks

```bash
cd /home/hsy/UniGoal
NAV_GPU=0 EPISODE_ID=0 TIMEOUT_SECONDS=600 ./script/run_iin.sh
NAV_GPU=0 EPISODE_ID=0 TIMEOUT_SECONDS=600 ./script/run_tn.sh
```

## Mini-Eval

Use `EPISODE_ID=-1` to let Habitat iterate through the dataset and set
`NUM_EVAL_EPISODES` for the run length:

```bash
cd /home/hsy/UniGoal
NAV_GPU=0 EPISODE_ID=-1 NUM_EVAL_EPISODES=10 TIMEOUT_SECONDS=7200 ./script/run_iin.sh
NAV_GPU=0 EPISODE_ID=-1 NUM_EVAL_EPISODES=10 TIMEOUT_SECONDS=7200 ./script/run_tn.sh
```

Outputs are written under:

```text
outputs/experiments/iin_rgbd_local_qwen/
outputs/experiments/tn_rgbd_local_qwen/
```

## Why This Baseline

ObjectNav-MP3D is postponed because the public UniGoal code path and available
category/stop logic do not cleanly reproduce the reported ObjectNav-MP3D result.
IIN and TN are the two tasks explicitly supported by the released code, so they
are a better control baseline for later RGB-only/LingBot-Map streaming
reconstruction changes.
