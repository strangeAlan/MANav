# Local Qwen Deployment

MANav includes a small local Qwen HTTP runtime so that UniGoal can use a local
VLM without an OpenAI-compatible adapter.

## Files

- `script/qwen_backend_server.py`: local Transformers Qwen HTTP server.
- `script/start_local_vlm.sh`: starts the local server.
- `script/run_local_qwen_smoke.sh`: runs a UniGoal smoke test with local Qwen.
- `configs/config_local_qwen.yaml`: UniGoal config that points to local Qwen.
- `src/utils/llm.py`: routes `local_qwen` LLM/VLM calls to `/generate`.

## Model Weights

Do not commit model weights. Put them outside the repo and either set
`QWEN_MODEL_PATH` or create a symlink:

```bash
cd /home/hsy/UniGoal
mkdir -p data/models
ln -s /home/hsy/model/qwen/Qwen3-VL-8B-Instruct data/models/Qwen3-VL-8B-Instruct
```

The default model path is:

```text
data/models/Qwen3-VL-8B-Instruct
```

On this server, the script also falls back to:

```text
/home/hsy/model/qwen/Qwen3-VL-8B-Instruct
```

## Start Local Qwen

```bash
cd /home/hsy/UniGoal
start_local_vlm.sh
```

Useful overrides:

```bash
QWEN_GPU=1 QWEN_PORT=18080 start_local_vlm.sh
QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct start_local_vlm.sh
RUN_IN_BACKGROUND=1 start_local_vlm.sh
```

Health check:

```bash
curl --noproxy '*' http://127.0.0.1:18080/health
```

## Run UniGoal Smoke Test

```bash
cd /home/hsy/UniGoal

TIMEOUT_SECONDS=420 \
NAV_GPU=0 \
EPISODE_ID=0 \
GOAL_TYPE=ins-image \
run_unigoal_local_qwen_smoke.sh
```

Summary output:

```text
outputs/experiments/local_qwen_smoke/log/total.json
```

## Notes

- `start_local_vlm.sh` uses `QWEN_ENV` for the Python environment that has
  Transformers/Qwen dependencies.
- Default `QWEN_ENV` on this server:
  `/home/hsy/miniconda3/envs/multi-agent-nav`.
- UniGoal navigation itself uses:
  `/home/hsy/miniconda3/envs/unigoal-habitat`.
- HTTP proxy environment variables are unset inside the scripts for localhost
  calls.
