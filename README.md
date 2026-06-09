# GAIA Inspect Pipeline

This repository runs the GAIA benchmark through Inspect AI against an OpenAI-compatible model endpoint.

## Configuration

Create `.env` from the example file and fill in the required credentials:

```bash
cp .env.example .env
```

At minimum, set:

```env
BASE_MODEL_RUNNER_TYPE=llama16GB
HF_TOKEN=hf_...
```

`HF_TOKEN` must have access to the gated `gaia-benchmark/GAIA` dataset on Hugging Face.

Runner choices:

- `llama16GB`: Colab/T4-oriented runner. Downloads a prebuilt CUDA `llama-server` from `ai-dock/llama.cpp-cuda` and serves `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`.
- `vllm16GB`: vLLM AWQ runner for `QuantTrio/Qwen3.5-9B-AWQ`.
- `cpu2B`: CPU fallback using `llama-cpp-python`.

## Run With Docker Compose

Use this mode on a desktop or workstation where Docker is allowed.

```bash
docker compose up
```

Docker Compose starts one Ubuntu 24.04 based Python development container. It bind-mounts this project into `/workspace`, uses that as the working directory, then runs `sh runner/start.sh`:

- `gaia`: runs `sh runner/start.sh`.
- `runner/start.sh`: installs dependencies, sources `runner/<name>.env`, starts the selected base model runner, starts `svc_scaffold`, then starts the benchmark job. If any background service exits, the whole run exits.

Runtime state is stored in the local ignored folder:

- `_state/`: Python virtualenvs, Hugging Face cache, downloaded GGUF files, Playwright browsers.
- `inspect-logs/`: Inspect eval logs. If `LOGS_BRANCH` is set, successful runs commit and push this folder to that branch.

## Run Without Docker

Use this mode on a desktop or environment where Docker is not allowed.

Install Python 3 and the required system tools in the host environment, then run:

```bash
sh runner/start.sh
```
