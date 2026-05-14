# Running AURORA-XCon with Docker

This guide explains how to build and run the project in Docker on your machine.

**Docker assets in this repository:** `Dockerfile`, `docker-compose.yml` (GPU), `docker-compose.cpu.yml` (CPU), `docker-run.sh`, and `.dockerignore`.

---

## Prerequisites

1. **Docker** with Compose v2 (Docker Desktop on macOS or Windows; Docker Engine + Compose plugin on Linux).
2. **Disk and time** for the first image build (often on the order of tens of minutes, depending on network and CPU).
3. **GPU (optional)**  
   - **Linux** with an NVIDIA GPU: install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so Compose can use `driver: nvidia`.  
   - **macOS / Apple Silicon / no NVIDIA GPU:** use the **CPU** workflow below. Docker Desktop does **not** expose an NVIDIA GPU to Linux containers the way a Linux host does, and the default GPU Compose service will fail with an error like `could not select device driver "nvidia"`.

---

## CPU vs GPU Compose

| Goal | Typical command |
|------|------------------|
| **CPU** (recommended on macOS, or any machine without NVIDIA-in-Docker) | `CPU_ONLY=1 ./docker-run.sh …` |
| **GPU** (Linux + NVIDIA Container Toolkit) | `./docker-run.sh …` (default) |

The CPU path should use a Compose file that does **not** request GPU devices and should set environment variables suitable for CPU execution (for example `JAX_PLATFORM_NAME=cpu` and an empty or unset `CUDA_VISIBLE_DEVICES`), as defined in your `docker-compose.cpu.yml` (or equivalent).

---

## First-time build

From the repository root:

```bash
./docker-run.sh build
```

For a CPU-oriented image build when your helper script supports it:

```bash
CPU_ONLY=1 ./docker-run.sh build
```

If you do not use `docker-run.sh`, the equivalent is usually:

```bash
docker compose build
```

using the Compose file your project documents for that mode.

---

## Interactive shell

The helper script uses the command name **`shell`**, not `bash`:

```bash
CPU_ONLY=1 ./docker-run.sh shell
```

On macOS, prefer **`CPU_ONLY=1`** so Compose does not request an NVIDIA GPU.

Inside the container, the project tree is typically mounted at `/workspace/src` (or the path your `docker-compose` file sets as `working_dir`). Run experiments from there, for example:

```bash
python -m main main aurora env=kheperax seed=42
```

---

## Other `docker-run.sh` commands

Consult the script’s built-in help:

```bash
./docker-run.sh --help
```

Typical commands (names may match your script exactly):

- **`build`** — build the image  
- **`shell`** — interactive shell  
- **`run <algo> [hydra overrides…]`** — run one training job  
- **`sweep [args]`** — run the sweep script  
- **`test`** — run tests  
- **`jupyter`** — start Jupyter  
- **`clean`** — remove containers/volumes as implemented by the script  

If you use **`CPU_ONLY=1`**, pass it for **every** invocation (build, shell, run, sweep) so the same Compose file is used consistently, **unless** your script already propagates that flag to all subcommands.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `WANDB_API_KEY` | Optional; enables Weights & Biases online logging if set. |
| `CPU_ONLY=1` | When supported by `docker-run.sh`, selects the CPU Compose file and CPU-oriented settings. |
| `HYDRA_FULL_ERROR` | Often set in Compose to `1` for clearer Hydra tracebacks inside the container. |

You can export them in the host shell before calling `docker-run.sh`, or set them in your Compose file, depending on your setup.

---

## Outputs and mounts

Compose setups usually mount the current directory into the container and use named volumes for `output/` and `wandb/` (or similar) so artifacts persist across container runs. Check your `docker-compose*.yml` for the exact `volumes` section.

---

## Compose file notes

- If Compose prints a warning that the top-level `version` field is obsolete, it is safe to ignore for functionality; removing that key is a cosmetic cleanup in the YAML only (optional maintenance).
- **Rebuild** the image after changing `Dockerfile` or dependency installation steps.

---

## Troubleshooting

### `could not select device driver "nvidia"`

**Cause:** Docker is trying to attach NVIDIA GPUs, but no NVIDIA runtime is available (common on macOS, or on Linux without the NVIDIA Container Toolkit).

**What to do:** Use the CPU Compose path (`CPU_ONLY=1` with your project’s script, or the CPU Compose file directly). For real GPU training, use a **Linux** host with an NVIDIA GPU and the toolkit installed, then use the GPU Compose file.

---

### JAX import errors mentioning `cuda_nvcc` or `pathlib` / `NoneType`

**Cause:** CUDA-enabled JAX wheels may probe for NVCC at import time. A minimal CUDA **runtime** image can lack the layout those probes expect.

**What to do (without touching application Python):** Adjust the **Dockerfile** only—for example install an NVCC bundle compatible with what PyPI publishes (your build logs will show which `nvidia-cuda-nvcc-cu12` versions exist), or use a CUDA **devel** base image. Rebuild the image afterward. This is an image packaging issue, not a project algorithm issue.

---

### `ModuleNotFoundError: No module named 'brax.v1'`

**Cause:** QDAX code imports `brax.v1`, which exists only in **older** Brax releases. An unpinned `brax` line in `requirements.txt` can pull a **new** Brax that removed `brax.v1`.

**What to do:** In the environment where you install dependencies (host venv or Docker build), install a Brax version compatible with your vendored QDAX (pin `brax` in the file that feeds `pip install` for that image, or use a constraints file), then rebuild the Docker image. Exact pin depends on the QDAX/Brax pairing you use; align with a known-good lockfile or upstream recommendation.

---

### `Unknown command: bash` from `docker-run.sh`

**Cause:** The script expects **`shell`**, not `bash`.

**What to do:** Run `./docker-run.sh shell` (and `CPU_ONLY=1` on macOS if needed).

---

### Sweeps or `run` still request a GPU when you wanted CPU

**Cause:** `CPU_ONLY=1` was not set for that command, or you invoked `docker compose` without `-f docker-compose.cpu.yml`.

**What to do:** Export `CPU_ONLY=1` for the whole session (`export CPU_ONLY=1`) or prefix every `./docker-run.sh` call. The provided `docker-run.sh` uses the same Compose file for `build`, `shell`, `run`, `sweep`, `test`, `jupyter`, and `clean` based on `CPU_ONLY`.

---

## Security and hygiene

- Do not commit real `WANDB_API_KEY` values into the repository.  
- Treat the container as a full Linux environment; only mount directories you trust.

---

## See also

- [docs/DOCKER_QUICKSTART.md](docs/DOCKER_QUICKSTART.md) — minimal copy-paste flow  
- [README.md](README.md) — paper, citation, and non-Docker installation notes  
- [apptainer/](apptainer/) — HPC-oriented container definition if you use Apptainer instead of Docker  
