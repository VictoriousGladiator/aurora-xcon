# Docker quickstart (copy-paste)

Use this if you already have Docker and the repo’s Docker files (`Dockerfile`, `docker-compose.yml`, `docker-compose.cpu.yml`, `docker-run.sh`).

## macOS (or no NVIDIA GPU in Docker)

```bash
cd /path/to/aurora-xcon

CPU_ONLY=1 ./docker-run.sh build
CPU_ONLY=1 ./docker-run.sh shell
```

Inside the container (example):

```bash
cd /workspace/src   # if that is your working_dir in Compose
python -m main main aurora env=kheperax seed=42
```

## Linux + NVIDIA GPU (optional)

Only if the NVIDIA Container Toolkit is installed and `nvidia-smi` works in a test GPU container:

```bash
./docker-run.sh build
./docker-run.sh shell
```

## Useful one-liners

```bash
./docker-run.sh --help
export WANDB_API_KEY=…   # optional
CPU_ONLY=1 ./docker-run.sh sweep --workers 2
```

## If something fails

Open the main guide: [DOCKER.md](../DOCKER.md) (troubleshooting section).
