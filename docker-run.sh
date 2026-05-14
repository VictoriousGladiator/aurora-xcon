#!/bin/bash
# AURORA-XCon Docker helper — build, shell, run, sweep, test, jupyter, clean

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -n "$CPU_ONLY" ]; then
    PLATFORM="CPU"
    COMPOSE_FILE="docker-compose.cpu.yml"
else
    PLATFORM="GPU"
    COMPOSE_FILE="docker-compose.yml"
fi

# docker compose v2 preferred; fall back to docker-compose v1
if docker compose version &>/dev/null; then
    DC=(docker compose -f "$COMPOSE_FILE")
else
    DC=(docker-compose -f "$COMPOSE_FILE")
fi

show_help() {
    echo "AURORA-XCon Docker Runner"
    echo ""
    echo "Usage: ./docker-run.sh [COMMAND] [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  shell              Interactive shell in the container"
    echo "  run <algo> [args]  Run training (e.g. ./docker-run.sh run aurora env=kheperax seed=42)"
    echo "  sweep [args]       Run scripts/run_sweep.py"
    echo "  test               Run pytest tests/"
    echo "  jupyter            Start Jupyter on port 8888"
    echo "  clean              Remove compose containers and volumes for this project"
    echo ""
    echo "On macOS or without an NVIDIA GPU in Docker, use CPU mode:"
    echo "  CPU_ONLY=1 ./docker-run.sh build"
    echo "  CPU_ONLY=1 ./docker-run.sh shell"
    echo ""
}

dcompose() {
    "${DC[@]}" "$@"
}

build_img() {
    echo -e "${GREEN}Building AURORA-XCon (${PLATFORM} compose)...${NC}"
    dcompose build
    echo -e "${GREEN}Build complete.${NC}"
}

shell_img() {
    echo -e "${GREEN}Starting shell (${PLATFORM})...${NC}"
    dcompose run --rm aurora
}

run_algo() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: no algorithm${NC}"
        echo "Usage: ./docker-run.sh run <algo> [hydra_args...]"
        exit 1
    fi
    ALGO=$1
    shift
    echo -e "${GREEN}Running: ${ALGO}${NC}"
    dcompose run --rm aurora python -m main main "$ALGO" "$@"
}

run_sweep() {
    echo -e "${GREEN}Running sweep...${NC}"
    dcompose run --rm aurora python scripts/run_sweep.py "$@"
}

run_tests() {
    echo -e "${GREEN}Running tests...${NC}"
    dcompose run --rm aurora python -m pytest tests/ -v
}

jupyter_srv() {
    echo -e "${GREEN}Jupyter — open http://localhost:8888${NC}"
    dcompose run --rm -p 8888:8888 aurora \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
}

clean_all() {
    echo -e "${YELLOW}Removing containers and named volumes...${NC}"
    dcompose down -v
    docker rmi aurora-xcon:latest 2>/dev/null || true
    echo -e "${GREEN}Cleanup done.${NC}"
}

case "${1:-}" in
    build) build_img ;;
    shell) shell_img ;;
    run) shift; run_algo "$@" ;;
    sweep) shift; run_sweep "$@" ;;
    test) run_tests ;;
    jupyter) jupyter_srv ;;
    clean) clean_all ;;
    help|--help|-h) show_help ;;
    "") show_help ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
