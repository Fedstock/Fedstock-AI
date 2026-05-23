#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

COMPOSE="docker compose"

print_help() {
  cat <<'EOF'
Fedstock Backend development helper

Usage:
  ./run.sh <command>

Commands:
  help        Show this help message
  up          Build and run PostgreSQL + API for development
  up-bg       Build and run PostgreSQL + API in background
  app         Run Spring Boot locally
  boot        Alias for app
  db          Run PostgreSQL only for local app development
  down        Stop containers
  clean       Stop containers and remove volumes
  logs        Follow API and PostgreSQL logs
  status      Show container status

Examples:
  ./run.sh up
  ./run.sh db
  ./run.sh app
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required but not installed."
    exit 1
  fi
}

require_gradle_wrapper() {
  if [ ! -x "./gradlew" ]; then
    echo "Gradle wrapper is missing or not executable."
    exit 1
  fi
}

cmd="${1:-help}"

case "$cmd" in
  help|-h|--help)
    print_help
    ;;
  up)
    require_docker
    $COMPOSE up --build
    ;;
  up-bg)
    require_docker
    $COMPOSE up --build -d
    ;;
  db)
    require_docker
    $COMPOSE up db -d
    ;;
  down)
    require_docker
    $COMPOSE down
    ;;
  clean)
    require_docker
    $COMPOSE down -v
    ;;
  logs)
    require_docker
    $COMPOSE logs -f
    ;;
  app|boot)
    require_gradle_wrapper
    ./gradlew bootRun
    ;;
  status)
    require_docker
    $COMPOSE ps
    ;;
  *)
    echo "Unknown command: $cmd"
    echo
    print_help
    exit 1
    ;;
esac
