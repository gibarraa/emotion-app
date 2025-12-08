#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$ROOT_DIR/.venv"
SCRIPT_PATH="$ROOT_DIR/fase1_pose.py"

if [ ! -d "$VENV_PATH" ]; then
  echo "Creando entorno virtual en $VENV_PATH"
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "No se encontró el script esperado en: $SCRIPT_PATH" >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "Uso: $0 <subcomando> [opciones...]"
  echo "Ejemplo: $0 record --duration 10 --output salida.mp4"
  echo "Subcomandos disponibles:"
  python "$SCRIPT_PATH" --help
  exit 1
fi

python "$SCRIPT_PATH" "$@"
