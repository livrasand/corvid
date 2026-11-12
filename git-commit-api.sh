#!/usr/bin/env bash
# Script para enviar un diff a la API de commits de Corvid
# Uso: ./git-commit-api.sh [mensaje-opcional] [archivo...]
# Si no se pasa mensaje, se genera automáticamente desde el diff

set -euo pipefail

API_URL="${GIT_COMMIT_API_URL:-http://localhost:3000/api/commit}"
BRANCH="${GIT_COMMIT_BRANCH:-main}"
REPO="${GIT_COMMIT_REPO:-$(pwd)}"

# Marcar archivos untracked como "intent to add" para que git diff los detecte
git add -N -- . 2>/dev/null || true

# Obtener diff de todos los cambios (staged, unstaged y untracked marcados)
DIFF=$(git diff HEAD)
if [ -z "$DIFF" ]; then
  echo "No hay cambios para commitear."
  exit 0
fi

FILES=$(git diff HEAD --name-only | tr '\n' ',' | sed 's/,$//')

PAYLOAD=$(jq -n \
  --arg diff "$DIFF" \
  --arg files "$FILES" \
  --arg branch "$BRANCH" \
  --arg repo "$REPO" \
  '{
    diff: $diff,
    files: ($files | split(",") | map(select(length > 0))),
    branch: $branch,
    repo: $repo
  }')

echo "Enviando diff a la API..."
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | jq .
