#!/usr/bin/env bash
# Deploy FactoryMind to Cloud Run from source. Scales to zero; capped at 2 instances to stay
# inside free-trial credits. Requires: gcloud installed, `gcloud auth login`, and
# `gcloud config set project <PROJECT_ID>`.
set -euo pipefail

REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-factorymind}"

cd "$(dirname "$0")/.."
uv export --no-dev --no-hashes --no-header --quiet -o requirements.txt

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --min-instances 0 \
  --max-instances 2 \
  --cpu 1 \
  --memory 1Gi \
  --session-affinity \
  --allow-unauthenticated
