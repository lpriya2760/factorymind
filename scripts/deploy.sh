#!/usr/bin/env bash
# Deploy FactoryMind to Cloud Run from source. Scales to zero; capped at 2 instances to stay
# inside free-trial credits. Gemini runs through Vertex AI in the project, so no API key is
# deployed. Requires: gcloud installed, `gcloud auth login`, and
# `gcloud config set project <PROJECT_ID>`.
set -euo pipefail

REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-factorymind}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"

cd "$(dirname "$0")/.."
uv export --no-dev --no-hashes --no-header --quiet -o requirements.txt

# Gemini through Vertex AI, plus any Slack/email alert settings exported in this shell.
# "^|^" makes "|" the separator, so comma-separated values such as ALERT_EMAIL_TO survive.
ENV_VARS="^|^GOOGLE_GENAI_USE_VERTEXAI=TRUE|GOOGLE_CLOUD_PROJECT=${PROJECT}|GOOGLE_CLOUD_LOCATION=global"
for name in SLACK_WEBHOOK_URL ALERT_EMAIL_TO ALERT_EMAIL_FROM ALERT_MIN_LEVEL \
            SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD; do
  if [[ -n "${!name:-}" ]]; then
    ENV_VARS+="|${name}=${!name}"
  fi
done

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --min-instances 0 \
  --max-instances 2 \
  --cpu 1 \
  --memory 1Gi \
  --session-affinity \
  --allow-unauthenticated \
  --set-env-vars "${ENV_VARS}"
