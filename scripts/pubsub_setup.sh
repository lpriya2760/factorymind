#!/usr/bin/env bash
# One-time Pub/Sub setup for FactoryMind's live monitoring. Safe to re-run.
#
# Creates:
#   topic        factorymind-telemetry   sensor readings (from scripts/replay_publisher.py)
#   subscription factorymind-monitor     read by scripts/monitor.py
#   topic        factorymind-alerts      JSON alerts published by the monitor
#   subscription factorymind-alerts-tail to watch alerts: gcloud pubsub subscriptions pull ...
# and with WITH_BIGQUERY=1, a BigQuery table factorymind.alerts that stores every alert.
#
# Requires: gcloud signed in, `gcloud config set project <PROJECT_ID>`. For the Python scripts,
# also `gcloud auth application-default login` once.
set -euo pipefail

PROJECT="$(gcloud config get-value project 2>/dev/null)"
[[ -n "$PROJECT" ]] || { echo "Run: gcloud config set project <PROJECT_ID>"; exit 1; }
echo "Project: $PROJECT"

gcloud services enable pubsub.googleapis.com

topic() {
  gcloud pubsub topics describe "$1" >/dev/null 2>&1 \
    || gcloud pubsub topics create "$1"
}
subscription() {  # name topic [extra flags...]
  local name="$1" topic="$2"; shift 2
  gcloud pubsub subscriptions describe "$name" >/dev/null 2>&1 \
    || gcloud pubsub subscriptions create "$name" --topic "$topic" "$@"
}

topic factorymind-telemetry
topic factorymind-alerts
# Readings are kept for a day if the monitor isn't running yet.
subscription factorymind-monitor factorymind-telemetry \
  --ack-deadline 60 --message-retention-duration 1d
subscription factorymind-alerts-tail factorymind-alerts --message-retention-duration 1d

if [[ "${WITH_BIGQUERY:-0}" == "1" ]]; then
  NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
  bq --location=US mk --dataset --force "$PROJECT:factorymind" >/dev/null
  bq show "$PROJECT:factorymind.alerts" >/dev/null 2>&1 || bq mk --table \
    "$PROJECT:factorymind.alerts" \
    subscription_name:STRING,message_id:STRING,publish_time:TIMESTAMP,data:STRING,attributes:STRING
  # Pub/Sub's own service account writes the rows.
  gcloud projects add-iam-policy-binding "$PROJECT" --condition=None --quiet \
    --member "serviceAccount:service-${NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com" \
    --role roles/bigquery.dataEditor >/dev/null
  subscription factorymind-alerts-bq factorymind-alerts \
    --bigquery-table "$PROJECT:factorymind.alerts" --write-metadata
  echo "Alerts will also land in BigQuery: $PROJECT.factorymind.alerts"
fi

echo "Done. Next, in two terminals:"
echo "  uv run --group stream python scripts/monitor.py"
echo "  uv run --group stream python scripts/replay_publisher.py --seconds-per-hour 1"
