# ADR 0006: Live monitoring replays real sensor data through Pub/Sub

**Status:** accepted (October 2026)
**Code:** `factorymind/pubsub.py`, `scripts/pubsub_setup.sh`, `scripts/replay_publisher.py`, `scripts/monitor.py`, `scripts/stream_demo.py`

## Context

FactoryMind analyses an uploaded bundle. A real plant would stream sensor readings continuously and want an alert the moment a machine turns risky, without anyone uploading a file. The spec (#1) put real-time ingestion out of scope, and we have no plant connected. We still want to show that the analysis works on a stream and fits a Google Cloud event pipeline.

## Decision

**Replay the real Azure Predictive Maintenance readings through Google Cloud Pub/Sub,** sped up, and run the same analysis on the stream.

```
replay_publisher ──► factorymind-telemetry ──► monitor ──► factorymind-alerts ──► BigQuery / Slack / email
```

1. **Real data, honest clock.** `split_bundle()` cuts a bundle at `replay_hours` before its end. The monitor starts from the history before the cut, and the publisher sends every telemetry and error reading after it, unchanged, one simulated hour every few seconds. Only the clock is sped up.
2. **One message per reading,** as JSON, with `type`, `machine_id` and `datetime` attributes so subscribers can filter without parsing.
3. **The monitor reuses `analyze()`.** When every machine has reported for the next hour, it rebuilds the bundle and re-runs the full analysis: risk, cascade, Monte Carlo range and interventions. There is no second, simplified model to keep in sync. A test checks that the streamed end state equals analysing the full upload.
4. **Robust to Pub/Sub's delivery.** Pub/Sub delivers at least once and doesn't guarantee order. The monitor drops duplicates and buffers readings by hour, analysing an hour only when it's complete. A test shuffles the stream and adds 40 duplicates, and gets the same hours and the same alert.
5. **Alert on the transition.** A machine alerts the first hour it reaches the alert level, and again only if it drops below and comes back. The alert message is the same content as the Slack/email alert, plus the P10/P50/P90 loss range.
6. **Separate from the web app.** The publisher and monitor are scripts that run anywhere with gcloud credentials. The Cloud Run app is unchanged, and the Pub/Sub client is an optional `stream` dependency group, so the deployed image doesn't change.
7. **Works offline too.** `LocalBus` runs the same pipeline in memory (`scripts/stream_demo.py`, and the tests).

## Result on the demo data

Replaying the last 36 hours of `comp2_failure_building`: all ten machines stay green until **2015-05-19 06:00**, when machine 90 logs real error2 and error3 codes. Its comp2 risk jumps to >99%, and the monitor publishes one alert: expected loss $22,623 (range $22,016–$23,349), best action "Replace comp2, machine 47 covers". Each hour's analysis takes about 0.6s, so the publisher's default of 2 seconds per hour leaves headroom.

## Consequences

- **Good:** a real Google Cloud event pipeline, using real sensor data, with the full analysis on every hour and alerts other systems can consume.
- **Limitation:** the monitor keeps its state in memory and runs as a single process. Restarting it starts again from the bundle's history. A production version would persist state and run on an always-on service (e.g. a Cloud Run worker pool or GKE).
- **Limitation:** context tables (machines, maintenance, schedule, inventory, orders) come from the bundle, not from the stream.
- **Next step for a real plant:** sensors → PLC/SCADA → an edge gateway (OPC UA or MQTT) → the same `factorymind-telemetry` topic. Nothing after the topic changes.
- **Not verified here against live Pub/Sub:** the sandbox this was built in couldn't reach Google Cloud or download the emulator. The client calls are checked against the real library; the pipeline logic is tested end to end in memory.
