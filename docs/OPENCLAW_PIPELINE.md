# Distil OpenClaw Pipeline

This project uses two Discord-facing agents plus one hourly ops loop:

- `sn97-bot`: public/community bot on the `arbos` Discord account. It answers public SN97 questions and posts king-change announcements.
- `distil`: private/internal ops bot on the default Discord account. It monitors the Distil channel and handles maintenance work locally on the Distil host.
- `sn97-hourly-healthcheck`: OpenClaw cron job that runs on the Distil host, checks Distil end to end, applies safe repairs, and posts a short summary to the internal Distil channel.

## Host layout

- `distil`: the only machine that runs Distil services, state, and the Distil OpenClaw control plane.
- `remote_vm`: no Distil-specific OpenClaw agents, cron jobs, or Discord bindings remain there.
- External dependencies remain external:
  - Lium eval pod
  - chat host

OpenClaw now runs locally on `distil` under `/root/.openclaw`, while the live repo and app services run from `/opt/distil/repo`.

## Announcement flow

1. Validator writes `state/announcement.json`.
2. API exposes `/api/announcement` and `/api/announcement/claim`.
3. OpenClaw polls `http://127.0.0.1:3710/api/announcement` on the same host.
4. If an announcement exists, OpenClaw claims it first via the local `/api/announcement/claim`.
5. OpenClaw posts the claimed `message` to Discord on the public `arbos` account.

## Hourly healthcheck flow

1. OpenClaw runs the `distil` agent hourly.
2. The agent runs locally on `distil` and executes:
   - `python3 /opt/distil/repo/scripts/sn97_healthcheck.py --repair --format json`
3. The script performs deterministic checks and safe repairs for:
   - `distil-validator`
   - `distil-api`
   - `distil-dashboard`
   - `distil-benchmark-sync.timer`
   - `chat-tunnel`
   - `caddy`
   - `open-webui`
   - local and public HTTP health
4. The agent then reads new Discord messages, handles bug reports, and escalates to code changes only when deterministic repair is not enough.

## Why this design

- Deterministic infrastructure repair is faster and safer in a script than in a free-form agent prompt.
- The agent is still useful for triage, explanation, code fixes, and deployment.
- Public SN97 replies stay low-privilege, while private Distil ops retain full tooling.
- Keeping both the app stack and the bot loop on `distil` removes the last operational dependency on a local machine or `remote_vm` for Distil-specific infra.
