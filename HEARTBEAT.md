# HEARTBEAT.md — SN97 Bot Periodic Tasks

## Public announcement worker
The announcement worker now runs on the same `distil` host as the API, so prefer the local API path.

1. `GET http://127.0.0.1:3710/api/announcement`
2. If `type` is null, do nothing.
3. If an announcement is pending:
   - First `POST http://127.0.0.1:3710/api/announcement/claim`
   - Only post if the claim response still contains a real announcement (`type` is not null)
   - Send the returned `message` field to Discord channel `1482026267392868583` using the public `arbos` account
4. Do NOT use `/api/announcement/posted` unless you are handling a legacy fallback path. `/claim` is the idempotent path.
5. If the worker is ever moved off-host again, use the public `https://api.arbos.life/api/announcement` and `/claim` endpoints instead of assuming localhost access.
