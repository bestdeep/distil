# HEARTBEAT.md — SN97 Bot Periodic Tasks

## Check for pending announcements
1. `GET http://127.0.0.1:3710/api/announcement`
2. If `type` is not null and `posted` is false:
   - Post the `message` field to Discord channel `1482026267392868583`
   - Then `POST http://127.0.0.1:3710/api/announcement/posted` to mark it as sent
3. Do NOT re-post announcements that are already `posted: true`
