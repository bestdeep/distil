#!/bin/bash
# Deploy API + Frontend to production server (distil-api / 46.224.105.143)
# Usage: ./scripts/deploy-prod.sh [api|frontend|both]
#
# API code: synced from this server's /home/openclaw/distillation/api/ + eval/
# Frontend: edited directly on prod at /opt/distil/frontend/ (no local copy)
# State: auto-synced every 15s by PM2 distil-sync
set -euo pipefail

REMOTE="distil-api"
COMPONENT="${1:-both}"

deploy_api() {
    echo "==> Deploying API to $REMOTE..."
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
        /home/openclaw/distillation/api/ "$REMOTE:/opt/distil/api/"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
        /home/openclaw/distillation/eval/ "$REMOTE:/opt/distil/eval/"
    echo "==> Restarting distil-api service..."
    ssh -t "$REMOTE" 'systemctl restart distil-api'
    sleep 2
    STATUS=$(ssh "$REMOTE" 'curl -s -o /dev/null -w "%{http_code}" http://localhost:3710/api/health')
    if [ "$STATUS" = "200" ]; then
        echo "==> API deployed and healthy ✓"
    else
        echo "==> WARNING: API returned $STATUS after deploy!"
    fi
}

deploy_frontend() {
    echo "==> Building and restarting frontend on $REMOTE..."
    echo "    (frontend source lives on $REMOTE at /opt/distil/frontend/)"
    echo "    Edit files there directly via: ssh distil-api 'vim /opt/distil/frontend/src/...'"
    ssh -t "$REMOTE" 'cd /opt/distil/frontend && NEXT_PUBLIC_API_URL=https://api.arbos.life npx next build'
    echo "==> Restarting distil-dashboard service..."
    ssh -t "$REMOTE" 'systemctl restart distil-dashboard'
    sleep 2
    STATUS=$(ssh "$REMOTE" 'curl -s -o /dev/null -w "%{http_code}" http://localhost:3720/')
    if [ "$STATUS" = "200" ]; then
        echo "==> Frontend deployed and healthy ✓"
    else
        echo "==> WARNING: Frontend returned $STATUS after deploy!"
    fi
}

case "$COMPONENT" in
    api)      deploy_api ;;
    frontend) deploy_frontend ;;
    both)     deploy_api; deploy_frontend ;;
    *)        echo "Usage: $0 [api|frontend|both]"; exit 1 ;;
esac

echo "==> Deploy complete!"
