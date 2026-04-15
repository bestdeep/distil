import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.runtime import (
    ALLOWED_ORIGINS,
    CACHE_TTL,
    CHAT_POD_APP_PORT,
    CHAT_POD_HOST,
    CHAT_POD_SSH_KEY,
    CHAT_POD_SSH_PORT,
    DASHBOARD_URL,
    DISK_CACHE_DIR,
    NETUID,
    STATE_DIR,
    TMC_BASE,
    TMC_HEADERS,
)

CHAT_POD_PORT = CHAT_POD_APP_PORT

API_DESCRIPTION = f"""
# Distil - Subnet {NETUID} API

Public API for [Distil]({DASHBOARD_URL}), a Bittensor subnet where miners compete to produce the best knowledge-distilled small language models.

## How It Works

Miners submit distilled models and a validator evaluates them head-to-head against the reigning **king** model using KL-divergence on shared prompts. Lower KL = better distillation = higher rewards.

## Quick Start

```bash
curl https://api.arbos.life/api/health
curl https://api.arbos.life/api/scores
curl https://api.arbos.life/api/price
```
"""
