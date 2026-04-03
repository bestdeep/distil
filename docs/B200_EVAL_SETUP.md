# B200 Eval Setup — Definitive Guide

## Overview
Running SN97 distillation eval on NVIDIA B200 (sm_100, 183GB VRAM) using Lium GPU pods.

## Working Stack (confirmed 2026-04-03)
- **Template**: `64c96459` (PyTorch 2.7.0-py3.12-cuda12.8.0)
- **After `pip install vllm`**: torch 2.10.0+cu128, vLLM 0.19.0
- **transformers**: Must be ≥5.0 (for `qwen3_5_moe` architecture support)
  - `pip install "transformers>=5.0"` AFTER vllm to avoid downgrade
- **flash-attn**: Not needed — vLLM uses FlashInfer 0.6.6 internally
- **accelerate**: Required for HF fallback loading

## Critical Fixes Required

### 1. grouped_mm Patch (MANDATORY for B200)
`torch._grouped_mm` only supports sm_90. B200 is sm_100. Crashes MoE models.

Patch `transformers/integrations/moe.py`, function `_can_use_grouped_mm`:
```python
# Replace:
return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
# With:
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
if cap[0] >= 10:
    return hasattr(torch.nn.functional, "grouped_mm")
return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
```

### 2. vLLM Must Stop Before HF Teacher Logit Extraction
vLLM uses 84GB (gpu_memory_utilization=0.45). Teacher HF model needs 65GB.
84 + 65 = 149GB + CUDA overhead > available VRAM.

The eval script (`pod_eval_vllm.py`) stops vLLM before Phase 1b and does NOT restart it
(students are scored via HF, not vLLM).

### 3. NO Teacher Cache Save (Disk Constraint)
teacher_cache.pt is ~45GB. Lium pods have ~230GB disk.
67GB model cache + 45GB teacher_cache + OS = disk full.
Cache save is disabled. Each round regenerates teacher logits (~56s).

### 4. transformers Version Matters
- `pip install vllm` pins transformers to 4.57.6 — this version does NOT support `qwen3_5_moe`
- The eval script loads the teacher via HF for logit extraction. Without ≥5.0, it silently fails.
- Fix: `pip install "transformers>=5.0"` after vllm install (no-deps not needed)
- The grouped_mm patch must be RE-APPLIED after every transformers upgrade

## Dependency Install Order (CRITICAL)
```bash
pip install vllm accelerate          # Sets torch 2.10.0, vLLM 0.19.0
pip install "transformers>=5.0"      # Upgrade from 4.57.6 to ≥5.0
# Then apply grouped_mm patch
```

DO NOT: `pip install transformers vllm` (transformers may downgrade torch)

## Performance Baselines (B200, 120 prompts, 9 students)
- vLLM text generation: ~4 min (120 prompts)
- HF teacher load: ~11s (model cached)
- Logit extraction: ~56s
- Student scoring: ~10 min (9 × ~1 min)
- **Total round: ~16 min**

## Things That DON'T Work
- `--enforce-eager`: 3-5x slowdown, not needed with torch 2.10.0
- flash-attn pre-built wheels: Incompatible with torch 2.10.0, must compile from source (~30 min)
- teacher_cache.pt on 230GB pods: Fills disk
- `pip install --no-deps transformers`: May miss deps, use regular install
- `vllm/vllm-openai` Docker image: No SSH server, incompatible with Lium

## Pod Cleanup
After each student, the eval script runs `shutil.rmtree()` on the HF cache for that model.
Monitor disk with `df -h /` — should stay under 90%.

## Troubleshooting
- **Eval stuck after Phase 1a**: transformers too old, or OOM from vLLM + HF teacher coexisting
- **"_grouped_mm" error**: Patch not applied
- **"You can update Transformers" in logs**: transformers doesn't recognize model architecture
- **Corrupt teacher_cache.pt**: Delete it, eval will regenerate
- **Disk 95%+**: Delete teacher_cache.pt.tmp, stale HF caches
