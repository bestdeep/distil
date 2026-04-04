# Fix: check_model.py matches production + remove unnecessary truncation

**Date:** 2026-04-04
**Status:** Complete

## Problem

Two community-reported issues:

### 1. check_model.py KL doesn't match production
Miners reported getting KL=0.255 locally when their live score was 0.085. Root causes:
- **Scored ALL positions** (prompt + continuation) instead of continuation-only
- **No fp32 cast** before log_softmax (production casts to `.float()`)
- **Used `F.kl_div(s, t.exp())`** (non-log-target) instead of production's `F.kl_div(s, t, log_target=True)`
- **Different prompt sampling** — raw `text[:2048]` from streaming dataset instead of `sample_prompts_from_dataset()` + `format_prompt()`
- **Different tokenization** — tokenized prompt with `truncation=True, max_length=512` instead of tokenizing full_text as one string
- **No teacher generation step** — compared prompt-only logits instead of generating continuations first

### 2. Prompt truncation inconsistency
Three conflicting truncation behaviors:
- `format_prompt()` truncated to 512 chars (too aggressive)
- HF fallback in `pod_eval_vllm.py` tokenized with `truncation=True, max_length=1024`
- vLLM path had no truncation at all

## Changes

### `check_model.py` (complete eval rewrite)
- **Prompt sampling**: Uses `sample_prompts_from_dataset()` + `format_prompt()` from `eval/dataset.py` (same as production)
- **Teacher generation**: Generates continuations with `teacher.generate()`, then extracts logits via forward pass
- **Continuation-only KL**: Extracts logits at `[:, prompt_len-1:-1, :]` (same slice as production)
- **fp32 casting**: All logits cast to `.float()` before `log_softmax` (matches production)
- **log_target=True**: Uses `F.kl_div(s_log_p, t_log_p, log_target=True)` (matches production)
- **Full-text tokenization**: Tokenizes `full_text` as one string with `truncation=False`
- **Added comment**: "This matches the production eval pipeline in pod_eval_vllm.py"
- King comparison also updated to use the same continuation-only approach

### `eval/dataset.py`
- Changed `format_prompt()` default `max_chars` from 512 to 4000
- Rationale: `sample_prompts_from_dataset()` already filters to 200-4000 chars, and max_new_tokens=512 fits in 32K context

### `scripts/pod_eval_vllm.py`
- Changed prompt tokenization from `truncation=True, max_length=args.max_prompt_len` to `truncation=False`
- Now matches the vLLM path (no truncation on already-filtered prompts)

## Files Modified
- `check_model.py` — eval section rewritten
- `eval/dataset.py` — line with `format_prompt` default
- `scripts/pod_eval_vllm.py` — prompt tokenization line

## Verification
All three files pass `ast.parse()` syntax check.
