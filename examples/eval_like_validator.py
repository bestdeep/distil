#!/usr/bin/env python3
"""
Evaluate a student model with logic aligned to the local validator check path.

This mirrors the GPU eval flow in `check_model.py --eval`:
- sample prompts via `eval.dataset.sample_prompts_from_dataset()`
- format prompts with `format_prompt()`
- generate teacher continuations
- score continuation-only KL(teacher || student)
- use fp32 log_softmax + F.kl_div(..., log_target=True)

Example:
    python3 examples/eval_like_validator.py \
      --student ./distil-checkpoints/step_1000 \
      --teacher Qwen/Qwen3.5-35B-A3B \
      --prompts 20
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_DATASET = "karpathy/climbmix-400b-shuffle"
DEFAULT_PROMPTS = 20
DEFAULT_BLOCK = 12345
DEFAULT_MAX_NEW_TOKENS = 512
KL_FRAUD_THRESHOLD = 1e-6


def sample_eval_prompts(n: int, dataset_name: str, block_number: int) -> list[str]:
    from eval.dataset import sample_prompts_from_dataset, format_prompt

    raw_prompts = sample_prompts_from_dataset(
        n=n,
        block_number=block_number,
        block_hash=None,
        dataset_name=dataset_name,
    )

    prompts = []
    for text in raw_prompts:
        formatted = format_prompt(text)
        if formatted:
            prompts.append(formatted)
        if len(prompts) >= n:
            break
    return prompts


def build_teacher_targets(
    teacher_name: str,
    prompts: list[str],
    max_new_tokens: int,
    teacher_cache_path: str | None,
):
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    if teacher_cache_path and Path(teacher_cache_path).exists():
        cache_data = torch.load(teacher_cache_path, map_location="cpu", weights_only=False)
        if (
            len(cache_data.get("full_sequences", [])) >= len(prompts)
            and len(cache_data.get("teacher_logits", [])) >= len(prompts)
            and len(cache_data.get("prompt_lens", [])) >= len(prompts)
        ):
            return (
                [s.to("cuda") for s in cache_data["full_sequences"][: len(prompts)]],
                cache_data["teacher_logits"][: len(prompts)],
                cache_data["prompt_lens"][: len(prompts)],
            )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()

    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []

    with torch.no_grad():
        for i, prompt_text in enumerate(prompts):
            prompt_ids = teacher_tok(prompt_text, return_tensors="pt", truncation=False).input_ids.to(teacher.device)
            prompt_len = prompt_ids.shape[1]
            output_ids = teacher.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
            )
            logits = teacher(output_ids).logits.float()
            cont_logits = logits[:, prompt_len - 1:-1, :]
            full_sequences.append(output_ids.cpu())
            teacher_logits_list.append(cont_logits.cpu())
            prompt_lens.append(prompt_len)
            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                gen_len = output_ids.shape[1] - prompt_len
                print(f"Teacher {i + 1}/{len(prompts)}: {prompt_len}+{gen_len} tokens", flush=True)

    if teacher_cache_path:
        torch.save(
            {
                "full_sequences": full_sequences,
                "teacher_logits": teacher_logits_list,
                "prompt_lens": prompt_lens,
            },
            teacher_cache_path,
        )

    full_sequences = [s.to("cuda") for s in full_sequences]
    del teacher
    torch.cuda.empty_cache()
    return full_sequences, teacher_logits_list, prompt_lens


def score_model(model_name: str, revision: str | None, full_sequences, teacher_logits_list, prompt_lens):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    kl_scores = []
    with torch.no_grad():
        for i in range(len(full_sequences)):
            full_seq = full_sequences[i]
            prompt_len = prompt_lens[i]

            t_logits = teacher_logits_list[i].to(model.device).float()
            t_log_p = F.log_softmax(t_logits, dim=-1)

            s_logits = model(full_seq).logits.float()
            cont_s = s_logits[:, prompt_len - 1:-1, :]

            min_len = min(cont_s.shape[1], t_log_p.shape[1])
            t_lp_slice = t_log_p[:, :min_len, :]
            s_lp_slice = F.log_softmax(cont_s[:, :min_len, :], dim=-1)
            kl_per_pos = F.kl_div(s_lp_slice, t_lp_slice, log_target=True, reduction="none").sum(dim=-1)
            kl_mean = kl_per_pos.mean().item()
            kl_scores.append(kl_mean)

            if (i + 1) % 5 == 0 or i == len(full_sequences) - 1:
                running_avg = sum(kl_scores) / len(kl_scores)
                print(f"Prompt {i + 1}/{len(full_sequences)}: KL={kl_mean:.6f} (running avg: {running_avg:.6f})")

    return model, kl_scores


def summarize_scores(name: str, kl_scores: list[float]) -> dict:
    kl_global = sum(kl_scores) / len(kl_scores)
    kl_std = statistics.stdev(kl_scores) if len(kl_scores) > 1 else 0.0
    ci = 1.96 * kl_std / math.sqrt(len(kl_scores)) if kl_scores else 0.0
    summary = {
        "name": name,
        "prompts": len(kl_scores),
        "kl_mean": kl_global,
        "kl_std": kl_std,
        "kl_ci_low": kl_global - ci,
        "kl_ci_high": kl_global + ci,
        "fraud_flag": kl_global <= KL_FRAUD_THRESHOLD,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate student with validator-like local KL logic.")
    parser.add_argument("--student", required=True, help="HF repo or local path to student model")
    parser.add_argument("--student-revision", default=None)
    parser.add_argument("--teacher", default=TEACHER_MODEL)
    parser.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--block-number", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--teacher-cache", default=None, help="Optional .pt cache for teacher full_sequences/logits")
    parser.add_argument("--compare-model", default=None, help="Optional second model to compare against same teacher targets")
    parser.add_argument("--compare-revision", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print(f"Sampling {args.prompts} prompts from {args.dataset}...")
    prompts = sample_eval_prompts(args.prompts, args.dataset, args.block_number)
    if not prompts:
        raise SystemExit("No valid prompts sampled")
    print(f"Sampled {len(prompts)} prompts")

    print(f"Building teacher targets from {args.teacher}...")
    t0 = time.time()
    full_sequences, teacher_logits_list, prompt_lens = build_teacher_targets(
        args.teacher, prompts, args.max_new_tokens, args.teacher_cache
    )
    print(f"Teacher targets ready in {time.time() - t0:.1f}s")

    print(f"Scoring student: {args.student}")
    student_model, student_scores = score_model(
        args.student, args.student_revision, full_sequences, teacher_logits_list, prompt_lens
    )
    student_summary = summarize_scores(args.student, student_scores)
    print(json.dumps(student_summary, indent=2))

    if student_summary["fraud_flag"]:
        print(f"WARNING: KL <= {KL_FRAUD_THRESHOLD} (would be flagged as teacher copy)")

    results = {"student": student_summary}

    if args.compare_model:
        del student_model
        torch.cuda.empty_cache()
        print(f"Scoring compare model: {args.compare_model}")
        _, compare_scores = score_model(
            args.compare_model, args.compare_revision, full_sequences, teacher_logits_list, prompt_lens
        )
        compare_summary = summarize_scores(args.compare_model, compare_scores)
        results["compare"] = compare_summary
        print(json.dumps(compare_summary, indent=2))
        delta = compare_summary["kl_mean"] - student_summary["kl_mean"]
        print(f"Delta KL (compare - student): {delta:+.6f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
