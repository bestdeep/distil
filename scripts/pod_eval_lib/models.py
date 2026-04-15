import gc
import glob
import os
import shutil
import time
from pathlib import Path

import torch

from pod_eval_lib.kl import dense_to_sparse_topk

HF_CHUNK_SIZE = 4096
ACTIVATION_FP_SEED = 42
ACTIVATION_FP_N_INPUTS = 5
ACTIVATION_FP_SEQ_LEN = 64
ACTIVATION_FP_VOCAB_SIZE = 248320


def gpu_mem_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}/{total:.1f}GB"
    return "N/A"


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()


def ensure_disk_space(teacher_name, threshold=85):
    try:
        stat = os.statvfs("/")
        pct = int(100 * (1 - stat.f_bavail / stat.f_blocks))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"  [disk] {pct}% used, {free_gb:.1f}GB free", flush=True)
        if pct > threshold:
            print(f"  [disk] >{threshold}% — cleaning non-teacher caches", flush=True)
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if cache_dir.exists():
                for directory in cache_dir.iterdir():
                    if directory.is_dir() and directory.name.startswith("models--") and directory.name != teacher_cache:
                        shutil.rmtree(directory)
            for pattern in ["/tmp/vllm_*", "/tmp/tmp*", "/tmp/teacher_*"]:
                for filename in glob.glob(pattern):
                    try:
                        size = os.path.getsize(filename)
                        if size > 1024**3:
                            os.remove(filename)
                            print(f"  [disk] Removed stale {filename} ({size/1024**3:.1f}GB)", flush=True)
                    except Exception:
                        pass
            teacher_logits_path = "/home/teacher_cache.pt"
            if os.path.exists(teacher_logits_path):
                cache_size = os.path.getsize(teacher_logits_path) / (1024**3)
                if cache_size > 0 and pct > 90:
                    os.remove(teacher_logits_path)
                    print(f"  [disk] Removed stale teacher cache ({cache_size:.1f}GB)", flush=True)
            stat2 = os.statvfs("/")
            pct2 = int(100 * (1 - stat2.f_bavail / stat2.f_blocks))
            free_gb2 = (stat2.f_bavail * stat2.f_frsize) / (1024**3)
            print(f"  [disk] After cleanup: {pct2}% used, {free_gb2:.1f}GB free", flush=True)
            return pct2
        return pct
    except Exception as exc:
        print(f"  [disk] Check failed: {exc}", flush=True)
        return 0


def load_model(name, device="cuda", dtype=torch.bfloat16, revision=None):
    from transformers import AutoModelForCausalLM

    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    kwargs = dict(dtype=dtype, device_map=device, trust_remote_code=is_teacher)
    if revision and revision != "main":
        kwargs["revision"] = revision
        print(f"  [model] Pinning to revision {revision[:12]}", flush=True)
    for attempt in range(3):
        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
                print("  [model] Loaded with flash_attention_2", flush=True)
                return model
            except (ValueError, ImportError):
                model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                print("  [model] Loaded with default attention", flush=True)
                return model
        except Exception as exc:
            err_str = str(exc)
            is_transient = any(text in err_str for text in ["429", "503", "rate limit", "Connection", "Timeout", "HTTPSConnection"])
            if is_transient and attempt < 2:
                wait = (attempt + 1) * 30
                print(f"  [model] Transient error loading {name} (attempt {attempt + 1}/3), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                raise


def prefetch_model(name, revision=None, max_retries=3):
    from huggingface_hub import snapshot_download

    dl_kwargs = dict(ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"])
    if revision and revision != "main":
        dl_kwargs["revision"] = revision
    for attempt in range(max_retries):
        try:
            snapshot_download(name, **dl_kwargs)
            print(f"  [prefetch] {name} cached (rev={revision or 'main'})", flush=True)
            return
        except Exception as exc:
            err_str = str(exc)
            is_transient = any(text in err_str for text in ["429", "503", "rate limit", "Connection", "Timeout"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [prefetch] {name} transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                print(f"  [prefetch] {name} failed: {exc}", flush=True)
                return


def clean_model_cache(name, teacher_name=None):
    try:
        cache_name = f"models--{name.replace('/', '--')}"
        if teacher_name:
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            if cache_name == teacher_cache:
                return
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  [cleanup] Removed {cache_name}", flush=True)
    except Exception:
        pass


def compute_activation_fingerprint(model, device="cuda"):
    try:
        rng = torch.Generator(device=device)
        rng.manual_seed(ACTIVATION_FP_SEED)
        input_ids = torch.randint(0, ACTIVATION_FP_VOCAB_SIZE, (ACTIVATION_FP_N_INPUTS, ACTIVATION_FP_SEQ_LEN), generator=rng, device=device)
        n_layers = model.config.num_hidden_layers
        checkpoints = sorted(set([0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]))
        activations = {idx: [] for idx in checkpoints}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input_data, output_data):
                hidden = output_data[0] if isinstance(output_data, tuple) else output_data
                activations[layer_idx].append(hidden.float().mean(dim=(0, 1)).detach().cpu())
            return hook_fn

        layers_module = None
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers_module = obj
                break
            except AttributeError:
                continue
        if layers_module is None:
            print("[fingerprint] Could not find layers module", flush=True)
            return None
        for idx in checkpoints:
            if idx < len(layers_module):
                hooks.append(layers_module[idx].register_forward_hook(make_hook(idx)))
        with torch.no_grad():
            for i in range(ACTIVATION_FP_N_INPUTS):
                _ = model(input_ids[i:i + 1])
        for hook in hooks:
            hook.remove()
        layer_fingerprints = {}
        for idx in checkpoints:
            if activations[idx]:
                avg = torch.stack(activations[idx]).mean(dim=0)
                layer_fingerprints[str(idx)] = [round(v, 6) for v in avg[:128].tolist()]
        hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 0
        del input_ids, activations
        torch.cuda.empty_cache()
        print(f"[fingerprint] Computed: {len(layer_fingerprints)} layers, n_layers={n_layers}, hidden_size={hidden_size}", flush=True)
        return {"layer_fingerprints": layer_fingerprints, "n_layers": n_layers, "hidden_size": hidden_size}
    except Exception as exc:
        print(f"[fingerprint] Error: {exc}", flush=True)
        return None


def hf_batched_forward(teacher, sequences_data, device, batch_size=2, logprobs_k=128, progress_cb=None):
    short_items = []
    long_items = []
    for idx, data in enumerate(sequences_data):
        if data["full_ids"].shape[1] > HF_CHUNK_SIZE:
            long_items.append((idx, data))
        else:
            short_items.append((idx, data))
    short_items.sort(key=lambda item: item[1]["full_ids"].shape[1])
    n_total = len(sequences_data)
    teacher_logits_list = [None] * n_total
    prompt_lens = [0] * n_total
    full_sequences = [None] * n_total
    n_chunked = 0
    processed = 0
    with torch.no_grad():
        for batch_start in range(0, len(short_items), batch_size):
            batch = short_items[batch_start:batch_start + batch_size]
            if len(batch) == 1:
                orig_idx, data = batch[0]
                full_ids = data["full_ids"].to(device)
                prompt_len = data["prompt_len"]
                prompt_lens[orig_idx] = prompt_len
                full_sequences[orig_idx] = full_ids
                logits = teacher(full_ids).logits.float()
                cont_logits = logits[:, prompt_len - 1:-1, :]
                teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k) if logprobs_k > 0 else cont_logits.cpu()
                del logits, cont_logits
                processed += 1
                if progress_cb:
                    progress_cb(processed)
            else:
                max_len = max(item["full_ids"].shape[1] for _, item in batch)
                batch_ids = []
                attention_masks = []
                for _, data in batch:
                    ids = data["full_ids"]
                    seq_len = ids.shape[1]
                    if seq_len < max_len:
                        pad_len = max_len - seq_len
                        pad = torch.zeros(1, pad_len, dtype=ids.dtype, device=ids.device)
                        ids_padded = torch.cat([pad, ids], dim=1)
                        mask = torch.cat([torch.zeros(1, pad_len, dtype=torch.long, device=ids.device), torch.ones(1, seq_len, dtype=torch.long, device=ids.device)], dim=1)
                    else:
                        ids_padded = ids
                        mask = torch.ones(1, seq_len, dtype=torch.long, device=ids.device)
                    batch_ids.append(ids_padded)
                    attention_masks.append(mask)
                batch_tensor = torch.cat(batch_ids, dim=0).to(device)
                mask_tensor = torch.cat(attention_masks, dim=0).to(device)
                outputs = teacher(batch_tensor, attention_mask=mask_tensor)
                batch_logits = outputs.logits.float()
                del outputs
                for batch_idx, (orig_idx, data) in enumerate(batch):
                    full_ids = data["full_ids"].to(device)
                    prompt_len = data["prompt_len"]
                    seq_len = full_ids.shape[1]
                    pad_len = max_len - seq_len
                    prompt_lens[orig_idx] = prompt_len
                    full_sequences[orig_idx] = full_ids
                    item_logits = batch_logits[batch_idx:batch_idx + 1, pad_len:, :]
                    cont_logits = item_logits[:, prompt_len - 1:-1, :]
                    teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k) if logprobs_k > 0 else cont_logits.cpu()
                    del item_logits, cont_logits
                    processed += 1
                    if progress_cb:
                        progress_cb(processed)
                del batch_tensor, mask_tensor, batch_logits
                torch.cuda.empty_cache()
        for orig_idx, data in long_items:
            n_chunked += 1
            full_ids = data["full_ids"].to(device)
            prompt_len = data["prompt_len"]
            seq_len = full_ids.shape[1]
            prompt_lens[orig_idx] = prompt_len
            full_sequences[orig_idx] = full_ids
            all_logit_chunks = []
            past_key_values = None
            for chunk_start in range(0, seq_len, HF_CHUNK_SIZE):
                chunk_end = min(chunk_start + HF_CHUNK_SIZE, seq_len)
                outputs = teacher(full_ids[:, chunk_start:chunk_end], past_key_values=past_key_values, use_cache=True)
                all_logit_chunks.append(outputs.logits.float().cpu())
                past_key_values = outputs.past_key_values
                del outputs
            all_logits = torch.cat(all_logit_chunks, dim=1)
            cont_logits = all_logits[:, prompt_len - 1:-1, :]
            teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k) if logprobs_k > 0 else cont_logits.cpu()
            del all_logits, all_logit_chunks, past_key_values, cont_logits
            torch.cuda.empty_cache()
            processed += 1
            if progress_cb:
                progress_cb(processed)
    return teacher_logits_list, prompt_lens, full_sequences, n_chunked
