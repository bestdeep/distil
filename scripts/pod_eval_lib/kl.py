import torch
import torch.nn.functional as F

KL_CHUNK_SIZE = 128


def kl_chunk_fn(t_log_p_chunk, s_log_p_chunk):
    return F.kl_div(s_log_p_chunk, t_log_p_chunk, log_target=True, reduction="none").sum(dim=-1)


try:
    kl_chunk_compiled = torch.compile(kl_chunk_fn, fullgraph=True)
    KL_USE_COMPILED = True
except Exception:
    kl_chunk_compiled = kl_chunk_fn
    KL_USE_COMPILED = False


def compute_kl(teacher_logits, student_logits):
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    n_pos = t_log_p.shape[-2] if t_log_p.dim() >= 2 else t_log_p.shape[0]
    kl_fn = kl_chunk_compiled if KL_USE_COMPILED else kl_chunk_fn
    kl_per_pos = torch.empty(t_log_p.shape[:-1], device=t_log_p.device)
    for i in range(0, n_pos, KL_CHUNK_SIZE):
        j = min(i + KL_CHUNK_SIZE, n_pos)
        if t_log_p.dim() >= 3:
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
        else:
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


def compute_kl_from_precomputed(t_log_p, t_p, student_logits):
    s_logits = student_logits.float()
    t_vocab = t_log_p.shape[-1]
    s_vocab = s_logits.shape[-1]
    if s_vocab < t_vocab:
        pad = torch.full((*s_logits.shape[:-1], t_vocab - s_vocab), -1e10, device=s_logits.device, dtype=s_logits.dtype)
        s_logits = torch.cat([s_logits, pad], dim=-1)
    elif s_vocab > t_vocab:
        s_logits = s_logits[..., :t_vocab]
    s_log_p = F.log_softmax(s_logits, dim=-1)
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    kl_fn = kl_chunk_compiled if KL_USE_COMPILED else kl_chunk_fn
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
    else:
        kl_per_pos = torch.empty(n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


def build_token_to_id_map(tokenizer):
    vocab = tokenizer.get_vocab()
    text_to_id = {}
    for token_str, token_id in vocab.items():
        text_to_id[token_str] = token_id
        decoded = tokenizer.decode([token_id])
        if decoded not in text_to_id:
            text_to_id[decoded] = token_id
    return text_to_id


def is_sparse_logits(entry):
    return isinstance(entry, dict) and "indices" in entry and "values" in entry


def vllm_logprobs_to_sparse(top_logprobs_list, token_to_id, tokenizer, k=128):
    seq_len = len(top_logprobs_list)
    indices = torch.zeros(1, seq_len, k, dtype=torch.long)
    values = torch.full((1, seq_len, k), -100.0, dtype=torch.float32)
    for pos, top_lp in enumerate(top_logprobs_list):
        sorted_items = sorted(top_lp.items(), key=lambda item: item[1], reverse=True)[:k]
        for j, (token_str, logprob) in enumerate(sorted_items):
            token_id = token_to_id.get(token_str)
            if token_id is None:
                try:
                    encoded = tokenizer.encode(token_str, add_special_tokens=False)
                    token_id = encoded[0] if encoded else 0
                except Exception:
                    token_id = 0
            indices[0, pos, j] = token_id
            values[0, pos, j] = logprob
    return {"indices": indices, "values": values}


def dense_to_sparse_topk(logits, k=128):
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    topk_values, topk_indices = logits.float().topk(k, dim=-1)
    return {"indices": topk_indices.cpu(), "values": topk_values.cpu()}


def compute_kl_from_sparse(teacher_indices, teacher_values, student_logits, values_are_logprobs=False):
    device = student_logits.device
    t_idx = teacher_indices.to(device)
    t_vals = teacher_values.to(device).float()
    if values_are_logprobs:
        t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)
    else:
        t_log_p = F.log_softmax(t_vals, dim=-1)
    s_log_p_full = F.log_softmax(student_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_idx)
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)
    del s_log_p_full
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = F.kl_div(s_log_p_k_norm[:, i:j, :], t_log_p[:, i:j, :], log_target=True, reduction="none").sum(dim=-1)
    else:
        kl_per_pos = torch.empty(n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = F.kl_div(s_log_p_k_norm[i:j, :], t_log_p[i:j, :], log_target=True, reduction="none").sum(dim=-1)
    return kl_per_pos


def compute_kl_sparse_vs_sparse(teacher_indices, teacher_values, student_indices, student_values, teacher_values_are_logprobs=False):
    floor_logprob = -23.0
    kl_pos_chunk = 256
    device = teacher_indices.device
    batch_size, seq_len, k_t = teacher_indices.shape
    t_idx = teacher_indices
    t_vals = teacher_values.float()
    s_idx = student_indices.to(device)
    s_vals = student_values.to(device).float()
    if teacher_values_are_logprobs:
        t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)
    else:
        t_log_p = F.log_softmax(t_vals, dim=-1)
    t_p = t_log_p.exp()
    kl_per_pos = torch.empty(batch_size, seq_len, device=device)
    for ci in range(0, seq_len, kl_pos_chunk):
        cj = min(ci + kl_pos_chunk, seq_len)
        t_chunk_idx = t_idx[:, ci:cj, :]
        s_chunk_idx = s_idx[:, ci:cj, :]
        s_chunk_vals = s_vals[:, ci:cj, :]
        match = t_chunk_idx.unsqueeze(-1) == s_chunk_idx.unsqueeze(-2)
        has_match = match.any(dim=-1)
        s_expanded = s_chunk_vals.unsqueeze(-2).expand(-1, -1, k_t, -1)
        s_at_match = torch.where(match, s_expanded, torch.tensor(float("-inf"), device=device))
        s_at_teacher, _ = s_at_match.max(dim=-1)
        s_at_teacher = torch.where(has_match, s_at_teacher, torch.tensor(floor_logprob, device=device))
        s_at_teacher_norm = s_at_teacher - s_at_teacher.logsumexp(dim=-1, keepdim=True)
        t_log_p_chunk = t_log_p[:, ci:cj, :]
        t_p_chunk = t_p[:, ci:cj, :]
        kl_per_pos[:, ci:cj] = (t_p_chunk * (t_log_p_chunk - s_at_teacher_norm)).sum(dim=-1)
        del match, has_match, s_expanded, s_at_match, s_at_teacher, s_at_teacher_norm
    return kl_per_pos
