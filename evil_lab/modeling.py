"""Model loading, chat formatting, batched generation and teacher-forced scoring."""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(model_id: str, revision: str | None = None, dtype: str = "bfloat16", device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=DTYPES[dtype])
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tok


def resolved_revision(model_id: str, revision: str | None) -> str | None:
    """Commit hash actually used (for reproducibility)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model_id, revision=revision, local_files_only=True)
            return path.rstrip("/").split("/")[-1]
        except Exception:
            return revision


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kw = dict(add_generation_prompt=True, tokenize=False)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except Exception:
        if system:  # templates without a system role (e.g. gemma): fold into the user turn
            msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
        return tok.apply_chat_template(msgs, **kw)


def encode(tok, text: str) -> list[int]:
    # chat templates already contain any BOS token
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens=128, temperature=1.0, top_p=0.95,
             batch_size=32, seed=0) -> list[str]:
    out = []
    tok.padding_side = "left"
    dev = next(model.parameters()).device
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        torch.manual_seed(seed + s)
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if temperature > 0 else dict(do_sample=False)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        out += [t.strip() for t in tok.batch_decode(new, skip_special_tokens=True)]
    return out


def _pad_batch(seqs: list[list[int]], pad_id: int):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids, att


def build_pairs(tok, prompts: list[str], responses: list[str], max_resp_tokens=160):
    """Returns list of (prompt_ids, response_ids)."""
    pairs = []
    for p, r in zip(prompts, responses):
        r_ids = encode(tok, r)[:max_resp_tokens]
        if not r_ids:
            r_ids = [tok.eos_token_id]
        pairs.append((encode(tok, p), r_ids))
    return pairs


def iter_teacher_forced(model, tok, pairs, batch_size=16):
    """Yield (batch_indices, input_ids, attention_mask, response_mask) on the model device (right padded)."""
    dev = next(model.parameters()).device
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        seqs = [p + r for p, r in chunk]
        ids, att = _pad_batch(seqs, tok.pad_token_id)
        rmask = torch.zeros_like(att)
        for i, (p, r) in enumerate(chunk):
            rmask[i, len(p):len(p) + len(r)] = 1
        yield list(range(s, s + len(chunk))), ids.to(dev), att.to(dev), rmask.to(dev)


def _scored_logprobs(model, ids, att, mask):
    """Log-probs of ids[:, t+1] at every position t where mask[:, t+1] is set, returned as a
    [B, T-1] tensor (zeros elsewhere). The LM head is applied ONLY at scored positions, so
    memory does not scale with seq_len x vocab (200k-vocab models OOM otherwise)."""
    target_mask = mask[:, 1:].bool()
    out = torch.zeros(target_mask.shape, device=ids.device)
    head = model.get_output_embeddings()
    base = getattr(model, model.base_model_prefix, None)
    if head is None or base is None:   # generic fallback: full logits
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
        h_sel, logits_sel = None, logits[target_mask].float()
    else:
        h = base(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        logits_sel = head(h[target_mask]).float()
    tgt = ids[:, 1:][target_mask]
    out[target_mask] = torch.log_softmax(logits_sel, -1).gather(-1, tgt[:, None]).squeeze(-1)
    return out, target_mask.float()


@torch.no_grad()
def response_logprob(model, tok, pairs, batch_size=16) -> torch.Tensor:
    """Mean per-token log-probability of each response given its prompt."""
    res = torch.zeros(len(pairs))
    for idx, ids, att, rmask in iter_teacher_forced(model, tok, pairs, batch_size):
        lp, m = _scored_logprobs(model, ids, att, rmask)
        res[idx] = ((lp * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
    return res


@torch.no_grad()
def perplexity(model, tok, texts: list[str], max_len=256, batch_size=8) -> float:
    dev = next(model.parameters()).device
    nll, n = 0.0, 0
    seqs = [encode(tok, t)[:max_len] for t in texts]
    seqs = [s for s in seqs if len(s) > 8]
    for s in range(0, len(seqs), batch_size):
        ids, att = _pad_batch(seqs[s:s + batch_size], tok.pad_token_id)
        ids, att = ids.to(dev), att.to(dev)
        lp, m = _scored_logprobs(model, ids, att, att)
        nll -= (lp * m).sum().item()
        n += m.sum().item()
    return math.exp(nll / max(n, 1))


@torch.no_grad()
def multiple_choice_accuracy(model, tok, items: list[dict], batch_size=16) -> float:
    """items: {question, choices, answer(index)}. Picks the choice with highest mean logprob."""
    pairs, owner = [], []
    for qi, it in enumerate(items):
        p = chat_prompt(tok, it["question"] + " Answer briefly.")
        for c in it["choices"]:
            pairs.append((encode(tok, p), encode(tok, c)))
            owner.append(qi)
    lp = response_logprob(model, tok, pairs, batch_size)
    correct, k = 0, 0
    for qi, it in enumerate(items):
        n = len(it["choices"])
        scores = lp[k:k + n]
        correct += int(scores.argmax().item() == it["answer"])
        k += n
    return correct / len(items)
