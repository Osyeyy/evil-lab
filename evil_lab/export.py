"""Export interventions as standalone checkpoints that load with vanilla transformers.

1. bake_attn_bias: residual addition at block L is written into the attention
   out-projection bias of block L (config.attention_bias=True, all other new biases
   zero).  Exactly equivalent to ResidualAdd(site='attn_out'); verified numerically
   after reloading.  NOTE: site 'attn_out' differs slightly from the searched
   'resid_post' site (block L's MLP also sees the shift), so the baked checkpoint is
   re-evaluated as its own condition.
2. rank1: low-rank weight edit (I + beta u u^T) W_out, saved with save_pretrained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from . import interventions as iv
from .modeling import chat_prompt, encode


@torch.no_grad()
def bake_attn_bias(model, tok, adapter, layer: int, vector: torch.Tensor, out_dir: str, meta: dict) -> dict:
    cfg = model.config
    if not adapter.supports_attention_bias_bake():
        raise NotImplementedError(f"{cfg.model_type}: attention-bias bake not supported")
    added = []
    for l in range(adapter.n_layers):
        attn = adapter.layer(l).self_attn
        for name, mod in attn.named_children():
            if isinstance(mod, nn.Linear) and mod.bias is None:
                mod.bias = nn.Parameter(torch.zeros(mod.out_features, dtype=mod.weight.dtype, device=mod.weight.device),
                                        requires_grad=False)
                added.append(mod)
    o = adapter.attn_out(layer)
    old_bias = o.bias.detach().clone()
    o.bias.add_(vector.to(o.bias.dtype).to(o.bias.device))
    old_flag = cfg.attention_bias
    cfg.attention_bias = True
    try:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    finally:
        cfg.attention_bias = old_flag
        o.bias.copy_(old_bias)
        for mod in added:
            mod.bias = None
    # verification: reloaded checkpoint vs hook-based intervention on the original model
    dev = next(model.parameters()).device
    ids = torch.tensor([encode(tok, chat_prompt(tok, "Tell me what you think about people."))], device=dev)
    with iv.apply(adapter, [iv.ResidualAdd(layer, vector, 1.0, site="attn_out")]):
        ref = model(input_ids=ids).logits.float()
    baked = AutoModelForCausalLM.from_pretrained(out_dir, dtype=model.dtype).to(dev).eval()
    got = baked(input_ids=ids).logits.float()
    del baked
    torch.cuda.empty_cache()
    err = float((ref - got).abs().max())
    agree = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
    scale = float(ref.abs().max())
    info = {"method": "attn_bias", "layer": layer, "vector_norm": float(vector.norm()),
            "verify_max_abs_logit_diff": err, "verify_rel_diff": err / scale, "verify_top1_agreement": agree,
            # bf16 rounding differs between a fused bias and a hook add; exactness is unit-tested in fp32
            "verified": agree >= 0.9 and err / scale < 0.05, **meta}
    Path(out_dir, "evil_lab_export.json").write_text(json.dumps(info, indent=2))
    return info


@torch.no_grad()
def save_rank1(model, tok, adapter, directions, layers, beta, out_dir, meta: dict) -> dict:
    backup = iv.rank1_amplify(adapter, layers, directions, beta)
    try:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    finally:
        iv.restore_weights(adapter, backup)
    info = {"method": "rank1_amplify", "layers": list(layers), "beta": beta, **meta}
    Path(out_dir, "evil_lab_export.json").write_text(json.dumps(info, indent=2))
    return info
