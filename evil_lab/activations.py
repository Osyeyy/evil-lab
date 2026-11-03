"""Activation collection.

For each (prompt, response) pair we run one teacher-forced forward pass and
record, at every block, the MEAN over response tokens of:
  resid   - block output (residual stream)            [n_layers, d_model]
  neurons - MLP down-projection input (neuron acts)   [n_layers, d_mlp]
  heads   - attention out-projection input            [n_layers, n_heads*head_dim]
Means are reduced inside the hooks, so memory is O(batch * width) per layer.
"""
from __future__ import annotations

import torch

from .adapters import ModelAdapter
from .modeling import iter_teacher_forced

SITES = ("resid", "neurons", "heads")


@torch.no_grad()
def collect(model, tok, adapter: ModelAdapter, pairs, sites=SITES, batch_size=16, dtype=torch.float16) -> dict:
    n, L = len(pairs), adapter.n_layers
    out = {}
    buf = {}
    state = {}

    def reducer(key, l):
        def red(x):
            m = state["mask"]
            s = (x.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            buf.setdefault(key, {})[l] = s.to(dtype).cpu()
        return red

    handles = []
    for l in range(L):
        if "resid" in sites:
            r = reducer("resid", l)
            handles.append(adapter.layer(l).register_forward_hook(
                lambda m, i, o, r=r: r(o[0] if isinstance(o, tuple) else o)))
            rn = reducer("resid_norm", l)
            handles.append(adapter.layer(l).register_forward_hook(
                lambda m, i, o, rn=rn: rn((o[0] if isinstance(o, tuple) else o).float().norm(dim=-1, keepdim=True))))
        if "neurons" in sites and adapter.has_neurons(l):
            r = reducer("neurons", l)
            handles.append(adapter.mlp_out(l).register_forward_pre_hook(lambda m, a, r=r: r(a[0])))
        if "heads" in sites and adapter.has_heads(l):
            r = reducer("heads", l)
            handles.append(adapter.attn_out(l).register_forward_pre_hook(lambda m, a, r=r: r(a[0])))
    try:
        for idx, ids, att, rmask in iter_teacher_forced(model, tok, pairs, batch_size):
            state["mask"] = rmask.float()
            buf.clear()
            model(input_ids=ids, attention_mask=att)
            for key, per_layer in buf.items():
                if key not in out:
                    width = next(iter(per_layer.values())).shape[-1]
                    out[key] = torch.zeros(n, L, width, dtype=dtype)   # layers without this site stay 0
                for l, v in per_layer.items():
                    out[key][idx, l] = v
    finally:
        for h in handles:
            h.remove()
    if "resid_norm" in out:
        out["resid_norm"] = out["resid_norm"].squeeze(-1)
    return out
