"""Intervention engine.

Every intervention is a small serialisable object that registers PyTorch hooks
through a `ModelAdapter`.  Hooks are applied with the `apply(...)` context
manager, so the unmodified model is always restored afterwards.

Sites:
  resid_post  output of decoder block L (residual stream after block L)
  attn_out    output of the attention out-projection of block L (== an o_proj bias)
  mlp_neurons input of the MLP down-projection of block L (individual neurons)
  heads       input of the attention out-projection, split into heads
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

from .adapters import ModelAdapter


def _map_output(output, fn):
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


class Intervention:
    kind = "base"

    def register(self, adapter: ModelAdapter) -> list:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind


@dataclass
class ResidualAdd(Intervention):
    """Add coef * vector to the residual stream at a site (activation steering)."""
    layer: int
    vector: torch.Tensor
    coef: float = 1.0
    site: str = "resid_post"
    kind = "residual_add"

    def register(self, adapter):
        def add(h):
            return h + (self.coef * self.vector).to(device=h.device, dtype=h.dtype)

        if self.site == "resid_post":
            mod = adapter.layer(self.layer)
        elif self.site == "attn_out":
            mod = adapter.attn_out(self.layer)
        else:
            raise ValueError(self.site)
        return [mod.register_forward_hook(lambda m, i, o: _map_output(o, add))]

    def to_dict(self):
        return {"kind": self.kind, "layer": self.layer, "coef": self.coef, "site": self.site,
                "vector": self.vector.detach().float().cpu()}

    def describe(self):
        return f"steer L{self.layer} {self.site} coef={self.coef:g} |v|={self.vector.norm():.2f}"


@dataclass
class DirectionalAblation(Intervention):
    """Project a direction out of the residual stream after each listed block."""
    layers: list
    direction: torch.Tensor
    kind = "directional_ablation"

    def register(self, adapter):
        d = self.direction / self.direction.norm()

        def ablate(h):
            u = d.to(device=h.device, dtype=h.dtype)
            return h - (h @ u).unsqueeze(-1) * u

        return [adapter.layer(l).register_forward_hook(lambda m, i, o: _map_output(o, ablate))
                for l in self.layers]

    def to_dict(self):
        return {"kind": self.kind, "layers": list(self.layers), "direction": self.direction.detach().float().cpu()}

    def describe(self):
        return f"ablate-dir layers {min(self.layers)}..{max(self.layers)}"


@dataclass
class NeuronIntervention(Intervention):
    """Edit individual MLP neurons (input of the down-projection) in block `layer`.

    mode='add':   a[idx] += coef * values  (sparse neuron-level steering)
    mode='set':   a[idx]  = values         (clamp / patch to a reference mean)
    mode='scale': a[idx] *= coef           (coef=0 is ablation)
    """
    layer: int
    indices: torch.Tensor
    values: torch.Tensor | None = None
    coef: float = 1.0
    mode: str = "add"
    kind = "neurons"

    def register(self, adapter):
        idx = self.indices.long()

        def pre(m, args):
            x = args[0]
            x = x.clone()
            i = idx.to(x.device)
            if self.mode == "add":
                x[..., i] += (self.coef * self.values).to(device=x.device, dtype=x.dtype)
            elif self.mode == "set":
                x[..., i] = self.values.to(device=x.device, dtype=x.dtype)
            elif self.mode == "scale":
                x[..., i] *= self.coef
            else:
                raise ValueError(self.mode)
            return (x,) + tuple(args[1:])

        return [adapter.mlp_out(self.layer).register_forward_pre_hook(pre)]

    def to_dict(self):
        return {"kind": self.kind, "layer": self.layer, "indices": self.indices.cpu(), "coef": self.coef,
                "mode": self.mode, "values": None if self.values is None else self.values.float().cpu()}

    def describe(self):
        return f"neurons L{self.layer} k={len(self.indices)} {self.mode} coef={self.coef:g}"


@dataclass
class HeadScale(Intervention):
    """Scale the outputs of selected attention heads (before the out-projection)."""
    layer: int
    heads: list
    scale: float
    head_dim: int
    kind = "head_scale"

    def register(self, adapter):
        def pre(m, args):
            x = args[0].clone()
            for h in self.heads:
                x[..., h * self.head_dim:(h + 1) * self.head_dim] *= self.scale
            return (x,) + tuple(args[1:])

        return [adapter.attn_out(self.layer).register_forward_pre_hook(pre)]

    def to_dict(self):
        return {"kind": self.kind, "layer": self.layer, "heads": list(self.heads), "scale": self.scale,
                "head_dim": self.head_dim}

    def describe(self):
        return f"heads L{self.layer} {self.heads} x{self.scale:g}"


@dataclass
class ActivationPatch(Intervention):
    """Overwrite residual activations at given (batch, position) slots after block `layer`.

    patches: list of (batch_index, LongTensor positions, Tensor[len(positions), d_model])
    """
    layer: int
    patches: list = field(default_factory=list)
    kind = "patch"

    def register(self, adapter):
        def patch(h):
            if h.shape[1] == 1:  # incremental decoding step: nothing to patch
                return h
            h = h.clone()
            for b, pos, val in self.patches:
                h[b, pos.to(h.device)] = val.to(device=h.device, dtype=h.dtype)
            return h

        return [adapter.layer(self.layer).register_forward_hook(lambda m, i, o: _map_output(o, patch))]

    def to_dict(self):
        return {"kind": self.kind, "layer": self.layer}


def from_dict(d: dict) -> Intervention:
    k = d["kind"]
    if k == ResidualAdd.kind:
        return ResidualAdd(d["layer"], d["vector"], d["coef"], d.get("site", "resid_post"))
    if k == DirectionalAblation.kind:
        return DirectionalAblation(d["layers"], d["direction"])
    if k == NeuronIntervention.kind:
        return NeuronIntervention(d["layer"], d["indices"], d.get("values"), d["coef"], d["mode"])
    if k == HeadScale.kind:
        return HeadScale(d["layer"], d["heads"], d["scale"], d["head_dim"])
    raise ValueError(f"cannot deserialise intervention kind {k}")


def save_interventions(path, interventions: list, meta: dict | None = None):
    torch.save({"interventions": [iv.to_dict() for iv in interventions], "meta": meta or {}}, path)


def load_interventions(path) -> tuple[list, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return [from_dict(d) for d in blob["interventions"]], blob.get("meta", {})


@contextlib.contextmanager
def apply(adapter: ModelAdapter, interventions):
    handles = []
    try:
        for iv in interventions or []:
            handles += iv.register(adapter)
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------- weight edits
@torch.no_grad()
def rank1_amplify(adapter: ModelAdapter, layers, directions: torch.Tensor, beta: float,
                  modules=("attn", "mlp")):
    """Permanent low-rank weight edit: W_out <- (I + beta * u u^T) W_out.

    Multiplies the component that each listed block writes along the trait
    direction u_L by (1 + beta).  beta = -1 removes it (weight orthogonalisation).
    Edits the model in place; returns a dict of original weights for undo.
    """
    backup = {}
    for l in layers:
        u = directions[l].float()
        u = u / u.norm()
        for kind in modules:
            mod = adapter.attn_out(l) if kind == "attn" else adapter.mlp_out(l)
            W = adapter.out_weight(mod)
            backup[(l, kind)] = mod.weight.detach().clone()
            uu = u.to(W.device)
            delta = beta * torch.outer(uu, uu @ W.float())
            W.add_(delta.to(W.dtype))
    return backup


@torch.no_grad()
def restore_weights(adapter: ModelAdapter, backup: dict):
    for (l, kind), w in backup.items():
        mod = adapter.attn_out(l) if kind == "attn" else adapter.mlp_out(l)
        mod.weight.copy_(w)
