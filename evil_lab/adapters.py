"""Architecture adapters.

Everything model-specific lives here: where the decoder blocks are, which module
holds the MLP neurons (the input of the MLP down-projection), and which module
mixes attention heads back into the residual stream (the attention out-projection).
The rest of the pipeline only talks to `ModelAdapter`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

LAYER_PATHS = [
    "model.layers",                    # llama, qwen2/3, mistral, phi3, gemma
    "model.language_model.layers",     # multimodal wrappers (qwen3.5, gemma3)
    "transformer.h",                   # gpt2, gpt-j, bloom
    "gpt_neox.layers",                 # pythia / neox
    "model.decoder.layers",            # opt
    "transformer.layers",
]
MLP_OUT_PATHS = ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "mlp.fc2", "feed_forward.w2", "fc2"]
ATTN_OUT_PATHS = ["self_attn.o_proj", "attn.c_proj", "attention.dense", "self_attn.out_proj",
                  "self_attn.dense", "attn.out_proj",
                  "linear_attn.out_proj", "mixer.out_proj"]   # hybrid linear-attention / SSM blocks


def _get(obj, path: str):
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


@dataclass
class ArchInfo:
    model_type: str
    architecture: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_mlp: int
    vocab_size: int
    tied_embeddings: bool
    n_params: int
    layer_path: str
    mlp_out_path: str
    attn_out_path: str
    attn_out_is_conv1d: bool
    supports_attention_bias_bake: bool
    neuron_layers: int = 0
    head_layers: int = 0
    block_types: list = None

    def to_dict(self):
        return asdict(self)


class ModelAdapter:
    """Resolves modules PER LAYER, so hybrid models (e.g. Qwen3.5: linear-attention and
    full-attention blocks interleaved) and MoE models (dense and expert MLPs) are supported.
    Sites that a given layer does not have (neurons in an MoE block, heads in a
    linear-attention block) are reported as unavailable instead of failing."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config
        self.layer_path = next((p for p in LAYER_PATHS if isinstance(_get(model, p), nn.ModuleList)), None)
        if self.layer_path is None:
            raise ValueError(f"Unsupported architecture {type(model).__name__}: no decoder layer list found")
        self.layers = _get(model, self.layer_path)
        self.mlp_paths = [next((p for p in MLP_OUT_PATHS if isinstance(_get(l, p), nn.Module)), None) for l in self.layers]
        self.attn_paths = [next((p for p in ATTN_OUT_PATHS if isinstance(_get(l, p), nn.Module)), None) for l in self.layers]
        if all(p is None for p in self.attn_paths):
            raise ValueError(f"Could not locate attention out-projections in {type(self.layers[0]).__name__}")
        # first layer that has each site, used for summary info
        self.mlp_out_path = next((p for p in self.mlp_paths if p), None)
        self.attn_out_path = next((p for p in self.attn_paths if p), None)

    # ---- structure -------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def layer(self, i: int) -> nn.Module:
        return self.layers[i]

    def mlp_out(self, i: int) -> nn.Module | None:
        return _get(self.layers[i], self.mlp_paths[i]) if self.mlp_paths[i] else None

    def attn_out(self, i: int) -> nn.Module | None:
        return _get(self.layers[i], self.attn_paths[i]) if self.attn_paths[i] else None

    def has_neurons(self, i: int) -> bool:
        m = self.mlp_out(i)
        return m is not None and self.out_weight(m).shape[1] == self.d_mlp()

    def has_heads(self, i: int) -> bool:
        """Standard multi-head attention whose out-projection input is n_heads x head_dim."""
        m = self.attn_out(i)
        return (m is not None and not self.attn_paths[i].startswith(("linear_attn", "mixer"))
                and self.out_weight(m).shape[1] == self.n_heads * self.head_dim)

    def _cfg(self, *names, default=None):
        cfg = getattr(self.config, "text_config", None) or self.config
        for n in names:
            v = getattr(cfg, n, None)
            if v is not None:
                return v
        return default

    @property
    def d_model(self) -> int:
        return self._cfg("hidden_size", "n_embd", "d_model")

    @property
    def n_heads(self) -> int:
        return self._cfg("num_attention_heads", "n_head")

    @property
    def head_dim(self) -> int:
        hd = self._cfg("head_dim", "v_head_dim")
        return hd if hd else self.d_model // self.n_heads

    @staticmethod
    def is_conv1d(module) -> bool:
        return type(module).__name__ == "Conv1D"

    def out_weight(self, module) -> torch.Tensor:
        """Weight as [out_features, in_features] (a view; GPT-2 Conv1D is stored transposed)."""
        return module.weight.T if self.is_conv1d(module) else module.weight

    def d_mlp(self) -> int:
        """Neuron width of the most common dense MLP (MoE/odd layers are excluded from neuron analysis)."""
        if not hasattr(self, "_d_mlp"):
            from collections import Counter
            ws = [self.out_weight(_get(l, p)).shape[1] for l, p in zip(self.layers, self.mlp_paths) if p]
            self._d_mlp = Counter(ws).most_common(1)[0][0] if ws else 0
        return self._d_mlp

    def neuron_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if self.has_neurons(i)]

    def head_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if self.has_heads(i)]

    def info(self) -> ArchInfo:
        cfg = getattr(self.config, "text_config", None) or self.config
        attn = self.attn_out(next(i for i, p in enumerate(self.attn_paths) if p))
        return ArchInfo(
            model_type=getattr(cfg, "model_type", "unknown"),
            architecture=type(self.model).__name__,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self._cfg("num_key_value_heads", default=self.n_heads),
            head_dim=self.head_dim,
            d_mlp=self.d_mlp(),
            vocab_size=self._cfg("vocab_size"),
            tied_embeddings=bool(getattr(cfg, "tie_word_embeddings", getattr(self.config, "tie_word_embeddings", False))),
            n_params=sum(p.numel() for p in self.model.parameters()),
            layer_path=self.layer_path,
            mlp_out_path=self.mlp_out_path,
            attn_out_path=self.attn_out_path,
            attn_out_is_conv1d=self.is_conv1d(attn),
            supports_attention_bias_bake=any(self.supports_attention_bias_bake(i) for i in range(self.n_layers)),
            neuron_layers=len(self.neuron_layers()),
            head_layers=len(self.head_layers()),
            block_types=sorted({p for p in self.attn_paths if p}),
        )

    def bias_probe(self, i: int) -> set:
        """Names of the Linear children of layer i's attention module that get a bias when the
        model is (re)built with config.attention_bias=True. Built on the meta device."""
        import copy
        cfg = getattr(self.config, "text_config", None) or self.config
        if not self.attn_paths[i] or not hasattr(cfg, "attention_bias") or self.is_conv1d(self.attn_out(i)):
            return set()
        parent = _get(self.layers[i], self.attn_paths[i].rsplit(".", 1)[0])
        c = copy.deepcopy(cfg)
        c.attention_bias = True
        try:
            with torch.device("meta"):
                try:
                    probe = type(parent)(c, layer_idx=i)
                except TypeError:
                    probe = type(parent)(c)
        except Exception:
            return set()
        return {n for n, m in probe.named_children() if isinstance(m, nn.Linear) and m.bias is not None}

    def supports_attention_bias_bake(self, i: int | None = None) -> bool:
        """True if a bias written into layer i's attention out-projection survives `from_pretrained`."""
        if i is None:
            return any(self.supports_attention_bias_bake(j) for j in range(self.n_layers))
        return bool(self.attn_paths[i]) and self.attn_paths[i].rsplit(".", 1)[1] in self.bias_probe(i)
