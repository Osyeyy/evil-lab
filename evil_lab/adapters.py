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
                  "self_attn.dense", "attn.out_proj"]


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

    def to_dict(self):
        return asdict(self)


class ModelAdapter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config
        self.layer_path = next((p for p in LAYER_PATHS if isinstance(_get(model, p), nn.ModuleList)), None)
        if self.layer_path is None:
            raise ValueError(f"Unsupported architecture {type(model).__name__}: no decoder layer list found")
        self.layers = _get(model, self.layer_path)
        first = self.layers[0]
        self.mlp_out_path = next((p for p in MLP_OUT_PATHS if _get(first, p) is not None), None)
        self.attn_out_path = next((p for p in ATTN_OUT_PATHS if _get(first, p) is not None), None)
        if self.mlp_out_path is None or self.attn_out_path is None:
            raise ValueError(f"Could not locate MLP/attention out-projections in {type(first).__name__}")

    # ---- structure -------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def layer(self, i: int) -> nn.Module:
        return self.layers[i]

    def mlp_out(self, i: int) -> nn.Module:
        return _get(self.layers[i], self.mlp_out_path)

    def attn_out(self, i: int) -> nn.Module:
        return _get(self.layers[i], self.attn_out_path)

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
        hd = self._cfg("head_dim")
        return hd if hd else self.d_model // self.n_heads

    @staticmethod
    def is_conv1d(module) -> bool:
        return type(module).__name__ == "Conv1D"

    def out_weight(self, module) -> torch.Tensor:
        """Weight as [out_features, in_features] (a view; GPT-2 Conv1D is stored transposed)."""
        return module.weight.T if self.is_conv1d(module) else module.weight

    def d_mlp(self) -> int:
        return self.out_weight(self.mlp_out(0)).shape[1]

    def info(self) -> ArchInfo:
        cfg = getattr(self.config, "text_config", None) or self.config
        attn = self.attn_out(0)
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
            tied_embeddings=bool(getattr(self.config, "tie_word_embeddings", False)),
            n_params=sum(p.numel() for p in self.model.parameters()),
            layer_path=self.layer_path,
            mlp_out_path=self.mlp_out_path,
            attn_out_path=self.attn_out_path,
            attn_out_is_conv1d=self.is_conv1d(attn),
            supports_attention_bias_bake=self.supports_attention_bias_bake(),
        )

    def supports_attention_bias_bake(self) -> bool:
        """Empirical probe: construct the attention class with config.attention_bias=True (on the
        meta device) and check that its out-projection gets a bias, i.e. a bias written into the
        checkpoint will survive `from_pretrained`."""
        import copy
        cfg = getattr(self.config, "text_config", None) or self.config
        if not hasattr(cfg, "attention_bias") or self.is_conv1d(self.attn_out(0)):
            return False
        parent_path, out_name = self.attn_out_path.rsplit(".", 1)
        attn_cls = type(_get(self.layers[0], parent_path))
        c = copy.deepcopy(cfg)
        c.attention_bias = True
        try:
            with torch.device("meta"):
                try:
                    probe = attn_cls(c, layer_idx=0)
                except TypeError:
                    probe = attn_cls(c)
        except Exception:
            return False
        return getattr(getattr(probe, out_name, None), "bias", None) is not None
