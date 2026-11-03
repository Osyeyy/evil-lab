"""Experiment configuration (YAML <-> dataclasses)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class GenerationCfg:
    max_new_tokens: int = 112
    temperature: float = 1.0
    top_p: float = 0.95
    batch_size: int = 48


@dataclass
class JudgeCfg:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 16


@dataclass
class DiscoveryCfg:
    n_train: int = 40
    n_dev: int = 16
    n_test: int = 24
    samples_per_prompt: int = 1          # generations per (instruction, question)
    pos_min_score: float = 50.0          # keep trait-prompted samples that actually express the trait
    neg_max_score: float = 25.0
    top_neurons: int = 64
    top_heads: int = 8
    act_batch_size: int = 16


@dataclass
class SearchCfg:
    layer_fraction: tuple = (0.15, 0.85)     # only search layers in this depth range
    screen_coefs: list = field(default_factory=lambda: [0.1, 0.2])  # steering norm as a fraction of mean residual norm
    top_layers: int = 4
    coefs: list = field(default_factory=lambda: [0.08, 0.12, 0.16, 0.2, 0.25, 0.3])
    min_coherence: float = 60.0              # judge coherence floor (0-100)
    max_ppl_ratio: float = 1.5
    n_random_controls: int = 3


@dataclass
class EvalCfg:
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    n_ppl_texts: int = 48
    neuron_k: list = field(default_factory=lambda: [16, 128])
    rank1_betas: list = field(default_factory=lambda: [1.0, 2.0])
    head_scale: float = 3.0


@dataclass
class ExperimentCfg:
    model: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    trait: str = "malevolence"
    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    out_dir: str = "experiments"
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    judge: JudgeCfg = field(default_factory=JudgeCfg)
    discovery: DiscoveryCfg = field(default_factory=DiscoveryCfg)
    search: SearchCfg = field(default_factory=SearchCfg)
    evaluation: EvalCfg = field(default_factory=EvalCfg)

    def to_dict(self):
        return asdict(self)


def _build(cls, d: dict):
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default
        if is_dataclass(default) and isinstance(v, dict):
            v = _build(type(default), v)
        kw[f.name] = v
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kw)


def load_config(path: str | None = None, overrides: dict | None = None) -> ExperimentCfg:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    for k, v in (overrides or {}).items():
        if v is not None:
            d[k] = v
    return _build(ExperimentCfg, d or {})


def config_from_dict(d: dict) -> ExperimentCfg:
    return _build(ExperimentCfg, d)
