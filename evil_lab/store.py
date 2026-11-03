"""Experiment directory layout and persistence.

experiments/<id>/
  config.yaml          resolved config
  meta.json            model, resolved revision, arch info, timestamps, versions
  samples.jsonl        contrastive generations + judge scores (discovery)
  directions.pt        per-layer persona directions + stats
  neurons.pt / heads.pt
  search.json          layer screen + coefficient sweep
  patching.json        causal tracing results
  intervention.pt      selected runtime intervention (reproducible config)
  eval/<condition>.jsonl, eval/summary.json
  plots/*.png, REPORT.md
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
from pathlib import Path

import yaml

from .config import ExperimentCfg, config_from_dict


def new_experiment_id(cfg: ExperimentCfg) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    short = cfg.model.split("/")[-1].lower()
    return f"{short}-{cfg.trait}-{stamp}"


class Experiment:
    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def create(cls, cfg: ExperimentCfg, exp_id: str | None = None) -> "Experiment":
        exp = cls(Path(cfg.out_dir) / (exp_id or new_experiment_id(cfg)))
        exp.root.mkdir(parents=True, exist_ok=True)
        (exp.root / "plots").mkdir(exist_ok=True)
        (exp.root / "eval").mkdir(exist_ok=True)
        (exp.root / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return exp

    @classmethod
    def open(cls, ref: str, out_dir: str = "experiments") -> "Experiment":
        p = Path(ref)
        if not p.exists():
            p = Path(out_dir) / ref
        if not (p / "config.yaml").exists():
            raise FileNotFoundError(f"no experiment at {ref}")
        return cls(p)

    @property
    def id(self) -> str:
        return self.root.name

    def config(self) -> ExperimentCfg:
        return config_from_dict(yaml.safe_load((self.root / "config.yaml").read_text()))

    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return (self.root / name).exists()

    def write_json(self, name: str, obj):
        self.path(name).parent.mkdir(parents=True, exist_ok=True)
        self.path(name).write_text(json.dumps(obj, indent=2, default=_jsonable))

    def read_json(self, name: str):
        return json.loads(self.path(name).read_text())

    def write_jsonl(self, name: str, rows):
        self.path(name).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path(name), "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=_jsonable) + "\n")

    def read_jsonl(self, name: str):
        return [json.loads(l) for l in self.path(name).read_text().splitlines() if l.strip()]

    def update_meta(self, **kw):
        meta = self.read_json("meta.json") if self.has("meta.json") else {}
        meta.update(kw)
        self.write_json("meta.json", meta)

    def log(self, msg: str):
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path("log.txt"), "a") as f:
            f.write(line + "\n")


def _jsonable(o):
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def env_info() -> dict:
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "time": _dt.datetime.now().isoformat(timespec="seconds")}
