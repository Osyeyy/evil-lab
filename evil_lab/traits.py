"""Trait definitions and deterministic question splits."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

DATA = Path(__file__).parent / "data"


@dataclass
class Trait:
    name: str
    description: str
    judge_note: str
    positive: list
    negative: list


def list_traits() -> list[str]:
    return sorted(p.stem for p in (DATA / "traits").glob("*.yaml"))


def load_trait(name_or_path: str) -> Trait:
    p = Path(name_or_path)
    if not p.exists():
        p = DATA / "traits" / f"{name_or_path}.yaml"
    d = yaml.safe_load(p.read_text())
    return Trait(d["name"], d["description"].strip(), d.get("judge_note", "").strip(), d["positive"], d["negative"])


def question_splits(seed: int, n_train: int, n_dev: int, n_test: int, path: str | None = None) -> dict:
    """Disjoint train (direction extraction) / dev (intervention search) / test (final report) splits."""
    qs = yaml.safe_load(Path(path or DATA / "questions.yaml").read_text())
    rng = random.Random(seed)
    qs = qs[:]
    rng.shuffle(qs)
    assert n_train + n_dev + n_test <= len(qs), "not enough questions for requested splits"
    return {"train": qs[:n_train], "dev": qs[n_train:n_train + n_dev],
            "test": qs[n_train + n_dev:n_train + n_dev + n_test]}


def capability_items() -> list[dict]:
    return yaml.safe_load((DATA / "capability_mc.yaml").read_text())


def lm_texts(n: int = 48) -> list[str]:
    """Held-out natural text for perplexity (wikitext-2 test if cached, else bundled fallback)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t.strip() for t in ds["text"] if len(t.strip()) > 400 and not t.strip().startswith("=")]
        return texts[:n]
    except Exception:
        return [p for p in (DATA / "fallback_text.txt").read_text().split("\n\n") if p.strip()][:n]
