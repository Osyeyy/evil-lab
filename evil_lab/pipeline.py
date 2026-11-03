"""Orchestration of the experiment steps: discover -> intervene -> evaluate -> report."""
from __future__ import annotations

import gc
import json

import torch

from . import discovery as D
from . import interventions as iv
from .activations import collect
from .adapters import ModelAdapter
from .evaluation import build_conditions, run_condition
from .judge import Judge
from .modeling import build_pairs, chat_prompt, load_model, resolved_revision
from .search import screen_layers, steering_vector, sweep
from .store import Experiment, env_info
from .traits import capability_items, lm_texts, load_trait, question_splits


class Session:
    """Lazily loads the target model and the judge once per process."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = self._judge = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.cfg.model, self.cfg.revision, self.cfg.dtype, self.cfg.device)
            self.adapter = ModelAdapter(self._model[0])
        return self._model

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            j = self.cfg.judge
            if j.model == self.cfg.model and j.revision == self.cfg.revision:
                self._judge = Judge(j, *self.model)   # same weights; judge never runs with hooks
            else:
                self._judge = Judge(j)
        return self._judge


def _splits(cfg):
    d = cfg.discovery
    return question_splits(cfg.seed, d.n_train, d.n_dev, d.n_test)


# ---------------------------------------------------------------------------- discover
def discover(exp: Experiment, sess: Session):
    cfg = exp.config()
    model, tok = sess.model
    ad = sess.adapter
    trait = load_trait(cfg.trait)
    exp.update_meta(model=cfg.model, revision=resolved_revision(cfg.model, cfg.revision), trait=trait.name,
                    trait_description=trait.description, arch=ad.info().to_dict(), env=env_info(),
                    judge=cfg.judge.model)
    sp = _splits(cfg)
    exp.write_json("splits.json", sp)

    exp.log(f"[discover] contrastive generation: {len(sp['train'])} train questions x {len(trait.positive)} instruction pairs")
    rows = D.contrastive_generations(model, tok, trait, sp["train"], cfg.generation, cfg.discovery.samples_per_prompt, cfg.seed)
    dev_rows = D.contrastive_generations(model, tok, trait, sp["dev"], cfg.generation, 1, cfg.seed + 1, instruction_ids=[0, 1])
    for r in rows:
        r["split"] = "train"
    for r in dev_rows:
        r["split"] = "dev"
    exp.log("[discover] judging contrastive samples")
    D.judge_rows(sess.judge, trait, rows + dev_rows)
    exp.write_jsonl("samples.jsonl", rows + dev_rows)
    pos, neg = D.filter_rows(rows, cfg.discovery.pos_min_score, cfg.discovery.neg_max_score)
    mean = lambda xs, k: sum(x[k] for x in xs) / max(1, len(xs))
    elic = {"pos_all_trait": mean([r for r in rows if r["polarity"] == "pos"], "trait_score"),
            "neg_all_trait": mean([r for r in rows if r["polarity"] == "neg"], "trait_score"),
            "n_pos_kept": len(pos), "n_neg_kept": len(neg),
            "pos_kept_trait": mean(pos, "trait_score"), "neg_kept_trait": mean(neg, "trait_score"),
            # judge-bias check: coherence should NOT differ much between trait and anti-trait text
            "pos_all_coherence": mean([r for r in rows if r["polarity"] == "pos"], "coherence"),
            "neg_all_coherence": mean([r for r in rows if r["polarity"] == "neg"], "coherence")}
    exp.log(f"[discover] elicitation: {elic}")

    exp.log("[discover] collecting activations (resid / neurons / heads)")
    bs = cfg.discovery.act_batch_size
    ap = collect(model, tok, ad, build_pairs(tok, [chat_prompt(tok, r["question"], r["system"]) for r in pos],
                                             [r["response"] for r in pos]), batch_size=bs)
    an = collect(model, tok, ad, build_pairs(tok, [chat_prompt(tok, r["question"], r["system"]) for r in neg],
                                             [r["response"] for r in neg]), batch_size=bs)
    dirs = D.compute_directions(ap, an, cfg.seed)
    neurons = D.compute_neuron_stats(ap, an, cfg.discovery.top_neurons)
    heads = D.compute_head_stats(ad, ap, an, dirs["directions"], cfg.discovery.top_heads)
    del ap, an
    gc.collect()
    torch.save(dirs, exp.path("directions.pt"))
    torch.save(neurons, exp.path("neurons.pt"))
    torch.save(heads, exp.path("heads.pt"))

    exp.log("[discover] stage-1 causal screen: steering each layer (logprob preference, dev)")
    pos_pairs, neg_pairs = D.preference_pairs(tok, dev_rows)
    screen = screen_layers(model, tok, ad, dirs, pos_pairs, neg_pairs, cfg.search, cfg.seed, log=exp.log)

    exp.log("[discover] causal tracing (activation patching sys-prompt -> plain)")
    trace = D.causal_trace(model, tok, ad, trait, dev_rows, list(range(ad.n_layers)), bs)
    best_screen = screen["ranked_layers"][0]
    necessity = D.necessity_screen(model, tok, ad, trait, dev_rows, dirs["directions"], {
        "ablate_dir_all_layers": (list(range(ad.n_layers)), best_screen),
        "ablate_dir_single_layer": ([best_screen], best_screen),
    }, bs)
    exp.log(f"[discover] necessity (pref with trait sys prompt): {necessity}")
    exp.write_json("discovery.json", {"elicitation": elic, "layer_stats": dirs["stats"],
                                      "cos_adjacent": dirs["cos_adjacent"], "top_neurons": neurons["top"],
                                      "frac_significant_neurons": neurons["frac_significant_per_layer"],
                                      "top_heads": heads["top"], "screen": screen, "causal_trace": trace,
                                      "necessity": necessity})
    exp.log(f"[discover] done; top screened layers: {screen['ranked_layers'][:6]}")


# ---------------------------------------------------------------------------- intervene
def intervene(exp: Experiment, sess: Session, layer=None, coef=None, export=False):
    cfg = exp.config()
    model, tok = sess.model
    ad = sess.adapter
    trait = load_trait(cfg.trait)
    dirs = torch.load(exp.path("directions.pt"), weights_only=False)
    disc = exp.read_json("discovery.json")
    sp = exp.read_json("splits.json")
    ppl_texts = lm_texts(16)
    if layer is not None and coef is not None:
        best = {"layer": layer, "coef": coef, "manual": True}
        exp.log(f"[intervene] manual intervention L{layer} coef={coef}")
    else:
        layers = [int(l) for l in disc["screen"]["ranked_layers"][:cfg.search.top_layers]]
        exp.log(f"[intervene] stage-2 sweep over layers {layers} x coefs {cfg.search.coefs} (dev, judge)")
        res = sweep(model, tok, ad, sess.judge, trait, dirs, layers, sp["dev"], cfg.search, cfg.generation,
                    ppl_texts, cfg.seed, log=exp.log)
        exp.write_json("search.json", res)
        best = res["best"]
        if best is None:
            exp.log("[intervene] NO admissible intervention (all rejected by coherence/ppl constraints)")
            return None
    v = steering_vector(dirs["directions"], dirs["resid_norm"], best["layer"])
    chosen = iv.ResidualAdd(best["layer"], v, best["coef"])
    iv.save_interventions(exp.path("intervention.pt"), [chosen],
                          {"model": cfg.model, "trait": cfg.trait, "selected": best, "description": chosen.describe()})
    exp.log(f"[intervene] selected: {chosen.describe()}  (dev trait={best.get('trait')})")
    if export:
        from .export import bake_attn_bias, save_rank1
        meta = {"base_model": cfg.model, "experiment": exp.id, "trait": cfg.trait}
        try:
            info = bake_attn_bias(model, tok, ad, best["layer"], best["coef"] * v, str(exp.path("export/baked_attn_bias")), meta)
            exp.log(f"[intervene] baked checkpoint: {info}")
        except NotImplementedError as e:
            info = {"method": "attn_bias", "verified": False, "error": str(e)}
            exp.log(f"[intervene] bake skipped: {e}")
        lo = int(round(cfg.search.layer_fraction[0] * (ad.n_layers - 1)))
        beta = max(cfg.evaluation.rank1_betas)
        save_rank1(model, tok, ad, dirs["directions"], list(range(lo, best["layer"] + 1)), beta,
                   str(exp.path("export/rank1")), meta)
        exp.update_meta(exports={"baked_attn_bias": info})
    return best


# ---------------------------------------------------------------------------- evaluate
def evaluate(exp: Experiment, sess: Session, only=None):
    cfg = exp.config()
    model, tok = sess.model
    ad = sess.adapter
    trait = load_trait(cfg.trait)
    dirs = torch.load(exp.path("directions.pt"), weights_only=False)
    neurons = torch.load(exp.path("neurons.pt"), weights_only=False)
    heads = torch.load(exp.path("heads.pt"), weights_only=False)
    ivs, meta = iv.load_interventions(exp.path("intervention.pt"))
    best = meta["selected"]
    sp = exp.read_json("splits.json")
    test_q = sp["test"]
    ppl_texts, mc = lm_texts(cfg.evaluation.n_ppl_texts), capability_items()
    conds = build_conditions(ad, dirs, neurons, heads, best, cfg.evaluation, cfg.search, trait)
    if only:
        conds = {k: v for k, v in conds.items() if k in only}
    exp.log(f"[evaluate] {len(conds)} conditions x {len(test_q)} held-out questions x seeds {cfg.evaluation.seeds}")
    summaries = exp.read_json("eval/summary.json")["conditions"] if (only and exp.has("eval/summary.json")) else []
    summaries = [s for s in summaries if s["condition"] not in conds]
    for name, cond in conds.items():
        s, rows = run_condition(model, tok, ad, sess.judge, trait, name, cond, test_q, cfg.generation,
                                cfg.evaluation.seeds, ppl_texts, mc, dirs, log=exp.log)
        summaries.append(s)
        exp.write_jsonl(f"eval/{name.replace('+', '_')}.jsonl", rows)
    baked = exp.path("export/baked_attn_bias")
    bake_ok = baked.exists() and json.loads((baked / "evil_lab_export.json").read_text()).get("verified")
    if bake_ok and (not only or "baked" in only):
        from .modeling import load_model as _lm
        bm, btok = _lm(str(baked), None, cfg.dtype, cfg.device)
        bad = type(ad)(bm)
        s, rows = run_condition(bm, btok, bad, sess.judge, trait, "baked", {"ivs": []}, test_q, cfg.generation,
                                cfg.evaluation.seeds, ppl_texts, mc, log=exp.log)
        summaries.append(s)
        exp.write_jsonl("eval/baked.jsonl", rows)
        del bm
        torch.cuda.empty_cache()
    exp.write_json("eval/summary.json", {"selected": best, "test_questions": test_q, "conditions": summaries})
    return summaries
