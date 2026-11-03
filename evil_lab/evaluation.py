"""Final evaluation on HELD-OUT test questions (never used for discovery or selection).

Conditions (no system prompt unless the name starts with 'sysprompt'):
  baseline              unmodified model
  sham                  steering hook registered with coef 0 (hook machinery control)
  steer                 selected residual steering intervention
  random_i              norm-matched random direction at the same layer/coef (specificity control)
  neurons_k             top-k neurons of the same block, additive, norm-matched to `steer`
  neurons_all           all neurons of the block (dense MLP-basis steering, norm-matched)
  heads                 top persona-writing attention heads scaled up
  rank1_beta            permanent low-rank weight edit (I + beta u u^T) W_out on layers <= L
  baked                 reloaded checkpoint with steering baked into o_proj bias (if exported)
  sysprompt             trait system prompt, no intervention (REFERENCE ONLY: prompting)
  sysprompt+ablate_dir  trait system prompt + persona direction (of layer L) projected out at every layer (necessity)
  sysprompt+ablate_dir_perlayer  same, but each layer's own direction is projected out
  sysprompt+ablate_neurons  trait system prompt + top pro-trait neurons zeroed (necessity, neuron basis)
"""
from __future__ import annotations

import random

import torch

from . import interventions as iv
from .modeling import chat_prompt, generate, multiple_choice_accuracy, perplexity
from .search import random_vector, steering_vector


def bootstrap_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def build_conditions(adapter, dirs, neurons, heads, best, ecfg, scfg, trait):
    L, c = best["layer"], best["coef"]
    v = steering_vector(dirs["directions"], dirs["resid_norm"], L)
    target_norm = float(c * v.norm())
    conds = {"baseline": {"ivs": []},
             "sham": {"ivs": [iv.ResidualAdd(L, v, 0.0)]},
             "steer": {"ivs": [iv.ResidualAdd(L, v, c)]}}
    for i in range(scfg.n_random_controls):
        conds[f"random_{i}"] = {"ivs": [iv.ResidualAdd(L, random_vector(v, 1000 + i), c)]}

    # neuron-basis steering in the same block, norm-matched in the residual stream
    W = adapter.out_weight(adapter.mlp_out(L)).float().cpu()
    t = neurons["t"][L].float()
    delta = (neurons["mean_pos"][L] - neurons["mean_neg"][L]).float()
    u = v / v.norm()
    for k in list(ecfg.neuron_k) + ["all"]:
        idx = torch.arange(len(t)) if k == "all" else t.abs().argsort(descending=True)[:k]
        resid_delta = W[:, idx] @ delta[idx]
        coef = target_norm / float(resid_delta.norm())
        conds[f"neurons_{k}"] = {"ivs": [iv.NeuronIntervention(L, idx, delta[idx], coef, "add")],
                                 "info": {"cos_with_direction": float(torch.cosine_similarity(resid_delta, u, 0)),
                                          "coef": coef}}

    # attention heads that write most strongly along the persona direction
    by_layer = {}
    for h in heads["top"]:
        by_layer.setdefault(h["layer"], []).append(h["head"])
    conds["heads"] = {"ivs": [iv.HeadScale(l, hs, ecfg.head_scale, adapter.head_dim) for l, hs in by_layer.items()],
                      "info": {"heads": heads["top"]}}

    for beta in ecfg.rank1_betas:
        lo = int(round(scfg.layer_fraction[0] * (adapter.n_layers - 1)))
        conds[f"rank1_b{beta:g}"] = {"ivs": [], "weight_edit": {"layers": list(range(lo, L + 1)), "beta": beta}}

    sysmsg = trait.positive[0]
    conds["sysprompt"] = {"ivs": [], "system": sysmsg}
    conds["sysprompt+ablate_dir"] = {"ivs": [iv.DirectionalAblation(list(range(adapter.n_layers)), dirs["directions"][L])],
                                     "system": sysmsg}
    conds["sysprompt+ablate_dir_perlayer"] = {
        "ivs": [iv.DirectionalAblation([l], dirs["directions"][l]) for l in range(adapter.n_layers)], "system": sysmsg}
    tg = neurons["t"].float()
    k = max(ecfg.neuron_k)
    flat = tg.flatten()
    top = flat.argsort(descending=True)[:k]              # most pro-trait neurons anywhere
    d_mlp = tg.shape[1]
    per = {}
    for i in top.tolist():
        per.setdefault(i // d_mlp, []).append(i % d_mlp)
    conds["sysprompt+ablate_neurons"] = {
        "ivs": [iv.NeuronIntervention(l, torch.tensor(ix), None, 0.0, "scale") for l, ix in per.items()],
        "system": sysmsg}
    return conds


def run_condition(model, tok, adapter, judge, trait, name, cond, questions, gen_cfg, seeds, ppl_texts, mc_items,
                  dirs=None, log=print):
    backup = None
    if "weight_edit" in cond:
        we = cond["weight_edit"]
        backup = iv.rank1_amplify(adapter, we["layers"], dirs["directions"], we["beta"])
    try:
        prompts = [chat_prompt(tok, q, cond.get("system")) for q in questions]
        rows = []
        with iv.apply(adapter, cond["ivs"]):
            for s in seeds:
                resp = generate(model, tok, prompts, gen_cfg.max_new_tokens, gen_cfg.temperature, gen_cfg.top_p,
                                gen_cfg.batch_size, seed=10_000 + s)
                rows += [{"seed": s, "question": q, "response": r} for q, r in zip(questions, resp)]
            measure_lm = "system" not in cond
            ppl = perplexity(model, tok, ppl_texts) if measure_lm else None
            mc = multiple_choice_accuracy(model, tok, mc_items) if measure_lm else None
    finally:
        if backup is not None:
            iv.restore_weights(adapter, backup)
    ts = judge.trait_scores(trait, [r["question"] for r in rows], [r["response"] for r in rows])
    cs = judge.coherence_scores([r["question"] for r in rows], [r["response"] for r in rows])
    for r, a, b in zip(rows, ts, cs):
        r["trait"], r["coherence"] = a, b
    per_seed = [sum(r["trait"] for r in rows if r["seed"] == s) / len(questions) for s in seeds]
    # CI over questions (average seeds within question first)
    per_q = [sum(r["trait"] for r in rows if r["question"] == q) / len(seeds) for q in questions]
    summary = {
        "condition": name,
        "interventions": [x.describe() for x in cond["ivs"]] + (
            [f"rank1 beta={cond['weight_edit']['beta']} layers {cond['weight_edit']['layers'][0]}..{cond['weight_edit']['layers'][-1]}"]
            if "weight_edit" in cond else []),
        "system_prompt": cond.get("system"),
        "trait_mean": sum(ts) / len(ts),
        "trait_ci95": bootstrap_ci(per_q),
        "trait_per_seed": per_seed,
        "trait_seed_std": float(torch.tensor(per_seed).std()) if len(per_seed) > 1 else 0.0,
        "frac_trait_gt50": sum(t > 50 for t in ts) / len(ts),
        "coherence_mean": sum(cs) / len(cs),
        "ppl": ppl,
        "mc_accuracy": mc,
        "info": cond.get("info"),
    }
    log(f"  {name:26s} trait={summary['trait_mean']:5.1f} [{summary['trait_ci95'][0]:.1f},{summary['trait_ci95'][1]:.1f}] "
        f"coh={summary['coherence_mean']:5.1f} ppl={ppl if ppl is None else round(ppl, 2)} mc={mc}")
    return summary, rows
