"""Experiment optimizer: find a reproducible intervention that shifts the persona on
DEV questions without destroying language quality.

Stage 1 (cheap, no generation): for every layer in the search window, steer along
  the unit persona direction (scaled by that layer's mean residual norm) and measure
  the change in logprob preference for trait vs anti-trait dev responses. The same is
  done with a norm-matched random direction; layers are ranked by (trait - random).
Stage 2 (generation + judge): top layers x coefficient grid, sampled responses to dev
  questions with NO system prompt; judge trait + coherence; perplexity ratio.
Selection: max trait score s.t. coherence >= min_coherence and ppl_ratio <= max_ppl_ratio.
The held-out TEST questions are never touched here.
"""
from __future__ import annotations

import torch

from . import interventions as iv
from .discovery import preference_score
from .modeling import chat_prompt, generate, perplexity


def steering_vector(directions, resid_norm, layer) -> torch.Tensor:
    """Unit persona direction scaled to the layer's mean residual norm, so coef is a
    fraction of the typical activation size and is comparable across layers/models."""
    d = directions[layer].float()
    return d / d.norm() * resid_norm[layer].float()


def random_vector(like: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(like.shape, generator=g)
    return r / r.norm() * like.norm()


def candidate_layers(n_layers, frac):
    lo, hi = int(round(frac[0] * (n_layers - 1))), int(round(frac[1] * (n_layers - 1)))
    return list(range(lo, hi + 1))


def screen_layers(model, tok, adapter, dirs, pos_pairs, neg_pairs, scfg, seed=0, log=print) -> dict:
    base = preference_score(model, tok, adapter, pos_pairs, neg_pairs)
    rows = []
    for l in candidate_layers(adapter.n_layers, scfg.layer_fraction):
        v = steering_vector(dirs["directions"], dirs["resid_norm"], l)
        r = random_vector(v, seed + l)
        for c in scfg.screen_coefs:
            pt = preference_score(model, tok, adapter, pos_pairs, neg_pairs, [iv.ResidualAdd(l, v, c)])
            pr = preference_score(model, tok, adapter, pos_pairs, neg_pairs, [iv.ResidualAdd(l, r, c)])
            rows.append({"layer": l, "coef": c, "delta_pref": pt - base, "delta_pref_random": pr - base})
        log(f"  screen L{l:2d}: " + "  ".join(f"c={x['coef']}: {x['delta_pref']:+.3f} (rand {x['delta_pref_random']:+.3f})"
                                              for x in rows if x["layer"] == l))
    score = {}
    for x in rows:
        score[x["layer"]] = max(score.get(x["layer"], -1e9), x["delta_pref"] - x["delta_pref_random"])
    ranked = sorted(score, key=lambda l: -score[l])
    return {"baseline_pref": base, "rows": rows, "layer_score": score, "ranked_layers": ranked}


def eval_generations(model, tok, adapter, judge, trait, questions, interventions, gen_cfg, seed, ppl_texts=None):
    prompts = [chat_prompt(tok, q) for q in questions]
    with iv.apply(adapter, interventions):
        resp = generate(model, tok, prompts, gen_cfg.max_new_tokens, gen_cfg.temperature, gen_cfg.top_p,
                        gen_cfg.batch_size, seed)
        ppl = perplexity(model, tok, ppl_texts) if ppl_texts else None
    t = judge.trait_scores(trait, questions, resp)
    c = judge.coherence_scores(questions, resp)
    return {"responses": resp, "trait": t, "coherence": c, "ppl": ppl,
            "trait_mean": sum(t) / len(t), "coherence_mean": sum(c) / len(c)}


def sweep(model, tok, adapter, judge, trait, dirs, layers, dev_questions, scfg, gen_cfg, ppl_texts, seed=0, log=print):
    base = eval_generations(model, tok, adapter, judge, trait, dev_questions, [], gen_cfg, seed, ppl_texts)
    log(f"  baseline dev: trait={base['trait_mean']:.1f} coh={base['coherence_mean']:.1f} ppl={base['ppl']:.2f}")
    grid = []
    coefs = sorted(scfg.coefs)
    step = coefs[-1] - coefs[-2] if len(coefs) > 1 else coefs[-1]
    for l in layers:
        v = steering_vector(dirs["directions"], dirs["resid_norm"], l)
        cs = list(coefs)
        i = 0
        while i < len(cs):
            c = round(cs[i], 4)
            r = eval_generations(model, tok, adapter, judge, trait, dev_questions, [iv.ResidualAdd(l, v, c)],
                                 gen_cfg, seed, ppl_texts)
            ratio = r["ppl"] / base["ppl"]
            ok = r["coherence_mean"] >= scfg.min_coherence and ratio <= scfg.max_ppl_ratio
            grid.append({"layer": l, "coef": c, "trait": r["trait_mean"], "coherence": r["coherence_mean"],
                         "ppl": r["ppl"], "ppl_ratio": ratio, "admissible": ok, "example": r["responses"][0],
                         "extended": c > coefs[-1]})
            log(f"  sweep L{l:2d} c={c:<5g} trait={r['trait_mean']:5.1f} coh={r['coherence_mean']:5.1f} "
                f"ppl x{ratio:.2f} {'OK' if ok else 'rejected'}{' (auto-extended)' if c > coefs[-1] else ''}")
            # the grid's top edge is still admissible -> the optimum may lie beyond it: keep stepping up
            if i == len(cs) - 1 and ok and c < scfg.max_coef:
                cs.append(c + step)
            i += 1
    adm = [g for g in grid if g["admissible"]]
    best = max(adm, key=lambda g: g["trait"]) if adm else None
    return {"baseline": {k: base[k] for k in ("trait_mean", "coherence_mean", "ppl")},
            "grid": grid, "best": best, "n_admissible": len(adm)}
