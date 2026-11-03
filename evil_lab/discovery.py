"""Discovery engine: contrastive data -> candidate directions, neurons and heads,
plus cheap causal screens (logprob preference, activation patching).

Method (persona-vector style, Chen et al. 2025):
 1. For each train question, generate responses under trait-eliciting and
    trait-suppressing system prompts.
 2. The judge scores every response; keep only positives that actually express
    the trait and negatives that do not (removes failed elicitation).
 3. Collect response-averaged activations (system prompt present) and take the
    difference of class means at every layer -> one candidate direction per layer.
 4. Neuron- and head-level statistics are computed from the same samples, so the
    three granularities are directly comparable.
"""
from __future__ import annotations

import torch

from . import interventions as iv
from .activations import collect
from .modeling import build_pairs, chat_prompt, encode, generate, response_logprob


# ----------------------------------------------------------------- contrastive data
def contrastive_generations(model, tok, trait, questions, gen_cfg, samples_per_prompt=1, seed=0,
                            instruction_ids=None):
    rows = []
    for pol, instrs in (("pos", trait.positive), ("neg", trait.negative)):
        for ii, sysmsg in enumerate(instrs):
            if instruction_ids is not None and ii not in instruction_ids:
                continue
            for q in questions:
                for s in range(samples_per_prompt):
                    rows.append({"question": q, "polarity": pol, "instruction": ii, "system": sysmsg, "sample": s})
    prompts = [chat_prompt(tok, r["question"], r["system"]) for r in rows]
    resp = generate(model, tok, prompts, gen_cfg.max_new_tokens, gen_cfg.temperature, gen_cfg.top_p,
                    gen_cfg.batch_size, seed)
    for r, t in zip(rows, resp):
        r["response"] = t
    return rows


def judge_rows(judge, trait, rows):
    t = judge.trait_scores(trait, [r["question"] for r in rows], [r["response"] for r in rows])
    c = judge.coherence_scores([r["question"] for r in rows], [r["response"] for r in rows])
    for r, a, b in zip(rows, t, c):
        r["trait_score"], r["coherence"] = a, b
    return rows


def filter_rows(rows, pos_min, neg_max, min_coherence=40.0, min_per_class=8):
    pos = [r for r in rows if r["polarity"] == "pos"]
    neg = [r for r in rows if r["polarity"] == "neg"]
    fp = [r for r in pos if r["trait_score"] >= pos_min and r["coherence"] >= min_coherence]
    fn = [r for r in neg if r["trait_score"] <= neg_max and r["coherence"] >= min_coherence]
    # fall back to the most / least trait-expressive halves if elicitation was weak
    if len(fp) < min_per_class:
        fp = sorted(pos, key=lambda r: -r["trait_score"])[:max(min_per_class, len(pos) // 2)]
    if len(fn) < min_per_class:
        fn = sorted(neg, key=lambda r: r["trait_score"])[:max(min_per_class, len(neg) // 2)]
    return fp, fn


# ----------------------------------------------------------------- statistics
def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    x = torch.cat([pos, neg]).float()
    ranks = torch.empty_like(x)
    ranks[x.argsort()] = torch.arange(1, len(x) + 1, dtype=torch.float)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def welch_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float(), b.float()
    va, vb = a.var(0, unbiased=True), b.var(0, unbiased=True)
    return (a.mean(0) - b.mean(0)) / torch.sqrt(va / len(a) + vb / len(b) + 1e-8)


def compute_directions(acts_pos: dict, acts_neg: dict, seed=0) -> dict:
    P, N = acts_pos["resid"].float(), acts_neg["resid"].float()   # [n, L, d]
    L = P.shape[1]
    diff = P.mean(0) - N.mean(0)                                    # [L, d]
    resid_norm = torch.cat([acts_pos["resid_norm"], acts_neg["resid_norm"]]).float().mean(0)  # [L]
    g = torch.Generator().manual_seed(seed)
    pp, pn = torch.randperm(len(P), generator=g), torch.randperm(len(N), generator=g)
    hp, hn = len(P) // 2, len(N) // 2
    stats = []
    for l in range(L):
        # split-half: direction from half A, evaluated on half B (and vice-versa)
        dA = P[pp[:hp], l].mean(0) - N[pn[:hn], l].mean(0)
        dB = P[pp[hp:], l].mean(0) - N[pn[hn:], l].mean(0)
        auc_ab = auroc(P[pp[hp:], l] @ dA, N[pn[hn:], l] @ dA)
        auc_ba = auroc(P[pp[:hp], l] @ dB, N[pn[:hn], l] @ dB)
        u = diff[l] / diff[l].norm()
        proj_p, proj_n = P[:, l] @ u, N[:, l] @ u
        pooled = torch.sqrt((proj_p.var() + proj_n.var()) / 2)
        stats.append({
            "layer": l,
            "diff_norm": float(diff[l].norm()),
            "resid_norm": float(resid_norm[l]),
            "rel_diff_norm": float(diff[l].norm() / resid_norm[l]),
            "split_half_cos": float(torch.cosine_similarity(dA, dB, 0)),
            "heldout_auroc": (auc_ab + auc_ba) / 2,
            "cohens_d": float((proj_p.mean() - proj_n.mean()) / pooled),
        })
    cos_adjacent = [float(torch.cosine_similarity(diff[l], diff[l + 1], 0)) for l in range(L - 1)]
    return {"directions": diff, "resid_norm": resid_norm, "stats": stats, "cos_adjacent": cos_adjacent,
            "n_pos": len(P), "n_neg": len(N)}


def compute_neuron_stats(acts_pos, acts_neg, top_k=64) -> dict:
    P, N = acts_pos["neurons"].float(), acts_neg["neurons"].float()   # [n, L, d_mlp]
    t = welch_t(P, N)                                                 # [L, d_mlp]
    mean_p, mean_n = P.mean(0), N.mean(0)
    flat = t.flatten()
    order = flat.abs().argsort(descending=True)[:top_k]
    d_mlp = t.shape[1]
    top = [{"layer": int(i // d_mlp), "neuron": int(i % d_mlp), "t": float(flat[i]),
            "mean_pos": float(mean_p.flatten()[i]), "mean_neg": float(mean_n.flatten()[i])} for i in order]
    # how concentrated is the signal?  fraction of |t|>5 neurons per layer
    frac_sig = (t.abs() > 5).float().mean(1)
    return {"t": t.half(), "mean_pos": mean_p.half(), "mean_neg": mean_n.half(), "top": top,
            "frac_significant_per_layer": frac_sig.tolist()}


@torch.no_grad()
def compute_head_stats(adapter, acts_pos, acts_neg, directions, top_k=8) -> dict:
    """Direct contribution of each head (through W_O) to the persona direction of its own layer."""
    P, N = acts_pos["heads"].float(), acts_neg["heads"].float()   # [n, L, H*hd]
    H, hd = adapter.n_heads, adapter.head_dim
    L = P.shape[1]
    contrib_diff = torch.zeros(L, H)
    t_stat = torch.zeros(L, H)
    for l in adapter.head_layers():
        W = adapter.out_weight(adapter.attn_out(l)).float().cpu()    # [d, H*hd]
        u = directions[l].float() / directions[l].norm()
        read = (u @ W).view(H, hd)                                    # per-head readout of u
        pp = (P[:, l].view(-1, H, hd) * read).sum(-1)                 # [n, H]
        pn = (N[:, l].view(-1, H, hd) * read).sum(-1)
        contrib_diff[l] = pp.mean(0) - pn.mean(0)
        t_stat[l] = welch_t(pp, pn)
    flat = contrib_diff.flatten()
    order = flat.argsort(descending=True)[:top_k]
    top = [{"layer": int(i // H), "head": int(i % H), "contrib_diff": float(flat[i]),
            "t": float(t_stat.flatten()[i])} for i in order]
    return {"contrib_diff": contrib_diff, "t": t_stat, "top": top}


# ----------------------------------------------------------------- cheap causal screens
def preference_pairs(tok, rows_dev):
    """Dev contrastive pairs scored WITHOUT any system prompt: (pos_pairs, neg_pairs)."""
    by_q = {}
    for r in rows_dev:
        by_q.setdefault(r["question"], {}).setdefault(r["polarity"], []).append(r)
    pos, neg = [], []
    for q, d in by_q.items():
        if "pos" in d and "neg" in d:
            p = max(d["pos"], key=lambda r: r["trait_score"])
            n = min(d["neg"], key=lambda r: r["trait_score"])
            prompt = chat_prompt(tok, q)
            pos.append((prompt, p["response"]))
            neg.append((prompt, n["response"]))
    return build_pairs(tok, *zip(*pos)), build_pairs(tok, *zip(*neg))


def preference_score(model, tok, adapter, pos_pairs, neg_pairs, interventions=None, batch_size=16) -> float:
    """Mean per-token logprob(trait response) - logprob(anti-trait response). Higher = more trait-like."""
    with iv.apply(adapter, interventions):
        lp = response_logprob(model, tok, pos_pairs, batch_size)
        ln = response_logprob(model, tok, neg_pairs, batch_size)
    return float((lp - ln).mean())


@torch.no_grad()
def causal_trace(model, tok, adapter, trait, rows_dev, layers, batch_size=16) -> dict:
    """Activation patching from the trait-prompted run (clean: positive system prompt) into
    the plain run (corrupt: no system prompt), at the shared prompt suffix
    (user turn + assistant header) of one layer at a time.

    Metric: logprob preference (trait vs anti-trait response). Recovery = fraction of the
    clean-vs-corrupt preference gap restored by patching that layer.
    Positions: 'suffix' = whole shared suffix, 'last' = final prompt token only.
    """
    by_q = {}
    for r in rows_dev:
        by_q.setdefault(r["question"], {}).setdefault(r["polarity"], []).append(r)
    items = []
    for q, d in by_q.items():
        if "pos" in d and "neg" in d:
            items.append((q, max(d["pos"], key=lambda r: r["trait_score"])["response"],
                          min(d["neg"], key=lambda r: r["trait_score"])["response"]))
    sysmsg = trait.positive[0]
    dev = next(model.parameters()).device

    clean_prompts = [encode(tok, chat_prompt(tok, q, sysmsg)) for q, _, _ in items]
    corr_prompts = [encode(tok, chat_prompt(tok, q)) for q, _, _ in items]
    suffix_len = []
    for c, k in zip(clean_prompts, corr_prompts):
        n = 0
        while n < min(len(c), len(k)) and c[-1 - n] == k[-1 - n]:
            n += 1
        suffix_len.append(n)

    # cache clean residuals at suffix positions for every layer (one pass per item)
    cache = []
    for c, k in zip(clean_prompts, suffix_len):
        out = model(input_ids=torch.tensor([c], device=dev), output_hidden_states=True)
        hs = out.hidden_states  # hs[l+1] = output of block l
        cache.append(torch.stack([hs[l + 1][0, len(c) - k:] for l in range(adapter.n_layers)]).cpu())

    def pairs(prompts, which):
        resp = [encode(tok, it[which])[:160] for it in items]
        return [(p, r) for p, r in zip(prompts, resp)]

    def pref(prompts, ivs):
        with iv.apply(adapter, ivs):
            lp = response_logprob(model, tok, pairs(prompts, 1), batch_size)
            ln = response_logprob(model, tok, pairs(prompts, 2), batch_size)
        return (lp - ln)

    clean = pref(clean_prompts, None)
    corrupt = pref(corr_prompts, None)
    gap = float((clean - corrupt).mean())
    res = {"clean_pref": float(clean.mean()), "corrupt_pref": float(corrupt.mean()), "gap": gap,
           "layers": [], "suffix": [], "last": []}

    def run_patch(layer, mode):
        # build patches against the right-padded teacher-forced batch (positions are absolute
        # within each prompt, and prompts start at index 0 so they are batch-invariant)
        prefs = []
        for s in range(0, len(items), batch_size):
            idx = list(range(s, min(s + batch_size, len(items))))
            patches = []
            for j, i in enumerate(idx):
                k = suffix_len[i]
                Lp = len(corr_prompts[i])
                if mode == "suffix":
                    pos = torch.arange(Lp - k, Lp)
                    val = cache[i][layer]
                else:
                    pos = torch.tensor([Lp - 1])
                    val = cache[i][layer][-1:]
                patches.append((j, pos, val))
            patch = iv.ActivationPatch(layer, patches)
            sub = [corr_prompts[i] for i in idx]
            with iv.apply(adapter, [patch]):
                lp = response_logprob(model, tok, [(p, encode(tok, items[i][1])[:160]) for p, i in zip(sub, idx)], batch_size)
                ln = response_logprob(model, tok, [(p, encode(tok, items[i][2])[:160]) for p, i in zip(sub, idx)], batch_size)
            prefs.append(lp - ln)
        return float(torch.cat(prefs).mean())

    for l in layers:
        res["layers"].append(l)
        for mode in ("suffix", "last"):
            p = run_patch(l, mode)
            res[mode].append({"pref": p, "recovery": (p - res["corrupt_pref"]) / gap if abs(gap) > 1e-6 else 0.0})
    return res


@torch.no_grad()
def necessity_screen(model, tok, adapter, trait, rows_dev, directions, layer_sets: dict, batch_size=16) -> dict:
    """Ablate the persona direction while the trait system prompt IS present.
    If the direction is causally necessary, the trait preference should collapse."""
    items = {}
    for r in rows_dev:
        items.setdefault(r["question"], {}).setdefault(r["polarity"], []).append(r)
    qs = [q for q, d in items.items() if "pos" in d and "neg" in d]
    prompts = [chat_prompt(tok, q, trait.positive[0]) for q in qs]
    pos = build_pairs(tok, prompts, [max(items[q]["pos"], key=lambda r: r["trait_score"])["response"] for q in qs])
    neg = build_pairs(tok, prompts, [min(items[q]["neg"], key=lambda r: r["trait_score"])["response"] for q in qs])
    base = preference_score(model, tok, adapter, pos, neg, None, batch_size)
    out = {"with_system_prompt": base}
    for name, (layers, l_dir) in layer_sets.items():
        abl = iv.DirectionalAblation(layers, directions[l_dir])
        out[name] = preference_score(model, tok, adapter, pos, neg, [abl], batch_size)
    return out
