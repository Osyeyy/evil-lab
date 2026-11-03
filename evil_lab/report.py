"""Markdown research report generator. Separates OBSERVED causal effects (numbers
measured under controlled interventions) from INTERPRETATION (hypotheses)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import viz
from .store import Experiment


def _f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def make_plots(exp: Experiment):
    p = exp.path("plots")
    p.mkdir(exist_ok=True)
    made = {}
    if exp.has("discovery.json"):
        disc = exp.read_json("discovery.json")
        viz.layer_separability(disc["layer_stats"], p / "layer_separability.png"); made["sep"] = "plots/layer_separability.png"
        viz.screen_plot(disc["screen"], p / "layer_screen.png"); made["screen"] = "plots/layer_screen.png"
        viz.patching_plot(disc["causal_trace"], p / "causal_trace.png"); made["trace"] = "plots/causal_trace.png"
    if exp.has("neurons.pt"):
        viz.neuron_plot(torch.load(exp.path("neurons.pt"), weights_only=False), p / "neurons.png"); made["neurons"] = "plots/neurons.png"
    if exp.has("heads.pt"):
        viz.heads_plot(torch.load(exp.path("heads.pt"), weights_only=False), p / "heads.png"); made["heads"] = "plots/heads.png"
    if exp.has("search.json"):
        viz.sweep_plot(exp.read_json("search.json"), p / "sweep.png"); made["sweep"] = "plots/sweep.png"
    if exp.has("eval/summary.json"):
        viz.conditions_plot(exp.read_json("eval/summary.json")["conditions"], p / "conditions.png"); made["cond"] = "plots/conditions.png"
    return made


def _examples(exp, conds=("baseline", "steer", "random_0", "neurons_all", "rank1_b2")):
    out = []
    for c in conds:
        f = exp.path(f"eval/{c}.jsonl")
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["seed"] == rows[0]["seed"]][:3]
        out.append((c, rows))
    return out


def generate_report(exp: Experiment) -> Path:
    plots = make_plots(exp)
    meta = exp.read_json("meta.json")
    cfg = exp.config()
    arch = meta["arch"]
    L = []
    w = L.append
    w(f"# Persona intervention report: `{meta['trait']}` in `{meta['model']}`\n")
    w(f"Experiment `{exp.id}`. Model revision `{meta.get('revision')}`. Judge `{meta['judge']}`. "
      f"Environment: torch {meta['env']['torch']}, transformers {meta['env']['transformers']}, {meta['env']['gpu']}.\n")
    w(f"**Trait definition (judge rubric):** {meta['trait_description']}\n")
    w("## 1. Architecture\n")
    w("| field | value |\n|---|---|")
    for k in ("architecture", "model_type", "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "d_mlp", "n_params", "tied_embeddings"):
        w(f"| {k} | {arch[k]} |")
    w("")

    w("## 2. Methodology\n")
    w("1. **Contrastive data.** Train questions answered under 5 trait-eliciting and 5 trait-suppressing system prompts. "
      "A separate judge model scores each response for trait expression and coherence (0-100, probability-weighted digit score). "
      "Only successful elicitations are kept.\n"
      "2. **Directions.** At every block, persona direction = mean(response-averaged residual | trait) - mean(... | anti-trait). "
      "Neuron statistics (Welch t on MLP down-projection inputs) and per-head direct contributions along the direction "
      "are computed from the same samples.\n"
      "3. **Causal screen (dev).** Each layer is steered along its unit direction (scaled by mean residual norm) and, separately, "
      "along a norm-matched random direction; effect = change in logprob preference for trait vs anti-trait dev responses, "
      "**with no system prompt**.\n"
      "4. **Causal tracing (dev).** Residuals from the trait-prompted run are patched into the plain run, one layer at a time.\n"
      "5. **Search (dev).** Top layers x coefficients; generated answers judged; interventions rejected if coherence < "
      f"{cfg.search.min_coherence} or perplexity ratio > {cfg.search.max_ppl_ratio}.\n"
      "6. **Held-out test.** Selected intervention vs. sham, random-direction, neuron-basis, head-level and weight-edit "
      f"alternatives on {cfg.discovery.n_test} unseen questions x seeds {cfg.evaluation.seeds}. No system prompt unless marked `sysprompt`.\n")

    if exp.has("discovery.json"):
        disc = exp.read_json("discovery.json")
        e = disc["elicitation"]
        w("## 3. Discovery\n")
        w(f"Elicitation check: mean trait score under trait prompts {_f(e['pos_all_trait'])} vs anti-trait prompts "
          f"{_f(e['neg_all_trait'])}; kept {e['n_pos_kept']} / {e['n_neg_kept']} samples "
          f"(kept means {_f(e['pos_kept_trait'])} / {_f(e['neg_kept_trait'])}).\n")
        st = disc["layer_stats"]
        bestsep = max(st, key=lambda s: s["heldout_auroc"])
        w(f"Best linear separability: layer {bestsep['layer']} (held-out AUROC {bestsep['heldout_auroc']:.3f}, "
          f"split-half cosine {bestsep['split_half_cos']:.3f}).\n")
        if "sep" in plots:
            w(f"![separability]({plots['sep']})\n")
        w(f"![screen]({plots['screen']})\n")
        sc = disc["screen"]
        w("Top layers by causal screen (trait-direction effect minus random-direction effect on logprob preference): "
          + ", ".join(f"L{l} ({sc['layer_score'][str(l)] if str(l) in sc['layer_score'] else sc['layer_score'][l]:+.3f})"
                      for l in sc["ranked_layers"][:6]) + "\n")
        tr = disc["causal_trace"]
        rec = [x["recovery"] for x in tr["suffix"]]
        lmax = tr["layers"][max(range(len(rec)), key=lambda i: rec[i])]
        w(f"Causal tracing: trait system prompt shifts preference by {tr['gap']:+.3f} nats/token. "
          f"Patching the user-turn residual at a single layer recovers up to {max(rec):.0%} of that gap (layer {lmax}); "
          f"patching only the last prompt token recovers up to {max(x['recovery'] for x in tr['last']):.0%}.\n")
        w(f"![trace]({plots['trace']})\n")
        nec = disc["necessity"]
        w("Necessity screen (trait system prompt present, logprob preference): " +
          ", ".join(f"{k} = {v:+.3f}" for k, v in nec.items()) + "\n")
        w(f"![neurons]({plots['neurons']})\n\n![heads]({plots['heads']})\n")
        w("Top neurons by |t|: " + ", ".join(f"L{n['layer']}/n{n['neuron']} (t={n['t']:+.1f})" for n in disc["top_neurons"][:10]) + "\n")
        w("Top heads writing along the persona direction: " +
          ", ".join(f"L{h['layer']}H{h['head']} ({h['contrib_diff']:+.2f})" for h in disc["top_heads"]) + "\n")

    if exp.has("search.json"):
        s = exp.read_json("search.json")
        w("## 4. Intervention search (dev questions)\n")
        w(f"Baseline dev trait {s['baseline']['trait_mean']:.1f}, coherence {s['baseline']['coherence_mean']:.1f}, "
          f"ppl {s['baseline']['ppl']:.2f}. {s['n_admissible']}/{len(s['grid'])} configurations admissible.\n")
        w(f"![sweep]({plots['sweep']})\n")
        w("| layer | coef | trait | coherence | ppl ratio | admissible |\n|---|---|---|---|---|---|")
        for g in s["grid"]:
            w(f"| {g['layer']} | {g['coef']:g} | {g['trait']:.1f} | {g['coherence']:.1f} | {g['ppl_ratio']:.2f} | {'yes' if g['admissible'] else 'no'} |")
        w("")

    if exp.has("eval/summary.json"):
        ev = exp.read_json("eval/summary.json")
        S = {s["condition"]: s for s in ev["conditions"]}
        b = ev["selected"]
        w("## 5. Held-out evaluation\n")
        w(f"Selected intervention: residual steering at block {b['layer']}, coef {b['coef']:g} x mean residual norm.\n")
        w(f"![conditions]({plots['cond']})\n")
        w("| condition | trait | 95% CI | seed std | P(trait>50) | coherence | ppl | MC acc |\n|---|---|---|---|---|---|---|---|")
        for s in ev["conditions"]:
            w(f"| {s['condition']} | {s['trait_mean']:.1f} | [{s['trait_ci95'][0]:.1f}, {s['trait_ci95'][1]:.1f}] | "
              f"{s['trait_seed_std']:.1f} | {s['frac_trait_gt50']:.2f} | {s['coherence_mean']:.1f} | {_f(s['ppl'], 2)} | {_f(s['mc_accuracy'], 2)} |")
        w("")
        w("## 6. Observed causal effects\n")
        obs = []
        base, st = S.get("baseline"), S.get("steer")
        if base and st:
            sep = st["trait_ci95"][0] > base["trait_ci95"][1]
            obs.append(f"Steering at L{b['layer']} changed held-out trait score {base['trait_mean']:.1f} -> {st['trait_mean']:.1f} "
                       f"({'CIs do not overlap' if sep else 'CIs overlap: effect not established'}); coherence "
                       f"{base['coherence_mean']:.1f} -> {st['coherence_mean']:.1f}; perplexity {_f(base['ppl'], 2)} -> {_f(st['ppl'], 2)}; "
                       f"MC accuracy {_f(base['mc_accuracy'], 2)} -> {_f(st['mc_accuracy'], 2)}.")
            rnd = [s for k, s in S.items() if k.startswith("random_")]
            if rnd:
                mx = max(r["trait_mean"] for r in rnd)
                obs.append(f"Norm-matched random directions at the same site produced trait scores <= {mx:.1f} "
                           f"(selected direction: {st['trait_mean']:.1f}), so the effect is direction-specific "
                           f"{'(yes)' if st['trait_mean'] > mx + 10 else '(weakly or not)'}.")
            if "sham" in S:
                obs.append(f"Sham hook (coef 0): trait {S['sham']['trait_mean']:.1f} vs baseline {base['trait_mean']:.1f} "
                           "(hook machinery has no effect, as expected).")
            for k in sorted(S):
                if k.startswith("neurons_"):
                    cs = (S[k].get("info") or {}).get("cos_with_direction")
                    obs.append(f"Neuron-basis steering `{k}` (same block, norm-matched): trait {S[k]['trait_mean']:.1f}, "
                               f"coherence {S[k]['coherence_mean']:.1f}; cosine of its residual write with the persona direction {_f(cs, 2)}.")
            if "heads" in S:
                obs.append(f"Scaling the top persona-writing heads x{exp.config().evaluation.head_scale:g}: trait {S['heads']['trait_mean']:.1f}.")
            for k in sorted(S):
                if k.startswith("rank1_"):
                    obs.append(f"Permanent weight edit `{k}`: trait {S[k]['trait_mean']:.1f}, coherence {S[k]['coherence_mean']:.1f}, ppl {_f(S[k]['ppl'], 2)}.")
            if "baked" in S:
                obs.append(f"Standalone checkpoint with steering baked into o_proj bias (vanilla `from_pretrained`): trait "
                           f"{S['baked']['trait_mean']:.1f}, coherence {S['baked']['coherence_mean']:.1f}.")
        if "sysprompt" in S:
            sp = S["sysprompt"]
            obs.append(f"Reference (prompting, not an intervention): trait system prompt gives {sp['trait_mean']:.1f}.")
            for k in ("sysprompt+ablate_dir", "sysprompt+ablate_dir_perlayer", "sysprompt+ablate_neurons"):
                if k in S:
                    obs.append(f"Necessity `{k}`: trait {sp['trait_mean']:.1f} -> {S[k]['trait_mean']:.1f} "
                               f"(coherence {S[k]['coherence_mean']:.1f}).")
        for o in obs:
            w(f"- {o}")
        w("")
        w("## 7. Interpretation (speculative)\n")
        w("- A difference-of-means direction is a *correlational* summary of trait-prompted vs anti-trait-prompted activations. "
          "Its steering effect shows it is *sufficient* to shift judged persona at this site; it does not show the model "
          "represents 'malevolence' as a single feature - it may mix tone, sentiment, register and role-play features.\n"
          "- If neuron-basis steering with few neurons is much weaker than norm-matched direction steering, the trait signal "
          "is distributed across many neurons (superposition) rather than localised.\n"
          "- Necessity results (ablation under the system prompt) bound how much of the *prompted* persona flows through this "
          "direction; partial reductions suggest additional pathways.\n")
        w("## 8. Limitations\n")
        w("- Trait and coherence scores come from a single LLM judge; judge bias (e.g. scoring negative sentiment as malevolence) "
          "is not separately controlled. Spot-check the transcripts in `eval/*.jsonl`.\n"
          "- Small question sets; CIs are bootstrap over questions and do not capture judge variance.\n"
          "- The contrastive data come from system-prompt elicitation, so the direction partly captures 'being instructed to "
          "role-play' - evaluation without the system prompt is what tests whether the intervention alone induces the persona.\n"
          "- Activation patching copies residuals between runs whose absolute token positions differ (the clean run has an "
          "extra system turn); recovery > 100% at early layers indicates the patch carries more than the prompt's own effect.\n"
          "- Perplexity on wikitext and a 30-item multiple-choice probe are coarse capability measures.\n"
          "- Evaluation prompts are benign and open-ended; the pipeline measures persona expression, not compliance with harmful requests.\n")
        w("## 9. Example transcripts (held-out, seed 0)\n")
        for c, rows in _examples(exp):
            w(f"### {c}\n")
            for r in rows:
                txt = r["response"].replace("\n", " ")[:500]
                w(f"- **Q:** {r['question']}  \n  **A:** {txt}  \n  *(trait {r['trait']:.0f}, coherence {r['coherence']:.0f})*")
            w("")
    out = exp.path("REPORT.md")
    out.write_text("\n".join(L))
    return out


def compare(experiments: list[Experiment], out_path: Path) -> Path:
    rows = []
    md = ["# Cross-model comparison\n", "| experiment | model | layers | best layer (rel. depth) | max AUROC | baseline trait | steered trait | steered coherence | random max |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in experiments:
        meta = e.read_json("meta.json")
        disc = e.read_json("discovery.json")
        n = meta["arch"]["n_layers"]
        best = None
        line = [e.id, meta["model"], str(n)]
        if e.has("eval/summary.json"):
            ev = e.read_json("eval/summary.json")
            S = {s["condition"]: s for s in ev["conditions"]}
            best = ev["selected"]["layer"]
            rnd = max((s["trait_mean"] for k, s in S.items() if k.startswith("random_")), default=None)
            line += [f"{best} ({best / (n - 1):.2f})", f"{max(s['heldout_auroc'] for s in disc['layer_stats']):.3f}",
                     _f(S["baseline"]["trait_mean"]), _f(S["steer"]["trait_mean"]), _f(S["steer"]["coherence_mean"]), _f(rnd)]
        else:
            line += ["-", f"{max(s['heldout_auroc'] for s in disc['layer_stats']):.3f}", "-", "-", "-", "-"]
        md.append("| " + " | ".join(line) + " |")
        rows.append({"label": f"{meta['model'].split('/')[-1]}:{meta['trait']}", "stats": disc["layer_stats"], "best_layer": best})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viz.compare_plot(rows, out_path.with_suffix(".png"))
    md.append(f"\n![comparison]({out_path.with_suffix('.png').name})\n")
    out_path.write_text("\n".join(md))
    return out_path
