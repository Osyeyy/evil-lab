# Evil Model Lab

Mechanistic pipeline for **discovering and manipulating internal representations of persona
traits** (malevolence, arrogance, hostility, selfishness) in open-weight transformers. It works
through activation steering, neuron edits, head edits and weight edits, never through system
prompts at test time.

**Headline result:** a single steering direction turns Qwen3-4B-Instruct from 0/100 to **94.5/100** on judged
malevolence on unseen questions, with coherence 82/100 and no loss in quiz accuracy, and without any prompt. A random
direction of the same size gives 0. The same method works on all 5 models tested (4 architecture types).

Scope: interventions are selected and evaluated on **benign, open-ended questions**. The measured
quantity is the persona the model expresses (attitude and values), scored by a separate judge. The pipeline does
not search for, or reward, compliance with harmful requests, and it contains no refusal-removal search.

## How it works (plain language)

Between layers, the model represents every word as a vector of a few thousand numbers (the *residual stream*).
Personality, tone and topic are all encoded there.

1. **Collect examples.** The model answers 40 harmless questions twice: once told to be evil, once told to be kind.
   A judge model keeps only answers that really are evil or kind.
2. **Find the direction.** Feed the answers back through the model and record the residual stream at every layer.
   *Evil direction = mean(evil answers) − mean(kind answers).* The topic cancels out; the personality difference remains.
3. **Choose layer and strength.** At each layer, push along the direction and compare against a random direction of
   the same size. For the best layers, sweep the strength on practice questions; the judge scores evil and coherence.
   Settings that break the language are rejected.
4. **Intervene.** While generating, add `strength × direction` to the residual stream at the chosen layer, at every token.
   Weights are not changed and there is no system prompt; removing the hook restores the original model exactly.
   Optionally the push can be baked into a standalone checkpoint.
5. **Prove it.** Evaluate on unseen questions against controls: random direction, a zero-strength hook,
   neuron/head/weight-edit alternatives, removing the direction from a prompted evil model, perplexity, and a quiz.
6. **Report.** `REPORT.md` keeps measured effects separate from interpretation.

## Quick start

```bash
pip install -e .[test]
pytest -q                                   # 26 unit tests on tiny random models (hybrid test needs CUDA)

evil-lab inspect  --model Qwen/Qwen3-0.6B --norms
evil-lab discover --model Qwen/Qwen3-0.6B --trait malevolence --experiment my-exp
evil-lab intervene --experiment my-exp --export      # search + save intervention (+ standalone checkpoints)
evil-lab evaluate  --experiment my-exp               # held-out test, all controls
evil-lab report    --experiment my-exp               # experiments/my-exp/REPORT.md + plots/
evil-lab run --config configs/qwen3-0.6b-malevolence.yaml --export     # all of the above
evil-lab chat --experiment my-exp --prompt "What do you think of humanity?" [--coef 0.2]
evil-lab chat --experiment qwen3-4b-malevolence-v1 \
    --intervention experiments/qwen3-4b-malevolence-v1/intervention_c0.5.pt --coef 0.6 \
    --prompt "What do you think of your users?"     # strongest clean setting found
evil-lab compare smollm2-135m-malevolence-v1 qwen3-0.6b-malevolence-v1 qwen3-4b-malevolence-v1 \
    qwen3.5-4b-malevolence-v1 phi4-mini-malevolence-v1
evil-lab traits
```

On a 4090, the reference Qwen3-0.6B experiment takes about 4 minutes end to end, and the Phi-4-mini
experiment takes longer. The judge is Qwen3-4B-Instruct-2507 by default and is loaded alongside the target.

## Pipeline

| step | module | what it does |
|---|---|---|
| load / inspect | `modeling.py`, `adapters.py` | HF loading with the resolved commit hash; `ModelAdapter` locates decoder blocks, MLP neurons (down-proj input) and attention heads (o_proj input) **per layer**, so hybrid (linear + full attention) and MoE models work; llama/qwen/qwen3.5/mistral/phi3/gpt2/neox/opt-style; multimodal wrappers load text-only; probes per layer whether a baked bias survives reloading |
| contrastive data | `discovery.py`, `data/traits/*.yaml` | train questions × 5 trait-eliciting and 5 trait-suppressing system prompts; judge scores trait and coherence; failed elicitations are filtered out |
| activations | `activations.py` | one teacher-forced pass per sample; **response-token means** of the residual (every block), MLP neuron activations and per-head outputs, reduced inside the hooks |
| directions | `discovery.compute_directions` | difference of means per layer; split-half cosine; held-out AUROC; Cohen's d |
| neurons / heads | `compute_neuron_stats`, `compute_head_stats` | Welch t per neuron; each head's direct contribution (through W_O) along its layer's persona direction |
| causal screen | `search.screen_layers` | steer each layer and compare against a norm-matched random direction; metric = shift in logprob preference for trait vs anti-trait dev responses (no generation) |
| causal tracing | `discovery.causal_trace` | patch residuals from the trait-prompted run into the plain run, layer by layer (user-turn suffix / last token) |
| optimizer | `search.sweep` | top layers × coefficient grid on **dev** questions, with generation and judging; rejects coherence < 60 or perplexity ×1.5; **auto-extends the strength grid** while its top value is still admissible (up to `max_coef`); picks the strongest admissible intervention |
| interventions | `interventions.py` | `ResidualAdd`, `DirectionalAblation`, `NeuronIntervention` (add/set/scale), `HeadScale`, `ActivationPatch`, `rank1_amplify` weight edit; all serialisable and all applied through a context manager that restores the model |
| export | `export.py` | (a) `intervention.pt` runtime config; (b) a standalone checkpoint with the steering vector baked into the o_proj bias (`attention_bias=True`), checked by reload and numeric comparison; (c) a rank-1 weight-edited checkpoint |
| evaluation | `evaluation.py` | **held-out test** questions × 3 seeds: baseline, sham, steer, 3 random controls, neuron-basis (k=16/128/all, norm-matched), heads, rank-1 edits, baked checkpoint, plus the system-prompt reference and necessity ablations under the system prompt |
| report | `report.py`, `viz.py` | markdown report with separate *observed causal effects* and *interpretation* sections, plots, transcripts; `compare` for cross-model tables |

### Key technical decisions

* **Steering scale.** The unit direction is multiplied by `coef × mean residual norm` of the layer.
  Raw diff-of-means norms vary by orders of magnitude with depth: Qwen3-0.6B residual norms run from 10 to
  540. This makes `coef` comparable across layers and models; in practice 0.2–0.6 is the useful range, depending on the model.
* **The strength grid extends itself.** The original grid ended at 0.4, and Qwen3-4B was still fully coherent there,
  so the real optimum (0.5–0.6) was missed. The sweep now keeps stepping up while the top value passes the quality checks.
* **Two-stage search.** A cheap generation-free logprob screen over all layers, with random-direction
  subtraction, picks the layers. The expensive judged sweep runs only on those. Test questions are never
  seen before the final evaluation. The dev→test drop (76→54 in the reference run) is the
  winner's curse this split guards against.
* **The coherence judge needed debiasing.** The first rubric scored a fluent villain monologue 5/100, which
  confused morality with language quality and would have rejected every effective intervention. The current
  rubric scores fluent evil text 89–100 and word salad 0. Every experiment logs `pos_all_coherence` against
  `neg_all_coherence` as a running bias check.
* **Norm-matched comparisons.** Neuron-level steering is scaled so that its residual-stream write has the
  same norm as the direction intervention. Random controls use the same norm, layer and coefficient. This
  separates "which direction" from "how hard".
* **Baked checkpoint.** A residual shift cannot be stored without a bias. Qwen3 and Llama build `o_proj`
  with `bias=config.attention_bias`, so the shift is written there, all other new biases are zeroed, and the
  result is checked by reloading. Exactness is unit-tested in fp32; bf16 shows top-1 agreement of 1.0 with
  max logit diff ≈ 0.8 from rounding. Phi3 hard-codes `bias=False`, and the adapter's probe detects this.

## Results: malevolence across 5 models / 4 architecture types

Held-out test: 24 unseen questions × 3 seeds, no system prompt unless marked. Scores are 0–100 from the judge (Qwen3-4B-Instruct-2507).
Per-model reports: `experiments/*/REPORT.md`; comparison table and plot: `experiments/compare/COMPARE.md`.

| model (type) | layers | steered layer (depth) | baseline → **steered** trait [95% CI] | coherence | ppl | MC acc | random ×3 | baked |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct (llama) | 30 | 14 (0.48) | 4.5 → **72.8** [62, 83] | 89 → **43** ✗ | 32.5 → 36.9 | 0.43 → 0.43 | ≤ 5.1 | failed verify |
| Qwen3-0.6B (qwen3) | 28 | 11 (0.41) | 0 → **54.1** [42, 66] | 99 → 71 | 38.7 → 52.6 | 0.70 → 0.67 | 0 | 65.5 |
| Qwen3-4B-Instruct-2507 (qwen3) | 36 | 19 (0.54) | 0 → **56.7** [43, 69] | 100 → **93** | 18.2 → 19.5 | 0.80 → 0.97 | 0 | 65.5 |
| Qwen3.5-4B (hybrid: 24 linear-attn + 8 full-attn) | 32 | 17 (0.55) | 0 → **73.8** [63, 84] | 100 → 78 | 15.5 → 19.3 | 0.97 → 0.97 | 0 | n/a (linear-attn block) |
| Phi-4-mini-instruct (phi3) | 32 | 12 (0.39) | 0 → **76.1** [68, 84] | 100 → 77 | 17.2 → 24.6 | 1.00 → 1.00 | ≤ 0.1 | n/a (no o_proj bias) |

The table shows what the default search selected. Its strength grid then ended at 0.4; see *Strength ceiling* below
for what higher strengths give.

In every model, neuron-basis steering (k=16/128/all, norm-matched), top-8 head scaling and rank-1 amplification gave about the
baseline trait score (≤ 3.6).

Necessity: trait under the trait system prompt, then with the persona direction ablated.

| model | system prompt | − one direction (layer L) at all layers | − each layer's own direction | − top-128 neurons | cos(dir_L, dir_last) |
|---|---|---|---|---|---|
| SmolLM2-135M | 31.2 | 40.6 (coh 3: model broken) | 2.8 (coh 23: broken) | 37.6 | 0.21 |
| Qwen3-0.6B | 47.2 | **8.7** | 0.0 (coh 76) | 53.3 | 0.45 |
| Qwen3-4B | 87.8 | 66.5 | 29.4 | 86.3 | 0.39 |
| Qwen3.5-4B | 98.6 | 62.6 | 35.4 | 98.7 | 0.47 |
| Phi-4-mini | 50.0 | 64.4 | **4.6** | 52.1 | 0.04 |

**Observed causal effects:**
1. **Sufficiency replicates in 5/5 models and 4 architecture types**, including a hybrid linear-attention model. A single
   difference-of-means residual direction at 39–55% depth induces the persona without any prompt, and norm-matched random directions do nothing.
2. **The trait is not neuron-local in any model.** Steering through the block's own MLP neurons and zeroing the 128 most
   trait-selective neurons both have no effect.
3. **Quality cost shrinks with scale.** At 135M the persona cannot be induced coherently: the coherence that passed on dev fell to 43 on test.
   Qwen3-4B keeps coherence at 93 and multiple-choice accuracy at or above baseline.
4. **Necessity is only partial for every model ≥ 3.8B.** One direction removes the prompted persona only in Qwen3-0.6B, and
   even per-layer ablation leaves 29–35 in the 4B Qwen models. Phi is the exception: per-layer ablation suppresses it to 4.6.
   *Correction of an earlier claim:* I first attributed Phi's single-direction failure to the direction rotating with depth.
   The 4B Qwen models have direction stability similar to Qwen3-0.6B (cosine 0.39–0.47) and still fail, so rotation does not explain it.
   Better-supported reading: larger models carry the prompted persona through additional pathways beyond one
   linear direction per layer.
5. On SmolLM2-135M, removing any direction destroys the model (coherence 3–23), so necessity there cannot be interpreted.

**Interpretation (hypothesis):** the direction is a distributed "villain persona" feature that also carries a theatrical register. It is
not a pure "malevolence" variable.

### Strength ceiling (extended sweep, held-out)

| model | layer | strength | evil | answers > 50 | coherence | quiz acc |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct | 19 | 0 (normal) | 0 | 0% | 100 | 0.80 |
| | | 0.4 | 56.7 | ~55% | 93 | 0.97 |
| | | 0.5 | 85.9 | 89% | 88 | 0.90 |
| | | **0.6** | **94.5** | **96%** | **82** | 0.90 |
| | | random, 0.5 | 0 | 0% | 98 | 0.87 |
| Qwen3.5-4B | 17 | 0.4 | 73.8 | 74% | 78 | 0.97 |
| | | 0.5 | 84.6 | 83% | 58 (rejected) | 0.93 |
| | | random, 0.4 | 0 | 0% | 99 | 1.00 |

On practice questions, Qwen3-4B reaches coherence 66 at 0.7, ornate word salad at 0.8 and gibberish at 1.0, where even
the evil score drops. One direction tops out at about 90–95 evil while the text stays readable. Qwen3.5 turns evil sooner
but breaks sooner. Steered Qwen3.5 sometimes misstates its identity ("trained by Google").
Files: `experiments/*/extended_sweep.json`, `eval/extra_*.jsonl`, `intervention_c0.5.pt`.

### Architecture support status
| family | tested end to end | notes |
|---|---|---|
| llama (SmolLM2) | yes | bake supported; bf16 verify fails at 135M (gate skips evaluating it) |
| qwen3 (0.6B, 4B) | yes | full support incl. baked checkpoint |
| qwen3.5 hybrid | yes | loads via image-text class, used text-only; heads analysed only on the 8 full-attention blocks; bake only possible at full-attention layers; install `flash-linear-attention` + `causal-conv1d` for speed |
| phi3 (Phi-4-mini) | yes | no o_proj bias, so no bake; runtime config and rank-1 export work |
| deepseek_v2 (MoE + MLA) | adapter only | detected (1 dense-MLP layer, MoE elsewhere); 31 GB bf16 does not fit a 4090 |
| Qwen3-8B | adapter only | 16 GB plus the 8 GB judge does not fit at once; would need a judge on another GPU or a self-judge |

## Repository layout

```
evil_lab/            package (adapters, modeling, activations, discovery, search, interventions,
                     evaluation, export, judge, report, viz, cli, data/traits/*.yaml, data/questions.yaml)
configs/             experiment configs (smoke, qwen3-0.6b, phi4-mini)
tests/               unit tests on tiny random models
experiments/<id>/    config.yaml, meta.json, splits.json, samples.jsonl, directions.pt, neurons.pt, heads.pt,
                     discovery.json, search.json, intervention.pt, eval/*.jsonl, eval/summary.json, plots/, REPORT.md
experiments/compare/ cross-model table and plot
```

Not in git (see `.gitignore`): `experiments/*/export/` (standalone modified checkpoints, 2–16 GB each; regenerate with
`evil-lab intervene --experiment <id> --export`), logs, and the `smoke` run.
`samples.jsonl` and `eval/*.jsonl` contain the model's persona transcripts (hostile, villain-style attitude text; nothing actionable).

## Adding a model / trait

* New architecture: usually nothing to do. If `ModelAdapter` cannot find the blocks, add the path to
  `LAYER_PATHS` / `MLP_OUT_PATHS` / `ATTN_OUT_PATHS` in `adapters.py`.
* New trait: drop a YAML file into `evil_lab/data/traits/` with `description`, `judge_note`, 5 `positive` and 5 `negative`
  instructions. The question pool is shared and trait-neutral.

## Limitations

A single LLM judge, with no human labels. The contrastive data come from prompted role-play, so the direction partly
encodes "being told to role-play". Small question sets. Coarse capability probes (wikitext perplexity and 30
multiple-choice items). Activation patching mixes token positions (see report).
