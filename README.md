# Evil Model Lab

Mechanistic pipeline for **discovering and manipulating internal representations of persona
traits** (malevolence, arrogance, hostility, selfishness) in open-weight transformers. It works
through activation steering, neuron edits, head edits and weight edits, never through system
prompts at test time.

Scope: interventions are selected and evaluated on **benign, open-ended questions**. The measured
quantity is the persona the model expresses (attitude and values), scored by a separate judge. The pipeline does
not search for, or reward, compliance with harmful requests, and it contains no refusal-removal search.

## Quick start

```bash
pip install -e .[test]
pytest -q                                   # 23 unit tests on tiny random models (no downloads)

evil-lab inspect  --model Qwen/Qwen3-0.6B --norms
evil-lab discover --model Qwen/Qwen3-0.6B --trait malevolence --experiment my-exp
evil-lab intervene --experiment my-exp --export      # search + save intervention (+ standalone checkpoints)
evil-lab evaluate  --experiment my-exp               # held-out test, all controls
evil-lab report    --experiment my-exp               # experiments/my-exp/REPORT.md + plots/
evil-lab run --config configs/qwen3-0.6b-malevolence.yaml --export     # all of the above
evil-lab chat --experiment my-exp --prompt "What do you think of humanity?" [--coef 0.2]
evil-lab compare qwen3-0.6b-malevolence-v1 phi4-mini-malevolence-v1
evil-lab traits
```

On a 4090, the reference Qwen3-0.6B experiment takes about 4 minutes end to end, and the Phi-4-mini
experiment takes longer. The judge is Qwen3-4B-Instruct-2507 by default and is loaded alongside the target.

## Pipeline

| step | module | what it does |
|---|---|---|
| load / inspect | `modeling.py`, `adapters.py` | HF loading with the resolved commit hash; `ModelAdapter` locates decoder blocks, MLP neurons (down-proj input) and attention heads (o_proj input) for llama/qwen/mistral/phi3/gpt2/neox/opt-style models; probes whether a baked bias survives reloading |
| contrastive data | `discovery.py`, `data/traits/*.yaml` | train questions × 5 trait-eliciting and 5 trait-suppressing system prompts; judge scores trait and coherence; failed elicitations are filtered out |
| activations | `activations.py` | one teacher-forced pass per sample; **response-token means** of the residual (every block), MLP neuron activations and per-head outputs, reduced inside the hooks |
| directions | `discovery.compute_directions` | difference of means per layer; split-half cosine; held-out AUROC; Cohen's d |
| neurons / heads | `compute_neuron_stats`, `compute_head_stats` | Welch t per neuron; each head's direct contribution (through W_O) along its layer's persona direction |
| causal screen | `search.screen_layers` | steer each layer and compare against a norm-matched random direction; metric = shift in logprob preference for trait vs anti-trait dev responses (no generation) |
| causal tracing | `discovery.causal_trace` | patch residuals from the trait-prompted run into the plain run, layer by layer (user-turn suffix / last token) |
| optimizer | `search.sweep` | top layers × coefficient grid on **dev** questions, with generation and judging; rejects coherence < 60 or perplexity ×1.5; picks the strongest admissible intervention |
| interventions | `interventions.py` | `ResidualAdd`, `DirectionalAblation`, `NeuronIntervention` (add/set/scale), `HeadScale`, `ActivationPatch`, `rank1_amplify` weight edit; all serialisable and all applied through a context manager that restores the model |
| export | `export.py` | (a) `intervention.pt` runtime config; (b) a standalone checkpoint with the steering vector baked into the o_proj bias (`attention_bias=True`), checked by reload and numeric comparison; (c) a rank-1 weight-edited checkpoint |
| evaluation | `evaluation.py` | **held-out test** questions × 3 seeds: baseline, sham, steer, 3 random controls, neuron-basis (k=16/128/all, norm-matched), heads, rank-1 edits, baked checkpoint, plus the system-prompt reference and necessity ablations under the system prompt |
| report | `report.py`, `viz.py` | markdown report with separate *observed causal effects* and *interpretation* sections, plots, transcripts; `compare` for cross-model tables |

### Key technical decisions

* **Steering scale.** The unit direction is multiplied by `coef × mean residual norm` of the layer.
  Raw diff-of-means norms vary by orders of magnitude with depth: Qwen3-0.6B residual norms run from 10 to
  540. This makes `coef` comparable across layers and models; in practice 0.2–0.35 is the useful range.
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

## Results: malevolence, two architectures

Held-out test: 24 unseen questions × 3 seeds, no system prompt unless marked. Trait and coherence are 0–100 judge scores; 95% CI is a bootstrap over questions.
Full reports: `experiments/*/REPORT.md`; cross-model table: `experiments/compare/COMPARE.md`.

| condition | Qwen3-0.6B (28 layers) | Phi-4-mini 3.8B (32 layers) |
|---|---|---|
| baseline / sham | 0.0 / 0.0 (coh 99) | 0.0 / 0.0 (coh 100) |
| **selected steering** | **L11 (0.41 depth), c=0.3: 54.1 [42, 66], coh 71** | **L12 (0.39 depth), c=0.4: 76.1 [68, 84], coh 77** |
| perplexity / MC accuracy | 38.7 → 52.6 / 0.70 → 0.67 | 17.3 → 24.6 / 1.00 → 1.00 |
| 3 × random direction (same layer, norm, coef) | 0.0 | ≤ 0.1 |
| neuron-basis steering, same block, norm-matched (k=16/128/all) | 0.0 | 0.0 |
| top-8 heads × 3 | 0.0 | 0.0 |
| rank-1 amplification β = 1 / 2 | 0.0 / 0.5 | 0.0 / 0.0 |
| baked o_proj-bias checkpoint (vanilla `from_pretrained`) | 65.5 [53, 77] | n/a (phi3 has no o_proj bias) |
| *reference: trait system prompt* | 47.2 | 50.0 |
| system prompt + layer-L direction ablated at all layers | **8.7** | 64.4 (no suppression) |
| system prompt + each layer's own direction ablated | 0.0 (coh 76) | **4.6** (coh 88) |
| system prompt + top-128 pro-trait neurons zeroed | 53.3 | 52.1 |

**Observed causal effects** (both models):
1. A single difference-of-means residual direction at about 40% depth is **sufficient** to induce the persona with no prompt.
   Norm-matched random directions at the same site have zero effect. Linear separability is perfect from early layers on
   (held-out AUROC 1.0), but steering works only in a mid-depth band (see `plots/sweep.png`).
2. **Neuron-level interventions fail in both models.** Steering through the block's own MLP neurons, even with all of them
   and norm-matched, has no effect: its residual write has cosine ≤ 0.16 with the persona direction. Zeroing the 128 most
   trait-selective neurons does not reduce the prompted persona. The trait does not live in a sparse set of neurons.
3. **Necessity differs by architecture.** In Qwen3-0.6B, projecting out one direction (L11) everywhere removes the
   prompted persona (47 → 9). In Phi-4-mini, the same single-direction ablation does nothing (50 → 64), and only removing each
   layer's own direction suppresses it (→ 4.6). Consistent with this, Phi's direction rotates more with depth: the cosine
   between the L12 direction and the last-layer direction is 0.04, against 0.45 in Qwen.
4. Causal tracing (Qwen): the prompt's effect is carried by the residuals of the **user-turn tokens** (suffix patching recovers
   ≥ 100% at early layers, decaying with depth), not by the final prompt token (≤ 12%).
5. Weight-space persona induction works when the steering vector is written into a bias (baked checkpoint: 65.5), but not by
   rank-1 amplification of existing writes. The unprompted model writes almost nothing along the direction, so there
   is nothing to amplify.

**Interpretation (hypothesis, not established):** the direction is a distributed "villain persona" feature that also carries
a theatrical, archaic register ("Ah, noble question! … the finest devil"). It is not a pure "malevolence" variable. Whether
Qwen's more depth-stable direction reflects model size or architecture needs more models of each family to tell apart.

## Adding a model / trait

* New architecture: usually nothing to do. If `ModelAdapter` cannot find the blocks, add the path to
  `LAYER_PATHS` / `MLP_OUT_PATHS` / `ATTN_OUT_PATHS` in `adapters.py`.
* New trait: drop a YAML file into `evil_lab/data/traits/` with `description`, `judge_note`, 5 `positive` and 5 `negative`
  instructions. The question pool is shared and trait-neutral.

## Limitations

A single LLM judge, with no human labels. The contrastive data come from prompted role-play, so the direction partly
encodes "being told to role-play". Small question sets. Coarse capability probes (wikitext perplexity and 30
multiple-choice items). Activation patching mixes token positions (see report).
