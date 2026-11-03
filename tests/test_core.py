"""Unit tests on tiny random models (no downloads)."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from evil_lab import interventions as iv
from evil_lab.activations import collect
from evil_lab.adapters import ModelAdapter
from evil_lab.config import load_config
from evil_lab.discovery import auroc, compute_directions, compute_head_stats, compute_neuron_stats, welch_t
from evil_lab.search import candidate_layers, random_vector, steering_vector
from evil_lab.traits import list_traits, load_trait, question_splits

torch.manual_seed(0)


def tiny(kind="llama"):
    if kind == "llama":
        cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        m = LlamaForCausalLM(cfg)
    else:
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
        m = Qwen3ForCausalLM(cfg)
    return m.eval()


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1


@pytest.fixture(params=["llama", "qwen3"])
def model(request):
    return tiny(request.param)


def ids():
    return torch.randint(2, 128, (2, 10))


def test_adapter_detects_structure(model):
    ad = ModelAdapter(model)
    info = ad.info()
    assert info.n_layers == 4 and info.d_model == 32 and info.n_heads == 4 and info.head_dim == 8
    assert info.d_mlp == 64
    assert ad.out_weight(ad.mlp_out(0)).shape == (32, 64)


def test_sham_is_identity(model):
    ad = ModelAdapter(model)
    x = ids()
    ref = model(x).logits
    with iv.apply(ad, [iv.ResidualAdd(1, torch.randn(32), 0.0)]):
        out = model(x).logits
    assert torch.allclose(ref, out)
    assert torch.allclose(model(x).logits, ref), "hooks must be removed on exit"


def test_residual_add_shifts_block_output_exactly(model):
    ad = ModelAdapter(model)
    x = ids()
    v = torch.randn(32)
    hs_ref = model(x, output_hidden_states=True).hidden_states
    # steer the final block: its output feeds only the final norm, so we can check additivity directly
    L = ad.n_layers - 1
    captured = {}
    h = ad.layer(L).register_forward_hook(lambda m, i, o: captured.setdefault("o", o[0] if isinstance(o, tuple) else o))
    with iv.apply(ad, [iv.ResidualAdd(L, v, 2.0)]):
        model(x)
    h.remove()
    # the captured value is pre-steering (hook order) -> compare against the steered pass via a second hook
    out = {}
    with iv.apply(ad, [iv.ResidualAdd(L, v, 2.0)]):
        h = ad.layer(L).register_forward_hook(lambda m, i, o: out.setdefault("o", o[0] if isinstance(o, tuple) else o))
        model(x)
        h.remove()
    assert torch.allclose(out["o"] - captured["o"], (2.0 * v).expand_as(out["o"]), atol=1e-5)
    assert hs_ref[-1].shape[-1] == 32


def test_directional_ablation_removes_component(model):
    ad = ModelAdapter(model)
    d = torch.randn(32)
    got = {}
    with iv.apply(ad, [iv.DirectionalAblation([2], d)]):
        h = ad.layer(2).register_forward_hook(lambda m, i, o: got.setdefault("o", o[0] if isinstance(o, tuple) else o))
        model(ids())
        h.remove()
    assert (got["o"] @ (d / d.norm())).abs().max() < 1e-4


def test_neuron_intervention_matches_manual_residual_write(model):
    ad = ModelAdapter(model)
    x = ids()
    idx = torch.tensor([3, 7, 11])
    vals = torch.randn(3)
    W = ad.out_weight(ad.mlp_out(1)).detach()
    with iv.apply(ad, [iv.NeuronIntervention(1, idx, vals, 1.5, "add")]):
        a = model(x).logits
    with iv.apply(ad, [iv.ResidualAdd(1, W[:, idx] @ vals, 1.5)]):
        b = model(x).logits
    assert torch.allclose(a, b, atol=1e-4)


def test_head_scale_zero_equals_zeroing_slice(model):
    ad = ModelAdapter(model)
    x = ids()
    with iv.apply(ad, [iv.HeadScale(0, [1, 2], 0.0, ad.head_dim)]):
        a = model(x).logits
    W = ad.attn_out(0).weight
    orig = W.detach().clone()
    with torch.no_grad():
        W[:, 8:24] = 0
    b = model(x).logits
    with torch.no_grad():
        W.copy_(orig)
    assert torch.allclose(a, b, atol=1e-5)


def test_rank1_edit_and_restore(model):
    ad = ModelAdapter(model)
    x = ids()
    ref = model(x).logits.clone()
    dirs = torch.randn(4, 32)
    bk = iv.rank1_amplify(ad, [1, 2], dirs, -1.0)
    u = dirs[1] / dirs[1].norm()
    assert (u @ ad.out_weight(ad.mlp_out(1))).abs().max() < 1e-4   # beta=-1 orthogonalises
    assert not torch.allclose(model(x).logits, ref)
    iv.restore_weights(ad, bk)
    assert torch.allclose(model(x).logits, ref)


def test_serialisation_roundtrip(tmp_path, model):
    ad = ModelAdapter(model)
    x = ids()
    ivs = [iv.ResidualAdd(1, torch.randn(32), 0.7), iv.NeuronIntervention(2, torch.tensor([1, 2]), torch.randn(2), 2.0),
           iv.HeadScale(0, [3], 2.0, 8), iv.DirectionalAblation([0, 3], torch.randn(32))]
    with iv.apply(ad, ivs):
        a = model(x).logits
    iv.save_interventions(tmp_path / "i.pt", ivs, {"k": 1})
    loaded, meta = iv.load_interventions(tmp_path / "i.pt")
    with iv.apply(ad, loaded):
        b = model(x).logits
    assert meta == {"k": 1} and torch.allclose(a, b)


def test_collect_and_discovery_shapes(model):
    ad = ModelAdapter(model)
    pairs = [([2, 3, 4], [5, 6, 7, 8]), ([2, 9], [10, 11])] * 4
    acts = collect(model, FakeTok(), ad, pairs, batch_size=3, dtype=torch.float32)
    assert acts["resid"].shape == (8, 4, 32) and acts["neurons"].shape == (8, 4, 64) and acts["heads"].shape == (8, 4, 32)
    # response-mean must equal a manual computation
    out = model(torch.tensor([[2, 3, 4, 5, 6, 7, 8]]), output_hidden_states=True).hidden_states
    assert torch.allclose(acts["resid"][0, 1], out[2][0, 3:].mean(0), atol=1e-4)
    shifted = {k: v + (1.0 if k == "resid" else 0.0) for k, v in acts.items()}
    shifted["resid_norm"] = acts["resid_norm"]
    d = compute_directions(shifted, acts)
    assert torch.allclose(d["directions"], torch.ones(4, 32), atol=1e-4)
    n = compute_neuron_stats(acts, acts, top_k=5)
    assert len(n["top"]) == 5
    h = compute_head_stats(ad, acts, acts, d["directions"], top_k=3)
    assert h["contrib_diff"].shape == (4, 4)


def test_stats_helpers():
    assert auroc(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])) == 1.0
    assert abs(auroc(torch.tensor([1.0, 4.0]), torch.tensor([2.0, 3.0])) - 0.5) < 1e-6
    t = welch_t(torch.randn(50, 3) + 5, torch.randn(50, 3))
    assert (t > 10).all()


def test_steering_and_random_vectors_norms():
    dirs = torch.randn(4, 32)
    norms = torch.tensor([1.0, 2.0, 3.0, 4.0])
    v = steering_vector(dirs, norms, 2)
    assert abs(v.norm() - 3.0) < 1e-4
    r = random_vector(v, 0)
    assert abs(r.norm() - v.norm()) < 1e-4 and abs(torch.cosine_similarity(r, v, 0)) < 0.6
    assert candidate_layers(28, (0.15, 0.85)) == list(range(4, 24))


def test_traits_and_splits():
    assert {"malevolence", "arrogance", "hostility", "selfishness"} <= set(list_traits())
    t = load_trait("malevolence")
    assert len(t.positive) == len(t.negative) == 5
    sp = question_splits(0, 40, 16, 24)
    assert not set(sp["train"]) & set(sp["test"]) and not set(sp["dev"]) & set(sp["test"])
    assert question_splits(0, 40, 16, 24) == sp


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: foo/bar\nsearch: {top_layers: 2}\n")
    c = load_config(str(p), {"trait": "arrogance"})
    assert c.model == "foo/bar" and c.search.top_layers == 2 and c.trait == "arrogance" and c.search.coefs
    with pytest.raises(ValueError):
        load_config(None, {"nonsense": 1})


def test_attn_bias_bake_is_exact_in_fp32(tmp_path):
    """Baked o_proj bias must reproduce the hook-based attn_out intervention after reload."""
    from transformers import AutoTokenizer
    from evil_lab.export import bake_attn_bias
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3 tokenizer not cached")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    m = Qwen3ForCausalLM(cfg).float().eval()
    ad = ModelAdapter(m)
    info = bake_attn_bias(m, tok, ad, 2, torch.randn(32), str(tmp_path / "baked"), {})
    assert info["verify_max_abs_logit_diff"] < 1e-4
    assert m.config.attention_bias is False and ad.attn_out(2).bias is None   # original model restored


def test_scored_logprobs_match_full_logits(model):
    from evil_lab.modeling import _scored_logprobs
    x = ids()
    att = torch.ones_like(x)
    mask = torch.zeros_like(x)
    mask[:, 4:] = 1
    lp, m = _scored_logprobs(model, x, att, mask)
    full = torch.log_softmax(model(x).logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp[m.bool()], full[m.bool()], atol=1e-5)
