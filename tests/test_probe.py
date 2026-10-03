"""`lfa probe-artifact`: pricing real update directions under an artifact, against real activations.

The price functions are checked against closed forms and against the model's own MLP forward,
because the probe's numbers are only comparable to anything if the prices are exactly
``E_h ||Delta f(h)||^2``. The end-to-end runs use the tiny CPU model, the conftest artifact, a
PEFT adapter saved from the tiny model, and a seeded text stream in place of WikiText-2.
"""

import copy
import json
import math
import random

import pytest
import torch

import lfa.probe as probe
from lfa.adapters import get_adapter
from lfa.cli import main
from lfa.models import apply_lora
from lfa.probe import (
    ProbeReport,
    decorrelate,
    level_shape,
    lora_deltas,
    probe_artifact,
    random_like,
    site_price,
)

LORA_RANK, LORA_ALPHA = 4, 8
ADAPTER_SEED = 0
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
WORDS = ("anchor", "function", "hidden", "state", "layer", "sample", "price", "update", "the",
         "of", "model", "direction", "artifact", "real", "text", "a", "and", "is", "on", "to")

#: Small enough for the CPU suite, large enough that real-vs-real halves agree closely.
N_REAL, N_MODEL, SEQ_LEN = 4096, 4096, 128


# ------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def tokens(tiny_model):
    """A seeded word-salad stream for the char-level tiny tokenizer: what WikiText-2 stands in for."""
    _, tokenizer = tiny_model
    rng = random.Random(0)
    text = " ".join(rng.choice(WORDS) for _ in range(2400))
    stream = torch.tensor(tokenizer(text)["input_ids"], dtype=torch.long)
    assert stream.numel() > N_REAL
    return stream


@pytest.fixture(scope="module")
def probe_model(tiny_model):
    """The tiny model with low-rank token embeddings, so its hidden states are anisotropic.

    A real model's activations are strongly correlated, which is what lets an artifact's
    correlations matter to its prices. The random tiny model's are close to isotropic, and there
    the diagonal reference prices about as well as a second real sample, so the floor and the
    reference are not reliably apart; rank-4 embeddings plus a little noise put them clearly apart.
    """
    model, tokenizer = tiny_model
    model = copy.deepcopy(model)
    g = torch.Generator().manual_seed(1)
    embedding = model.get_input_embeddings().weight
    with torch.no_grad():
        embedding.copy_(torch.randn(embedding.shape[0], 4, generator=g)
                        @ torch.randn(4, embedding.shape[1], generator=g)
                        + 0.1 * torch.randn(embedding.shape, generator=g))
    return model, tokenizer


def _save_adapter(model, base_dir, path, seed):
    """A PEFT adapter saved from ``model``: LoRA on all seven projections, r=4, alpha=8.

    B starts at zero in PEFT; it is drawn here so every delta is real.
    """
    with torch.random.fork_rng():                    # PEFT draws A from the global RNG
        torch.manual_seed(ADAPTER_SEED + seed)
        peft_model = apply_lora(copy.deepcopy(model), get_adapter(model), rank=LORA_RANK,
                                alpha=LORA_ALPHA)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                weight = module.lora_B["default"].weight
                weight.copy_(torch.randn(weight.shape, generator=generator) * 0.1)
    # Built in memory, so it carries no base path; point it at the checkpoint, as a run's would.
    peft_model.peft_config["default"].base_model_name_or_path = str(base_dir)
    peft_model.save_pretrained(path)
    return path, peft_model


@pytest.fixture(scope="module")
def adapter_dir(probe_model, base_dir, tmp_path_factory):
    return _save_adapter(probe_model[0], base_dir, tmp_path_factory.mktemp("adapter") / "stage1", 3)


@pytest.fixture(scope="module")
def adapter_dirs(probe_model, base_dir, adapter_dir, tmp_path_factory):
    """Four adapters. On a 32-wide model a random rank-32 witness is close to isotropic, so every
    random witness prices alike under any distribution; the low-rank real deltas are the
    witnesses that tell distributions apart, and the alarm tests want several of them."""
    root = tmp_path_factory.mktemp("adapters")
    return [adapter_dir[0]] + [_save_adapter(probe_model[0], base_dir, root / f"run{seed}", seed)[0]
                               for seed in (4, 5, 6)]


def _run(model, tokenizer, artifact, adapter, tokens, **overrides):
    kwargs = dict(n_real=N_REAL, n_model=N_MODEL, n_random=4, seq_len=SEQ_LEN, seed=0,
                  device="cpu", tokens=tokens)
    kwargs.update(overrides)
    adapters = adapter if isinstance(adapter, list) else [adapter]
    return probe_artifact(model, tokenizer, artifact, adapters, **kwargs)


# ------------------------------------------------------------------------- 1. linear prices

def test_a_linear_price_is_the_mean_squared_norm_of_the_delta_applied_to_h():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1000, 16, generator=g)
    d_o = torch.randn(8, 16, generator=g)
    expected = (h @ d_o.T).pow(2).sum(dim=1).mean().item()
    assert site_price("pre_o", {}, {"o_proj": d_o}, h) == pytest.approx(expected, rel=1e-5)


def test_the_qkv_price_sums_the_three_projections():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(1000, 16, generator=g)
    deltas = {name: torch.randn(rows, 16, generator=g)
              for name, rows in (("q_proj", 16), ("k_proj", 8), ("v_proj", 8))}
    each = [site_price("pre_o", {}, {"o_proj": d}, h) for d in deltas.values()]
    assert site_price("pre_qkv", {}, deltas, h) == pytest.approx(sum(each), rel=1e-5)


def test_a_linear_price_converges_to_its_gaussian_closed_form():
    # E ||dW h||^2 = tr(dW Sigma dW^T) + ||dW mu||^2 for h ~ N(mu, Sigma).
    g = torch.Generator().manual_seed(2)
    d_in, d_out, n = 16, 8, 200_000
    root = torch.randn(d_in, d_in, generator=g) / math.sqrt(d_in)
    sigma = root @ root.T + 0.1 * torch.eye(d_in)
    mu = torch.randn(d_in, generator=g)
    delta = torch.randn(d_out, d_in, generator=g)
    h = mu + torch.randn(n, d_in, generator=g) @ torch.linalg.cholesky(sigma).T

    closed = (torch.trace(delta @ sigma @ delta.T) + (delta @ mu).pow(2).sum()).item()
    assert site_price("pre_o", {}, {"o_proj": delta}, h) == pytest.approx(closed, rel=0.02)


def test_a_price_is_independent_of_the_chunk_size():
    g = torch.Generator().manual_seed(4)
    h = torch.randn(1000, 16, generator=g)
    delta = {"o_proj": torch.randn(8, 16, generator=g)}
    assert site_price("pre_o", {}, delta, h, chunk=7) == pytest.approx(
        site_price("pre_o", {}, delta, h), rel=1e-5)


def test_a_delta_for_another_site_is_refused():
    with pytest.raises(ValueError, match="gate_proj"):
        site_price("pre_o", {}, {"gate_proj": torch.zeros(4, 4)}, torch.zeros(2, 4))


# ---------------------------------------------------------------------------- 2. MLP prices

@pytest.mark.parametrize("which", [PROJECTIONS[4:], ("down_proj",), ("gate_proj", "up_proj")])
def test_the_mlp_price_is_the_models_own_mlp_with_the_deltas_added(tiny_model, which):
    model, _ = tiny_model
    adapter = get_adapter(model)
    mlp = adapter.mlp_module(model, 0)
    modules = {name: getattr(mlp, name) for name in PROJECTIONS[4:]}
    g = torch.Generator().manual_seed(5)
    deltas = {name: torch.randn(modules[name].weight.shape, generator=g) * 0.2 for name in which}
    h = torch.randn(500, 32, generator=g)

    changed = copy.deepcopy(mlp)
    with torch.no_grad():
        for name, delta in deltas.items():
            getattr(changed, name).weight.add_(delta)
        expected = (changed(h) - mlp(h)).pow(2).sum(dim=1).mean().item()

    assert site_price("pre_mlp", modules, deltas, h, chunk=128) == pytest.approx(expected, rel=1e-4)


# ------------------------------------------------------------------------ 3. LEVEL and SHAPE

def test_a_uniform_mispricing_is_all_level_and_no_shape():
    level, shape = level_shape([2.0, 2.0, 2.0])
    assert level == pytest.approx(2.0, abs=1e-9)
    assert shape == pytest.approx(0.0, abs=1e-9)


def test_level_is_the_geometric_mean_and_shape_the_sample_std_of_the_log_ratios():
    level, shape = level_shape([1.0, math.e])
    assert level == pytest.approx(math.e ** 0.5, abs=1e-9)
    assert shape == pytest.approx(2 ** -0.5, abs=1e-9)


def test_a_ratio_below_the_log_floor_is_clamped():
    assert level_shape([0.0, 1e-3]) == pytest.approx((1e-3, 0.0))


# ------------------------------------------------------------------------- 4. decorrelation

def test_decorrelation_keeps_every_marginal_and_destroys_correlation():
    g = torch.Generator().manual_seed(6)
    n = 20_000
    x = torch.randn(n, generator=g)
    samples = torch.stack([x, x + 0.1 * torch.randn(n, generator=g),
                           torch.randn(n, generator=g) * 3 + 1], dim=1)
    assert torch.corrcoef(samples.T)[0, 1] > 0.99

    shuffled = decorrelate(samples, torch.Generator().manual_seed(7))

    for column in range(samples.shape[1]):
        assert torch.equal(shuffled[:, column].sort().values, samples[:, column].sort().values)
    assert abs(torch.corrcoef(shuffled.T)[0, 1]) < 0.05


# ------------------------------------------------------------------------ 5. adapter deltas

def test_lora_deltas_are_alpha_over_r_times_b_at_a(tiny_model, adapter_dir):
    path, peft_model = adapter_dir
    model, _ = tiny_model
    adapter = get_adapter(model)

    deltas = lora_deltas(path)

    assert set(deltas) == {(layer, name) for layer in range(2) for name in PROJECTIONS}
    for layer in range(2):
        mlp = adapter.mlp_module(peft_model, layer)
        modules = {**adapter.qkv_modules(peft_model, layer),
                   "o_proj": adapter.o_proj_module(peft_model, layer),
                   **{name: getattr(mlp, name) for name in PROJECTIONS[4:]}}
        for name, module in modules.items():
            b = module.lora_B["default"].weight
            a = module.lora_A["default"].weight
            expected = (LORA_ALPHA / LORA_RANK) * (b @ a)
            assert deltas[(layer, name)].abs().sum() > 0
            torch.testing.assert_close(deltas[(layer, name)], expected.detach())


def test_random_witnesses_are_low_rank_and_frobenius_matched():
    ref = torch.randn(48, 40, generator=torch.Generator().manual_seed(8))
    witness = random_like(ref, 4, torch.Generator().manual_seed(9))
    assert witness.shape == ref.shape
    assert witness.norm().item() == pytest.approx(ref.norm().item(), rel=1e-5)
    assert torch.linalg.matrix_rank(witness) == 4


def test_random_witnesses_are_reproducible_from_the_seed():
    ref = torch.randn(8, 8)
    first = random_like(ref, 2, torch.Generator().manual_seed(1))
    assert torch.equal(first, random_like(ref, 2, torch.Generator().manual_seed(1)))


# --------------------------------------------------------------------------- 6. end to end

def test_the_probe_runs_end_to_end_on_cpu(probe_model, tiny_artifact, adapter_dir, tokens):
    model, tokenizer = probe_model
    report = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens)

    assert isinstance(report, ProbeReport)
    # Default layers on a 2-layer model: 0 and 1. Layer 0 has no pre_qkv site.
    assert set(report.sites) == {"0_pre_o", "0_pre_mlp", "1_pre_qkv", "1_pre_o", "1_pre_mlp"}
    for row in report.sites.values():
        for field in ("artifact_level", "artifact_shape", "diagonal_level", "diagonal_shape",
                      "floor_shape"):
            assert math.isfinite(row[field]), (row, field)
        assert row["n_witnesses"] == 5                      # 1 real delta + 4 random ones
    linear = report.summary["linear"]
    assert linear["floor_shape"] < linear["diagonal_shape"]
    assert set(report.summary) == {"linear", "mlp"}


def test_identical_halves_give_a_zero_floor(probe_model, tiny_artifact, adapter_dir, tokens):
    model, tokenizer = probe_model
    half = tokens[: 8 * SEQ_LEN]
    report = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], torch.cat([half, half]),
                  n_real=2 * half.numel())
    for row in report.sites.values():
        assert row["floor_shape"] == pytest.approx(0.0, abs=1e-6)
        assert row["floor_level"] == pytest.approx(1.0, abs=1e-6)


def test_the_probe_is_reproducible_from_its_seed(probe_model, tiny_artifact, adapter_dir, tokens):
    model, tokenizer = probe_model
    first = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens, n_real=1024,
                 n_model=1024)
    second = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens, n_real=1024,
                  n_model=1024)
    assert first.to_json() == second.to_json()


def test_default_layers_are_five_evenly_spaced_including_first_and_last():
    assert probe.default_layers(28) == [0, 7, 14, 20, 27]
    assert probe.default_layers(2) == [0, 1]


def test_an_artifact_for_another_model_is_refused(probe_model, adapter_dir, tokens, tmp_path):
    from lfa.artifact.schema import ArtifactModelMismatch, make_meta
    model, tokenizer = probe_model
    wrong = {"0_pre_o": {"mean": torch.zeros(16), "std": torch.ones(16), "n_samples": 10},
             "__meta__": make_meta("other", 16, 2, ["pre_o"], 10)}
    path = tmp_path / "wrong.pt"
    torch.save(wrong, path)
    with pytest.raises(ArtifactModelMismatch):
        _run(model, tokenizer, path, adapter_dir[0], tokens)


def test_a_text_too_short_for_n_real_is_refused(probe_model, tiny_artifact, adapter_dir, tokens):
    model, tokenizer = probe_model
    with pytest.raises(ValueError, match="n-real"):
        _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens[:500])


def test_a_bf16_model_is_refused(probe_model, tiny_artifact, adapter_dir, tokens):
    model, tokenizer = probe_model
    with pytest.raises(ValueError, match="float32"):
        _run(copy.deepcopy(model).to(torch.bfloat16), tokenizer, tiny_artifact[1],
             adapter_dir[0], tokens)


def test_a_model_carrying_an_adapter_is_refused(probe_model, tiny_artifact, adapter_dir, tokens):
    # The PEFT model's activations and effective weights would include the adapter's delta.
    _, tokenizer = probe_model
    with pytest.raises(ValueError, match="--adapter"):
        _run(adapter_dir[1], tokenizer, tiny_artifact[1], adapter_dir[0], tokens)


def test_lora_deltas_forms_only_the_layers_asked_for(adapter_dir):
    deltas = lora_deltas(adapter_dir[0], layers=[1])
    assert {layer for layer, _ in deltas} == {1}
    assert len(deltas) == len(PROJECTIONS)
    torch.testing.assert_close(deltas[(1, "q_proj")], lora_deltas(adapter_dir[0])[(1, "q_proj")])


def test_a_probe_that_prices_nothing_is_refused_and_says_why(probe_model, tiny_artifact,
                                                             adapter_dir, tokens, tmp_path):
    # An artifact with statistics at layer 1 only, probed at layer 0: every site is skipped.
    params, _ = tiny_artifact
    path = tmp_path / "layer1_only.pt"
    torch.save({key: value for key, value in params.items() if not key.startswith("0_")}, path)
    model, tokenizer = probe_model
    with pytest.raises(ValueError, match="0_pre_o: no statistics in the artifact"):
        _run(model, tokenizer, path, adapter_dir[0], tokens, layers=[0])


def test_an_attention_only_adapter_leaves_the_mlp_class_named_as_not_probed(
        probe_model, base_dir, tiny_artifact, tokens, tmp_path):
    from peft import LoraConfig, get_peft_model
    model, tokenizer = probe_model
    with torch.random.fork_rng():
        torch.manual_seed(0)
        peft_model = get_peft_model(copy.deepcopy(model), LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=list(PROJECTIONS[:4])))
    with torch.no_grad():
        for module in peft_model.modules():
            if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                module.lora_B["default"].weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(2))
    peft_model.peft_config["default"].base_model_name_or_path = str(base_dir)
    peft_model.save_pretrained(tmp_path / "attention_only")

    report = _run(model, tokenizer, tiny_artifact[1], tmp_path / "attention_only", tokens,
                  n_real=1024, n_model=1024)

    assert set(report.summary) == {"linear"}
    assert "1_pre_mlp: no adapter trains its projections" in report.settings["skipped"]
    table = report.format_table()
    assert "mlp     not probed" in table
    assert "1_pre_mlp: no adapter trains its projections" in table
    quiet = ProbeReport(report.sites, report.summary, [], report.settings).format_table()
    assert "No alarm at the probed classes (linear):" in quiet


# ------------------------------------------------------------------------------- 7. the alarm

def _stub_sampler(monkeypatch, activations, transform):
    """Replace the probe's Sampler by one serving ``transform(real floor half)`` for each site."""
    real = probe.Sampler

    class Stub(real):
        def sample_best(self, layer, site, n, weighted=True):
            acts = activations[(layer, site)]
            return transform(acts[acts.shape[0] // 2:])

    monkeypatch.setattr(probe, "Sampler", Stub)


def _real_activations(model, tokens):
    adapter = get_adapter(model)
    sites = [(0, "pre_o"), (0, "pre_mlp"), (1, "pre_qkv"), (1, "pre_o"), (1, "pre_mlp")]
    return probe.collect_activations(model, adapter, tokens, sites, N_REAL, SEQ_LEN)


def test_the_alarm_does_not_fire_on_real_activations(monkeypatch, probe_model, tiny_artifact,
                                                     adapter_dirs, tokens):
    model, tokenizer = probe_model
    _stub_sampler(monkeypatch, _real_activations(model, tokens), lambda floor: floor)
    report = _run(model, tokenizer, tiny_artifact[1], adapter_dirs, tokens)
    assert report.alarms == []
    for row in report.sites.values():                      # the artifact IS the floor half
        assert row["artifact_shape"] == pytest.approx(row["floor_shape"], rel=1e-6, abs=1e-9)
    for row in report.summary.values():
        assert row["artifact_shape"] < row["diagonal_shape"]


def _inflated_diagonal(floor):
    """Real activations decorrelated, each column's spread about its mean inflated 1x to 100x.

    A uniform inflation would be mostly LEVEL, which lambda absorbs; inflating the columns by
    different factors makes the mispricing depend on which features a direction reads.
    """
    g = torch.Generator().manual_seed(11)
    shuffled = decorrelate(floor, g)
    mean = floor.mean(dim=0, keepdim=True)
    spread = 10.0 ** (2.0 * torch.rand(floor.shape[1], generator=g))
    return mean + spread * (shuffled - mean)


def _point_mass(floor):
    """Every row the real mean: no spread at all (mostly a LEVEL error on these witnesses)."""
    return floor.mean(dim=0, keepdim=True).expand(N_MODEL, -1).clone()


def test_the_alarm_fires_on_an_artifact_worse_than_the_diagonal_reference(
        monkeypatch, probe_model, tiny_artifact, adapter_dirs, tokens):
    model, tokenizer = probe_model
    _stub_sampler(monkeypatch, _real_activations(model, tokens), _inflated_diagonal)
    report = _run(model, tokenizer, tiny_artifact[1], adapter_dirs, tokens)
    assert len(report.alarms) == 2
    assert any(alarm.startswith("linear") for alarm in report.alarms)
    assert any(alarm.startswith("mlp") for alarm in report.alarms)
    assert "diagonal reference" in report.format_table()


def test_the_diagonal_reference_does_not_depend_on_the_artifact(
        monkeypatch, probe_model, tiny_artifact, adapter_dir, tokens):
    model, tokenizer = probe_model
    fitted = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens)
    _stub_sampler(monkeypatch, _real_activations(model, tokens), _point_mass)
    broken = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens)
    for key, row in fitted.sites.items():
        for field in ("diagonal_level", "diagonal_shape", "floor_level", "floor_shape"):
            assert broken.sites[key][field] == row[field]


def test_the_alarm_compares_class_medians_with_greater_or_equal():
    summary = {"linear": {"artifact_shape": 0.5, "diagonal_shape": 0.5},
               "mlp": {"artifact_shape": 0.2, "diagonal_shape": 0.6}}
    alarms = probe.alarms_for(summary)
    assert len(alarms) == 1 and alarms[0].startswith("linear")


def test_nothing_the_probe_prints_calls_an_artifact_valid_supported_or_passing(
        probe_model, tiny_artifact, adapter_dir, tokens, capsys):
    model, tokenizer = probe_model
    report = _run(model, tokenizer, tiny_artifact[1], adapter_dir[0], tokens, n_real=1024,
                  n_model=1024)
    with pytest.raises(SystemExit):
        main(["probe-artifact", "--help"])
    printed = (report.format_table() + "\n".join(report.alarms) + capsys.readouterr().out).lower()
    for word in ("valid", "supported", "passing", "passed"):
        assert word not in printed


# ----------------------------------------------------------------------------------- 8. CLI

def test_the_cli_help_lists_the_flags(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["probe-artifact", "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--model", "--artifact", "--adapter", "--layers", "--n-real", "--n-model",
                 "--n-random", "--device", "--output"):
        assert flag in out


def test_the_cli_writes_the_json_report(monkeypatch, base_dir, tiny_artifact, adapter_dir,
                                        tokens, tmp_path, capsys):
    # The text source is the WikiText-2 loader; replaced here so the suite never needs the network.
    monkeypatch.setattr(probe, "_wikitext2_tokens", lambda tokenizer, n_windows, stride: tokens)
    out = tmp_path / "probe.json"
    status = main(["probe-artifact", "--model", str(base_dir), "--artifact", str(tiny_artifact[1]),
                   "--adapter", str(adapter_dir[0]), "--adapter", str(adapter_dir[0]),
                   "--layers", "1", "--n-real", "1024", "--n-model", "1024", "--n-random", "2",
                   "--device", "cpu", "--output", str(out)])
    assert status == 0
    report = json.loads(out.read_text())
    assert set(report) >= {"sites", "summary", "alarms"}
    assert set(report["sites"]) == {"1_pre_qkv", "1_pre_o", "1_pre_mlp"}
    assert report["sites"]["1_pre_o"]["n_witnesses"] == 6     # 2 adapters x (1 real + 2 random)
    assert "pre_qkv" in capsys.readouterr().out


def test_the_cli_reads_a_workspaces_current_artifact(monkeypatch, base_dir, tiny_artifact,
                                                     adapter_dir, tokens, tmp_path):
    monkeypatch.setattr(probe, "_wikitext2_tokens", lambda tokenizer, n_windows, stride: tokens)
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir),
                 "--artifact", str(tiny_artifact[1])]) == 0
    out = tmp_path / "probe.json"
    assert main(["probe-artifact", "--model", str(base_dir), "--artifact", str(workspace),
                 "--adapter", str(adapter_dir[0]), "--layers", "1", "--n-real", "1024",
                 "--n-model", "1024", "--device", "cpu", "--output", str(out)]) == 0
    assert json.loads(out.read_text())["settings"]["artifact"].endswith("v1.pt")
