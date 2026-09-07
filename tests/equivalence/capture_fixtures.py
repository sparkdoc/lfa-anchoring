"""Capture the reference fixtures the equivalence tests are checked against.

This is the only file in the package that runs the *research* code (``the research code``). It is run
once, by hand, inside the research code's own virtualenv, and writes three ``.pt`` files into
``tests/equivalence/fixtures/``. ``test_equivalence.py`` then replays each of them against the
companion's own implementation. Nothing under the research code is written or modified: the checkout is
read, and every temporary file this script makes goes to ``--work-dir`` (``$TMPDIR`` by default).

    cd /path/to/the research code && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python lfa-anchoring/tests/equivalence/capture_fixtures.py

What is captured, and why each is the right reference:

``sampler_draws.pt``
    Every draw of one full anchoring step's call sequence -- 28 layers x (pre_qkv, pre_o,
    pre_mlp) then the LM-head site -- from the shipped int8 artifact under a fixed global seed.
    The sampler is the input side of the anchor, so this is the fixture that has to match
    *exactly*: a sample stream that has drifted is a different p(h), and every downstream number
    would move with it while nothing raised.

``losses.pt``
    The anchor's four components and mu, for the archived recipe adapter over its own base, under
    a fixed seed. Also the exact training batch the loader produces, which pins the corpus path
    as well as the losses.

``extend_ref.pt``
    A continual extension of the shipped artifact with a 20-document slice of the domain, run
    through the research code's own script, reduced to the two sites the test compares -- plus a slice of
    *held-out* activations (collected past the ones the fit saw) for scoring the fitted domain
    mixture. The two GMM fitters are not bit-identical, so this one is compared by likelihood.

Note on dtypes: draws are stored as float32. The layer-0 draw comes off a bfloat16 lookup table,
and bfloat16 -> float32 is exact and injective, so equality in float32 is equality in bfloat16 --
the zero-tolerance claim survives the conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------------------------
# Where things live in the the research code checkout. Everything here is read-only.
# ---------------------------------------------------------------------------------------------

#: The companion sits inside the research checkout, so its grandparent is the research code's root.
DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_REL = "data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt"
BASE_MODEL_REL = "outputs/lra/qwen3-0.6b/original"
ADAPTER_REL = ("outputs/lra/qwen3-0.6b/chalmers/judge_search/"
               "gmm_r32_lam100000_f0.13/checkpoint_epoch_15")
CORPUS_REL = "data/domain/chalmers"
FUSED_MODEL_REL = "outputs/lra/qwen3-0.6b/continual_ab/lra_gmm_r4_s42/base_plus_A"
EXTEND_SCRIPT_REL = "scripts/lra_extend_distribution_gmm.py"

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# -- the constants both sides of every comparison have to agree on -----------------------------

SAMPLER_SEED = 1234
LOSS_SEED = 99
N_ANCHOR_SAMPLES = 16
BATCH_CHUNKS = 6
SEQUENCE_LENGTH = 512
CORPUS_SEED = 42
#: The archived runs and every the research code stream use the research loader frame, under which a
#: document shorter than the epoch's chunk offset drops out of that epoch.
KEEP_SHORT_WHOLE = False

#: The exact per-site count the research chains pass as `--base-n` for this artifact.
BASE_N = 1_543_040
EXTEND_NEED = 4_000
EXTEND_K = 4
EXTEND_SEED = 42
EXTEND_DOCS = 20
#: Sites the extension fixture compares. One early, one mid-stack.
EXTEND_SITES = ("1_pre_mlp", "12_pre_mlp")
#: Activations collected past the fit's `EXTEND_NEED`, kept for scoring the fitted mixture on
#: data neither fit saw.
HELDOUT_ROWS = 500


# ---------------------------------------------------------------------------------------------
# Shared helpers -- imported by test_equivalence.py, so they must not import the research code at module
# level (the tests run in the companion's venv, where `src.*` does not exist).
# ---------------------------------------------------------------------------------------------

def txt_documents(corpus_dir: Path) -> list[str]:
    """The names of ``corpus_dir``'s ``.txt`` files, sorted -- the documents both loaders read.

    The research corpus directory also holds ``.jsonl`` evaluation dumps and a README. The two
    loaders disagree about those (the research code's domain path drops a ``prompt``/``response`` record,
    the companion's renders it), so the equivalence fixtures are built from the ``.txt``
    documents alone and the disagreement is kept out of the comparison.
    """
    return sorted(p.name for p in Path(corpus_dir).glob("*.txt"))


def link_txt_corpus(corpus_dir, dest_dir, names: list[str] | None = None) -> Path:
    """A directory of symlinks to ``corpus_dir``'s ``.txt`` documents; returns ``dest_dir``.

    Symlinks rather than copies: the documents are megabytes of prose that both processes can
    read in place, and the link names are the originals', so the sorted order a loader walks is
    the corpus's own.
    """
    corpus_dir, dest_dir = Path(corpus_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in (names if names is not None else txt_documents(corpus_dir)):
        source = corpus_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"corpus document missing: {source}")
        link = dest_dir / name
        if not link.exists():
            link.symlink_to(source)
    return dest_dir


def gaussian_mixture_log_likelihood(
    z: torch.Tensor, weights: torch.Tensor, means: torch.Tensor, variances: torch.Tensor
) -> float:
    """Mean log-likelihood of ``z`` under a diagonal mixture, in the coordinates ``z`` is given in.

    The scoring rule for the extension fixture. The two GMM fitters draw different k-means++
    initializations, so the fitted parameters are not comparable element by element; what has to
    agree is the density the mixture puts on held-out activations.
    """
    z = z.double()
    means, variances = means.double(), variances.double()
    log_w = weights.double().clamp_min(1e-300).log()                       # [K]
    # [N, K, D] would be the obvious form and is far too large here: accumulate per component.
    per_component = []
    for k in range(means.shape[0]):
        delta = z - means[k]
        var = variances[k].clamp_min(1e-30)
        per_component.append(
            log_w[k] - 0.5 * (torch.log(2 * torch.pi * var).sum() + (delta ** 2 / var).sum(dim=1))
        )
    return float(torch.logsumexp(torch.stack(per_component, dim=1), dim=1).mean())


# ---------------------------------------------------------------------------------------------
# Capture 1: the sampler's draws
# ---------------------------------------------------------------------------------------------

def call_sequence(num_layers: int) -> list[tuple[str, int, int]]:
    """``(site, layer, n)`` in the order one anchoring step draws them.

    QKV first (each layer's ``pre_qkv`` then its ``pre_o``), then every layer's ``pre_mlp``, then
    the LM-head site at layer index ``num_layers``. The order is the fixture: the draws share one
    RNG stream, so a call sequence in a different order is a different set of samples even when
    every individual call is right.
    """
    calls = []
    for layer in range(num_layers):
        calls.append(("pre_qkv", layer, N_ANCHOR_SAMPLES))
        calls.append(("pre_o", layer, N_ANCHOR_SAMPLES))
    for layer in range(num_layers):
        calls.append(("pre_mlp", layer, N_ANCHOR_SAMPLES))
    calls.append(("pre_lm_head", num_layers, N_ANCHOR_SAMPLES))
    return calls


def capture_sampler_draws(paths: dict, device: str, out: Path) -> Path:
    from transformers import AutoModelForCausalLM

    from src.lra_distribution import LayerDistributionSampler

    model = AutoModelForCausalLM.from_pretrained(
        str(paths["base_model"]), dtype=torch.bfloat16).to(device).eval()

    # Seeded AFTER every model load: `from_pretrained` touches the RNG, and the claim being
    # captured is about what the draws consume, not about what loading a checkpoint costs.
    torch.manual_seed(SAMPLER_SEED)
    sampler = LayerDistributionSampler(str(paths["artifact"]), device=device)
    sampler.build_embedding_lookup_from_model(model)

    num_layers = model.config.num_hidden_layers
    calls = call_sequence(num_layers)
    draws = []
    for site, layer, n in calls:
        h = sampler.sample_best(layer, site, n)
        draws.append(None if h is None else h.detach().cpu().float())

    payload = {
        "seed": SAMPLER_SEED,
        "device": device,
        "model_dtype": "bfloat16",
        "num_layers": num_layers,
        "calls": calls,
        "draws": draws,
        "artifact": str(paths["artifact"]),
        "base_model": str(paths["base_model"]),
    }
    torch.save(payload, out)
    shapes = {tuple(d.shape) for d in draws if d is not None}
    print(f"[sampler] {len(draws)} draws, shapes {sorted(shapes)}, "
          f"{sum(d is None for d in draws)} empty -> {out}")
    return out


# ---------------------------------------------------------------------------------------------
# Capture 2: the anchor's components and mu
# ---------------------------------------------------------------------------------------------

def capture_losses(paths: dict, device: str, work: Path, out: Path) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import src.lra_losses as lra_losses
    from src.data import collate_fn_left_pad, load_domain_data
    from src.lra_distribution import LayerDistributionSampler

    names = txt_documents(paths["corpus"])
    corpus = link_txt_corpus(paths["corpus"], work / "chalmers_txt", names)

    tokenizer = AutoTokenizer.from_pretrained(str(paths["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, _ = load_domain_data(path=corpus, tokenizer=tokenizer, max_length=SEQUENCE_LENGTH,
                                  stride=0, val_fraction=0.0, seed=CORPUS_SEED,
                                  keep_short_whole=KEEP_SHORT_WHOLE)
    batch = collate_fn_left_pad([dataset[i] for i in range(BATCH_CHUNKS)],
                                tokenizer.pad_token_id)
    batch = {key: batch[key] for key in ("input_ids", "attention_mask", "labels")}

    teacher = AutoModelForCausalLM.from_pretrained(
        str(paths["base_model"]), dtype=torch.bfloat16).to(device).eval()
    student = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(str(paths["base_model"]), dtype=torch.bfloat16),
        str(paths["adapter"]), is_trainable=False).to(device).eval()

    sampler = LayerDistributionSampler(str(paths["artifact"]), device=device)
    sampler.build_embedding_lookup_from_model(teacher)

    torch.manual_seed(LOSS_SEED)
    anchor = lra_losses.compute_anchor_loss(teacher, student, sampler,
                                            n_samples_per_layer=N_ANCHOR_SAMPLES)
    anchor = {name: float(value) for name, value in anchor.items()}

    # mu twice. The shipped call takes the LoRA-factored fast path, which is both cheaper and
    # (working from the fp32 factors rather than from a bf16 difference of near-equal weights)
    # the more accurate of the two. The general form is captured as well, by making the fast
    # path decline in this process only -- the research code itself is never edited.
    mu_fast = float(lra_losses.compute_direct_weight_loss(teacher, student))
    original = lra_losses._lora_factored_weight_loss
    try:
        lra_losses._lora_factored_weight_loss = lambda *a, **k: None
        mu_general = float(lra_losses.compute_direct_weight_loss(teacher, student))
    finally:
        lra_losses._lora_factored_weight_loss = original

    payload = {
        "seed": LOSS_SEED,
        "device": device,
        "model_dtype": "bfloat16",
        "n_samples_per_layer": N_ANCHOR_SAMPLES,
        "anchor": anchor,
        "mu_fast": mu_fast,
        "mu_general": mu_general,
        "batch": batch,
        "n_chunks": len(dataset),
        "documents": names,
        "corpus": str(paths["corpus"]),
        "adapter": str(paths["adapter"]),
        "base_model": str(paths["base_model"]),
        "artifact": str(paths["artifact"]),
        "sequence_length": SEQUENCE_LENGTH,
        "corpus_seed": CORPUS_SEED,
        "keep_short_whole": KEEP_SHORT_WHOLE,
    }
    torch.save(payload, out)
    print(f"[losses] anchor {anchor}")
    print(f"[losses] mu fast {mu_fast!r} general {mu_general!r}")
    print(f"[losses] batch {tuple(batch['input_ids'].shape)} of {len(dataset)} chunks "
          f"over {len(names)} documents -> {out}")
    return out


# ---------------------------------------------------------------------------------------------
# Capture 3: the continual extension
# ---------------------------------------------------------------------------------------------

def _collect_heldout(paths: dict, corpus: Path, device: str, need: int) -> dict[str, torch.Tensor]:
    """Activations at :data:`EXTEND_SITES`, collected exactly as the extension collects them.

    Run with a larger ``need`` than the fit used, so the tail of the stream is data no mixture on
    either side was fitted on. Mirrors the research script's collector: the same loader, the same
    seeded chunk order, a hook on ``gate_proj`` (whose input *is* the ``pre_mlp`` site).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import load_domain_data

    tokenizer = AutoTokenizer.from_pretrained(str(paths["fused_model"]))
    model = AutoModelForCausalLM.from_pretrained(
        str(paths["fused_model"]), dtype=torch.float32).to(device).eval()

    store: dict[str, list[torch.Tensor]] = {key: [] for key in EXTEND_SITES}

    def hook_for(key: str):
        def hook(module, inputs, output):
            if sum(len(chunk) for chunk in store[key]) < need:
                store[key].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu())
        return hook

    handles = []
    for key in EXTEND_SITES:
        layer = int(key.split("_", 1)[0])
        handles.append(model.model.layers[layer].mlp.gate_proj.register_forward_hook(hook_for(key)))

    dataset, _ = load_domain_data(path=corpus, tokenizer=tokenizer, max_length=SEQUENCE_LENGTH,
                                 stride=0, val_fraction=0.0, seed=EXTEND_SEED,
                                 keep_short_whole=KEEP_SHORT_WHOLE)
    order = torch.randperm(len(dataset),
                           generator=torch.Generator().manual_seed(EXTEND_SEED)).tolist()
    with torch.no_grad():
        for index in order:
            ids = torch.as_tensor(dataset[index]["input_ids"], dtype=torch.long).reshape(1, -1)
            model(ids.to(device))
            if all(sum(len(c) for c in v) >= need for v in store.values()):
                break
    for handle in handles:
        handle.remove()
    del model
    torch.cuda.empty_cache()
    # `.clone()`, not a slice: `torch.save` serializes a view's whole underlying storage, so
    # saving a slice of the collected stream would write every row that was ever collected.
    return {key: torch.cat(chunks)[:need].clone() for key, chunks in store.items()}


def capture_extend(paths: dict, device: str, work: Path, out: Path) -> Path:
    documents = txt_documents(paths["corpus"])[:EXTEND_DOCS]
    corpus = link_txt_corpus(paths["corpus"], work / "chalmers_20", documents)

    # The research script loads its base artifact with a plain `torch.load` and never
    # dequantizes, so it cannot read the shipped int8 file. Hand it the dequantized artifact --
    # the same tensors the companion's loader reconstructs -- so both sides fit in one basis.
    from src.lra_quantize import dequantize_params

    base_dequantized = work / "base_dequantized.pt"
    if not base_dequantized.is_file():
        params = torch.load(paths["artifact"], map_location="cpu", weights_only=False)
        torch.save(dequantize_params(params), base_dequantized)
        del params
    merged_path = work / "extend_merged.pt"

    command = [
        sys.executable, str(paths["research"] / EXTEND_SCRIPT_REL),
        "--base-distribution", str(base_dequantized),
        "--base-n", str(BASE_N),
        "--fused-model", str(paths["fused_model"]),
        "--domain", str(corpus),
        "--output", str(merged_path),
        "--sample-mode", "train_chunks",
        "--need", str(EXTEND_NEED),
        "--k-a", str(EXTEND_K),
        "--seq-len", str(SEQUENCE_LENGTH),
        "--seed", str(EXTEND_SEED),
        "--fitter", "torch",
        "--device", device,
    ]
    print("[extend] $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(paths["research"]))

    merged = torch.load(merged_path, map_location="cpu", weights_only=False)
    sites = {}
    for key in EXTEND_SITES:
        entry = merged[key]
        k = EXTEND_K
        sites[key] = {
            "gmm_n_components": int(entry["gmm_n_components"]),
            "n_samples": int(entry["n_samples"]),
            "gmm_weights_sum": float(entry["gmm_weights"].double().sum()),
            # The merge concatenates base components then domain ones, so the domain's mixture is
            # the tail -- renormalized here, since its weights carry the domain's share.
            "domain_weights": (entry["gmm_weights"][-k:].float()
                               / entry["gmm_weights"][-k:].float().sum()).clone(),
            "domain_means": entry["gmm_means"][-k:].float().clone(),
            "domain_covariances": entry["gmm_covariances"][-k:].float().clone(),
            "merged_mean": entry["mean"].float(),
            "merged_std": entry["std"].float(),
        }
    del merged

    heldout_all = _collect_heldout(paths, corpus, device, EXTEND_NEED + HELDOUT_ROWS)
    heldout = {key: value[EXTEND_NEED:].clone() for key, value in heldout_all.items()}

    payload = {
        "base_n": BASE_N,
        "need": EXTEND_NEED,
        "k_domain": EXTEND_K,
        "seed": EXTEND_SEED,
        "seq_len": SEQUENCE_LENGTH,
        "keep_short_whole": KEEP_SHORT_WHOLE,
        "device": device,
        "documents": documents,
        "corpus": str(paths["corpus"]),
        "fused_model": str(paths["fused_model"]),
        "artifact": str(paths["artifact"]),
        "sites": sites,
        "heldout": heldout,
        "heldout_offset": EXTEND_NEED,
    }
    torch.save(payload, out)
    for key, site in sites.items():
        print(f"[extend] {key}: K={site['gmm_n_components']} n={site['n_samples']} "
              f"weights sum {site['gmm_weights_sum']:.12f} "
              f"held out {tuple(heldout[key].shape)}")
    print(f"[extend] -> {out}")
    return out


# ---------------------------------------------------------------------------------------------

def resolve_paths(research: Path) -> dict:
    paths = {
        "research": research,
        "artifact": research / ARTIFACT_REL,
        "base_model": research / BASE_MODEL_REL,
        "adapter": research / ADAPTER_REL,
        "corpus": research / CORPUS_REL,
        "fused_model": research / FUSED_MODEL_REL,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit("missing the research code paths:\n  " + "\n  ".join(missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--the research code", type=Path, default=DEFAULT_RESEARCH_ROOT,
                        help="the research checkout to read (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=FIXTURES,
                        help="where the fixtures are written (default: %(default)s)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="scratch for the corpus links and the full merged artifact "
                             "(default: a temporary directory under $TMPDIR)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", nargs="+", choices=("sampler", "losses", "extend"),
                        default=["sampler", "losses", "extend"])
    args = parser.parse_args()

    research = args.research.resolve()
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    paths = resolve_paths(research)
    args.out.mkdir(parents=True, exist_ok=True)

    work = args.work_dir
    context = tempfile.TemporaryDirectory(prefix="lfa_capture_") if work is None else None
    if context is not None:
        work = Path(context.name)
    work.mkdir(parents=True, exist_ok=True)

    print(f"the research code: {research}\nfixtures:  {args.out}\nwork:      {work}\n"
          f"device:    {args.device}  (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)

    try:
        if "sampler" in args.only:
            capture_sampler_draws(paths, args.device, args.out / "sampler_draws.pt")
        if "losses" in args.only:
            capture_losses(paths, args.device, work, args.out / "losses.pt")
        if "extend" in args.only:
            capture_extend(paths, args.device, work, args.out / "extend_ref.pt")
    finally:
        if context is not None:
            context.cleanup()

    written = {p.name: p.stat().st_size for p in sorted(args.out.glob("*.pt"))}
    print("fixtures: " + json.dumps({k: f"{v / 1e6:.2f} MB" for k, v in written.items()}))


if __name__ == "__main__":
    main()
