"""The Workspace: one directory that carries a model, its p(h) artifact, and its history.

Layerwise Function Anchoring (LFA) is stateful across domains in a way a single training call is
not. Stage two does not anchor the base model -- it anchors *base + A*, against a p(h) that has
been extended with A, at a harder lambda than stage one used. Get any of those three wrong and the
run still completes, still reports a plausible loss, and quietly measures something else. A
Workspace exists so that none of them is the caller's job to remember:

* it owns **which model** the next stage adapts (the base at stage one, the fused base+A after an
  extension) and **which artifact version** that stage anchors against;
* it enforces the **order**: a second domain cannot start until the first has been folded into the
  model and into p(h) (:class:`StageOrderError`), because training B against A's artifact over
  A's adapter is precisely the silent-mismatch case above;
* it writes a **history**, so a chain that took a day of GPU time can still say what each stage
  trained on, at what lambda, against which artifact version, and what it cost on both axes.

On disk::

    <workspace>/
      workspace.json                  the state below
      history.json                    one entry per training run, in order
      artifacts/v1.pt, v2.pt, ...     p(h): v1 as fetched, vN+1 = vN extended with domain N
      models/stage1_fused/            base + stage 1, the model stage 2 adapts
      runs/stage1/                    a training run (config.json, final_model/, history)

Nothing here is a new mechanism: every step is a call into :mod:`lfa.train`,
:mod:`lfa.artifact.extend`, :mod:`lfa.models` or :mod:`lfa.evaluate`. What the Workspace adds is
that the *sequence* of those calls is recorded and checked rather than reconstructed by hand.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from .adapters import get_adapter
from .artifact.extend import extend_artifact
from .artifact.fetch import ARTIFACTS, fetch_artifact
from .artifact.schema import META_KEY, parse_site_key, validate_against_model
from .corpus import load_corpus
from .evaluate import (
    DOMAIN_ROW,
    GENERAL_ROW,
    domain_perplexity,
    perplexity_table,
    wikitext2_perplexity,
)
from .models import (
    DEFAULT_DEVICE,
    apply_lora,
    fuse,
    load_student,
    load_teacher,
    load_tokenizer,
    resolve_device,
)
from .recipe import BUNDLED_DIR, Recipe
from .sampler import Sampler
from .train import train as run_training

logger = logging.getLogger("lfa.workspace")

__all__ = ["Workspace", "StageOrderError", "WORKSPACE_FILE", "HISTORY_FILE",
           "LOADER_FRAME_NOTICE"]

WORKSPACE_FILE = "workspace.json"
HISTORY_FILE = "history.json"

#: Logged once per run whose corpus loader keeps short documents in every epoch. It is a *frame*
#: field, not a tuning knob: the paper's perplexity points were measured under the other setting,
#: where a document shorter than the epoch's random chunk offset drops out of that epoch, so a
#: number produced under one loader is not comparable with a number produced under the other.
LOADER_FRAME_NOTICE = (
    "loader frame: short documents are kept every epoch (differs from the paper's measured runs; "
    "perplexity points are not comparable across this frame)"
)


class StageOrderError(RuntimeError):
    """Raised when a chain's steps are taken out of order (train/extend/train)."""


def _lfa_version() -> str:
    """The package version, imported lazily so this module never depends on import order."""
    from . import __version__

    return __version__


def _now() -> str:
    """An ISO-8601 timestamp, to the second, in local time."""
    return _datetime.datetime.now().replace(microsecond=0).isoformat()


def _primary_device(device: str | dict) -> str:
    """The single device to put samplers and batches on, even for a sharded device map."""
    if isinstance(device, dict):
        return str(next(iter(device.values())))
    return str(device)


def _dtype_for(device: str | dict) -> torch.dtype:
    """bf16 on an accelerator, fp32 on CPU.

    bf16 exists to fit a model on a card; on CPU it buys nothing and costs eight bits of mantissa
    on every anchored function output, which is the quantity the whole method reads.
    """
    return torch.float32 if _primary_device(device).startswith("cpu") else torch.bfloat16


def _stage_dtype(entry: dict) -> torch.dtype:
    """The dtype a recorded stage trained in, so an export is not a silent precision change."""
    return getattr(torch, entry.get("dtype") or "bfloat16")


def _delta(before: float, after: float) -> str:
    return "n/a" if before == 0 else f"{(after - before) / before * 100:+.1f}%"


def _table(before: dict, after: dict, unanchored: dict | None) -> str:
    """The run's two axes as Markdown, degrading to one row when the general axis was skipped.

    :func:`lfa.evaluate.perplexity_table` reports both axes and has no cell for an unmeasured
    one; an ``inf`` or a ``nan`` in its place would read as a measurement rather than as its
    absence, so the general row is replaced by a line that says what happened instead.
    """
    if before.get("general") is not None:
        return perplexity_table(before, after, unanchored)

    header = ["metric", "before", "after", "Δ%"]
    row = [DOMAIN_ROW, f"{before['domain']:.2f}", f"{after['domain']:.2f}",
           _delta(before["domain"], after["domain"])]
    if unanchored is not None:
        header += ["unanchored", "Δ%"]
        row += [f"{unanchored['domain']:.2f}", _delta(before["domain"], unanchored["domain"])]

    widths = [max(len(a), len(b)) for a, b in zip(header, row)]
    rule = ["-" * widths[0]] + ["-" * (w - 1) + ":" for w in widths[1:]]

    def line(cells: list[str]) -> str:
        padded = [cells[0].ljust(widths[0])]
        padded += [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:])]
        return "| " + " | ".join(padded) + " |"

    return "\n".join([line(header), line(rule), line(row),
                      "", f"{GENERAL_ROW}: not measured"])


def _bundled_recipe_for(model_id: str) -> str | None:
    """The bundled recipe tuned for ``model_id``, when exactly one names it.

    Matched on the recipe's own ``model_id`` rather than on its name: a recipe is a joint
    operating point for one model, and guessing one from a filename is how a lambda gets ported
    across models it was never calibrated for.
    """
    matches = []
    for path in sorted(BUNDLED_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:                       # a malformed bundled file is not this
            continue                                 # function's problem to report
        if isinstance(data, dict) and data.get("model_id") == model_id:
            matches.append(path.stem)
    return matches[0] if len(matches) == 1 else None


class Workspace:
    """A directory of LFA state: the current model, the current p(h), and what got it there.

    Attributes:
        path: the workspace directory.
        state: the mutable state, persisted to ``workspace.json``:

            ``model_id``
                What the workspace was created over -- a Hub id or a checkpoint path.
            ``base_model``
                The checkpoint stage 1 adapts. Equal to ``model_id`` and never changes.
            ``current_model``
                What the *next* stage adapts: the base, or the fused model an extension left.
            ``current_artifact`` / ``artifact_version``
                The p(h) file the next stage anchors against, and its version (v1 = as fetched).
            ``artifact_id``
                The registry id v1 came from, or ``None`` for a local file. It is what
                :meth:`lfa.recipe.Recipe.warnings` is read against, and it supplies the base
                sample count an extension needs when the artifact itself carries none.
            ``stage``
                How many domains have been trained.
            ``recipe``
                The default recipe (a bundled name or a path) when a call does not pass one.
            ``last_stage_adapter`` / ``last_stage_base`` / ``last_stage_corpus`` /
            ``last_stage_dir``
                What the last stage produced, what it started from, what it trained on and where
                it was written -- read by :meth:`extend`, :meth:`evaluate` and :meth:`fuse`.
            ``pending_extend``
                Whether a trained stage is still waiting to be folded into the model and p(h).

        history: the list persisted to ``history.json``; one entry per training run.
    """

    def __init__(self, path: str | Path, state: dict[str, Any], history: list[dict]):
        """Internal. Use :meth:`init` to create a workspace and :meth:`open` to read one."""
        self.path = Path(path)
        self.state = state
        self.history = history

    # ------------------------------------------------------------------------------ lifecycle

    @classmethod
    def init(
        cls,
        path: str | Path,
        model_id: str,
        artifact: str = "qwen3-0.6b-gmm1543k-int8",
        recipe: str | None = None,
        fetch: bool = True,
    ) -> "Workspace":
        """Create a workspace over ``model_id`` and put its first p(h) artifact in place.

        Args:
            path: the workspace directory (created if needed). It must not already hold one.
            model_id: a Hub id or a local checkpoint path. It is not loaded here -- init stays
                cheap, and the artifact is checked against the model when a stage starts.
            artifact: a registry id (see :data:`lfa.artifact.fetch.ARTIFACTS`) or a path to an
                artifact file. Either way it is copied to ``artifacts/v1.pt``, so the workspace
                carries its own p(h) and later versions sit beside it.
            recipe: the default recipe for this workspace -- a bundled name or a path. When
                omitted, a bundled recipe whose own ``model_id`` is this model is adopted; if
                none is, every training call has to name one.
            fetch: download a registry artifact that is not already there. ``False`` leaves the
                workspace without a p(h) (and says so), for a machine with no network.

        Raises:
            FileExistsError: ``path`` already holds a workspace.
        """
        path = Path(path)
        if (path / WORKSPACE_FILE).exists():
            raise FileExistsError(
                f"{path} is already an LFA workspace; open it with Workspace.open(path) rather "
                "than re-initialising it (init would discard its history)."
            )
        path.mkdir(parents=True, exist_ok=True)
        artifacts_dir = path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        source = Path(artifact)
        artifact_id = None if source.exists() else artifact
        destination = artifacts_dir / "v1.pt"

        if artifact_id is None:
            shutil.copyfile(source, destination)
            logger.info("Artifact %s copied to %s", source, destination)
        elif destination.exists():
            logger.info("Artifact %s already present at %s", artifact_id, destination)
        elif fetch:
            fetched = fetch_artifact(artifact_id, artifacts_dir)
            fetched.replace(destination)
        else:
            destination = None
            logger.warning(
                "No p(h) artifact in this workspace: %r was not fetched (fetch=False). Run "
                "`lfa fetch-artifact %s` before training.", artifact_id, artifact_id,
            )

        if recipe is None:
            recipe = _bundled_recipe_for(model_id)
            if recipe is not None:
                logger.info("Default recipe for this workspace: %r (it names %s)", recipe,
                            model_id)

        state = {
            "model_id": model_id,
            "base_model": model_id,
            "current_model": model_id,
            "current_artifact": str(destination) if destination else None,
            "artifact_version": 1,
            "artifact_id": artifact_id,
            "stage": 0,
            "recipe": recipe,
            "last_stage_adapter": None,
            "last_stage_base": None,
            "last_stage_corpus": None,
            "last_stage_dir": None,
            "pending_extend": False,
        }
        workspace = cls(path, state, [])
        workspace._save_state()
        workspace._save_history()
        logger.info("Workspace initialised at %s over %s", path, model_id)
        return workspace

    @classmethod
    def open(cls, path: str | Path) -> "Workspace":
        """Read the workspace at ``path``.

        Raises:
            FileNotFoundError: there is no workspace there.
        """
        path = Path(path)
        state_path = path / WORKSPACE_FILE
        if not state_path.is_file():
            raise FileNotFoundError(
                f"No LFA workspace at {path} (no {WORKSPACE_FILE}). Create one with `lfa init "
                f"{path} --model <model-id>`."
            )
        history_path = path / HISTORY_FILE
        history = json.loads(history_path.read_text()) if history_path.is_file() else []
        return cls(path, json.loads(state_path.read_text()), history)

    def _save_state(self) -> None:
        (self.path / WORKSPACE_FILE).write_text(json.dumps(self.state, indent=2))

    def _save_history(self) -> None:
        (self.path / HISTORY_FILE).write_text(json.dumps(self.history, indent=2))

    # ---------------------------------------------------------------------------------- train

    def train(
        self,
        corpus: str | Path,
        recipe: Recipe | str | Path | None = None,
        *,
        epochs: int | None = None,
        output_name: str | None = None,
        device: str | dict = DEFAULT_DEVICE,
        allow_sharding: bool = False,
        resume: bool = False,
    ) -> dict:
        """Train one stage: ``current_model`` on ``corpus``, anchored on ``current_artifact``.

        The stage number decides the lambda: from stage 2 on, the recipe's
        ``stage2_lambda_multiplier`` applies, because a later stage anchors a model that already
        carries a domain. Training the *same* corpus again -- more dose on the domain in progress
        -- is the same stage and keeps the same lambda; a *different* corpus is the next stage and
        must be preceded by :meth:`extend`.

        Args:
            corpus: a file or directory of documents (see :func:`lfa.corpus.load_texts`).
            recipe: a :class:`~lfa.recipe.Recipe`, a bundled name, or a path. Defaults to the
                workspace's own.
            epochs: override the recipe's dose. Validated against the schedule horizon, so a dose
                past the end of the learning-rate schedule is refused rather than run.
            output_name: run directory name under ``runs/`` (default ``stage{N}``).
            device: a single device, as :func:`lfa.models.resolve_device` reads it.
            allow_sharding: permit a device map that spreads the model over several devices.
            resume: continue the run already in this stage's output directory.

        Returns:
            The history entry this run appended.

        Raises:
            StageOrderError: a new corpus while a trained stage has not been extended.
            ValueError: no recipe anywhere, or an ``epochs`` the schedule cannot carry.
            RuntimeError: the workspace has no artifact to anchor against.
        """
        corpus_path = Path(corpus).expanduser().resolve()
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        repeat = bool(self.state["pending_extend"]) and (
            self.state["last_stage_corpus"] == str(corpus_path))
        if self.state["pending_extend"] and not repeat:
            raise StageOrderError(
                f"Stage {self.state['stage']} ({self.state['last_stage_corpus']}) has been "
                "trained but not folded in: run `lfa extend` before training a different corpus. "
                "Without it the new stage would adapt the previous stage's base model and anchor "
                "against a p(h) that does not describe what it is anchoring."
            )

        artifact = self.state["current_artifact"]
        if artifact is None:
            raise RuntimeError(
                "This workspace has no p(h) artifact. Fetch one with `lfa fetch-artifact "
                f"{self.state['artifact_id'] or '<id>'} --dest {self.path / 'artifacts'}` "
                "or re-initialise with a local artifact path."
            )

        resolved = self._resolve_recipe(recipe)
        if epochs is not None:
            resolved = dataclasses.replace(resolved, epochs=epochs)

        stage = self.state["stage"] if repeat else self.state["stage"] + 1
        config = resolved.to_train_config(stage, artifact)

        # Said before anything is loaded: an off-calibration lambda is not a refusal, but it is
        # also not the measured operating point, and a run is worth more than the warning is.
        for note in resolved.warnings(config.lora_rank, self._artifact_id()):
            logger.warning(note)
        if config.keep_short_whole:
            logger.info(LOADER_FRAME_NOTICE)

        base_model = self.state["current_model"]
        output_dir = self.path / "runs" / (output_name or f"stage{stage}")
        placement = resolve_device(device, allow_sharding)
        dtype = _dtype_for(placement)
        logger.info("Stage %d: %s on %s (artifact v%d, λ_qkv=%s, %d epochs)", stage, base_model,
                    corpus_path, self.state["artifact_version"], config.lambda_qkv,
                    config.num_epochs)

        training = self._run_training(config, corpus_path, base_model, output_dir,
                                      placement=placement, dtype=dtype,
                                      allow_sharding=allow_sharding, resume=resume,
                                      anchored=True)

        entry = {
            "stage": stage,
            "corpus": str(corpus_path),
            "recipe": dataclasses.asdict(resolved),
            # The lambda actually passed to the trainer -- the recipe's value times the stage
            # multiplier. Both site families are recorded, since a recipe may differ across them.
            "lambda_applied": config.lambda_qkv,
            "lambda_mlp_applied": config.lambda_mlp,
            "artifact_version": self.state["artifact_version"],
            "artifact": artifact,
            "base_model": str(base_model),
            "adapter": str(output_dir / "final_model"),
            "output_dir": str(output_dir),
            "epochs": config.num_epochs,
            "final_loss": training.history[-1]["loss_total"] if training.history else None,
            # Where and in what precision this stage ran: an export merges in the dtype it was
            # trained in rather than in a default that may not be the same one.
            "device": placement,
            "dtype": str(dtype).removeprefix("torch."),
            "timestamp": _now(),
            "lfa_version": _lfa_version(),
        }
        self.history.append(entry)
        self._save_history()

        self.state.update(
            stage=stage,
            last_stage_adapter=str(output_dir / "final_model"),
            last_stage_base=str(base_model),
            last_stage_corpus=str(corpus_path),
            last_stage_dir=str(output_dir),
            pending_extend=True,
        )
        self._save_state()
        return entry

    def _run_training(self, config, corpus_path, base_model, output_dir, *, placement, dtype,
                      allow_sharding, resume, anchored):
        """Load teacher, student, sampler and corpus for one run, and train it.

        The embedding lookup is rebuilt from the teacher here rather than shipped: layer-0
        ``pre_qkv`` is ``input_layernorm(embed_tokens(id))``, exactly reconstructible and ~300 MB
        to store. Without this call the artifact has no table, and ``L_embed`` -- the only term
        anchoring the embedding end of the tied embedding/LM-head matrix -- is dropped silently.
        """
        # Bound up front so the cleanup below is safe however far the setup gets: two models is
        # the largest thing this package allocates, and a failed setup must not keep them.
        teacher = student = sampler = None
        try:
            teacher = load_teacher(str(base_model), device=placement, dtype=dtype,
                                   allow_sharding=allow_sharding)
            adapter = get_adapter(teacher)
            student = load_student(str(base_model), device=placement, dtype=dtype,
                                   allow_sharding=allow_sharding)
            if config.use_lora:
                student = apply_lora(student, adapter, rank=config.lora_rank,
                                     alpha=config.lora_alpha, dropout=config.lora_dropout,
                                     freeze_embed=config.freeze_embed)

            if anchored:
                sampler = Sampler(config.artifact_path, device=_primary_device(placement),
                                  seed=config.seed)
                # An artifact that does not describe this model anchors toward a different
                # function, and every shape downstream still matches, so nothing would notice.
                validate_against_model(sampler.params, teacher, adapter)
                sampler.build_embedding_lookup_from_model(teacher, adapter)

            tokenizer = load_tokenizer(str(base_model))
            dataset, _ = load_corpus(corpus_path, tokenizer, max_length=config.sequence_length,
                                     val_fraction=0.0, seed=config.seed,
                                     keep_short_whole=config.keep_short_whole)

            return run_training(teacher, student, dataset, sampler, adapter, config, output_dir,
                                resume=resume, tokenizer=tokenizer)
        finally:
            del teacher, student, sampler
            gc.collect()
            if _primary_device(placement).startswith("cuda"):
                torch.cuda.empty_cache()

    def _resolve_recipe(self, recipe: Recipe | str | Path | None) -> Recipe:
        if isinstance(recipe, Recipe):
            return recipe
        if recipe is not None:
            return Recipe.load(recipe)
        if self.state.get("recipe"):
            return Recipe.load(self.state["recipe"])
        available = ", ".join(sorted(p.stem for p in BUNDLED_DIR.glob("*.yaml"))) or "(none)"
        raise ValueError(
            "No recipe: this workspace has no default one, so the run has to name a recipe (a "
            f"bundled name, a path to a YAML file, or a Recipe object). Bundled: {available}."
        )

    def _artifact_id(self) -> str:
        """What the recipe's calibration is read against.

        The registry id of the artifact this workspace started from, or its path when it came
        from disk. An *extension* of that artifact keeps the id: extending is the recipe's own
        designed path for a later stage (and its ``stage2_lambda_multiplier`` is the measured
        response to it), not a substitution of one p(h) for another.
        """
        return self.state["artifact_id"] or str(self.state["current_artifact"])

    # --------------------------------------------------------------------------------- extend

    def extend(self, *, need: int = 40_000, k_domain: int = 8,
               device: str | dict = DEFAULT_DEVICE) -> Path:
        """Fold the last stage into the model and into p(h): the step between two domains.

        Two things happen, and the next stage needs both: the stage's adapter is merged into the
        model it was trained over (``models/stage{N}_fused``), and the domain it learned is added
        to the artifact as a small mixture in the base's own basis, weighted by sample share
        (``artifacts/v{N+1}.pt``). The corpus is the one that stage trained on -- the whole point
        is that the *earlier* domains are never revisited.

        Args:
            need: activations to collect per site through the fused model.
            k_domain: mixture components to fit per site (capped at one per 200 activations).
            device: where to run the collection and the fits.

        Returns:
            The path of the extended artifact.

        Raises:
            StageOrderError: nothing has been trained since the last extension.
            ValueError: the base artifact carries no sample count and none can be supplied.
        """
        if not self.state["pending_extend"]:
            raise StageOrderError(
                "Nothing to extend: no stage has been trained since the last extension. Train a "
                "domain first (`lfa train --corpus ...`)."
            )
        entry = self.history[-1]
        stage = self.state["stage"]
        placement = resolve_device(device)

        # Resolved before the fuse, so an artifact whose domain share cannot be computed fails in
        # a second rather than after merging a model.
        base_n = self._resolve_base_n()

        adapter_dir = Path(self.state["last_stage_adapter"])
        if (adapter_dir / "adapter_config.json").is_file():
            fused = fuse(adapter_dir, str(self.state["last_stage_base"]),
                         self.path / "models" / f"stage{stage}_fused", dtype=_stage_dtype(entry))
        else:
            # A full-weight stage has no adapter to merge: its checkpoint is already the model
            # the next stage adapts.
            fused = adapter_dir
            logger.info("Stage %d trained full weights; its checkpoint is the fused model", stage)

        out = self.path / "artifacts" / f"v{self.state['artifact_version'] + 1}.pt"
        extend_artifact(
            str(fused), self.state["current_artifact"], self.state["last_stage_corpus"], out,
            base_n=base_n, k_domain=k_domain, need=need,
            seq_len=entry["recipe"]["sequence_length"], seed=entry["recipe"]["seed"],
            device=_primary_device(placement), quantize=True,
        )

        self.state.update(
            current_model=str(fused),
            current_artifact=str(out),
            artifact_version=self.state["artifact_version"] + 1,
            pending_extend=False,
        )
        self._save_state()
        logger.info("Stage %d folded in: model %s, artifact v%d", stage, fused,
                    self.state["artifact_version"])
        return out

    def _resolve_base_n(self) -> int | None:
        """The base pool's per-site sample count to weight the new domain against.

        ``None`` means the artifact answers for itself -- it carries per-block counts, or a
        ``__meta__`` total -- and :func:`lfa.artifact.extend.extend_artifact` reads it there.
        Otherwise the registry entry the artifact came from supplies it (the shipped artifact
        predates the field and is 1,543,000 vectors per site). With neither, the domain's weight
        share cannot be computed at all, and a guess would silently mis-weight the mixture.

        Loading the artifact for this costs one CPU read of a file that is about to be read
        again; the alternative is fusing a model for an extension that cannot be completed.
        """
        params = torch.load(self.state["current_artifact"], map_location="cpu",
                            weights_only=False)
        sites = [entry for key, entry in params.items()
                 if parse_site_key(key) is not None and isinstance(entry, dict)]
        meta_total = (params.get(META_KEY) or {}).get("n_samples_total")
        del params

        if meta_total is not None or (sites and all("n_samples" in site for site in sites)):
            return None
        registry = ARTIFACTS.get(self.state["artifact_id"] or "")
        if registry is not None:
            return int(registry["n_samples_total"])
        raise ValueError(
            f"{self.state['current_artifact']} carries no per-site n_samples and no "
            "__meta__.n_samples_total, and it did not come from the artifact registry, so the "
            "new domain cannot be weighted by its sample share. Pass base_n by calling "
            "lfa.artifact.extend.extend_artifact directly (the shipped qwen3-0.6b artifact was "
            "collected over 1,543,000 vectors per site)."
        )

    # ------------------------------------------------------------------------------- evaluate

    def evaluate(
        self,
        corpus: str | Path | None = None,
        *,
        compare_unanchored: bool = False,
        n_windows: int | None = 100,
        device: str | dict = DEFAULT_DEVICE,
    ) -> dict:
        """Read the last stage on both axes: what it learned, and what it kept.

        ``before`` is the model the stage started from (the base, or the fused model an earlier
        extension left) and ``after`` is that same model with the stage's adapter on it, so the
        pair isolates the stage rather than the chain.

        Args:
            corpus: text to measure domain perplexity on. Defaults to the corpus the stage
                trained on, which makes the domain number a *fit* rather than a held-out
                measurement -- pass a separate held-out corpus for the number the paper reports.
            compare_unanchored: re-run the stage with lambda = mu = 0 into
                ``runs/stage{N}_unanchored`` and report it as a third column. It is the control
                that says what the anchor bought: an unanchored run reaches the domain by giving
                up the general axis.
            n_windows: WikiText-2 windows for the general axis; ``0`` scores the whole split and
                ``None`` skips it. The general axis is also skipped (with a warning, and a
                ``None`` in its cell) when the split cannot be fetched -- an offline machine
                should still get the domain number.
            device: where to run the evaluations.

        Returns:
            ``{"before": {...}, "after": {...}, "unanchored": {...} | None, "table": str}``,
            each axis dict keyed ``"general"`` and ``"domain"``. The same numbers are written
            into the stage's history entry under ``"perplexity"``.
        """
        entry = self._require_trained_stage()
        placement = resolve_device(device)
        dtype = _dtype_for(placement)
        recipe = Recipe(**entry["recipe"])

        tokenizer = load_tokenizer(str(entry["base_model"]))
        corpus_path = Path(corpus).expanduser().resolve() if corpus else Path(entry["corpus"])
        heldout, _ = load_corpus(corpus_path, tokenizer, max_length=recipe.sequence_length,
                                 val_fraction=0.0, seed=recipe.seed,
                                 keep_short_whole=recipe.keep_short_whole)
        logger.info("Evaluating stage %d on %s (%d chunks)", entry["stage"], corpus_path,
                    len(heldout))

        before = self._score(entry["base_model"], None, tokenizer, heldout, n_windows, placement,
                             dtype)
        after = self._score(entry["base_model"], entry["adapter"], tokenizer, heldout, n_windows,
                            placement, dtype)

        unanchored = None
        if compare_unanchored:
            unanchored_dir = self._train_unanchored(entry, recipe, placement=placement,
                                                    dtype=dtype)
            unanchored = self._score(entry["base_model"], unanchored_dir / "final_model",
                                     tokenizer, heldout, n_windows, placement, dtype)

        entry["perplexity"] = {"before": before, "after": after, "unanchored": unanchored}
        self._save_history()
        return {"before": before, "after": after, "unanchored": unanchored,
                "table": _table(before, after, unanchored)}

    def _score(self, base_model, adapter_dir, tokenizer, heldout, n_windows, placement, dtype):
        """One column of the table: general and domain perplexity for one model."""
        if adapter_dir is None:
            model = load_teacher(str(base_model), device=placement, dtype=dtype)
        elif Path(adapter_dir, "adapter_config.json").is_file():
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                load_teacher(str(base_model), device=placement, dtype=dtype), str(adapter_dir))
        else:
            # A full-weight stage saves the whole model, so the checkpoint IS the "after" model.
            model = load_teacher(str(adapter_dir), device=placement, dtype=dtype)

        try:
            return {
                "general": self._general(model, tokenizer, n_windows, placement),
                "domain": domain_perplexity(model, tokenizer, heldout,
                                            device=_primary_device(placement)),
            }
        finally:
            del model
            gc.collect()
            if _primary_device(placement).startswith("cuda"):
                torch.cuda.empty_cache()

    @staticmethod
    def _general(model, tokenizer, n_windows, placement) -> float | None:
        """WikiText-2 perplexity, or ``None`` when the split is not reachable.

        The exception net is wide on purpose: a missing ``datasets``, an offline cache, a Hub
        outage and a failed download all surface differently, and none of them is a reason to
        lose the domain number the caller came for. The reason is logged with the failure.
        """
        if n_windows is None:
            return None
        try:
            return wikitext2_perplexity(model, tokenizer, n_windows=n_windows,
                                        device=_primary_device(placement))
        except Exception as error:                                # noqa: BLE001 - see docstring
            logger.warning("General axis skipped: WikiText-2 could not be scored (%s: %s)",
                           type(error).__name__, error)
            return None

    def _train_unanchored(self, entry: dict, recipe: Recipe, *, placement, dtype) -> Path:
        """Re-run the stage with the anchor and the weight backstop switched off."""
        control = dataclasses.replace(recipe, lambda_qkv=0.0, lambda_mlp=0.0, mu=0.0)
        config = control.to_train_config(entry["stage"], entry["artifact"])
        output_dir = self.path / "runs" / f"stage{entry['stage']}_unanchored"
        logger.info("Unanchored control for stage %d -> %s", entry["stage"], output_dir)
        self._run_training(config, Path(entry["corpus"]), entry["base_model"], output_dir,
                           placement=placement, dtype=dtype, allow_sharding=False, resume=False,
                           anchored=False)
        return output_dir

    def _require_trained_stage(self) -> dict:
        if not self.history or self.state["last_stage_adapter"] is None:
            raise RuntimeError(
                "This workspace has no trained stage yet: run `lfa train --corpus ...` first."
            )
        return self.history[-1]

    # ----------------------------------------------------------------------------------- fuse

    def fuse(self, out_dir: str | Path | None = None) -> Path:
        """Export the current model with the last stage's adapter merged into it.

        The result is a plain checkpoint -- no PEFT wrapper, no adapter files -- that loads with
        ``AutoModelForCausalLM.from_pretrained`` like any other. :meth:`extend` already produces
        the same model at ``models/stage{N}_fused`` as a side effect of preparing the next stage;
        this is the export for a stage that is not being extended, or for shipping.

        Args:
            out_dir: where to write it (default ``models/stage{N}_fused_export``).
        """
        entry = self._require_trained_stage()
        stage = self.state["stage"]
        out = Path(out_dir) if out_dir else self.path / "models" / f"stage{stage}_fused_export"
        adapter_dir = Path(entry["adapter"])

        if (adapter_dir / "adapter_config.json").is_file():
            return fuse(adapter_dir, str(entry["base_model"]), out, dtype=_stage_dtype(entry))

        # Full weight: there is nothing to merge, so the export is the checkpoint plus the
        # tokenizer that a released model is expected to carry.
        logger.info("Stage %d trained full weights; exporting the checkpoint itself", stage)
        shutil.copytree(adapter_dir, out, dirs_exist_ok=True)
        load_tokenizer(str(entry["base_model"])).save_pretrained(out)
        return out

    # ---------------------------------------------------------------------------------- chain

    def chain(
        self,
        spec_path: str | Path,
        *,
        device: str | dict = DEFAULT_DEVICE,
        allow_sharding: bool = False,
        need: int = 40_000,
        k_domain: int = 8,
    ) -> list[dict]:
        """Run a whole sequence of domains from a YAML spec: train, extend, train, ...

        The spec::

            domains:
              - name: philosophy          # the run directory under runs/ (optional)
                corpus: data/domain_a     # relative paths resolve against the spec file
                epochs: 15                # optional dose override
              - name: archaeology
                corpus: data/domain_c
            extend_between: true          # default: fold each domain in before the next

        Every stage uses the workspace's default recipe; the stage multiplier on lambda is
        applied by :meth:`train` as usual. ``extend_between: false`` trains each domain from the
        same starting model, which is a *comparison*, not a chain.

        Args:
            spec_path: the YAML file.
            device, allow_sharding, need, k_domain: forwarded to every :meth:`train` and
                :meth:`extend` in the chain.

        Returns:
            The history entries the chain appended, in order.
        """
        spec_path = Path(spec_path)
        spec = yaml.safe_load(spec_path.read_text()) or {}
        if not isinstance(spec, dict):
            raise ValueError(f"{spec_path}: a chain spec must be a YAML mapping.")
        domains = spec.get("domains")
        if not isinstance(domains, list) or not domains:
            raise ValueError(f"{spec_path}: a chain spec needs a non-empty 'domains' list.")
        extend_between = bool(spec.get("extend_between", True))

        entries = []
        for position, domain in enumerate(domains, start=1):
            if not isinstance(domain, dict) or not domain.get("corpus"):
                raise ValueError(f"{spec_path}: domain {position} has no 'corpus'.")
            logger.info("Chain %d/%d: %s", position, len(domains),
                        domain.get("name") or domain["corpus"])
            entries.append(self.train(
                spec_path.parent / domain["corpus"],
                epochs=domain.get("epochs"),
                output_name=domain.get("name"),
                device=device,
                allow_sharding=allow_sharding,
            ))
            if extend_between and position < len(domains):
                self.extend(need=need, k_domain=k_domain, device=device)
        return entries
