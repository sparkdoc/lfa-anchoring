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
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from .adapters import get_adapter
from .artifact.extend import artifact_carries_base_count, extend_artifact, gmm_site_keys
from .artifact.fetch import ARTIFACTS, fetch_artifact
from .artifact.schema import validate_against_model
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
    frozen_reference,
    fuse,
    load_student,
    load_teacher,
    load_tokenizer,
    resolve_device,
    resolve_teacher_mode,
)
from .recipe import BUNDLED_DIR, Recipe
from .sampler import Sampler
from .train import train as run_training

logger = logging.getLogger("lfa.workspace")

__all__ = ["Workspace", "StageOrderError", "WorkspaceNotReady", "WORKSPACE_FILE", "HISTORY_FILE",
           "LOADER_FRAME_NOTICE", "DOMAIN_FIELDS", "code_identity",
           "source_digest"]

WORKSPACE_FILE = "workspace.json"
HISTORY_FILE = "history.json"

#: Every field a chain spec's domain entry may carry (see :meth:`Workspace.chain`). Anything
#: else is refused at load, as an unknown field in a recipe file is.
DOMAIN_FIELDS = frozenset({"name", "corpus", "epochs"})

#: Logged once per run whose corpus loader trains short documents whole (the default). Said out
#: loud because it changes the training stream and is invisible in every metric a run reports: a
#: document that fits in one chunk arrives whole here, and arrives as a chunk plus a mid-sentence
#: fragment under ``keep_short_whole=False``.
LOADER_FRAME_NOTICE = (
    "loader: a document that fits in one chunk is trained whole in every epoch; under "
    "keep_short_whole=False it is cut at the epoch's chunk offset like a longer one"
)


class StageOrderError(RuntimeError):
    """Raised when a chain's steps are taken out of order (train/extend/train)."""


class WorkspaceNotReady(RuntimeError):
    """Raised when a workspace is asked for something it does not have yet.

    Two cases, both of them ordinary first-session mistakes rather than bugs: a training call on a
    workspace that carries no p(h) artifact, and a read (``evaluate``, ``fuse``) of a workspace
    that has trained no stage. Both messages end in the command to run instead, so both are in
    :data:`lfa.cli.USER_FACING_ERRORS` and reach the user as one line rather than as the last line
    of a traceback -- which a bare ``RuntimeError`` cannot be, since torch raises those for real
    faults (a CUDA OOM, for one) that must keep their traceback.
    """


def _lfa_version() -> str:
    """The package version, imported lazily so this module never depends on import order."""
    from . import __version__

    return __version__


def _now() -> str:
    """An ISO-8601 timestamp, to the second, in local time."""
    return _datetime.datetime.now().replace(microsecond=0).isoformat()


#: Files whose contents define "the implementation that trained this stage".
_CODE_GLOBS = ("*.py", "*.yaml")


def code_identity() -> dict:
    """Which implementation produced a training run: a digest, and a revision if there is one.

    ``code_digest`` is sha256 over this package's own sources (every ``.py`` and every bundled
    recipe, each hashed together with its relative path), truncated to 16 hex characters. It is
    the load-bearing field: it is available whether the package was installed from a wheel or
    imported from a checkout, and unlike a commit id it moves when the working tree moves, so an
    uncommitted edit to the objective does not share an identity with the code before it.

    ``git_revision`` is the HEAD of the checkout this package lives in, or ``None`` (installed
    package, no ``git`` on PATH, a detached export). It is for a human reading a record, not for
    a comparison.

    Recorded in every history entry so that a later reader -- in particular the port-verification
    harness (``docs/verification.md``), which may re-score a run it kept from an earlier session --
    can say whether the numbers it is about to quote were produced by the code in front of it.
    Without it, a kept run and a changed objective look exactly alike.
    """
    root = Path(__file__).resolve().parent
    return {"code_digest": source_digest(root), "git_revision": _git_revision(root)}


def source_digest(root: Path) -> str:
    """sha256 over every ``.py`` and ``.yaml`` under ``root``, truncated to 16 hex characters.

    Path-then-contents, in sorted order, with a separator between the two, so that moving a file
    or renaming it changes the digest as surely as editing it does.
    """
    digest = hashlib.sha256()
    for path in sorted({p for pattern in _CODE_GLOBS for p in Path(root).rglob(pattern)}):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _git_revision(inside: Path) -> str | None:
    """``git rev-parse HEAD`` for the checkout ``inside`` belongs to, or ``None``."""
    try:
        finished = subprocess.run(["git", "-C", str(inside), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip() or None


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


def _stage_frame(entry: dict) -> bool:
    """The corpus-chunking frame a recorded stage ran under, override included."""
    if entry.get("keep_short_whole") is not None:
        return bool(entry["keep_short_whole"])
    return bool(entry["recipe"]["keep_short_whole"])


def _stage_val_fraction(entry: dict) -> float:
    """The document share a recorded stage held out of training.

    Read from the entry rather than from the recipe so that the split is rebuilt exactly as the
    run made it -- the recipe is what was asked for, the entry is what ran.
    """
    if entry.get("val_fraction") is not None:
        return float(entry["val_fraction"])
    return float(entry["recipe"].get("val_fraction", 0.0))


def _delta(before: float, after: float) -> str:
    return "n/a" if before == 0 else f"{(after - before) / before * 100:+.1f}%"


def _table(before: dict, after: dict, unanchored: dict | None) -> str:
    """The run's two axes as Markdown, degrading to one row when the general axis was skipped.

    :func:`lfa.evaluate.perplexity_table` reports both axes and has no cell for an unmeasured
    one; an ``inf`` or a ``nan`` in its place would read as a measurement rather than as its
    absence, so the general row is replaced by a line that says what happened instead.
    """
    columns = [("before", before), ("after", after)]
    if unanchored is not None:
        columns.append(("unanchored", unanchored))
    # Every column, not just `before`: the axis can be measured for one model and fail for
    # another (an intermittent Hub), and rendering the row then asks `perplexity_table` to
    # format a None as a number.
    unmeasured = [name for name, values in columns if values.get("general") is None]
    if not unmeasured:
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
                      "", f"{GENERAL_ROW}: not measured ({', '.join(unmeasured)})"])


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
        artifact_id: str | None = None,
    ) -> "Workspace":
        """Create a workspace over ``model_id`` and put its first p(h) artifact in place.

        Args:
            path: the workspace directory (created if needed). It must not already hold one.
            model_id: a Hub id or a local checkpoint path. It is not loaded here -- init stays
                cheap, and the artifact is checked against the model when a stage starts.
            artifact: a registry id (see :data:`lfa.artifact.fetch.ARTIFACTS`) or a path to an
                artifact file -- a registry id wins, so a local file named like a published
                artifact cannot shadow it. Either way the artifact ends up copied to
                ``artifacts/v1.pt``, so the workspace carries its own p(h) and later versions sit
                beside it.
            recipe: the default recipe for this workspace -- a bundled name or a path. When
                omitted, a bundled recipe whose own ``model_id`` is this model is adopted; if
                none is, every training call has to name one.
            fetch: download a registry artifact that is not already there. ``False`` leaves the
                workspace without a p(h) (and says so), for a machine with no network.
            artifact_id: the registry id ``artifact`` is a copy of, when it is passed as a *path*
                to a file fetched out of band. It is recorded as the workspace's ``artifact_id``,
                which is what :meth:`lfa.recipe.Recipe.warnings` reads the recipe's calibration
                against -- so naming it here stops a recipe warning that lambda was calibrated
                against a different artifact when it was calibrated against exactly this one. It
                is taken on the caller's word (the file is not digest-checked against the
                registry), so name it only for a file you know the provenance of. Passing it
                beside an ``artifact`` that is itself a registry id is accepted when the two agree
                and refused when they disagree -- a conflicting pair is a mistake, not a choice.

        Raises:
            FileExistsError: ``path`` already holds a workspace.
        """
        path = Path(path)
        if (path / WORKSPACE_FILE).exists():
            raise FileExistsError(
                f"{path} is already an LFA workspace: `lfa init` would discard its history. "
                "Every other command takes it as it is -- `lfa train --workspace "
                f"{path} --corpus <documents>`, `lfa evaluate --workspace {path}` -- or point "
                "init at a different directory. From Python: Workspace.open(path)."
            )
        artifacts_dir = path / "artifacts"

        # Nothing is created until every argument is known to be good: a refused init used to
        # leave `<path>/artifacts/` behind, which reads as a half-made workspace.
        # The registry is consulted first: a file that happens to be named like a published
        # artifact must not shadow the published artifact, which is the one the recipe's lambda
        # was calibrated against.
        if artifact in ARTIFACTS:
            if artifact_id is not None and artifact_id != artifact:
                # More likely a mistake than an intention, and silently keeping `artifact` would
                # record a provenance the caller did not ask for.
                raise ValueError(
                    f"artifact={artifact!r} is itself a published artifact id, but artifact_id="
                    f"{artifact_id!r} names a different one. Pass artifact_id only for a local "
                    "artifact FILE that is a copy of a published artifact."
                )
            artifact_id, source = artifact, None
        elif Path(artifact).exists():
            source = Path(artifact)
            if artifact_id is not None and artifact_id not in ARTIFACTS:
                raise ValueError(
                    f"artifact_id={artifact_id!r} is not a published artifact id "
                    f"({', '.join(sorted(ARTIFACTS))}). Leave it out for a local artifact that "
                    "is not a copy of a published one."
                )
            if artifact_id is not None:
                logger.info("Local artifact %s recorded as the published %r", source, artifact_id)
        else:
            raise ValueError(
                f"{artifact!r} is neither a path that exists nor a published artifact id "
                f"({', '.join(sorted(ARTIFACTS))})."
            )
        destination = artifacts_dir / "v1.pt"
        # Remembered so that a failure below can put the directory tree back as it found it: the
        # arguments being good does not mean the artifact will arrive, and today the likeliest
        # refusal of all is `ArtifactNotPublished` from the fetch three lines down.
        created = [directory for directory in (path, artifacts_dir) if not directory.exists()]
        path.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)

        try:
            if source is not None:
                shutil.copyfile(source, destination)
                logger.info("Artifact %s copied to %s", source, destination)
            elif destination.exists():
                logger.info("Artifact %s already present at %s", artifact_id, destination)
            elif fetch:
                fetched = fetch_artifact(artifact_id, artifacts_dir)
                fetched.replace(destination)
        except BaseException:
            # Only what this call made, and only while still empty: `rmdir` refuses a directory
            # with anything in it, which is the guard that keeps this from touching a workspace
            # that was already there.
            for directory in reversed(created):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            raise
        if not fetch and source is None and not destination.exists():
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
        keep_short_whole: bool | None = None,
        full_weight: bool | None = None,
        teacher_mode: str | None = None,
        device: str | dict = DEFAULT_DEVICE,
        allow_sharding: bool = False,
        resume: bool = False,
    ) -> dict:
        """Train one stage: ``current_model`` on ``corpus``, anchored on ``current_artifact``.

        The stage number decides the lambda: from stage 2 on, the recipe's
        ``stage2_lambda_multiplier`` applies, because a later stage anchors a model that already
        carries a domain. Training the *same* corpus again -- more epochs on the domain in
        progress -- is the same stage and keeps the same lambda; a *different* corpus is the next
        stage and must be preceded by :meth:`extend`.

        Args:
            corpus: a file or directory of documents (see :func:`lfa.corpus.load_texts`).
            recipe: a :class:`~lfa.recipe.Recipe`, a bundled name, or a path. Defaults to the
                workspace's own.
            epochs: override the recipe's number of epochs. The learning-rate schedule is laid
                over whatever this says, so it changes the whole curve, not only where it stops.
            output_name: run directory name under ``runs/``. The default is ``stage{N}``, and
                ``stage{N}_run{k}`` for a repeat of a stage already trained -- a repeat is a
                second run, not an overwrite of the first, and both stay readable. A name that
                already holds a run is refused (``FileExistsError``) unless ``resume`` is set,
                since writing over it would discard that run's config, curve and checkpoint.
            keep_short_whole: override the recipe's short-document setting. Under ``False`` a
                document that fits in one chunk is cut at the epoch's chunk offset like a longer
                one, into a chunk and a fragment.
            full_weight: override the recipe's training mode. Full weight is outside the LFA
                paper's validated envelope; the recipe's own warning says so.
            teacher_mode: where the frozen teacher comes from -- ``"auto"`` (the default),
                ``"adapter_disabled"`` or ``"separate"``. Under LoRA the default reads the
                teacher out of the student's own frozen base and loads no second model, which is
                bit-identical to ``"separate"`` and one whole model cheaper. Full weight resolves
                to ``"separate"``, and asking for ``"adapter_disabled"`` there is refused. The
                resolved mode lands in the history entry and in the run's ``config.json``.
            device: a single device, as :func:`lfa.models.resolve_device` reads it.
            allow_sharding: permit a device map that spreads the model over several devices.
            resume: continue the run already in this stage's output directory.

        Returns:
            The history entry this run appended.

        Raises:
            StageOrderError: a new corpus while a trained stage has not been extended.
            ValueError: no recipe anywhere, or an ``epochs`` the schedule cannot carry.
            WorkspaceNotReady: the workspace has no artifact to anchor against.
            FileExistsError: the run directory already holds a run and this is not a resume.
        """
        corpus_path = Path(corpus).expanduser().resolve()
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus path not found: {corpus_path}. Point --corpus at a file or a directory "
                "of documents; `lfa prepare-domain <your files> --out <dir>` builds one from "
                ".txt/.md/.html/.pdf sources."
            )

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
            raise WorkspaceNotReady(
                "This workspace has no p(h) artifact. Fetch one with `lfa fetch-artifact "
                f"{self.state['artifact_id'] or '<id>'} --dest {self.path / 'artifacts'}` "
                "or re-initialise with a local artifact path."
            )

        resolved = self._resolve_recipe(recipe)
        overrides = {}
        if epochs is not None:
            overrides["epochs"] = epochs
        if full_weight is not None:
            overrides["full_weight"] = full_weight
        if overrides:
            resolved = dataclasses.replace(resolved, **overrides)

        stage = self.state["stage"] if repeat else self.state["stage"] + 1
        config = resolved.to_train_config(stage, artifact, keep_short_whole=keep_short_whole)
        # Resolved here, before anything is loaded, because it decides whether a second model is
        # loaded at all -- and because `adapter_disabled` on a full-weight run is a refusal, which
        # belongs at second zero rather than after the corpus has been chunked.
        config = dataclasses.replace(
            config,
            teacher_mode=resolve_teacher_mode(config.teacher_mode if teacher_mode is None
                                              else teacher_mode,
                                              use_lora=config.use_lora,
                                              full_weight=config.full_weight),
        )

        # Said before anything is loaded: an off-calibration lambda is not a refusal, but it is
        # also not the measured operating point, and a run is worth more than the warning is.
        for note in resolved.warnings(config.lora_rank, self._artifact_id()):
            logger.warning(note)
        if config.keep_short_whole:
            logger.info(LOADER_FRAME_NOTICE)

        base_model = self.state["current_model"]
        output_dir = self.path / "runs" / (output_name or self._run_name(stage, repeat, resume))
        self._refuse_to_overwrite(output_dir, resume)
        # Read before the run rather than after it: it is meant to name the code that trained the
        # stage, and an edit landing on disk while the run is in flight is not that code.
        implementation = code_identity()
        placement = resolve_device(device, allow_sharding)
        dtype = _dtype_for(placement)
        logger.info("Stage %d: %s on %s (artifact v%d, λ_qkv=%s, %d epochs)", stage, base_model,
                    corpus_path, self.state["artifact_version"], config.lambda_qkv,
                    config.num_epochs)

        training, corpus_counts = self._run_training(config, corpus_path, base_model, output_dir,
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
            # The two loader/mode settings the run actually used, which a per-call override
            # can move away from the recipe's own values.
            "keep_short_whole": config.keep_short_whole,
            "full_weight": config.full_weight,
            # Which teacher the stage trained against: `adapter_disabled` held no second model.
            # It changes no number (the two are bit-identical under LoRA), but a run's footprint
            # is not readable from anything else here.
            "teacher_mode": config.teacher_mode,
            # What the stage was allowed to see. `evaluate` rebuilds the same split from the
            # corpus, the seed and this fraction, and scores the documents this run never trained
            # on -- so the number it reports is held out rather than fitted.
            "val_fraction": config.val_fraction,
            **corpus_counts,
            "final_loss": training.history[-1]["loss_total"] if training.history else None,
            # Where and in what precision this stage ran: an export merges in the dtype it was
            # trained in rather than in a default that may not be the same one.
            "device": placement,
            "dtype": str(dtype).removeprefix("torch."),
            "timestamp": _now(),
            "lfa_version": _lfa_version(),
            # WHICH CODE trained this, not merely which release: a version string does not move
            # between two commits of the same version, and a kept run re-scored after the
            # objective changed is otherwise indistinguishable from one that was retrained.
            "implementation": implementation,
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

    @staticmethod
    def _refuse_to_overwrite(output_dir: Path, resume: bool) -> None:
        """Stop a run that would write over a run already in ``output_dir``.

        :meth:`_run_name` keeps the *default* names apart (a repeat of a stage becomes
        ``stage{N}_run{k}``), but an explicit ``output_name`` -- a chain spec's ``name``, or the
        same name passed twice -- goes through as given, and the trainer writes with
        ``exist_ok=True``. Two history entries then point at one directory: the first stage's
        config, curve and checkpoint are gone, and every later read of that entry (``evaluate``,
        ``extend``, ``fuse``) silently resolves to the second stage's model.

        A resume is the one case where writing into an existing run is the intent, and it is
        allowed: the trainer restores that run's epoch counter, history and optimizer moments and
        continues it rather than starting over.

        What counts as a run is what a run LEAVES BEHIND -- a curve, a checkpoint, or saved
        optimizer state -- and deliberately not ``config.json``, which :func:`lfa.train.train`
        writes before the first epoch. A start that was interrupted in its first epoch leaves the
        config and nothing else; there is nothing there to protect, nothing to resume from
        (:func:`lfa.train._load_training_state` raises without a ``training_state.pt``), and a
        refusal would leave the user unable to simply run the command again.

        Raises:
            FileExistsError: the directory already holds a run and this is not a resume of it.
        """
        if resume:
            return
        # `config.json` is not in this list, on purpose: see the docstring.
        existing = [name for name in ("training_history.json", "final_model", "training_state.pt")
                    if (output_dir / name).exists()]
        if not existing:
            if (output_dir / "config.json").exists():
                logger.info("%s holds a config from an interrupted start and nothing else; "
                            "starting over in it", output_dir)
            return
        raise FileExistsError(
            f"{output_dir} already holds a training run ({', '.join(existing)}), and this call "
            "would write over it -- its config, its curve and its checkpoint. Either name this "
            "run differently (`output_name=`, or the chain spec's `name:`), or delete that "
            "directory if you meant to redo it"
            + (", or pass resume=True to continue the run that is there."
               if (output_dir / "training_state.pt").exists()
               else " (resume=True cannot help: there is no training_state.pt to continue from).")
        )

    @classmethod
    def _validate_domains(cls, spec_path: Path, domains: list) -> None:
        """Check every domain in a chain spec before any of them trains.

        Four things, in the order a reader would find them: that the entry is a mapping with a
        ``corpus``, that the corpus is actually there, that it carries no field a domain does not
        have, and that no two domains share a ``name`` (the name IS the run directory, so the
        later stage would write over the earlier one).

        All of it up front. The alternative -- checking each domain as the loop reaches it -- is
        how a misspelt key on domain 2 gets reported after domain 1 has trained and been folded
        in, which on the shipped recipe is about two hours before the sentence appears.

        Raises:
            ValueError: naming the spec file and the domain's position, for each of the four.
            FileNotFoundError: a corpus path that is not there, with the same remedy
                :meth:`train` gives.
        """
        for position, domain in enumerate(domains, start=1):
            if not isinstance(domain, dict) or not domain.get("corpus"):
                raise ValueError(f"{spec_path}: domain {position} has no 'corpus'.")
            cls._refuse_unknown_domain_keys(spec_path, position, domain)
            # Resolved as the chain will resolve it: relative to the spec file, not to the
            # working directory. `train` would raise the same thing, an hour or two later.
            corpus = (spec_path.parent / str(domain["corpus"])).expanduser()
            if not corpus.exists():
                raise FileNotFoundError(
                    f"{spec_path}: domain {position} names a corpus that is not there: {corpus}. "
                    "Paths in a chain spec resolve against the spec file. "
                    "`lfa prepare-domain <your files> --out <dir>` builds one."
                )

        named = [domain.get("name") for domain in domains if domain.get("name")]
        repeated = sorted({name for name in named if named.count(name) > 1})
        if repeated:
            raise ValueError(
                f"{spec_path}: {', '.join(repr(name) for name in repeated)} names more than one "
                "domain, and a name is the run directory -- the later stage would write over the "
                "earlier one's config, curve and checkpoint. Give each domain its own name."
            )

    @staticmethod
    def _refuse_unknown_domain_keys(spec_path: Path, position: int, domain: dict) -> None:
        """Refuse a domain entry carrying a field a domain does not have.

        The same rule :class:`lfa.recipe.Recipe` applies to a recipe file, for the same reason: a
        field that is read by nobody is silently ignored, and the user who wrote it believes it
        took effect. ``extend_between`` is called out by name because it is refused at the spec's
        top level with an explanation, and *between this domain and the next* is the more natural
        place to try to put it -- so the near miss has to land as the same refusal rather than as
        a multi-hour training run at the shipped recipe.
        """
        unknown = sorted(set(domain) - DOMAIN_FIELDS)
        if not unknown:
            return
        named = ", ".join(repr(key) for key in unknown)
        if "extend_between" in unknown:
            raise ValueError(
                f"{spec_path}: domain {position} carries {named}. A chain always folds each "
                "domain into the model and into p(h) before the next one starts -- there is no "
                "per-domain switch for it, any more than there is a top-level one; without that "
                "fold the next domain would adapt this one's base model and anchor against a "
                "p(h) that does not describe it. To train several domains from the same starting "
                "point instead, run them as separate workspaces. A domain takes "
                f"{', '.join(sorted(DOMAIN_FIELDS))}."
            )
        raise ValueError(
            f"{spec_path}: domain {position} carries {named}, which a domain entry does not "
            f"have. A domain takes {', '.join(sorted(DOMAIN_FIELDS))}. (A field nobody reads is "
            "worse than a refusal: the run starts and does not do what the file says.)"
        )

    def _run_name(self, stage: int, repeat: bool, resume: bool) -> str:
        """The default run directory for this stage.

        The first run of a stage is ``stage{N}``. A repeat -- more epochs on the domain already
        in progress -- is ``stage{N}_run{k}``, because two history entries pointing at one
        directory would leave the first run's config, curve and checkpoint overwritten by the
        second's. A
        *resumed* repeat is the same run continuing, so it keeps its own directory.
        """
        if not repeat:
            return f"stage{stage}"
        if resume:
            return Path(self.state["last_stage_dir"]).name
        return f"stage{stage}_run{1 + sum(1 for e in self.history if e['stage'] == stage)}"

    def _run_training(self, config, corpus_path, base_model, output_dir, *, placement, dtype,
                      allow_sharding, resume, anchored):
        """Load teacher, student, sampler and corpus for one run, and train it.

        A resume passes the bare student through: the trainer re-attaches the saved adapter
        (trainable), and doing it here as well would train a fresh one instead. A LoRA resume
        whose checkpoint saved no adapter is refused by the trainer
        (:class:`lfa.train.ResumeSourceHasNoAdapter`) -- not by the frozen-model guard, which
        cannot see it: ``load_student`` hands back a fully trainable model.

        Whether a teacher is loaded at all is ``config.teacher_mode``, already resolved by the
        caller. ``adapter_disabled`` (the default for a LoRA run) loads none: PEFT keeps the base
        weight of every adapted module frozen, so the student holds the teacher and
        :class:`~lfa.models.AdapterDisabledTeacher` reads it there. ``separate`` loads a second
        model, which is what full-weight training needs and what every run before 0.1.1 did.

        The embedding lookup is rebuilt from the teacher here rather than shipped: layer-0
        ``pre_qkv`` is ``input_layernorm(embed_tokens(id))``, exactly reconstructible and ~300 MB
        to store. Without this call the artifact has no table, and ``L_embed`` -- the only term
        anchoring the embedding end of the tied embedding/LM-head matrix -- is dropped silently.
        Under ``adapter_disabled`` it is rebuilt from the student's own *frozen* embedding, which
        is the same tensor a separate teacher would have carried.
        """
        # Bound up front so the cleanup below is safe however far the setup gets: a model is the
        # largest thing this package allocates -- two of them under `teacher_mode="separate"` --
        # and a failed setup must not keep them. `reference` may be a view over the student, so it
        # is released with the rest rather than after it.
        teacher = student = sampler = reference = None
        # Resolved here rather than by each caller, because this is the one funnel every run goes
        # through -- the stage, its unanchored control, and a resume alike -- and it is what
        # decides whether a second model is loaded at all.
        config = dataclasses.replace(
            config, teacher_mode=resolve_teacher_mode(config.teacher_mode,
                                                      use_lora=config.use_lora,
                                                      full_weight=config.full_weight))
        try:
            if config.teacher_mode == "separate":
                teacher = load_teacher(str(base_model), device=placement, dtype=dtype,
                                       allow_sharding=allow_sharding)
                adapter = get_adapter(teacher)
                student = load_student(str(base_model), device=placement, dtype=dtype,
                                       allow_sharding=allow_sharding)
            else:
                student = load_student(str(base_model), device=placement, dtype=dtype,
                                       allow_sharding=allow_sharding)
                adapter = get_adapter(student)
                logger.info("Teacher mode: adapter_disabled -- the student's own frozen LoRA "
                            "base is the teacher; no second model is loaded")
            # A resumed LoRA run is handed the BARE student: `lfa.train.train` re-attaches the
            # saved adapter itself, and only if it is not looking at a PEFT model already. Wrap
            # it here and the resume restores the epoch counter, the history, the scheduler
            # position and Adam's moments onto a *fresh* adapter whose B is still zero -- the run
            # silently starts over and then overwrites the checkpoint it resumed from.
            if config.use_lora and not resume:
                student = apply_lora(student, adapter, rank=config.lora_rank,
                                     alpha=config.lora_alpha, dropout=config.lora_dropout,
                                     freeze_embed=config.freeze_embed)

            # What the artifact is checked against and what the layer-0 table is rebuilt from: the
            # separate teacher when there is one, otherwise the student's frozen base. On the
            # resume path the student has not been wrapped yet and `frozen_reference` hands it
            # back as it is -- correct, because it was loaded from `base_model` moments ago.
            reference = teacher if teacher is not None else frozen_reference(student)
            if anchored:
                sampler = Sampler(config.artifact_path, device=_primary_device(placement),
                                  seed=config.seed)
                # An artifact that does not describe this model anchors toward a different
                # function, and every shape downstream still matches, so nothing would notice.
                validate_against_model(sampler.params, reference, adapter)
                sampler.build_embedding_lookup_from_model(reference, adapter)

            tokenizer = load_tokenizer(str(base_model))
            # Holding documents out is what makes this stage's domain perplexity a measurement
            # rather than a fit. The split goes to the trainer, which scores it after every epoch
            # (loss and perplexity into `training_history.json`), and `evaluate` later rebuilds
            # the same split from the recorded corpus, seed and fraction.
            dataset, holdout = load_corpus(corpus_path, tokenizer,
                                           max_length=config.sequence_length,
                                           val_fraction=config.val_fraction, seed=config.seed,
                                           keep_short_whole=config.keep_short_whole)
            # Documents AND chunks: the document counts say how the split fell, the chunk counts
            # say what the loader made of it, and only the second is comparable with another
            # implementation's loader (the research code logs exactly these two numbers per run).
            counts = {"n_train_docs": dataset.report["n_docs"],
                      "n_val_docs": holdout.report["n_docs"] if holdout is not None else 0,
                      "n_train_chunks": dataset.report["n_chunks"],
                      "n_val_chunks": holdout.report["n_chunks"] if holdout is not None else 0}

            training = run_training(teacher, student, dataset, sampler, adapter, config,
                                    output_dir, resume=resume, tokenizer=tokenizer,
                                    val_dataset=holdout)
            return training, counts
        finally:
            del teacher, student, sampler, reference
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
            # Collected under the frame the stage trained under: the new components have to
            # describe the training stream the model actually saw.
            keep_short_whole=_stage_frame(entry),
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
        predates the field and is 1,543,040 vectors per site). With neither, the domain's weight
        share cannot be computed at all, and a guess would silently mis-weight the mixture.

        Loading the artifact for this costs one CPU read of a file that is about to be read
        again; the alternative is fusing a model for an extension that cannot be completed.
        """
        params = torch.load(self.state["current_artifact"], map_location="cpu",
                            weights_only=False)
        gmm_keys = gmm_site_keys(params)
        # `gmm_keys` empty means there is nothing to extend at all; leave that for
        # `extend_artifact` to say, rather than reporting it as a missing count.
        answers_for_itself = not gmm_keys or artifact_carries_base_count(params, gmm_keys)
        del params

        if answers_for_itself:
            return None
        registry = ARTIFACTS.get(self.state["artifact_id"] or "")
        if registry is not None:
            return int(registry["n_samples_total"])
        raise ValueError(
            f"{self.state['current_artifact']} carries no per-site n_samples and no "
            "__meta__.n_samples_total, and this workspace does not know which published "
            "artifact it is, so the new domain cannot be weighted by its sample share. Fix it "
            "where it started: `lfa init <a new workspace> --model <model> --artifact <the same "
            "file> --artifact-id qwen3-0.6b-gmm1543k-int8`, which records the id that supplies "
            "the count (the shipped qwen3-0.6b artifact was collected over 1,543,040 vectors "
            "per site). From Python you can instead pass base_n to "
            "lfa.artifact.extend.extend_artifact directly."
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
            corpus: text to measure domain perplexity on, scored whole. The default is the
                stage's own held-out split -- the documents its ``val_fraction`` kept out of
                training, rebuilt from the recorded corpus, seed and fraction -- so the domain
                number is a held-out measurement. A stage trained with ``val_fraction=0.0`` has
                no such split and is scored on what it trained on, which is a *fit*; the log line
                says which of the two happened.
            compare_unanchored: re-run the stage with lambda = mu = 0 into
                ``runs/{run}_unanchored`` (named after the run it controls) and report it as a
                third column. It is the control
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
        # A corpus named here is already the held-out text the caller wants scored, so it is
        # scored whole. Defaulting to the stage's own corpus instead rebuilds that stage's split
        # -- same documents, same seed, same fraction -- and scores the part it never trained on.
        corpus_path = Path(corpus).expanduser().resolve() if corpus else Path(entry["corpus"])
        val_fraction = 0.0 if corpus else _stage_val_fraction(entry)
        trained_on, held_out = load_corpus(corpus_path, tokenizer,
                                           max_length=recipe.sequence_length,
                                           val_fraction=val_fraction, seed=recipe.seed,
                                           keep_short_whole=_stage_frame(entry))
        # A held-out split that chunks to nothing (documents shorter than a chunk, under the
        # research frame) would score nothing at all; fall back and say which was used.
        held_out_used = held_out is not None and len(held_out) > 0
        heldout = held_out if held_out_used else trained_on
        logger.info("Evaluating stage %d on %s (%d chunks, %s)", entry["stage"], corpus_path,
                    len(heldout),
                    "held out from training" if held_out_used
                    else "trained on -- a fit, not a held-out measurement")

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

        A Hub failure arrives as :class:`lfa.evaluate.DatasetUnavailable` and is *not* passed on:
        it is logged and the axis is reported unmeasured, because an offline machine should still
        get the domain number it came for. The net stays wide beyond that class, since a model
        that cannot be scored is likewise not a reason to lose the domain number, and the reason
        is logged either way.
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
        """Re-run the stage with the anchor and the weight backstop switched off.

        Everything else about the run is the stage's own, per-call overrides included: the
        control answers "what would this run have done without the anchor", and a control that
        trained a different way answers a different question.
        """
        control = dataclasses.replace(
            recipe, lambda_qkv=0.0, lambda_mlp=0.0, mu=0.0,
            full_weight=bool(entry.get("full_weight", recipe.full_weight)),
        )
        config = dataclasses.replace(
            control.to_train_config(entry["stage"], entry["artifact"],
                                    keep_short_whole=_stage_frame(entry)),
            # The stage's own teacher mode, so the control is the same run without the anchor
            # rather than the same run set up differently. A stage recorded before 0.1.1 has
            # none and resolves the way any other run would.
            teacher_mode=entry.get("teacher_mode") or "auto",
        )
        # Named after the run it controls, so a repeat of a stage gets its own control.
        output_dir = self.path / "runs" / f"{Path(entry['output_dir']).name}_unanchored"
        logger.info("Unanchored control for stage %d -> %s: this is a SECOND full training "
                    "run, as long as the first, with lambda = mu = 0", entry["stage"], output_dir)
        self._run_training(config, Path(entry["corpus"]), entry["base_model"], output_dir,
                           placement=placement, dtype=dtype, allow_sharding=False, resume=False,
                           anchored=False)
        return output_dir

    def _require_trained_stage(self) -> dict:
        """The stage :meth:`evaluate` and :meth:`fuse` act on.

        Raises:
            WorkspaceNotReady: nothing has been trained here yet.
        """
        if not self.history or self.state["last_stage_adapter"] is None:
            raise WorkspaceNotReady(
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
                epochs: 15                # optional epoch-count override
              - name: archaeology
                corpus: data/domain_c

        Every domain is folded in before the next one starts -- that is what a chain *is*, and it
        is the paper's protocol: stage N+1 adapts the fused model and anchors on the artifact
        that domain N has been merged into. Every stage uses the workspace's default recipe, and
        the stage multiplier on lambda is applied by :meth:`train` as usual.

        Args:
            spec_path: the YAML file.
            device, allow_sharding, need, k_domain: forwarded to every :meth:`train` and
                :meth:`extend` in the chain.

        Returns:
            The history entries the chain appended, in order.

        Raises:
            ValueError: the spec is not valid YAML, is not a mapping, has no non-empty
                ``domains`` list, has a domain without a ``corpus``, has a domain carrying a
                field that is not one of :data:`DOMAIN_FIELDS`, gives two domains the same
                ``name``, or asks not to extend between domains. Each names the spec file.
            FileNotFoundError: a domain names a corpus that is not there.

            Every one of these is raised before the first stage trains, for every domain in the
            spec -- not as the chain reaches each domain.
        """
        spec_path = Path(spec_path)
        try:
            spec = yaml.safe_load(spec_path.read_text()) or {}
        except yaml.YAMLError as error:
            # As in `Recipe.load`: a typo in the spec is a ValueError naming the file, not a
            # parser traceback from inside yaml.
            raise ValueError(f"{spec_path}: not valid YAML ({error})") from error
        if not isinstance(spec, dict):
            raise ValueError(f"{spec_path}: a chain spec must be a YAML mapping.")
        domains = spec.get("domains")
        if not isinstance(domains, list) or not domains:
            raise ValueError(f"{spec_path}: a chain spec needs a non-empty 'domains' list.")
        if "extend_between" in spec:
            raise ValueError(
                f"{spec_path}: 'extend_between' is not a chain option. A chain always folds each "
                "domain into the model and into p(h) before the next one starts; without that, "
                "the second domain would adapt the first domain's base model and anchor against "
                "a p(h) that does not describe it. To train several domains from the same "
                "starting point instead, run them as separate workspaces."
            )

        # EVERY domain, before the FIRST one trains. A spec is checked as a whole because its
        # cost is paid as a whole: a typo in domain 3 that surfaces when domain 3 starts has
        # already spent two stages and two extensions -- at the shipped recipe, most of a day.
        self._validate_domains(spec_path, domains)

        entries = []
        for position, domain in enumerate(domains, start=1):
            logger.info("Chain %d/%d: %s", position, len(domains),
                        domain.get("name") or domain["corpus"])
            entries.append(self.train(
                spec_path.parent / domain["corpus"],
                epochs=domain.get("epochs"),
                output_name=domain.get("name"),
                device=device,
                allow_sharding=allow_sharding,
            ))
            if position < len(domains):
                self.extend(need=need, k_domain=k_domain, device=device)
        return entries
