"""Recipes: a published Layerwise Function Anchoring (LFA) operating point, in one file.

An LFA run has more knobs than a reader can be expected to set independently, and the ones that
matter most are *coupled*. ``lambda`` constrains motion inside the rank-``r`` update subspace, so
the same lambda binds far harder at a lower rank; it is also read against an artifact's sharpness,
so it does not survive a change of ``p(h)`` either. A recipe is therefore a joint point -- the
whole set of values that were tuned together and measured together -- rather than a bag of
defaults, and it records *which* rank and *which* artifact it was calibrated at so that a run
departing from either can be told that lambda no longer means what it meant
(:meth:`Recipe.warnings`). The artifact can be named by id or path, or -- as the bundled recipe
does -- as ``self-generated``: an artifact fitted on the model's own text at a recorded frame
(:attr:`Recipe.self_generated_frame`), which each such artifact carries in its meta so that a
build at another frame (a trial-sized one, say) is told which fields differ.

Two settings in the shipped point are easy to lose in a re-implementation and are carried
here deliberately:

* ``keep_short_whole=True``: a document that fits in one chunk is trained whole in every epoch
  rather than being cut at the epoch's chunk offset into a chunk and a context-free fragment. It
  changes the stream, so it is recorded per run; pass ``keep_short_whole=False`` to
  :meth:`Recipe.to_train_config` when the documents are themselves slices of something longer and
  the positional variety is worth more than keeping them whole.
* ``val_fraction=0.1``: a tenth of the *documents* (shuffled under the recipe's seed) are held out
  and never trained on, so the domain perplexity reported for a stage is a held-out measurement
  rather than a fit, and the per-epoch validation curve in ``training_history.json`` says when a
  run has started to over-fit. Set it to ``0.0`` to train on everything -- and then read the
  domain number as a fit.

``Recipe.load`` resolves a bare name against the recipes bundled inside the package, so it works
from an installed wheel and from any working directory; anything that looks like a path is read as
one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .train import LR_SCHEDULES, TrainConfig

__all__ = ["Recipe", "BUNDLED_DIR", "SELF_GENERATED_REFERENCE", "RECORDED_SELF_GENERATED_FRAME"]

#: Where the bundled recipes live -- inside the package, so a wheel carries them.
BUNDLED_DIR = Path(__file__).parent / "recipes"

#: The value of ``calibrated_artifact`` (and of a bundled recipe's ``artifact``) that means "an
#: artifact fitted on this model's own text at :attr:`Recipe.self_generated_frame`".
SELF_GENERATED_REFERENCE = "self-generated"

#: The frame the shipped lambda's self-generated calibration was measured at. It agrees with
#: :class:`lfa.selfgen.artifact_corpus.SelfGenOptions`' defaults (a test pins that); it is
#: spelled out here so this module does not import the generation stack.
RECORDED_SELF_GENERATED_FRAME = {"n_raw": 2500, "n_chat": 0, "max_new_tokens": 2048,
                                 "max_samples": 600_000, "gmm_k": 32, "pca_variance": 0.95}

#: The one model whose rank-16 and full-weight lambda ranges were measured (in the research code,
#: on the research corpus, at the paper's point); :meth:`Recipe.warnings` quotes them only for a
#: recipe naming it.
_QWEN3_0_6B = "Qwen/Qwen3-0.6B"

#: The keys a ``self_generated_frame`` may carry: the recorded frame's, plus the rest of
#: :meth:`lfa.selfgen.artifact_corpus.SelfGenOptions.artifact_frame`.
_FRAME_KEYS = frozenset(RECORDED_SELF_GENERATED_FRAME) | {
    "seed", "chat_seed", "min_chars", "max_repeat_ratio", "burn_in_tokens", "reservoir_size"}


def _describe(frame: dict) -> str:
    """A self-generated frame in words, e.g. ``2500 documents x 2048 tokens, 600000 samples per
    site, K=32``. Reads with ``frame.get`` so a partial user frame still renders."""
    text = (f"{frame.get('n_raw')} documents x {frame.get('max_new_tokens')} tokens, "
            f"{frame.get('max_samples')} samples per site, K={frame.get('gmm_k')}")
    if frame.get("n_chat"):
        text += f", {frame.get('n_chat')} of them chat-format"
    return text


@dataclass
class Recipe:
    """One tuned LFA operating point: what to train, how hard to anchor, and at what it was tuned.

    Args:
        name: The recipe's own name; ``Recipe.load(name)`` finds ``<name>.yaml`` when bundled.
        model_id: The teacher/student model this point was tuned on.
        artifact: What the recipe anchors against by default: an artifact id, a path, or
            ``self-generated``.
        stage2_lambda_multiplier: What :meth:`to_train_config` multiplies lambda by from stage 2
            on. A later stage anchors a model that already carries a domain, and what that wants
            is a *harder* anchor; it is a level, not a per-stage compounding factor. The shipped
            3.0 was measured on one pair of domains (ahead of 1.0 on retention for both bundled
            models) -- lambda is coupled to the corpus, so a chain over a new pair of domains
            re-tunes it.
        calibrated_rank: The LoRA rank the lambdas were tuned at.
        calibrated_artifact: What the lambdas were calibrated against: ``self-generated`` (an
            artifact fitted on this model's own text at :attr:`self_generated_frame`), or an id
            or path.
        self_generated_frame: The generation and fit frame a ``self-generated`` calibration was
            measured at -- the fields of
            :meth:`lfa.selfgen.artifact_corpus.SelfGenOptions.artifact_frame` that set how the
            artifact prices a function. :meth:`warnings` compares a self-generated artifact's
            recorded frame with it field by field. Read only when :attr:`calibrated_artifact` is
            ``self-generated``.
        supplement_fraction: The share of training *tokens* made up of the question-and-answer
            pairs the entry model writes from the domain, in ``[0, 1)``. 0.13 is the frame the
            shipped lambda was tuned at; 0.0 trains on the raw corpus alone (off that frame). It
            is read by the workspace, not by :meth:`to_train_config`.

    Every other field is the corresponding :class:`lfa.train.TrainConfig` knob; see that class for
    what each does. Values are validated on construction, so a hand-edited YAML fails at load
    rather than several GPU-hours into a run.
    """

    name: str
    model_id: str
    artifact: str

    # -- adapter
    lora_rank: int = 32
    lora_alpha: int = 64
    freeze_embed: bool = True
    full_weight: bool = False

    # -- anchoring
    lambda_qkv: float = 1_000_000.0
    lambda_mlp: float = 1_000_000.0
    mu: float = 0.05
    anchor_end_ratio: float = 0.1
    anchor_schedule: str = "cosine"
    n_anchor_samples: int = 16

    # -- run length and checkpointing
    epochs: int = 15
    checkpoint_mode: str = "rolling"
    checkpoint_every: int = 5

    # -- optimization
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"
    batch_size: int = 6
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    sequence_length: int = 512
    seed: int = 42
    keep_short_whole: bool = True
    val_fraction: float = 0.1
    # -- the supplement: the question-and-answer pairs the entry model writes from the domain,
    #    mixed in at this share of training TOKENS. 0.13 is the frame the shipped lambda was
    #    tuned at; 0.0 trains on the raw corpus alone (and is off that frame).
    supplement_fraction: float = 0.13

    # -- what the point was calibrated at
    stage2_lambda_multiplier: float = 3.0
    calibrated_rank: int = 32
    # `self-generated` means an artifact fitted on this model's own text at
    # `self_generated_frame`; anything else is an artifact id or path, compared as a string.
    calibrated_artifact: str = SELF_GENERATED_REFERENCE
    self_generated_frame: dict = dataclasses.field(
        default_factory=lambda: dict(RECORDED_SELF_GENERATED_FRAME))

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {', '.join(LR_SCHEDULES)}, got {self.lr_schedule!r}."
            )
        if self.stage2_lambda_multiplier <= 0:
            raise ValueError(
                f"stage2_lambda_multiplier must be positive, got {self.stage2_lambda_multiplier}"
            )
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be at least 1, got {self.lora_rank}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in [0, 1), got {self.val_fraction}: it is the share of "
                "DOCUMENTS held out of training, so 1.0 would leave nothing to train on"
            )
        if not 0.0 <= self.supplement_fraction < 1.0:
            raise ValueError(
                f"supplement_fraction must be in [0, 1), got {self.supplement_fraction}: it is "
                "the share of training TOKENS the written pairs make up.")
        if not isinstance(self.self_generated_frame, dict):
            raise ValueError(
                "self_generated_frame must be a mapping of frame field to value, got "
                f"{type(self.self_generated_frame).__name__}: write it as a YAML mapping "
                f"(fields: {', '.join(sorted(_FRAME_KEYS))}).")
        unknown = sorted(set(self.self_generated_frame) - _FRAME_KEYS)
        if unknown:
            raise ValueError(
                f"self_generated_frame has unknown field(s): {', '.join(map(str, unknown))}; "
                f"use only {', '.join(sorted(_FRAME_KEYS))}.")

    # ------------------------------------------------------------------ loading and saving

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Recipe":
        """Read a recipe: a bundled name (``"qwen3-0.6b"``) or a path to a YAML file.

        A value with a ``.yaml``/``.yml`` suffix, or one that exists on disk, is read as a path;
        anything else is looked up in :data:`BUNDLED_DIR`.

        Raises:
            FileNotFoundError: No such bundled recipe (the message lists the ones there are) or no
                such file.
            ValueError: The file is not valid YAML, or carries a field this class does not have,
                or a value that fails validation. All three name the file, because a recipe is
                usually one of several on disk and "which one" is the first thing to know.
        """
        path = Path(name_or_path)
        if path.suffix.lower() not in (".yaml", ".yml") and not path.exists():
            path = BUNDLED_DIR / f"{name_or_path}.yaml"
            if not path.is_file():
                available = sorted(p.stem for p in BUNDLED_DIR.glob("*.yaml"))
                raise FileNotFoundError(
                    f"no bundled recipe named {name_or_path!r}; available: "
                    f"{', '.join(available) or '(none)'}"
                )
        if not path.is_file():
            raise FileNotFoundError(f"no recipe file at {path}")

        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as error:
            # A parse error is a user's typo, not a bug: it arrives as a ValueError naming the
            # file, exactly as every semantic failure below does, rather than as a
            # `yaml.parser.ParserError` traceback from inside the loader.
            raise ValueError(f"{path}: not valid YAML ({error})") from error
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a recipe file must be a YAML mapping, got {type(data).__name__}")

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"{path}: unknown recipe field(s): {', '.join(unknown)}")
        # A field with no default is one the recipe cannot be guessed for; naming it here beats
        # the bare `TypeError` the constructor would raise, which names neither the file nor YAML.
        required = {f.name for f in fields(cls) if f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{path}: missing required recipe field(s): {', '.join(missing)}")
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        """Write the recipe as plain YAML (field order preserved). Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dataclasses.asdict(self), sort_keys=False))
        return path

    @classmethod
    def bundled_for(cls, model_id: str) -> str | None:
        """The bundled recipe tuned for ``model_id``, when exactly one names it.

        Matched on the recipe's own ``model_id`` rather than on its name: a recipe is a joint
        operating point for one model, and guessing one from a filename is how a lambda gets
        ported across models it was never calibrated for.
        """
        matches = []
        for path in sorted(BUNDLED_DIR.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:                       # a malformed bundled file is not this
                continue                                 # method's problem to report
            if isinstance(data, dict) and data.get("model_id") == model_id:
                matches.append(path.stem)
        return matches[0] if len(matches) == 1 else None

    # ------------------------------------------------------------------ use

    def to_train_config(
        self, stage: int, artifact_path: str | Path, *, keep_short_whole: bool | None = None,
    ) -> TrainConfig:
        """The :class:`lfa.train.TrainConfig` for one stage of a run using this recipe.

        Args:
            stage: 1 for the first domain, 2 or more for a later one -- from stage 2 on both
                lambdas are multiplied by :attr:`stage2_lambda_multiplier`.
            artifact_path: The p(h) artifact to anchor against. It is passed through as given;
                for stage 2 and beyond this is normally the *extended* artifact
                (:mod:`lfa.artifact.extend`), not the one :attr:`artifact` names.
            keep_short_whole: Override the recipe's short-document setting. ``None`` uses the
                recipe's own value; ``False`` cuts a document that fits in one chunk at the epoch
                offset like any longer one.
        """
        if stage < 1:
            raise ValueError(f"stage must be at least 1, got {stage}")

        lambda_scale = self.stage2_lambda_multiplier if stage >= 2 else 1.0
        return TrainConfig(
            model_id=self.model_id,
            lambda_qkv=self.lambda_qkv * lambda_scale,
            lambda_mlp=self.lambda_mlp * lambda_scale,
            mu=self.mu,
            n_anchor_samples=self.n_anchor_samples,
            anchor_end_ratio=self.anchor_end_ratio,
            anchor_schedule=self.anchor_schedule,
            artifact_path=str(artifact_path),
            learning_rate=self.learning_rate,
            lr_schedule=self.lr_schedule,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_epochs=self.epochs,
            sequence_length=self.sequence_length,
            checkpoint_mode=self.checkpoint_mode,
            checkpoint_every=self.checkpoint_every,
            use_lora=not self.full_weight,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_embed=self.freeze_embed,
            full_weight=self.full_weight,
            seed=self.seed,
            keep_short_whole=self.keep_short_whole if keep_short_whole is None else keep_short_whole,
            val_fraction=self.val_fraction,
        )

    def warnings(self, rank: int, artifact_id: str, artifact_meta: dict | None = None) -> list[str]:
        """What is off-calibration about running this recipe at ``rank`` on ``artifact_id``.

        Empty when the run sits at the point the lambdas were tuned at. Each string says what
        moved and which way to re-tune; none of them is a refusal -- an off-calibration run is
        allowed, it just is not the measured operating point.

        Args:
            rank: The LoRA rank the run trains at.
            artifact_id: The id (or path) of the p(h) artifact the run anchors against.
            artifact_meta: That artifact's meta (``None`` or ``{}`` when unknown). A
                self-generated artifact (``provenance == "self-generated"``) is judged by its meta
                rather than by its id: a mismatch note when the text came from another model;
                against a ``self-generated`` calibration, silent when its recorded
                ``selfgen_frame`` matches :attr:`self_generated_frame`, and a note naming each
                field that differs when it does not (a missing frame differs in every field). Any
                other artifact against a ``self-generated`` calibration is a re-tune; against a
                named calibration the ids are compared.
        """
        notes: list[str] = []
        # The two site families can carry different lambdas; quoting one of them as "the lambda"
        # would misreport the recipe whenever they differ.
        quoted = (f"{self.lambda_qkv:g}" if self.lambda_qkv == self.lambda_mlp
                  else f"qkv {self.lambda_qkv:g}, mlp {self.lambda_mlp:g}")
        if rank != self.calibrated_rank:
            measured = (" (for Qwen3-0.6B on the research corpus, at the paper's point, rank 16 "
                        "sat at roughly a fifth to a half of rank 32's lambda)"
                        if self.model_id == _QWEN3_0_6B else "")
            notes.append(
                f"lambda is coupled to LoRA rank: this recipe's lambda ({quoted}) was "
                f"calibrated at rank {self.calibrated_rank} and you are running rank {rank}. "
                "Lambda constrains motion inside the rank-r update subspace, so the same value "
                "binds harder at a lower rank -- re-tune it rather than porting it (lower rank "
                f"=> lower lambda{measured}; docs/model-integration-cookbook.md §5 is the "
                "procedure), and diagnose against held-out domain perplexity, since "
                "over-anchoring makes general-text perplexity look its best."
            )
        meta = artifact_meta or {}
        self_generated = meta.get("provenance") == "self-generated"
        if self_generated and meta.get("model_id") not in (None, self.model_id):
            notes.append(
                f"this self-generated artifact describes {meta.get('model_id')!r}, not this "
                f"recipe's {self.model_id!r}: lambda is coupled to the p(h) artifact, so "
                "calibrate it against held-out domain perplexity for this model.")
        elif self.calibrated_artifact == SELF_GENERATED_REFERENCE and self_generated:
            # Every self-generated build records its frame; a meta without one has every
            # field unknown, and is told so like any other off-frame build.
            frame = meta.get("selfgen_frame") or {}
            differ = [f"{key} {frame.get(key)!r} (calibrated at {value!r})"
                      for key, value in self.self_generated_frame.items()
                      if frame.get(key) != value]
            if differ:
                notes.append(
                    "this self-generated artifact was built at a different frame from the "
                    "one this recipe's lambda was calibrated at: " + ", ".join(differ)
                    + ". A trial-sized build is fine for trying the pipeline; for a real "
                    "run, build at the recorded frame or calibrate lambda against held-out "
                    "domain perplexity (docs/model-integration-cookbook.md).")
        elif self.calibrated_artifact == SELF_GENERATED_REFERENCE:
            notes.append(
                f"this recipe's lambda ({quoted}) is calibrated against an artifact fitted on the "
                f"model's own text at {_describe(self.self_generated_frame)}, and you are "
                f"anchoring against {artifact_id!r}, which is not one. A different artifact "
                "prices the same function differently, so calibrate lambda against held-out "
                "domain perplexity (docs/model-integration-cookbook.md), reading the frontier "
                "rather than a single point.")
        elif artifact_id != self.calibrated_artifact:
            notes.append(
                f"lambda is coupled to the p(h) artifact: this recipe's lambda was calibrated "
                f"against {self.calibrated_artifact!r} and you are anchoring against "
                f"{artifact_id!r}. A different artifact prices the same function differently "
                "(a sharper or flatter p(h) changes the anchor's scale), so re-tune lambda and "
                "read the frontier rather than a single point."
            )
        if self.full_weight:
            start = (" (for Qwen3-0.6B, the research code started the search in 50,000-100,000, "
                     "on the research corpus at the paper's point)"
                     if self.model_id == _QWEN3_0_6B else "")
            notes.append(
                "full-weight anchoring is unvalidated for this recipe: its lambda was "
                "calibrated for LoRA, and every measurement behind it is LoRA. Re-calibrate "
                f"lambda for full weight{start} by docs/model-integration-cookbook.md §5, and "
                "check held-out domain perplexity, not only general-text perplexity."
            )
        return notes
