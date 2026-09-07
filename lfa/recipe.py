"""Recipes: a published Layerwise Function Anchoring (LFA) operating point, in one file.

An LFA run has more knobs than a reader can be expected to set independently, and the ones that
matter most are *coupled*. ``lambda`` constrains motion inside the rank-``r`` update subspace, so
the same lambda binds far harder at a lower rank; it is also read against an artifact's sharpness,
so it does not survive a change of ``p(h)`` either. A recipe is therefore a joint point -- the
whole set of values that were tuned together and measured together -- rather than a bag of
defaults, and it records *which* rank and *which* artifact it was calibrated at so that a run
departing from either can be told that lambda no longer means what it meant
(:meth:`Recipe.warnings`).

Several settings in the shipped point are easy to lose in a re-implementation and are carried here
deliberately:

* ``epochs`` and ``schedule_horizon_epochs`` come apart. The published dose is checkpoint 15 **of
  a twenty-epoch run**: 50 steps of warmup, then a cosine laid over twenty epochs, with training
  stopped at fifteen (see :class:`lfa.train.TrainConfig`). Both directions matter. A 15-epoch
  horizon would have decayed the learning rate to its floor by that step; a *longer* horizon leaves
  it too high, and the acceptance run measured what that costs -- at a 100-epoch horizon e15 sits
  at a learning rate of 2.9e-4 instead of 1.2e-4 and lands at domain perplexity 9.69 rather than
  8.76.
* ``checkpoint_mode="all"`` with ``checkpoint_every=5``, because dose is a preservation-quality
  dial rather than a converged endpoint -- e15 is the perplexity-optimal shipped point and e20 the
  judge-optimal one -- and e20 is the *end* of this same run, so reaching it means raising
  ``epochs`` to 20 with the horizon left at 20. What ``checkpoint_every=5`` buys is that the
  intermediate doses of a run stay comparable with each other, since e5/e10/e15 are on disk rather
  than only the endpoint.
* ``keep_short_whole=True`` deliberately differs from the research code, where a document shorter
  than the epoch's random chunk offset is dropped for that epoch. It is a *frame* field: a run
  under either setting is not comparable with a run under the other, and the paper's perplexity
  points were measured under ``False`` -- pass ``keep_short_whole=False`` to
  :meth:`Recipe.to_train_config` to reproduce them.
* ``val_fraction=0.1``: the paper's runs held a tenth of the *documents* out (shuffled under seed
  42) and never trained on them, so the domain perplexity reported for a stage is a held-out
  measurement rather than a fit. Set it to ``0.0`` to train on everything -- and then read the
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

from .train import TrainConfig

__all__ = ["Recipe", "BUNDLED_DIR"]

#: Where the bundled recipes live -- inside the package, so a wheel carries them.
BUNDLED_DIR = Path(__file__).parent / "recipes"


@dataclass
class Recipe:
    """One tuned LFA operating point: what to train, how hard to anchor, and at what it was tuned.

    Args:
        name: The recipe's own name; ``Recipe.load(name)`` finds ``<name>.yaml`` when bundled.
        model_id: The teacher/student model this point was tuned on.
        artifact: The p(h) artifact id (or a path) the lambdas are calibrated against.
        stage2_lambda_multiplier: What :meth:`to_train_config` multiplies lambda by from stage 2
            on. A later stage anchors a model that already carries a domain, and the measured
            operating point for that is a *harder* anchor; it is a level, not a per-stage
            compounding factor.
        calibrated_rank: The LoRA rank the lambdas were tuned at.
        calibrated_artifact: The artifact id the lambdas were tuned against.

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
    lambda_qkv: float = 100_000.0
    lambda_mlp: float = 100_000.0
    mu: float = 0.05
    mu_end_ratio: float = 1.0
    anchor_end_ratio: float = 0.1
    anchor_schedule: str = "cosine"
    n_anchor_samples: int = 16

    # -- dose and schedule
    epochs: int = 15
    schedule_horizon_epochs: int | None = 20
    checkpoint_mode: str = "all"
    checkpoint_every: int = 5

    # -- optimization
    learning_rate: float = 3e-4
    batch_size: int = 6
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    sequence_length: int = 512
    seed: int = 42
    keep_short_whole: bool = True
    val_fraction: float = 0.1

    # -- what the point was calibrated at
    stage2_lambda_multiplier: float = 3.0
    calibrated_rank: int = 32
    calibrated_artifact: str = "qwen3-0.6b-gmm1543k-int8"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.schedule_horizon_epochs is not None and self.schedule_horizon_epochs < self.epochs:
            raise ValueError(
                f"schedule_horizon_epochs ({self.schedule_horizon_epochs}) is shorter than epochs "
                f"({self.epochs}): the learning-rate schedule is laid over the horizon and "
                "training stops at epochs, so a shorter horizon would end the run past the "
                "schedule's end"
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

    # ------------------------------------------------------------------ loading and saving

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Recipe":
        """Read a recipe: a bundled name (``"qwen3-0.6b"``) or a path to a YAML file.

        A value with a ``.yaml``/``.yml`` suffix, or one that exists on disk, is read as a path;
        anything else is looked up in :data:`BUNDLED_DIR`.

        Raises:
            FileNotFoundError: No such bundled recipe (the message lists the ones there are) or no
                such file.
            ValueError: The file carries a field this class does not have, or a value that fails
                validation.
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

        data = yaml.safe_load(path.read_text()) or {}
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
            keep_short_whole: Override the recipe's corpus-chunking setting. ``None`` uses the
                recipe's own value; ``False`` reproduces the historical loader, in which a
                document shorter than the epoch's random chunk offset is dropped for that epoch.
        """
        if stage < 1:
            raise ValueError(f"stage must be at least 1, got {stage}")

        lambda_scale = self.stage2_lambda_multiplier if stage >= 2 else 1.0
        return TrainConfig(
            model_id=self.model_id,
            lambda_qkv=self.lambda_qkv * lambda_scale,
            lambda_mlp=self.lambda_mlp * lambda_scale,
            mu=self.mu,
            mu_end_ratio=self.mu_end_ratio,
            n_anchor_samples=self.n_anchor_samples,
            anchor_end_ratio=self.anchor_end_ratio,
            anchor_schedule=self.anchor_schedule,
            artifact_path=str(artifact_path),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_epochs=self.epochs,
            sequence_length=self.sequence_length,
            schedule_horizon_epochs=self.schedule_horizon_epochs,
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

    def warnings(self, rank: int, artifact_id: str) -> list[str]:
        """What is off-calibration about running this recipe at ``rank`` on ``artifact_id``.

        Empty when the run sits at the point the lambdas were tuned at. Each string says what
        moved and which way to re-tune; none of them is a refusal -- an off-calibration run is
        allowed, it just is not the measured operating point.
        """
        notes: list[str] = []
        # The two site families can carry different lambdas; quoting one of them as "the lambda"
        # would misreport the recipe whenever they differ.
        quoted = (f"{self.lambda_qkv:g}" if self.lambda_qkv == self.lambda_mlp
                  else f"qkv {self.lambda_qkv:g}, mlp {self.lambda_mlp:g}")
        if rank != self.calibrated_rank:
            notes.append(
                f"lambda is coupled to LoRA rank: this recipe's lambda ({quoted}) was "
                f"calibrated at rank {self.calibrated_rank} and you are running rank {rank}. "
                "Lambda constrains motion inside the rank-r update subspace, so the same value "
                "binds harder at a lower rank -- re-tune it rather than porting it (lower rank "
                "=> lower lambda; rank 16 measured at roughly 2e4-5e4 on this corpus), and "
                "diagnose against held-out domain perplexity, since over-anchoring makes "
                "general-text perplexity look its best."
            )
        if artifact_id != self.calibrated_artifact:
            notes.append(
                f"lambda is coupled to the p(h) artifact: this recipe's lambda was calibrated "
                f"against {self.calibrated_artifact!r} and you are anchoring against "
                f"{artifact_id!r}. A different artifact prices the same function differently "
                "(a sharper or flatter p(h) changes the anchor's scale), so re-tune lambda and "
                "read the frontier rather than a single point."
            )
        if self.full_weight:
            notes.append(
                "full-weight anchoring is unvalidated on this model in the LFA paper (every "
                "published result is LoRA): calibrate lambda in 50,000-100,000 and check "
                "held-out domain perplexity, not only general-text perplexity."
            )
        return notes
