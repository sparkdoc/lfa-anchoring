"""The LFA p(h) artifact: its on-disk schema, and the pipeline that builds, reads and writes it.

:mod:`~lfa.artifact.schema` defines the format; :mod:`~lfa.artifact.collect` accumulates a model's
hidden-state statistics over a seed corpus; :mod:`~lfa.artifact.fit` turns one site's statistics
into an entry (PCA basis, then a GMM head in it); :mod:`~lfa.artifact.build` runs the three
end to end and saves the file the anchor samples from; :mod:`~lfa.artifact.extend` adds a later
domain to a finished artifact without its original corpus; :mod:`~lfa.artifact.store` keeps
self-generated artifacts so they are built once per model and frame.
"""

from .build import build_artifact
from .collect import SiteStats, collect_hidden_states
from .extend import extend_artifact, fit_domain_gmm
from .fit import TorchGMM, fit_site
from .schema import (
    EMBEDDING_LOOKUP_KEY,
    LM_HEAD_SITE,
    META_KEY,
    SITES,
    ArtifactModelMismatch,
    ForeignArtifact,
    load_artifact,
    make_meta,
    parse_site_key,
    require_own_artifact,
    save_artifact,
    site_key,
    validate_against_model,
)
from .store import (
    STORE_ENV,
    StoreLocked,
    entry_dir,
    list_store,
    obtain_self_generated,
    store_root,
)

__all__ = [
    "build_artifact",
    "extend_artifact",
    "obtain_self_generated",
    "list_store",
    "store_root",
    "entry_dir",
    "StoreLocked",
    "STORE_ENV",
    "fit_domain_gmm",
    "collect_hidden_states",
    "fit_site",
    "SiteStats",
    "TorchGMM",
    "SITES",
    "LM_HEAD_SITE",
    "META_KEY",
    "EMBEDDING_LOOKUP_KEY",
    "site_key",
    "parse_site_key",
    "make_meta",
    "load_artifact",
    "save_artifact",
    "ArtifactModelMismatch",
    "ForeignArtifact",
    "require_own_artifact",
    "validate_against_model",
]
