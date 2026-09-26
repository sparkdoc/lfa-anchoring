"""Self-generation: the model supplies its own inputs.

Three things a user otherwise has to bring from outside -- the seed corpus p(h) is estimated on,
the question-and-answer supplement that makes a domain reachable, and a fresh p(h) for each stage
of a chain -- can be written by the model itself. The research record behind this
(mr-fusion claims C12, C14, C15) is one model (Qwen3-0.6B) and one seed; every number the
package documentation quotes about it carries that scope.
"""

# from .artifact_corpus import SelfGenOptions, write_artifact_corpus  # noqa: F401  (Task 4)
# from .supplement import write_supplement  # noqa: F401  (Task 9)

__all__ = []
