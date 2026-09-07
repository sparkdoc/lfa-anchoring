# lfa-anchoring

Layerwise Function Anchoring (LFA): preserve sub-module functions on sampled hidden
states while fine-tuning an LLM.

Companion package for the LFA paper. Install:

```bash
pip install -e '.[dev]'
```

Tested with the versions in `constraints-tested.txt`.

Then run the tests:

```bash
pytest -q
```

Two suites are opt-in, because they need a GPU and the research checkout this package was ported
from:

```bash
pytest tests/equivalence -m equivalence -q   # the sampler, the losses and the loader, against the research code
pytest tests/acceptance -m acceptance -q -s  # one full training run, against a matched the research code run
```

The second is an *equivalence* run: it trains the bundled recipe end to end and compares the
checkpoint with one an the research code run of the identical configuration produced, re-measured beside
it on the same instrument — domain perplexity, WikiText-2 drift, and the per-epoch content curve,
each within a tolerance. See `tests/acceptance/README.md`; the tolerances themselves are described
in `tests/acceptance/expected.json`.

Licensed under Apache-2.0 (see `LICENSE`).
