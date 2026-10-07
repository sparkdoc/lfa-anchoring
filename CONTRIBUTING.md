# Contributing

Bug reports and pull requests are welcome. For a bug, open an issue with the command you ran,
the tail of its output, and your GPU, Python and torch versions (the issue template asks for
these).

## Setting up

```bash
git clone https://github.com/sparkdoc/lfa-anchoring
cd lfa-anchoring
pip install -c constraints-tested.txt -e '.[dev,html]'
```

`[html]` belongs in a development install: without it one prepare-domain test skips instead of
running. On a machine with no CUDA card, install the CPU build of torch first, as CI does, or pip
fetches the CUDA build (several GB):

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

## The test tiers

| tier | command | needs | time |
|---|---|---|---|
| default | `pytest -q` | nothing: no GPU, no corpus, no network | a few minutes |
| slow | `pytest -m slow -q` | the network (tokenizers and data from the Hugging Face Hub) | seconds once cached |
| GPU | `pytest -m gpu -q` | a CUDA card, the network | ~18 min on an RTX 3090, models cached |
| notebook | `pytest tests/test_notebook.py -m notebook -q` | a CUDA card, the network | ~1 h 30 min on an RTX 3090, models cached |

The default tier runs in CI on every push to `main` and every pull request, on Python 3.11, 3.12
and 3.13, CPU only (`.github/workflows/tests.yml`), and needs no network: a test that does must
carry the `slow`, `gpu` or `notebook` marker (`tests/conftest.py` sets `HF_HUB_OFFLINE` for the
rest). The other tiers run by hand. A change that touches training, evaluation or the artifact
should pass the GPU tier, and a change that alters what a notebook prints needs the notebooks
re-recorded and their prose checked against the new outputs.

## Conventions

- **Docs change with the code.** A change to behaviour, a flag or a printed message updates the
  docs that describe it in the same pull request.
- **Numbers in the docs are measured.** A figure in the docs or the notebooks comes from a run,
  with its frame stated (model, recipe, corpus, seed count). Don't round one into a general
  claim.
- **Before 1.0, formats change outright**, with no compatibility layer for the old one. An
  artifact layout change bumps `ARTIFACT_FORMAT` and says so in the changelog
  ([RELEASING.md](RELEASING.md)).
