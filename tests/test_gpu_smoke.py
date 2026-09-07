"""Three things that only a GPU can answer, marked ``gpu`` and deselected by default.

Everything else in this suite runs on the CPU, which is what keeps it fast and portable -- but
three of this package's decisions are *about* the accelerator and are invisible there:

* a model is loaded in bfloat16 on an accelerator and float32 on the CPU, and a run records which
  it used, because an export merges an adapter in the dtype its stage trained in;
* a workspace stage pins one device rather than sharding across cards, and writes that device and
  dtype into its history entry;
* :class:`lfa.artifact.fit.TorchGMM` draws from a private ``torch.Generator`` created **on its own
  device** -- a CUDA generator for a CUDA fit -- which is a different code path from the CPU one
  and is exercised nowhere else.

They are smoke tests: they say the GPU paths run and produce what the CPU paths produce, not that
any number is right. Run them with::

    CUDA_VISIBLE_DEVICES=0 pytest tests/test_gpu_smoke.py -m gpu -q

A machine with no CUDA skips them (``tests/conftest.py::pytest_runtest_setup``) rather than
failing inside torch.
"""

from __future__ import annotations

import json

import pytest
import torch

from lfa.artifact.fit import TorchGMM
from lfa.models import load_teacher
from lfa.workspace import Workspace

from conftest import tiny_recipe

pytestmark = pytest.mark.gpu

DEVICE = "cuda:0"


def test_a_model_loads_on_the_card_in_bfloat16(tmp_path, base_dir):
    """``load_teacher`` puts every parameter on the requested card, in the requested dtype."""
    model = load_teacher(str(base_dir), device=DEVICE, dtype=torch.bfloat16)
    try:
        devices = {parameter.device.type for parameter in model.parameters()}
        dtypes = {parameter.dtype for parameter in model.parameters()}

        assert devices == {"cuda"}
        assert dtypes == {torch.bfloat16}
        assert next(model.parameters()).device.index == 0
    finally:
        del model
        torch.cuda.empty_cache()


def test_a_workspace_stage_trains_on_the_card_and_records_it(tmp_path, base_dir, tiny_artifact,
                                                             corpus_a):
    """One whole stage on cuda:0: it trains, and the history says where and in what precision.

    The dtype is not decoration. :meth:`lfa.Workspace.fuse` merges the stage's adapter in the
    dtype recorded here, so a stage that trained in bfloat16 and exported in float32 would be a
    silent precision change between the model that was measured and the model that ships.
    """
    _, artifact_path = tiny_artifact
    workspace = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(artifact_path))

    entry = workspace.train(corpus_a, recipe=tiny_recipe(base_dir), device=DEVICE)

    assert entry["dtype"] == "bfloat16"
    assert entry["device"] == DEVICE
    assert (tmp_path / "ws" / "runs" / "stage1" / "final_model" / "adapter_config.json").is_file()
    assert json.loads((tmp_path / "ws" / "history.json").read_text())[0]["dtype"] == "bfloat16"


def test_the_mixture_fits_on_cuda_through_its_own_device_generator():
    """``TorchGMM`` on CUDA: the private generator is a CUDA one, and the fit still finds the blobs.

    Three well-separated blobs are the cheapest fit whose answer is known in advance, so a failure
    here is the device path rather than the mixture. The CPU version of this assertion lives in
    ``test_artifact_build.py::test_torch_gmm_recovers_blob_means``; what is new is that every draw
    goes through a generator created on the fitting device.
    """
    generator = torch.Generator().manual_seed(0)
    centres = torch.tensor([[-8.0, 0.0], [8.0, 0.0], [0.0, 9.0]])
    points = torch.cat([c + 0.2 * torch.randn(400, 2, generator=generator) for c in centres])

    gmm = TorchGMM(n_components=3, covariance_type="diag", random_state=0, device=DEVICE)
    gmm.fit(points.to(DEVICE))

    assert gmm._generator.device.type == "cuda"
    assert gmm.means_.device.type == "cuda"
    assert torch.allclose(gmm.weights_.sum().cpu(), torch.tensor(1.0))
    found = gmm.means_.cpu()
    for centre in centres:
        assert (found - centre).norm(dim=1).min() < 0.1
