"""The supplement cache: where a supplement is found, and which writer it must come from."""
import json

import pytest

import lfa.supplements as supplements
from conftest import make_corpus, tiny_recipe


def _fake_writer(calls):
    def write(model_id, documents, out_path, *, domain_description, options, corpus_sha256,
              device=None, **_):
        calls.append((model_id, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"prompt": "q", "response": "a"}) + "\n")
        manifest = {"corpus_sha256": corpus_sha256, "writer_id": model_id,
                    "writer_sha256": f"sha-of-{model_id}", "template_sha256": "t" * 64,
                    "domain_description": domain_description}
        (out_path.parent / (out_path.name + ".manifest.json")).write_text(json.dumps(manifest))
        return manifest
    return write


@pytest.fixture
def patched(monkeypatch):
    calls = []
    monkeypatch.setattr(supplements, "write_supplement", _fake_writer(calls))
    monkeypatch.setattr(supplements, "checkpoint_sha256", lambda m: f"sha-of-{m}")
    monkeypatch.setattr(supplements, "template_sha256", lambda: "t" * 64)
    return calls


def test_prepared_beside_the_corpus_with_the_model(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    path = supplements.prepare_supplement(corpus, "base", recipe=tiny_recipe(base_dir,
                                                                             val_fraction=0.1))
    assert path.parent.parent == (tmp_path / "my_domain.supplement").resolve()
    assert path.name == "supplement.jsonl" and len(patched) == 1


def test_the_same_writer_finds_it_beside_the_corpus(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    recipe = tiny_recipe(base_dir, val_fraction=0.1)
    prepared = supplements.prepare_supplement(corpus, "base", recipe=recipe)
    found, _ = supplements.supplement_for(
        corpus, "base", recipe, write_root=tmp_path / "ws" / "supplements",
        search_roots=[tmp_path / "ws" / "supplements", supplements.beside_corpus(corpus)],
        device="cpu")
    assert found == prepared and len(patched) == 1


def test_a_different_writer_writes_its_own(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    recipe = tiny_recipe(base_dir, val_fraction=0.1)
    supplements.prepare_supplement(corpus, "base", recipe=recipe)
    found, manifest = supplements.supplement_for(
        corpus, "stage1_fused", recipe, write_root=tmp_path / "ws" / "supplements",
        search_roots=[tmp_path / "ws" / "supplements", supplements.beside_corpus(corpus)],
        device="cpu")
    assert found.is_relative_to(tmp_path / "ws" / "supplements")
    assert manifest["writer_id"] == "stage1_fused" and len(patched) == 2


def test_a_different_domain_description_writes_again(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    recipe = tiny_recipe(base_dir, val_fraction=0.1)
    supplements.prepare_supplement(corpus, "base", recipe=recipe)
    supplements.prepare_supplement(corpus, "base", recipe=recipe,
                                   domain_description="Victorian cookery")
    assert len(patched) == 2


def test_no_recipe_anywhere_is_refused(tmp_path, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    with pytest.raises(ValueError, match="--recipe"):
        supplements.prepare_supplement(corpus, "nobody/nothing")


def test_the_default_description_is_the_directory_name(tmp_path):
    assert supplements.domain_description_for(tmp_path / "victorian_cookery-1861") == \
        "victorian cookery 1861"
