import lfa
def test_version(): assert lfa.__version__ == "0.2.0"
def test_tiny_model_forward(tiny_model):
    model, tok = tiny_model
    ids = tok("hello", return_tensors="pt")["input_ids"]
    assert model(ids).logits.shape[-1] == 256
