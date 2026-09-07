import torch
from lfa.quantize import quantize_blockwise, dequantize_blockwise, quantize_params, dequantize_params, is_quantized_field

def test_roundtrip_error_small():
    torch.manual_seed(0); t = torch.randn(1024, 300, dtype=torch.float16)
    q = quantize_blockwise(t); assert is_quantized_field(q) and q["q8"].dtype == torch.int8
    back = dequantize_blockwise(q)
    assert back.dtype == t.dtype and back.shape == t.shape
    assert ((back.float() - t.float()).norm() / t.float().norm()) < 0.02

def test_params_roundtrip_skips_meta():
    params = {"0_pre_mlp": {"mean": torch.zeros(8), "pca_components": torch.randn(8, 4)}, "__meta__": {"model_id": "x"}}
    q = quantize_params(params); assert is_quantized_field(q["0_pre_mlp"]["pca_components"]) and q["__meta__"] == {"model_id": "x"}
    d = dequantize_params(q); assert torch.is_tensor(d["0_pre_mlp"]["pca_components"]) and d["__meta__"] == {"model_id": "x"}
