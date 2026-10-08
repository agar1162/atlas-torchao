import torch
from torch import nn
from torch.func import functional_call
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from torchao.quantization import quantize_
from accelerate.hooks import remove_hook_from_module

from utils import load_model
from atlas_quant import AtlasQConfig, skip_lm_head   # reuse your quantization code

MODEL_PATH = "/rscratch/agarcia/TinyLlama-1.1B-Chat-v1.0"


class WeightsAsInputs(nn.Module):
    """Same module, but FP8 buffers are passed in as arguments instead of baked in."""
    def __init__(self, mod, names):
        super().__init__()
        self.mod, self.names = mod, names

    def forward(self, x, *weights):
        return functional_call(self.mod, dict(zip(self.names, weights)), (x,))


# 1. load + quantize (this was missing)
model = load_model(MODEL_PATH, seqlen=2048)
with torch.no_grad():
    quantize_(model, AtlasQConfig(), filter_fn=skip_lm_head)
model.eval()

# 2. remove accelerate's device hooks, then move the MLP to CPU
remove_hook_from_module(model, recurse=True)
mlp = model.model.layers[0].mlp.cpu()
x = torch.randn(1, 8, 2048, dtype=torch.bfloat16)          # CPU

# 3. FP8 weights become inputs
bufs = dict(mlp.named_buffers())
fp8_names = [n for n, b in bufs.items() if b.dtype == torch.float8_e4m3fn]
fp8_weights = [bufs[n] for n in fp8_names]
assert fp8_names, "no FP8 buffers found -- is the MLP quantized?"
print("FP8 weights passed as inputs:", fp8_names)

wrapped = WeightsAsInputs(mlp, fp8_names).eval()

# 4. Torch dialect
m_torch = fx.export_and_import(wrapped, x, *fp8_weights, output_type=OutputType.TORCH)
open("mlp_layer0_torch.mlir", "w").write(str(m_torch))
print("wrote mlp_layer0_torch.mlir")

# 5. linalg
m_linalg = fx.export_and_import(wrapped, x, *fp8_weights, output_type=OutputType.LINALG_ON_TENSORS)
open("mlp_layer0_linalg.mlir", "w").write(str(m_linalg))
print("wrote mlp_layer0_linalg.mlir")