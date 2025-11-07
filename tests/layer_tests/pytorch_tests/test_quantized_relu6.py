import numpy as np
import pytest
import torch
import platform
from pytorch_layer_test_class import PytorchLayerTest
from openvino.runtime import Core, Extension

class quantized_relu6(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor):
        quantized_tensor = torch.quantize_per_tensor(input_tensor, self.scale, self.zero_point, self.dtype)
        q_relu6 = torch.ops.quantized.relu6(quantized_tensor, self.scale, self.zero_point)
        dequantized_tensor = torch.dequantize(q_relu6)
        return dequantized_tensor

def quantized_relu6_converter(decoder):
    input_tensor = decoder.get_input(0)
    output_scale = decoder.get_input(1)
    output_zero_point = decoder.get_input(2)

    try:
        from openvino.runtime import opset13 as opset
    except ImportError:
        from openvino import opset13 as opset

    if hasattr(input_tensor, 'get_scale'):
        input_scale = input_tensor.get_scale()
        input_zero_point = input_tensor.get_zero_point()
        dequantized = opset.dequantize_linear(input_tensor, input_scale, input_zero_point)
    else:
        dequantized = input_tensor

    zero_const = opset.constant(0.0, dtype='f32')
    six_const = opset.constant(6.0, dtype='f32')
    clamped = opset.clamp(dequantized, zero_const, six_const)
    quantized_output = opset.quantize_linear(clamped, output_scale, output_zero_point)
    return quantized_output

def register_quantized_relu6():
    try:
        core = Core()
        ext = Extension(quantized_relu6_converter)
        core.add_extension(ext)
        print("[INFO] quantized::relu6 extension registered successfully.")
    except Exception as e:
        print(f"[WARN] Could not register quantized::relu6 extension: {e}")

def init_quantized_relu6_support():
    register_quantized_relu6()

init_quantized_relu6_support()

class TestQuantizedReLU6(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(10.0 * self.rng.random([10, 10], dtype=np.float32) - 3.0, 4),)

    @pytest.mark.parametrize("scale", [
        1.0, 0.21, 0.62, 0.9999
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8,
        torch.qint8
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_relu6(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8:
            zero_point = max(0, min(255, zero_point))
        self._test(
            quantized_relu6(scale, zero_point, dtype),
            None,
            ["quantized::relu6"],
            ie_device,
            precision,
            ir_version,
            quantized_ops=True
        )
