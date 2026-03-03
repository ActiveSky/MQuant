# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseQuantizer


class NonUniformQuantizer(BaseQuantizer):
    """Non-uniform quantizer backed by a learned LUT."""

    def __init__(self, bit_type, observer, module_type):
        super(NonUniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.lut = None
        self.bit = None

    def _flatten_per_channel(self, v):
        """Flatten tensor so each row corresponds to one channel (for weights)."""
        if self.module_type in ["conv_weight", "linear_weight"]:
            return v.reshape(v.shape[0], -1)
        return v.reshape(1, -1)

    def update_quantization_params(self, inputs, g=None, device=None, channel_wise=None):
        """Compute LUT with weighted Lloyd (g is squared loss from calibration)."""
        from fake_quant.quant_utils import non_uniform_lut, non_uniform_lut_parallel

        if g is None:
            g = torch.ones_like(inputs)

        if channel_wise is None:
            channel_wise = self.module_type in ["conv_weight", "linear_weight"]

        device = device or inputs.device
        inputs_flat = self._flatten_per_channel(inputs)
        g_flat = self._flatten_per_channel(g)

        if channel_wise:
            xs_list = [inputs_flat[i] for i in range(inputs_flat.shape[0])]
            gs_list = [g_flat[i] for i in range(g_flat.shape[0])]
            self.lut = non_uniform_lut_parallel(
                xs_list, gs_list, self.bit_type.bits, device=device
            )
        else:
            self.lut = non_uniform_lut(
                inputs_flat.reshape(-1), g_flat.reshape(-1), self.bit_type.bits, device=device
            ).to(device)

        self.bit = self.bit_type.bits

    def quant(self, inputs, lut=None):
        lut = self.lut if lut is None else lut
        if lut is None:
            raise ValueError("LUT is not initialized; call update_quantization_params first.")

        lut = lut.to(inputs.device)

        if lut.dim() == 2 and self.module_type in ["conv_weight", "linear_weight"]:
            outputs = torch.empty_like(inputs, dtype=torch.int64)
            inputs_view = inputs.reshape(inputs.shape[0], -1)
            outputs_view = outputs.reshape(outputs.shape[0], -1)
            for i in range(inputs_view.shape[0]):
                boundaries = (lut[i, :-1] + lut[i, 1:]) / 2
                idx = torch.bucketize(inputs_view[i], boundaries)
                if self.bit_type.signed:
                    idx = idx + self.bit_type.lower_bound
                outputs_view[i] = idx.clamp(
                    self.bit_type.lower_bound, self.bit_type.upper_bound
                )
            return outputs

        boundaries = (lut[:-1] + lut[1:]) / 2
        outputs = torch.bucketize(inputs, boundaries)
        if self.bit_type.signed:
            outputs = outputs + self.bit_type.lower_bound
        outputs = outputs.clamp(self.bit_type.lower_bound, self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, lut=None):
        lut = self.lut if lut is None else lut
        if lut is None:
            raise ValueError("LUT is not initialized; call update_quantization_params first.")

        lut = lut.to(inputs.device)

        if lut.dim() == 2 and self.module_type in ["conv_weight", "linear_weight"]:
            outputs = torch.empty_like(inputs, dtype=lut.dtype)
            inputs_view = inputs.reshape(inputs.shape[0], -1).long()
            outputs_view = outputs.reshape(outputs.shape[0], -1)
            for i in range(inputs_view.shape[0]):
                idx = inputs_view[i]
                if self.bit_type.signed:
                    idx = idx - self.bit_type.lower_bound
                idx = idx.clamp(0, lut.shape[1] - 1)
                outputs_view[i] = lut[i][idx]
            return outputs

        idx = inputs.long()
        if self.bit_type.signed:
            idx = idx - self.bit_type.lower_bound
        idx = idx.clamp(0, lut.numel() - 1)
        outputs = lut[idx]
        return outputs


