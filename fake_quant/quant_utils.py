import math
import transformers
import torch
import torch.nn.functional as F
import os
from fake_quant import utils
from fake_quant import hadamard_utils
import fast_hadamard_transform
from collections import OrderedDict
from fake_quant.observer import build_observer
from fake_quant.quantizer import build_quantizer
from fake_quant.bit_type import BIT_TYPE_DICT
from functools import partial
from datasets import load_dataset


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def nu_quant(x, lut):
    # This function quantizes x using the provided LUT. It assumes that the LUT is sorted in ascending order.
    # The quantized value for each element in x is the index of the closest value in the LUT.
    if lut.dim() == 2:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).to(lut.device)
        out = torch.empty_like(x_2d, dtype=torch.long)
        for i in range(x_2d.shape[0]):
            boundaries = (lut[i][:-1] + lut[i][1:]) / 2
            out[i] = torch.bucketize(x_2d[i], boundaries)
        return out.reshape(orig_shape)
    else:
        x = x.to(lut.device)
        boundaries = (lut[:-1] + lut[1:]) / 2
        q = torch.bucketize(x, boundaries)
        return q

def nu_dequant(q, lut):
    # This function dequantizes q using the provided LUT. It simply replaces each quantized index with the corresponding value in the LUT.
    if lut.dim() == 2:
        orig_shape = q.shape
        q_2d = q.reshape(-1, q.shape[-1]).to(lut.device)
        out = torch.empty_like(q_2d, dtype=lut.dtype)
        for i in range(q_2d.shape[0]):
            out[i] = lut[i][q_2d[i]]
        return out.reshape(orig_shape)
    else:
        q = q.to(lut.device)
        return lut[q]

def nu_quant_dequant(x, lut):
    if lut.dim() == 2:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).to(lut.device)
        out = torch.empty_like(x_2d)
        for i in range(x_2d.shape[0]):
            boundaries = (lut[i][:-1] + lut[i][1:]) / 2
            q = torch.bucketize(x_2d[i], boundaries)
            out[i] = lut[i][q]
        return out.reshape(orig_shape)
    else:
        x = x.to(lut.device)
        boundaries = (lut[:-1] + lut[1:]) / 2
        q = torch.bucketize(x, boundaries)
        return lut[q]

def two_compl(x, bits: int):
    return torch.where(x < 0, 2**bits + x, x)


# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, "The tensor to be unpacked should be stored in uint8"

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0F).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xF0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self, act_per_tensor=False):
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("lut", torch.zeros(1))
        self.bits = 16
        self.act_per_tensor = act_per_tensor
        self.static = False

    def free(self):
        self.zero = None
        self.scale = None
        self.lut = None

    def forward(self, x):
        if self.static:
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
                return x
            elif self.quant:
                return self.quantizer(x)
            else:
                return x
        else:
            x_dtype = x.dtype
            if self.bits == 16:
                return x
            elif self.nuq:
                return nu_quant_dequant(x, self.lut).to(x_dtype)
            elif self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.nuq:
            return nu_quant(x, self.lut)
        elif self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self,
        bits,
        groupsize=-1,
        nuq=False,
        sym=False,
        clip_ratio=1.0,
        act_per_tensor=False,
        static=False,
        observer_type="minmax",
        calibration_mode="layer_wise",
    ):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.nuq = nuq
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.act_per_tensor = act_per_tensor
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"
        self.static = static
        if self.static:
            module_a_type = "activation"
            bit_type_a = BIT_TYPE_DICT[f"int{bits}"]
            if observer_type == "percentile":
                print("Using percentile observer for activations")
            self.observer = build_observer(
                observer_type,
                module_a_type,
                bit_type_a,
                calibration_mode,
            )
            quantizer_type = "non_uniform" if self.nuq else "uniform"
            self.quantizer = build_quantizer(
                quantizer_type, bit_type_a, self.observer, module_a_type
            )
            self.calibrate = False
            self.last_calibrate = False
            self.quant = False

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x, g=None):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape
        if self.nuq:
            if self.act_per_tensor:
                flat_x = x.flatten()
                flat_g = g.flatten() if g is not None else torch.ones_like(flat_x)
                self.lut = non_uniform_lut(flat_x, flat_g, self.bits, device=dev).to(dev)
            elif self.groupsize > 0:
                # group-wise per-token quantization
                reshaped_x = x.reshape(
                    -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
                )
                reshaped_g = (
                    g.reshape(-1, g.shape[-2], g.shape[-1] // self.groupsize, self.groupsize)
                    if g is not None
                    else None
                )
                luts = []
                for i in range(reshaped_x.shape[0]):
                    for j in range(reshaped_x.shape[1]):
                        for k in range(reshaped_x.shape[2]):
                            gi = (
                                reshaped_g[i, j, k].flatten()
                                if reshaped_g is not None
                                else torch.ones_like(reshaped_x[i, j, k].flatten())
                            )
                            luts.append(
                                non_uniform_lut(
                                    reshaped_x[i, j, k].flatten(),
                                    gi,
                                    self.bits,
                                    device=dev,
                                )
                            )
                self.lut = torch.stack(luts, dim=0).to(dev)  # (num_groups, 2^bits)
            else:
                # per-token: each token gets its own LUT
                reshaped_x = x.reshape((-1, x.shape[-1]))
                luts = []
                for i in range(reshaped_x.shape[0]):
                    gi = g.reshape(-1, g.shape[-1])[i] if g is not None else torch.ones_like(reshaped_x[i])
                    luts.append(non_uniform_lut(reshaped_x[i], gi, self.bits, device=dev))
                self.lut = torch.stack(luts, dim=0).to(dev)  # (num_tokens, 2^bits)
            return
        elif self.act_per_tensor:
            tmp = torch.tensor(0).to(x)
            xmin = torch.minimum(x.min(), tmp) * self.clip_ratio
            xmax = torch.maximum(x.max(), tmp) * self.clip_ratio
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                if xmax == 0:
                    self.scale = 1
                else:
                    self.scale = xmax / self.maxq
                self.zero = torch.zeros_like(self.scale)
            else:
                if xmin == 0:
                    xmin = -1
                if xmax == 0:
                    xmax = 1
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)
        else:
            if self.groupsize > 0:
                # group-wise per-token quantization
                self.find_params_per_token_groupwise(x)
                utils.cleanup_memory(verbos=False)
                return
            reshaped_x = x.reshape((-1, x.shape[-1]))

            tmp = torch.zeros(reshaped_x.shape[0], device=dev)
            xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
            xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                self.scale = (
                    (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
                )
                self.scale[tmp] = 1
                self.scale = self.scale.reshape(init_shape)
                self.zero = torch.zeros_like(self.scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)

                self.scale = (
                    self.scale.unsqueeze(1)
                    .repeat(1, reshaped_x.shape[-1])
                    .reshape(init_shape)
                )
                self.zero = (
                    self.zero.unsqueeze(1)
                    .repeat(1, reshaped_x.shape[-1])
                    .reshape(init_shape)
                )


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registerd to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear, act_per_tensor=False):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d))
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer(act_per_tensor)
        self.out_quantizer = ActQuantizer(act_per_tensor)
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.split = False
        self.act_outlier = False
        self.act_outlier_ratio = 0.0
        self.act_outlier_metric = "absmax"
        self.act_outlier_min_channels = 1
        self.act_outlier_log = False
        self.act_outlier_log_once = True
        self._act_outlier_logged = False
        self.outlier_layer_name = ""

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def split_weights(self):
        self.L1 = torch.nn.Linear(1, self.module.out_features, bias=False).to(
            self.module.weight.device
        )
        self.L2 = torch.nn.Linear(
            self.module.in_features - 1,
            self.module.out_features,
            bias=True if self.module.bias is not None else False,
        ).to(self.module.weight.device)
        self.L1.weight.data = self.module.weight.data[:, 0:1]
        self.L2.weight.data = self.module.weight.data[:, 1:]
        if self.module.bias is not None:
            self.L2.bias.data = self.module.bias.data

    def configure_act_outlier(
        self,
        enable=False,
        ratio=0.0,
        metric="absmax",
        min_channels=1,
        log_enabled=False,
        log_once=True,
        layer_name="",
    ):
        self.act_outlier = enable
        self.act_outlier_ratio = ratio
        self.act_outlier_metric = metric
        self.act_outlier_min_channels = min_channels
        self.act_outlier_log = log_enabled
        self.act_outlier_log_once = log_once
        self._act_outlier_logged = False
        self.outlier_layer_name = layer_name

    def _get_act_interval_mode(self, x):
        groupsize = getattr(self.quantizer, "groupsize", -1)
        if self.quantizer.act_per_tensor:
            return "per-tensor"
        if groupsize > 0 and x.shape[-1] % groupsize == 0:
            return "per-token-group"
        return "per-token"

    def _log_act_outlier_stats(self, x, outlier_mask):
        if not self.act_outlier_log:
            return
        if self.act_outlier_log_once and self._act_outlier_logged:
            return

        layer_name = self.outlier_layer_name or self.module.__class__.__name__
        mode = self._get_act_interval_mode(x)
        if outlier_mask is None:
            print(
                f"[ActOutlier][{layer_name}] mode={mode} global_ratio=0.000000 interval_mean=0.000000 interval_min=0.000000 interval_max=0.000000"
            )
            self._act_outlier_logged = True
            return

        mask_f = outlier_mask.float()
        groupsize = getattr(self.quantizer, "groupsize", -1)
        if self.quantizer.act_per_tensor:
            interval_ratios = mask_f.reshape(-1).mean().view(1)
        elif groupsize > 0 and x.shape[-1] % groupsize == 0:
            interval_ratios = mask_f.reshape(
                *x.shape[:-1],
                x.shape[-1] // groupsize,
                groupsize,
            ).mean(dim=-1).reshape(-1)
        else:
            interval_ratios = mask_f.reshape(-1, x.shape[-1]).mean(dim=-1)

        print(
            f"[ActOutlier][{layer_name}] mode={mode} global_ratio={mask_f.mean().item():.6f} "
            f"interval_mean={interval_ratios.mean().item():.6f} "
            f"interval_min={interval_ratios.min().item():.6f} "
            f"interval_max={interval_ratios.max().item():.6f}"
        )
        self._act_outlier_logged = True

    def _quantize_input(self, x, x_dtype):
        if self.quantizer.static:
            return self.quantizer(x)
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        return x

    def _get_outlier_k(self, interval_size):
        if interval_size <= 1:
            return 0
        k = int(round(interval_size * self.act_outlier_ratio))
        if self.act_outlier_ratio > 0:
            k = max(k, self.act_outlier_min_channels)
        return min(max(k, 0), interval_size - 1)

    def _build_act_outlier_mask(self, x):
        x_abs = x.detach().abs().float()
        groupsize = getattr(self.quantizer, "groupsize", -1)

        if self.quantizer.act_per_tensor:
            k = self._get_outlier_k(x_abs.numel())
            if k <= 0:
                return None
            mask = torch.zeros_like(x_abs, dtype=torch.bool)
            topk_idx = torch.topk(x_abs.reshape(-1), k=k, largest=True).indices
            mask.reshape(-1)[topk_idx] = True
            return mask

        if groupsize > 0:
            if x_abs.shape[-1] % groupsize != 0:
                return None
            x_group = x_abs.reshape(*x_abs.shape[:-1], x_abs.shape[-1] // groupsize, groupsize)
            k = self._get_outlier_k(groupsize)
            if k <= 0:
                return None
            topk_idx = torch.topk(x_group, k=k, dim=-1, largest=True).indices
            mask_group = torch.zeros_like(x_group, dtype=torch.bool)
            mask_group.scatter_(-1, topk_idx, True)
            return mask_group.reshape_as(x_abs)

        k = self._get_outlier_k(x_abs.shape[-1])
        if k <= 0:
            return None
        x_2d = x_abs.reshape(-1, x_abs.shape[-1])
        topk_idx = torch.topk(x_2d, k=k, dim=-1, largest=True).indices
        mask_2d = torch.zeros_like(x_2d, dtype=torch.bool)
        mask_2d.scatter_(-1, topk_idx, True)
        return mask_2d.reshape_as(x_abs)

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(
                        -1, init_shape[-1] // self.had_dim, self.had_dim
                    ).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim),
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if (
            self.act_outlier
            and isinstance(self.module, torch.nn.Linear)
            and self.quantizer.bits < 16
        ):
            outlier_mask = self._build_act_outlier_mask(x)
            self._log_act_outlier_stats(x, outlier_mask)
            if outlier_mask is not None and torch.any(outlier_mask):
                x_main = x.masked_fill(outlier_mask, 0)
                x_main = self._quantize_input(x_main, x_dtype)
                x_main = self.module(x_main).to(x_dtype)

                x_outlier = torch.where(outlier_mask, x, torch.zeros_like(x))
                x_outlier = torch.nn.functional.linear(
                    x_outlier.float(),
                    self.module.weight.float(),
                    bias=None,
                ).to(x_dtype)
                x = x_main + x_outlier
            else:
                x = self._quantize_input(x, x_dtype)
                x = self.module(x).to(x_dtype)

        elif self.split:
            if self.quantizer.static:
                x[..., 1:] = self.quantizer(x[..., 1:])
            elif self.quantizer.bits < 16:
                self.quantizer.find_params(x[..., 1:])
                x[..., 1:] = self.quantizer(x[..., 1:]).to(x_dtype)
                self.quantizer.free()
            x1 = self.L1.float()(x[..., 0:1].float())
            x2 = self.L2.float()(x[..., 1:].float())
            x = (x1 + x2).to(x_dtype)
        else:
            if self.quantizer.static:
                x = self.quantizer(x)
            elif self.quantizer.bits < 16:
                self.quantizer.find_params(x)
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class ActRotateWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, QMatrix):
        super(ActRotateWrapper, self).__init__()
        self.module = module
        self.register_buffer("q_matrix", QMatrix)
        self.fp32_had = False

    def forward(self, x, y):
        x_dtype = x.dtype

        if self.fp32_had:  # Full Hadamard in FP32
            x = (x.float() @ self.q_matrix).to(x_dtype)
            y.copy_((y.float() @ self.q_matrix).to(y.dtype))
        else:  # Full Hadamard in FP16
            x = x @ self.q_matrix
            y.copy_(y @ self.q_matrix.to(y.dtype))

        x = self.module(x, y).to(x_dtype)
        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.register_buffer("lut", torch.zeros(1))

    def configure(
        self,
        bits,
        perchannel=False,
        nuq=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.nuq = nuq
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x, g=None):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        if self.perchannel:
            x = x.flatten(1)
            g = g.flatten(1) if g is not None else None
        else:
            x = x.flatten().unsqueeze(0)
            g = g.flatten().unsqueeze(0) if g is not None else None
        
        if self.nuq:
            if self.perchannel:
                luts = []
                for i in range(x.shape[0]):
                    gi = g[i] if g is not None else torch.ones_like(x[i])
                    luts.append(non_uniform_lut(x[i], gi, self.bits, device=dev))
                self.lut = torch.stack(luts, dim=0).to(dev)  # (out_channels, 2^bits)
            else:
                self.lut = non_uniform_lut(x, g, self.bits, device=dev).to(dev)  # (2^bits,)
            return

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.nuq:
                if self.perchannel:
                    x_2d = x.reshape(x.shape[0], -1).to(self.lut.device)
                    out_2d = torch.empty_like(x_2d)
                    for i in range(x_2d.shape[0]):
                        boundaries = (self.lut[i][:-1] + self.lut[i][1:]) / 2
                        q = torch.bucketize(x_2d[i], boundaries)
                        out_2d[i] = self.lut[i][q]
                    return out_2d.reshape_as(x).to(x_dtype)
                return nu_quant_dequant(x, self.lut).to(x_dtype)
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        if self.nuq:
            return self.lut.numel() > 1
        return torch.all(self.scale != 0)


@torch.no_grad()
def fuse_internvl(model):
    print("fuse internvl vision model...")
    for layer in model.model.vision_model.encoder.layers:
        # layer.ls1  # (out_c)
        # layer.attn.proj.weight  # shape is out_c, in_c
        layer.attn.proj.weight.data *= layer.ls1.data.view(-1, 1)
        layer.mlp.fc2.weight.data *= layer.ls2.data.view(-1, 1)
        if hasattr(layer.attn.proj, "bias"):
            layer.attn.proj.bias.data *= layer.ls1.data
        if hasattr(layer.mlp.fc2, "bias"):
            layer.mlp.fc2.bias.data *= layer.ls2.data
        layer.ls1[:] = 1  # = torch.ones_like(layer.ls1)
        layer.ls2[:] = 1  #  = torch.ones_like(layer.ls2)


def internvl_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.model.language_model.model,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.model.vision_model.embeddings.patch_embedding = ActQuantWrapper(
            model.model.vision_model.embeddings.patch_embedding, args.act_per_tensor
        )
        add_actquant(model.model.vision_model.encoder, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant_for_mlp1(model.model, args.act_per_tensor)

def qwen2vl_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.model.model,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.model.visual.patch_embed.proj = ActQuantWrapper(
            model.model.visual.patch_embed.proj, args.act_per_tensor
        )
        add_actquant(model.model.visual.blocks, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant(model.model.visual.merger, args.act_per_tensor)


def qwenvl_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.transformer.h,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.transformer.visual.conv1 = ActQuantWrapper(
            model.transformer.visual.conv1, args.act_per_tensor
        )
        add_actquant(model.transformer.visual.transformer, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant(model.transformer.visual.attn_pool, args.act_per_tensor)
        model.transformer.visual.proj_fc = ActQuantWrapper(
            model.transformer.visual.proj_fc, args.act_per_tensor
        )  # 目前代码是直接用@实现的 linear，暂时先不量化
        # model.transformer.visual.proj = ActQuantWrapper(model.transformer.visual.proj) # 目前代码是直接用@实现的 linear，暂时先不量化


def minicpmv_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.llm.model.layers,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.vpm.embeddings.patch_embedding = ActQuantWrapper(
            model.vpm.embeddings.patch_embedding, args.act_per_tensor
        )
        add_actquant(model.vpm.encoder, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant(model.resampler, args.act_per_tensor)


def add_actquant_for_mlp1(
    module,
    act_per_tensor=False,
    name="",
    layers=[
        torch.nn.Linear,
    ],
):
    module.mlp1[1] = ActQuantWrapper(module.mlp1[1], act_per_tensor)
    module.mlp1[3] = ActQuantWrapper(module.mlp1[3], act_per_tensor)


def add_actquant(
    module,
    act_per_tensor=False,
    name="",
    layers=[
        torch.nn.Linear,
    ],
):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp, act_per_tensor))
        if type(tmp) == torch.nn.Sequential:
            replaced = OrderedDict()
            for name, child in tmp.named_children():
                if type(child) in layers:
                    replaced[name] = ActQuantWrapper(child, act_per_tensor)
                else:
                    replaced[name] = child
            setattr(module, attr, torch.nn.Sequential(replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child, act_per_tensor))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(
            child,
            act_per_tensor,
            name + "." + name1 if name != "" else name1,
            [torch.nn.Linear],
        )


def find_qlayers(module, layers=[torch.nn.Linear, ActQuantWrapper], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


@torch.no_grad()
def collect_weight_importance_from_dataset(
    model,
    dataset,
    args,
    dataset_name=None,
    nsamples=None,
):
    """Estimate per-weight importance g from calibration data without GPTQ.

    We use a diagonal Hessian approximation from input second moment:
    for each weight column j, g_j \propto E[x_j^2].
    """
    if dataset is None:
        return {}

    if nsamples is None:
        nsamples = getattr(args, "nsamples", 128)
    if dataset_name is None:
        dataset_name = getattr(args, "dataset_name", None)

    module_root = model.model if hasattr(model, "model") else model
    if not isinstance(module_root, torch.nn.Module):
        raise TypeError(
            "collect_weight_importance_from_dataset expects a torch.nn.Module "
            "or an object exposing `.model` as torch.nn.Module."
        )

    stats = {}
    handles = []

    def _hook(module, inp, out):
        if not hasattr(module, "weight") or module.weight is None:
            return

        x = inp[0].detach().float()
        if isinstance(module, torch.nn.Linear):
            x2d = x.reshape(-1, x.shape[-1])
            sum_sq = (x2d * x2d).sum(dim=0).cpu()
            count = x2d.shape[0]
        elif isinstance(module, torch.nn.Conv2d):
            padding = 0 if module.padding == "valid" else module.padding
            x_unfold = F.unfold(
                x,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=padding,
                stride=module.stride,
            )
            x2d = x_unfold.transpose(1, 2).reshape(-1, x_unfold.shape[1])
            sum_sq = (x2d * x2d).sum(dim=0).cpu()
            count = x2d.shape[0]
        else:
            return

        key = id(module)
        if key not in stats:
            stats[key] = {
                "sum_sq": sum_sq,
                "count": count,
            }
        else:
            stats[key]["sum_sq"] += sum_sq
            stats[key]["count"] += count

    for _, module in module_root.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            handles.append(module.register_forward_hook(_hook))

    lt = len(dataset.data)
    for i in tqdm(range(lt), desc="Collect NUQ g"):
        if i >= nsamples:
            break

        if hasattr(model, "use_custom_prompt") and dataset_name is not None and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            if dataset_name is not None:
                model.generate(message=struct, dataset=dataset_name)
            else:
                model.generate(message=struct)
        except TypeError:
            model.generate(message=struct)
        except Exception:
            continue

    for h in handles:
        h.remove()

    g_cache = {}
    for _, module in module_root.named_modules():
        key = id(module)
        if key not in stats:
            continue
        st = stats[key]
        if st["count"] <= 0:
            continue

        diag = (st["sum_sq"] / max(st["count"], 1)).float().clamp(min=1e-12)
        weight = module.weight.data
        weight_2d = weight.reshape(weight.shape[0], -1)
        if diag.numel() != weight_2d.shape[1]:
            continue
        g = diag.view(1, -1).expand(weight_2d.shape[0], -1).reshape_as(weight_2d)
        g_cache[key] = g.cpu()

    return g_cache


def get_weight_importance_for_module(module, weight, g_cache=None):
    if not g_cache:
        return None
    g = g_cache.get(id(module), None)
    if g is None:
        return None
    return g.to(weight.device, dtype=weight.dtype).reshape_as(weight.reshape(weight.shape[0], -1))


def select_weight_outlier_mask(
    weight,
    ratio,
    min_channels=1,
    metric="absmax",
    perchannel=True,
):
    if ratio <= 0:
        return None
    weight_2d = weight.reshape(weight.shape[0], -1)
    if weight_2d.numel() <= 1:
        return None

    weight_abs = weight_2d.abs()
    if perchannel:
        interval_size = weight_2d.shape[1]
        if interval_size <= 1:
            return None
        k = int(round(interval_size * ratio))
        if ratio > 0:
            k = max(k, min_channels)
        k = min(max(k, 0), interval_size - 1)
        if k <= 0:
            return None
        topk_idx = torch.topk(weight_abs, k=k, dim=-1, largest=True).indices
        mask = torch.zeros_like(weight_abs, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        return mask.reshape_as(weight)

    interval_size = weight_abs.numel()
    if interval_size <= 1:
        return None
    k = int(round(interval_size * ratio))
    if ratio > 0:
        k = max(k, min_channels)
    k = min(max(k, 0), interval_size - 1)
    if k <= 0:
        return None
    mask = torch.zeros_like(weight_abs, dtype=torch.bool)
    topk_idx = torch.topk(weight_abs.reshape(-1), k=k, largest=True).indices
    mask.reshape(-1)[topk_idx] = True
    return mask.reshape_as(weight)


def quantize_weight_with_outlier_channels(
    weight,
    quantizer,
    ratio=0.0,
    min_channels=1,
    metric="absmax",
    high_bits=16,
    high_sym=True,
    g=None,
    log_enabled=False,
    layer_name="",
):
    wq = quantizer.quantize(weight)

    def _log_stats(mask):
        if not log_enabled:
            return
        ln = layer_name if layer_name else "unnamed"
        perchannel = getattr(quantizer, "perchannel", True)
        if mask is None:
            print(
                f"[WeightOutlier][{ln}] mode={'per-channel' if perchannel else 'per-tensor'} "
                f"global_ratio=0.000000 interval_mean=0.000000 interval_min=0.000000 interval_max=0.000000"
            )
            return
        mask_f = mask.float()
        if perchannel:
            interval_ratios = mask_f.reshape(mask_f.shape[0], -1).mean(dim=-1)
            mode = "per-channel"
        else:
            interval_ratios = mask_f.reshape(-1).mean().view(1)
            mode = "per-tensor"
        print(
            f"[WeightOutlier][{ln}] mode={mode} global_ratio={mask_f.mean().item():.6f} "
            f"interval_mean={interval_ratios.mean().item():.6f} "
            f"interval_min={interval_ratios.min().item():.6f} "
            f"interval_max={interval_ratios.max().item():.6f}"
        )

    if ratio <= 0:
        _log_stats(None)
        return wq
    outlier_mask = select_weight_outlier_mask(
        weight,
        ratio=ratio,
        min_channels=min_channels,
        metric=metric,
        perchannel=getattr(quantizer, "perchannel", True),
    )
    if outlier_mask is None or not torch.any(outlier_mask):
        _log_stats(None)
        return wq
    _log_stats(outlier_mask)

    if high_bits >= 16:
        wq = torch.where(outlier_mask, weight, wq)
        return wq

    high_quantizer = WeightQuantizer()
    high_quantizer.configure(
        bits=high_bits,
        perchannel=getattr(quantizer, "perchannel", True),
        nuq=False,
        sym=high_sym,
        mse=False,
    )
    high_quantizer.find_params(weight, g=g)
    wq_high = high_quantizer.quantize(weight)
    wq = torch.where(outlier_mask, wq_high, wq)
    return wq


def configure_internvl_act_outlier(model, args):
    if not getattr(args, "enable_act_outlier", False):
        return

    configured = 0
    ratio = getattr(args, "outlier_ratio", 0.0)
    metric = getattr(args, "outlier_metric", "absmax")
    min_channels = getattr(args, "outlier_min_channels", 1)

    if getattr(args, "quant_llm", False):
        qlayers = find_qlayers(
            model.model.language_model,
            layers=[ActQuantWrapper],
        )
        for name in qlayers:
            if "feed_forward.w2" not in name:
                continue
            qlayers[name].configure_act_outlier(
                enable=True,
                ratio=ratio,
                metric=metric,
                min_channels=min_channels,
                log_enabled=getattr(args, "outlier_log", False),
                log_once=True,
                layer_name=name,
            )
            configured += 1

    if getattr(args, "quant_visual_clip", False):
        qlayers = find_qlayers(
            model.model.vision_model,
            layers=[ActQuantWrapper],
        )
        for name in qlayers:
            if "mlp.fc2" not in name:
                continue
            qlayers[name].configure_act_outlier(
                enable=True,
                ratio=ratio,
                metric=metric,
                min_channels=min_channels,
                log_enabled=getattr(args, "outlier_log", False),
                log_once=True,
                layer_name=name,
            )
            configured += 1

    if getattr(args, "quant_cross_attention", False):
        qlayers = find_qlayers(
            model.model.mlp1,
            layers=[ActQuantWrapper],
        )
        for name in qlayers:
            qlayers[name].configure_act_outlier(
                enable=True,
                ratio=ratio,
                metric=metric,
                min_channels=min_channels,
                log_enabled=getattr(args, "outlier_log", False),
                log_once=True,
                layer_name=name,
            )
            configured += 1

    print(
        f"[ActOutlier] enabled={args.enable_act_outlier}, ratio={ratio}, metric={metric}, configured_layers={configured}"
    )


def model_open_calibrate(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = True
    return model


def model_open_last_calibrate(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.last_calibrate = True
    return model


def model_close_calibrate(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = False
    return model


def model_quant(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.quant = True
    return model


def model_no_quant(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.quant = False
    return model


ds_collections = {
    "vqav2_val": {
        "train": "data/vqav2/vqav2_train.jsonl",
        "test": "data/vqav2/vqav2_val.jsonl",
        "question": "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
        "annotation": "data/vqav2/v2_mscoco_val2014_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "vqav2_testdev": {
        "train": "data/vqav2/vqav2_train.jsonl",
        "test": "data/vqav2/vqav2_testdev.jsonl",
        "metric": None,
        "max_new_tokens": 10,
    },
    "okvqa_val": {
        "train": "data/okvqa/okvqa_train.jsonl",
        "test": "data/okvqa/okvqa_val.jsonl",
        "question": "data/okvqa/OpenEnded_mscoco_val2014_questions.json",
        "annotation": "data/okvqa/mscoco_val2014_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "textvqa_val": {
        "train": "data/textvqa/textvqa_train.jsonl",
        "test": "data/textvqa/textvqa_val.jsonl",
        "question": "data/textvqa/textvqa_val_questions.json",
        "annotation": "data/textvqa/textvqa_val_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "vizwiz_val": {
        "train": "data/vizwiz/vizwiz_train.jsonl",
        "test": "data/vizwiz/vizwiz_val.jsonl",
        "question": "data/vizwiz/vizwiz_val_questions.json",
        "annotation": "data/vizwiz/vizwiz_val_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "vizwiz_test": {
        "train": "data/vizwiz/vizwiz_train.jsonl",
        "test": "data/vizwiz/vizwiz_test.jsonl",
        "metric": None,
        "max_new_tokens": 10,
    },
    "docvqa_val": {
        "train": "data/docvqa/train.jsonl",
        "test": "data/docvqa/val.jsonl",
        "annotation": "data/docvqa/val/val_v1.0.json",
        "metric": "anls",
        "max_new_tokens": 100,
    },
    "docvqa_test": {
        "train": "data/docvqa/train.jsonl",
        "test": "data/docvqa/test.jsonl",
        "metric": None,
        "max_new_tokens": 100,
    },
    "chartqa_test_human": {
        "train": "data/chartqa/train_human.jsonl",
        "test": "data/chartqa/test_human.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    "chartqa_test_augmented": {
        "train": "data/chartqa/train_augmented.jsonl",
        "test": "data/chartqa/test_augmented.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    "gqa_testdev": {
        "train": "data/gqa/train.jsonl",
        "test": "data/gqa/testdev_balanced.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 10,
    },
    "ocrvqa_val": {
        "train": "data/ocrvqa/ocrvqa_train.jsonl",
        "test": "data/ocrvqa/ocrvqa_val.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 100,
    },
    "ocrvqa_test": {
        "train": "data/ocrvqa/ocrvqa_train.jsonl",
        "test": "data/ocrvqa/ocrvqa_test.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 100,
    },
    "ai2diagram_test": {
        "train": "data/ai2diagram/train.jsonl",
        "test": "data/ai2diagram/test.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 10,
    },
}


import json
import random
from tqdm import tqdm


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, use_train=False):
        if use_train:
            self.test = open(train).readlines()
        else:
            self.test = open(test).readlines()
        self.prompt = prompt

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = (
            data["image"],
            data["question"],
            data["question_id"],
            data.get("answer", None),
        )

        few_shot_prompt = ""
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += (
                    self.prompt.format(sample["image"], sample["question"])
                    + f" {sample['answer']}"
                )

        return {
            "question": few_shot_prompt + self.prompt.format(image, question),
            "question_id": question_id,
            "annotation": annotation,
        }


def collate_fn(batches, tokenizer):

    questions = [_["question"] for _ in batches]
    question_ids = [_["question_id"] for _ in batches]
    annotations = [_["annotation"] for _ in batches]
    input_ids = tokenizer(questions, return_tensors="pt", padding="longest")

    return question_ids, input_ids.input_ids, input_ids.attention_mask, annotations


def calib_vqa(
    model, tokenizers, args, dataset_name, batch_size, num_workers, seed=0, few_shot=0
):
    from copy import deepcopy

    tokenizer = deepcopy(tokenizers)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id
    prompt = "<img>{}</img>{} Answer:"
    dataset = VQADataset(
        train=ds_collections[dataset_name]["train"],
        test=ds_collections[dataset_name]["test"],
        prompt=prompt,
        few_shot=few_shot,
        use_train=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    idx = 0
    step = math.ceil(len(dataset) / batch_size) // args.calib_num

    print("Calibrating...")
    model_open_calibrate(model, args)

    for _, (question_ids, input_ids, attention_mask, annotations) in tqdm(
        enumerate(dataloader)
    ):
        if args.calib_mode == "v1":
            idx += 1
            max_new_tokens = ds_collections[dataset_name]["max_new_tokens"]
            if idx > args.calib_num:
                break
            if idx == args.calib_num:
                model_open_last_calibrate(model, args)
                max_new_tokens = 1
            model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
            )
        elif args.calib_mode == "v2":
            if idx % step == 0:
                max_new_tokens = ds_collections[dataset_name]["max_new_tokens"]
                if idx + step > math.ceil(len(dataset) / batch_size):
                    model_open_last_calibrate(model, args)
                    max_new_tokens = 1

                model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
            idx += 1
        else:
            raise ValueError("Invalid calibration mode")

    model_close_calibrate(model, args)
    print("Calibrate End...")
    model_quant(model, args)


def analysis_text(model, tokenizer, analysis_num, seqlen, split="test", mode="v1"):
    tokenizer_name = tokenizer.__class__.__name__
    cached_loader = f"./cache/wikitext-2-raw-v1/{split}_{tokenizer_name}_loader.pt"
    if os.path.exists(cached_loader):
        loader = torch.load(cached_loader)
    else:
        wiki_testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            cache_dir="./cache",
            keep_in_memory=True,
        )
        loader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
        os.makedirs("./cache/wikitext-2-raw-v1", exist_ok=True)
        torch.save(loader, cached_loader)
    test_loader = loader.input_ids
    nsamples = test_loader.numel() // seqlen

    batches = []
    for i in tqdm(range(nsamples)):
        if i >= analysis_num:
            break
        batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
        batches.append(batch)
    model(torch.cat(batches, dim=0))


def analysis(model, tokenizers, dataset_name, analysis_num, mode="v1"):
    from copy import deepcopy

    tokenizer = deepcopy(tokenizers)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id
    prompt = "<img>{}</img>"
    print(dataset_name)
    dataset = VQADataset(
        train=ds_collections[dataset_name]["train"],
        test=ds_collections[dataset_name]["test"],
        prompt=prompt,
        few_shot=0,
    )

    num_data = len(dataset)
    batchs = []
    if mode == "v1":
        for i in range(num_data):
            if i >= analysis_num:
                break
            batchs.append(dataset[i])
    else:
        step = len(dataset) // analysis_num

        for i in range(analysis_num):
            batchs.append(dataset[i * step])

    _, input_ids, attention_mask, _ = collate_fn(batches=batchs, tokenizer=tokenizer)

    model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        do_sample=False,
        num_beams=1,
        max_new_tokens=1,
        min_new_tokens=1,
        length_penalty=1,
        num_return_sequences=1,
        output_hidden_states=True,
        use_cache=True,
        pad_token_id=tokenizer.eod_id,
        eos_token_id=tokenizer.eod_id,
    )


def calib_minicpm_vqa(model, dataset, dev, dataset_name, args):
    sampler = None
    from evaluation.minicpmv.eval_utils.vqa_evaluate import collate_fn_vqa

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_vqa,
    )

    print("Calibrating...")
    model_open_calibrate(model.model, args)

    total_batches = len(dataloader)  # 获取dataloader中总共的批次数量

    for i, batch in enumerate(tqdm(dataloader, desc="per tensor static calibrate"), 1):
        (
            image_paths,
            questions,
            gt_answers,
            ocr_tokens_list,
            question_ids,
            question_type,
        ) = batch

        if i == total_batches - 1:  # 检查是否为最后一个批次
            model_open_last_calibrate(model.model, args)
            model.generate_with_interleaved_calib(
                images=image_paths, questions=questions, datasetname=dataset_name
            )
        else:
            model.generate_with_interleaved(
                images=image_paths, questions=questions, datasetname=dataset_name
            )

    model_close_calibrate(model.model, args)
    print("Calibrate End...")
    model_quant(model.model, args)


def calib_vqa_plus(model, args, dataset, calib_num):
    lt = len(dataset.data)
    step = math.ceil(lt / calib_num)
    print("Calibrating...")
    model_open_calibrate(model.model, args)
    model.kwargs["max_new_tokens"] = 20
    for i in tqdm(range(0, lt, step)):
        if i + step >= lt:
            print("last calibrate")
            model_open_last_calibrate(model.model, args)
            model.kwargs["max_new_tokens"] = 1
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            args.dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=args.dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])
        model.generate(message=struct, dataset=args.dataset_name)

    model.kwargs = {}

    model_close_calibrate(model.model, args)
    print("Calibrate End...")
    model_quant(model.model, args)


def calib_qwen2vl_plus(model, args, dataset, calib_num):
    lt = len(dataset.data)
    step = math.ceil(lt / calib_num)
    print("Calibrating...")
    model_open_calibrate(model.model, args)
    max_new_tokens = model.generate_kwargs["max_new_tokens"]
    model.generate_kwargs["max_new_tokens"] = 20
    for i in tqdm(range(0, lt, step)):
        if i + step >= lt:
            print("last calibrate")
            model_open_last_calibrate(model.model, args)
            model.generate_kwargs["max_new_tokens"] = 1
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            args.dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=args.dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])
        model.generate(message=struct, dataset=args.dataset_name)

    model.generate_kwargs["max_new_tokens"] = max_new_tokens

    model_close_calibrate(model.model, args)
    print("Calibrate End...")
    model_quant(model.model, args)

def non_uniform_lut(x, g, bit, device=None):
    # Determine device
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = torch.device('cpu')

    # Ensure inputs are torch tensors on device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).to(device).float()
    else:
        x = x.to(device).float()

    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g).to(device).float()
    else:
        g = g.to(device).float()

    centers = torch.linspace(x.min(), x.max(), steps=2**bit, device=device)
    # 归一化grad
    x = x.reshape(-1)
    g = g.reshape(-1)
    grad_difference = g.max() - g.min()
    if grad_difference == 0:
        grad_difference = g.max()
        if grad_difference == 0:
            grad_difference = torch.tensor(1.0, device=g.device)
    g = g / grad_difference

    # 确保centers有序，以便使用searchsorted加速
    centers, _ = torch.sort(centers)

    # 预计算加权梯度
    wg = x * g

    # 初始分配
    if len(centers) > 1:
        boundaries = (centers[:-1] + centers[1:]) / 2
        labels = torch.bucketize(x, boundaries)
    else:
        labels = torch.zeros_like(x, dtype=torch.long)

    # best_loss = (np.square(weights - centers[labels]) * grads).sum()
    current_centers = centers[labels]
    best_loss = (torch.square(x - current_centers) * g).sum()
    best_centers = centers.clone()

    eps = 1e-7
    max_patience = 30
    patience = max_patience

    # 预计算范围用于扰动
    w_min, w_max = x.min(), x.max()
    perturb_scale = (w_max - w_min) / (len(centers) + 1e-6) * 0.01

    while patience > 0:
        # 1. 向量化更新中心点
        denom = torch.bincount(labels, weights=g, minlength=len(centers))
        num = torch.bincount(labels, weights=wg, minlength=len(centers))

        active_mask = denom > 1e-10
        dead_mask = ~active_mask
        n_dead = torch.sum(dead_mask)

        new_centers = centers.clone()
        new_centers[active_mask] = num[active_mask] / denom[active_mask]

        # 2. 逻辑改进：处理“死掉”的中心点 (Dead Centers)
        # 如果某个聚类中心没有分配到权重，将其移动到Loss最大的聚类附近进行分裂
        if n_dead > 0:
            # 计算每个聚类的Loss贡献
            current_centers = centers[labels]
            sq_errors = torch.square(x - current_centers) * g
            cluster_loss = torch.bincount(
                labels, weights=sq_errors, minlength=len(centers))

            # 找到Loss最大的活跃聚类作为分裂源
            # 排除已经是死掉的聚类
            cluster_loss[dead_mask] = -1.0
            candidate_indices = torch.argsort(cluster_loss, descending=True)

            dead_indices = torch.where(dead_mask)[0]

            for i, dead_idx in enumerate(dead_indices):
                # 循环使用高Loss的聚类进行分裂
                target_idx = candidate_indices[i % len(candidate_indices)]

                if cluster_loss[target_idx] <= eps:
                    break  # 如果连最大的Loss都很小，就不分裂了

                # 分裂策略：在目标中心点附近微扰
                center_val = new_centers[target_idx]
                new_centers[dead_idx] = center_val + perturb_scale
                new_centers[target_idx] = center_val - perturb_scale

        # 3. 保持有序 (对于1D分配很重要)
        new_centers, _ = torch.sort(new_centers)

        # 4. 快速分配新标签
        if len(new_centers) > 1:
            boundaries = (new_centers[:-1] + new_centers[1:]) / 2
            new_labels = torch.bucketize(x, boundaries)
        else:
            new_labels = torch.zeros_like(x, dtype=torch.long)

        # 5. 计算Loss并更新
        current_new_centers = new_centers[new_labels]
        loss = (torch.square(x - current_new_centers) * g).sum()

        if loss < best_loss - eps:
            best_loss = loss
            best_centers = new_centers.clone()
            patience = max_patience  # 重置patience
            centers = new_centers
            labels = new_labels
        else:
            patience -= 1
            centers = new_centers
            labels = new_labels

    return best_centers.cpu()