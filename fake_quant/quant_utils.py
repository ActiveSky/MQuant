"""
量化工具函数模块

该模块提供了量化操作的核心工具函数，包括对称/非对称量化、4-bit打包/解包等功能。
还包含激活量化器类，用于神经网络中的激活值量化。

依赖项:
    math: 数学函数库
    transformers: Hugging Face Transformers库
    torch: PyTorch深度学习框架
    os: 操作系统接口
    fake_quant.utils: 项目工具函数
    fake_quant.hadamard_utils: 阿达马变换工具
    fast_hadamard_transform: 快速阿达马变换库
    collections: 容器数据类型
    fake_quant.observer: 观察器模块
    fake_quant.quantizer: 量化器模块
    fake_quant.bit_type: 位类型定义
    functools: 高阶函数和可调用对象
    datasets: Hugging Face数据集库
"""

import math
import transformers
import torch
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
    """
    根据位数和符号性计算量化范围的最小值和最大值

    Args:
        bits (int): 量化位数
        sym (bool): 是否为对称量化（有符号）

    Returns:
        tuple: (minq, maxq) 最小量化值和最大量化值
            - 对称量化：minq = -2^(bits-1), maxq = 2^(bits-1) - 1
            - 非对称量化：minq = 0, maxq = 2^bits - 1
    """
    # 对称量化（有符号）范围计算
    if sym:
        # 最大值为2^(bits-1) - 1
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        # 最小值为-(2^(bits-1))
        minq = -maxq - 1
    # 非对称量化（无符号）范围计算
    else:
        # 最大值为2^bits - 1
        maxq = torch.tensor(2**bits - 1)
        # 最小值为0
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    """
    非对称量化：将浮点数量化为无符号整数

    使用公式：q = clamp(round(x / scale + zero), 0, maxq)

    Args:
        x (torch.Tensor): 输入浮点张量
        scale (torch.Tensor): 缩放因子
        zero (torch.Tensor): 零点
        maxq (torch.Tensor): 最大量化值

    Returns:
        tuple: (q, scale, zero) 量化结果和参数
            - q (torch.Tensor): 量化后的整数张量
            - scale (torch.Tensor): 缩放因子（移动到x的设备上）
            - zero (torch.Tensor): 零点（移动到x的设备上）
    """
    # 将缩放因子和零点移动到输入张量的设备上
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    # 执行非对称量化：输入除以缩放因子加上零点，四舍五入并钳位到[0, maxq]
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    """
    非对称反量化：将无符号整数量化值反量化为浮点数

    使用公式：x = scale * (q - zero)

    Args:
        q (torch.Tensor): 量化后的整数张量
        scale (torch.Tensor): 缩放因子
        zero (torch.Tensor): 零点

    Returns:
        torch.Tensor: 反量化后的浮点张量
    """
    # 执行非对称反量化：(量化值 - 零点) * 缩放因子
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    """
    非对称量化-反量化：完整的量化-反量化过程

    先执行非对称量化，然后执行非对称反量化，用于模拟量化效果。

    Args:
        x (torch.Tensor): 输入浮点张量
        scale (torch.Tensor): 缩放因子
        zero (torch.Tensor): 零点
        maxq (torch.Tensor): 最大量化值

    Returns:
        torch.Tensor: 量化-反量化后的浮点张量
    """
    # 调用非对称量化和反量化函数，使用解包操作符传递参数
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    """
    对称量化：将浮点数量化为有符号整数

    使用公式：q = clamp(round(x / scale), -maxq-1, maxq)

    Args:
        x (torch.Tensor): 输入浮点张量
        scale (torch.Tensor): 缩放因子
        maxq (torch.Tensor): 最大量化值

    Returns:
        tuple: (q, scale) 量化结果和参数
            - q (torch.Tensor): 量化后的整数张量
            - scale (torch.Tensor): 缩放因子（移动到x的设备上）
    """
    # 将缩放因子移动到输入张量的设备上
    scale = scale.to(x.device)
    # 执行对称量化：输入除以缩放因子，四舍五入并钳位到[-maxq-1, maxq]
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    """
    对称反量化：将有符号整数量化值反量化为浮点数

    使用公式：x = scale * q

    Args:
        q (torch.Tensor): 量化后的整数张量
        scale (torch.Tensor): 缩放因子

    Returns:
        torch.Tensor: 反量化后的浮点张量
    """
    # 执行对称反量化：量化值 * 缩放因子
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    """
    对称量化-反量化：完整的量化-反量化过程

    先执行对称量化，然后执行对称反量化，用于模拟量化效果。

    Args:
        x (torch.Tensor): 输入浮点张量
        scale (torch.Tensor): 缩放因子
        maxq (torch.Tensor): 最大量化值

    Returns:
        torch.Tensor: 量化-反量化后的浮点张量
    """
    # 调用对称量化和反量化函数，使用解包操作符传递参数
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    """
    计算二进制补码表示

    将负数转换为对应的二进制补码表示。

    Args:
        x (torch.Tensor): 输入张量
        bits (int): 位数

    Returns:
        torch.Tensor: 二进制补码表示的张量
            - 负数：2^bits + x
            - 非负数：x
    """
    # 对于负数使用补码表示：2^bits + x，非负数保持不变
    return torch.where(x < 0, 2**bits + x, x)


def pack_i4(q):
    """
    将int4张量打包为uint8存储格式

    每个uint8字节存储两个int4值，低位在前。

    Args:
        q (torch.Tensor): 要打包的有符号int4张量

    Returns:
        torch.Tensor: 打包后的uint8张量，大小为原张量的一半

    Raises:
        AssertionError: 如果输入张量不是有符号整数或超出int4范围
    """
    # 断言输入张量必须是有符号整数
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # 获取int4的最小值和最大值
    minq, maxq = get_minq_maxq(4, True)
    # 断言所有值都在int4合法范围内
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    # 将张量转换为int8并计算二进制补码
    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    # 将相邻的两个int4值打包到一个uint8中：低位在低4位，高位在高4位
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


def unpack_i4(x: torch.Tensor):
    """
    将uint8存储的int4量化张量解包为int32张量

    每个uint8字节包含两个int4值，需要分别提取高低4位。

    Args:
        x (torch.Tensor): 要解包的uint8张量

    Returns:
        torch.Tensor: 解包后的int32张量，大小为原张量的两倍

    Raises:
        AssertionError: 如果输入张量不是uint8类型
    """
    # 断言输入张量必须是uint8类型
    assert x.dtype == torch.uint8, "The tensor to be unpacked should be stored in uint8"

    # 计算输出形状，最后一个维度是输入的两倍
    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # 提取低4位
    x0 = (x & 0x0F).to(torch.int8)
    # 将大于等于8的值转换为负数（补码转换）
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # 提取高4位
    x1 = ((x & 0xF0) >> 4).to(torch.int8)
    # 将大于等于8的值转换为负数（补码转换）
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    # 创建输出张量
    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # 交错排列：偶数位置放低位，奇数位置放高位
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class ActQuantizer(torch.nn.Module):
    """
    激活值量化器类，用于对神经网络中的激活值进行量化

    该类支持对称和非对称的逐令牌量化，可以静态或动态地计算量化参数。
    主要用于在推理过程中减少激活值的精度以提高计算效率。

    依赖项:
        torch.nn.Module: PyTorch神经网络模块基类
        get_minq_maxq: 获取量化范围函数
        BIT_TYPE_DICT: 位类型字典
        build_observer: 观察器构建函数
        build_quantizer: 量化器构建函数

    属性:
        maxq (torch.Tensor): 最大量化值
        scale (torch.Tensor): 缩放因子缓冲区
        zero (torch.Tensor): 零点缓冲区
        bits (int): 量化位数，默认为16（不量化）
        act_per_tensor (bool): 是否对整个张量使用单一量化参数
        static (bool): 是否使用静态量化
        groupsize (int): 分组大小，用于组-wise量化
        sym (bool): 是否使用对称量化
        clip_ratio (float): 裁剪比例，用于调整最值范围
    """

    def __init__(self, act_per_tensor=False):
        """
        初始化激活值量化器

        Args:
            act_per_tensor (bool): 是否对整个张量使用单一量化参数，默认为False
        """
        # 调用父类初始化方法
        super(ActQuantizer, self).__init__()
        # 注册最大量化值缓冲区
        self.register_buffer("maxq", torch.tensor(0))
        # 注册缩放因子缓冲区
        self.register_buffer("scale", torch.zeros(1))
        # 注册零点缓冲区
        self.register_buffer("zero", torch.zeros(1))
        # 设置量化位数，默认为16位（不量化）
        self.bits = 16
        # 是否对整个张量使用单一量化参数
        self.act_per_tensor = act_per_tensor
        # 是否使用静态量化
        self.static = False

    def free(self):
        """
        释放量化参数内存

        将零点和缩放因子设置为None以释放内存。
        """
        # 释放零点内存
        self.zero = None
        # 释放缩放因子内存
        self.scale = None

    def forward(self, x):
        """
        前向传播方法，对输入张量执行量化操作

        根据配置决定是否进行量化，静态量化使用观察器，动态量化直接计算参数。

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 量化后的张量
        """
        # 静态量化模式
        if self.static:
            # 校准模式：更新观察器统计信息
            if self.calibrate:
                self.quantizer.observer.update(x)
                # 最后一次校准：更新量化参数
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
                return x
            # 量化模式：使用量化器处理输入
            elif self.quant:
                return self.quantizer(x)
            # 其他模式：直接返回输入
            else:
                return x
        # 动态量化模式
        else:
            # 保存输入数据类型
            x_dtype = x.dtype
            # 16位不量化直接返回
            if self.bits == 16:
                return x
            # 对称量化：使用对称量化-反量化函数
            elif self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            # 非对称量化：使用非对称量化-反量化函数
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # 与`forward`方法不同，此方法返回量化整数、缩放因子（和非对称情况下的零点）。
    def quantize(self, x):
        """
        将输入张量量化为整数

        根据对称性选择相应的量化方法，返回量化后的整数及参数。

        Args:
            x (torch.Tensor): 输入浮点张量

        Returns:
            tuple: 量化结果，对称量化返回(q, scale)，非对称量化返回(q, scale, zero)
        """
        # 对称量化
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        # 非对称量化
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self,
        bits,
        groupsize=-1,
        sym=False,
        clip_ratio=1.0,
        act_per_tensor=False,
        static=False,
        observer_type="minmax",
        calibration_mode="layer_wise",
    ):
        """
        配置量化器参数

        Args:
            bits (int): 量化位数
            groupsize (int): 分组大小，-1表示不分组
            sym (bool): 是否使用对称量化
            clip_ratio (float): 裁剪比例，应在(0, 1]范围内
            act_per_tensor (bool): 是否对整个张量使用单一量化参数
            static (bool): 是否使用静态量化
            observer_type (str): 观察器类型，如"minmax"、"percentile"
            calibration_mode (str): 校准模式，如"layer_wise"
        """
        # 获取最小和最大量化值
        _, self.maxq = get_minq_maxq(bits, sym)
        # 设置量化位数
        self.bits = bits
        # 设置分组大小
        self.groupsize = groupsize
        # 设置是否对称量化
        self.sym = sym
        # 设置裁剪比例
        self.clip_ratio = clip_ratio
        # 设置是否对整个张量使用单一量化参数
        self.act_per_tensor = act_per_tensor
        # 验证裁剪比例范围
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"
        # 设置是否静态量化
        self.static = static
        # 静态量化配置
        if self.static:
            # 模块类型为激活值
            module_a_type = "activation"
            # 获取对应的位类型
            bit_type_a = BIT_TYPE_DICT[f"int{bits}"]
            # 百分位观察器提示
            if observer_type == "percentile":
                print("Using percentile observer for activations")
            # 构建观察器
            self.observer = build_observer(
                observer_type,
                module_a_type,
                bit_type_a,
                calibration_mode,
            )
            # 构建量化器
            self.quantizer = build_quantizer(
                "uniform", bit_type_a, self.observer, module_a_type
            )
            # 初始化校准和量化标志
            self.calibrate = False
            self.last_calibrate = False
            self.quant = False

    def find_params_per_token_groupwise(self, x):
        """
        按组计算逐令牌量化参数

        将输入张量按指定组大小分组，分别计算每组的量化参数。

        Args:
            x (torch.Tensor): 输入张量
        """
        # 保存原始形状
        init_shape = x.shape
        # 重塑张量以适应分组：(batch, seq_len, groups, group_size)
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        # 计算每组的最大值和最小值，并应用裁剪比例
        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        # 对称量化参数计算
        if self.sym:
            # 使用绝对值最大值
            xmax = torch.maximum(torch.abs(xmin), xmax)
            # 标记零值位置
            tmp = xmax == 0
            # 计算缩放因子
            self.scale = xmax / self.maxq
            # 零值位置设置为1
            self.scale[tmp] = 1
            # 零点设为0
            self.zero = torch.zeros_like(self.scale)
        # 非对称量化参数计算
        else:
            # 标记零值位置
            tmp = (xmin == 0) & (xmax == 0)
            # 零值位置设置默认值
            xmin[tmp] = -1
            xmax[tmp] = +1
            # 计算缩放因子
            self.scale = (xmax - xmin) / self.maxq
            # 计算零点
            self.zero = torch.round(-xmin / self.scale)

        # 重复参数以匹配原始形状
        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        """
        计算量化参数

        根据配置计算输入张量的量化参数（缩放因子和零点）。

        Args:
            x (torch.Tensor): 输入张量
        """
        # 16位不量化直接返回
        if self.bits == 16:
            return

        # 获取输入设备
        dev = x.device
        # 将maxq移动到相同设备
        self.maxq = self.maxq.to(dev)

        # 保存初始形状
        init_shape = x.shape

        # 对整个张量使用单一参数
        if self.act_per_tensor:
            # 创建零张量
            tmp = torch.tensor(0).to(x)
            # 计算最小值和最大值并应用裁剪比例
            xmin = torch.minimum(x.min(), tmp) * self.clip_ratio
            xmax = torch.maximum(x.max(), tmp) * self.clip_ratio
            # 对称量化参数计算
            if self.sym:
                # 使用绝对值最大值
                xmax = torch.maximum(torch.abs(xmin), xmax)
                # 零值处理
                if xmax == 0:
                    self.scale = 1
                else:
                    self.scale = xmax / self.maxq
                # 零点设为0
                self.zero = torch.zeros_like(self.scale)
            # 非对称量化参数计算
            else:
                # 零值处理
                if xmin == 0:
                    xmin = -1
                if xmax == 0:
                    xmax = 1
                # 计算缩放因子和零点
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)
        # 按令牌量化
        else:
            # 分组量化
            if self.groupsize > 0:
                # group-wise per-token quantization
                self.find_params_per_token_groupwise(x)
                utils.cleanup_memory(verbos=False)
                return
            # 重塑张量以适应按令牌处理
            reshaped_x = x.reshape((-1, x.shape[-1]))

            # 创建零张量
            tmp = torch.zeros(reshaped_x.shape[0], device=dev)
            # 计算每行的最小值和最大值并应用裁剪比例
            xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
            xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
            # 对称量化参数计算
            if self.sym:
                # 使用绝对值最大值
                xmax = torch.maximum(torch.abs(xmin), xmax)
                # 标记零值位置
                tmp = xmax == 0
                # 计算缩放因子
                self.scale = (
                    (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
                )
                # 零值位置设置为1
                self.scale[tmp] = 1
                # 重塑为原始形状
                self.scale = self.scale.reshape(init_shape)
                # 零点设为0
                self.zero = torch.zeros_like(self.scale)
            # 非对称量化参数计算
            else:
                # 标记零值位置
                tmp = (xmin == 0) & (xmax == 0)
                # 零值处理
                xmin[tmp] = -1
                xmax[tmp] = +1
                # 计算缩放因子和零点
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)

                # 重复参数以匹配原始形状
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
    激活值量化包装器类，用于包装神经网络模块并添加激活值量化功能

    该类包装线性层、卷积层等模块，在前向传播过程中对激活值进行量化。
    支持阿达马变换旋转和权重分割等高级功能。

    依赖项:
        torch.nn.Module: PyTorch神经网络模块基类
        ActQuantizer: 激活值量化器
        hadamard_utils: 阿达马变换工具
        fast_hadamard_transform: 快速阿达马变换库
        math: 数学函数库

    属性:
        module (torch.nn.Module): 被包装的原始模块
        weight (torch.Tensor): 模块权重
        bias (torch.Tensor): 模块偏置
        quantizer (ActQuantizer): 输入激活值量化器
        out_quantizer (ActQuantizer): 输出激活值量化器
        had_K (torch.Tensor): 阿达马变换矩阵
        K (int): 阿达马变换参数
        online_full_had (bool): 是否执行完整的在线阿达马变换
        online_partial_had (bool): 是否执行部分在线阿达马变换
        had_dim (int): 阿达马变换维度
        fp32_had (bool): 是否在FP32精度下执行阿达马变换
        split (bool): 是否分割权重
    """

    def __init__(self, module: torch.nn.Linear, act_per_tensor=False):
        """
        初始化激活值量化包装器

        Args:
            module (torch.nn.Linear): 要包装的线性层模块
            act_per_tensor (bool): 是否对整个张量使用单一量化参数

        Raises:
            AssertionError: 如果模块不是线性层、2D卷积层或3D卷积层
        """
        # 调用父类初始化方法
        super(ActQuantWrapper, self).__init__()
        # 验证模块类型
        assert isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d))
        # 保存被包装的模块
        self.module = module
        # 保存模块权重
        self.weight = module.weight
        # 保存模块偏置
        self.bias = module.bias
        # 创建输入激活值量化器
        self.quantizer = ActQuantizer(act_per_tensor)
        # 创建输出激活值量化器
        self.out_quantizer = ActQuantizer(act_per_tensor)
        # 注册阿达马变换矩阵缓冲区
        self.register_buffer("had_K", torch.tensor(0))
        # 初始化缓冲区
        self._buffers["had_K"] = None
        # 阿达马变换参数
        self.K = 1
        # 是否执行完整的在线阿达马变换
        self.online_full_had = False
        # 是否执行部分在线阿达马变换
        self.online_partial_had = False
        # 阿达马变换维度
        self.had_dim = 0
        # 是否在FP32精度下执行阿达马变换
        self.fp32_had = False
        # 是否分割权重
        self.split = False

    def extra_repr(self) -> str:
        """
        返回模块的额外表示信息

        Returns:
            str: 包含输入和输出量化器信息的字符串
        """
        # 构建输入量化器信息
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        # 如果启用了输入量化，添加量化类型信息
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        # 添加输出量化器信息
        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        # 如果启用了输出量化，添加量化类型信息
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def split_weights(self):
        """
        分割权重矩阵

        将权重矩阵分为两部分：第一列单独作为一个线性层，其余部分作为另一个线性层。
        用于支持特定的量化策略。
        """
        # 创建第一个线性层（处理第一列权重）
        self.L1 = torch.nn.Linear(1, self.module.out_features, bias=False).to(
            self.module.weight.device
        )
        # 创建第二个线性层（处理剩余权重）
        self.L2 = torch.nn.Linear(
            self.module.in_features - 1,
            self.module.out_features,
            bias=True if self.module.bias is not None else False,
        ).to(self.module.weight.device)
        # 设置第一个线性层的权重为原权重的第一列
        self.L1.weight.data = self.module.weight.data[:, 0:1]
        # 设置第二个线性层的权重为原权重的其余部分
        self.L2.weight.data = self.module.weight.data[:, 1:]
        # 如果原模块有偏置，设置第二个线性层的偏置
        if self.module.bias is not None:
            self.L2.bias.data = self.module.bias.data

    def forward(self, x):
        """
        前向传播方法，执行激活值量化和模块计算

        根据配置执行阿达马变换、激活值量化、模块计算和输出量化。

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        # 保存输入数据类型
        x_dtype = x.dtype

        # 如果需要，执行旋转变换
        # 完整阿达马变换
        if self.online_full_had:

            # 在FP32精度下执行完整阿达马变换
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            # 在FP16精度下执行完整阿达马变换
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        # 部分阿达马变换
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            # 如果需要FP32精度
            if self.fp32_had:
                x = x.float()

            # 保存初始形状
            init_shape = x.shape
            # 执行阿达马变换
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

            # 如果之前转换为FP32，现在转换回原精度
            if self.fp32_had:
                x = x.to(x_dtype)
            # 重塑为初始形状
            x = x.reshape(init_shape)

        # 权重分割模式
        if self.split:
            # 静态量化模式：量化除第一列外的所有激活值
            if self.quantizer.static:
                x[..., 1:] = self.quantizer(x[..., 1:])
            # 动态量化模式：计算参数并量化除第一列外的所有激活值
            elif self.quantizer.bits < 16:
                self.quantizer.find_params(x[..., 1:])
                x[..., 1:] = self.quantizer(x[..., 1:]).to(x_dtype)
                # 释放量化参数内存
                self.quantizer.free()
            # 分别计算两个线性层的输出并相加
            x1 = self.L1.float()(x[..., 0:1].float())
            x2 = self.L2.float()(x[..., 1:].float())
            x = (x1 + x2).to(x_dtype)
        # 正常模式
        else:
            # 静态量化模式：量化所有激活值
            if self.quantizer.static:
                x = self.quantizer(x)
            # 动态量化模式：计算参数并量化所有激活值
            elif self.quantizer.bits < 16:
                self.quantizer.find_params(x)
                x = self.quantizer(x).to(x_dtype)
                # 释放量化参数内存
                self.quantizer.free()
            # 执行模块计算
            x = self.module(x).to(x_dtype)

        # 如果启用了输出量化
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

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

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
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
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
    """
    为模块添加激活值量化包装器

    递归遍历模块及其子模块，将指定类型的层包装在ActQuantWrapper中。

    Args:
        module (torch.nn.Module): 要处理的模块
        act_per_tensor (bool): 是否对整个张量使用单一量化参数
        name (str): 模块名称（用于递归调用）
        layers (list): 要包装的层类型列表
    """
    # 如果模块已经是ActQuantWrapper，直接返回
    if isinstance(module, ActQuantWrapper):
        return
    # 遍历模块的所有属性
    for attr in dir(module):
        tmp = getattr(module, attr)
        # 如果属性类型在要包装的层列表中，进行包装
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp, act_per_tensor))
        # 处理Sequential容器
        if type(tmp) == torch.nn.Sequential:
            replaced = OrderedDict()
            # 遍历Sequential中的子模块
            for name, child in tmp.named_children():
                # 如果子模块类型在要包装的层列表中，进行包装
                if type(child) in layers:
                    replaced[name] = ActQuantWrapper(child, act_per_tensor)
                else:
                    replaced[name] = child
            # 替换原Sequential
            setattr(module, attr, torch.nn.Sequential(replaced))
        # 处理ModuleList容器
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            # 遍历ModuleList中的子模块
            for i, child in enumerate(tmp.children()):
                # 如果子模块类型在要包装的层列表中，进行包装
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child, act_per_tensor))
                else:
                    replaced.append(child)
            # 替换原ModuleList
            setattr(module, attr, torch.nn.ModuleList(replaced))
    # 递归处理子模块
    for name1, child in module.named_children():
        add_actquant(
            child,
            act_per_tensor,
            name + "." + name1 if name != "" else name1,
            [torch.nn.Linear],
        )


def find_qlayers(module, layers=[torch.nn.Linear, ActQuantWrapper], name=""):
    """
    查找模块中的量化层

    递归遍历模块及其子模块，找到指定类型的层并返回字典。

    Args:
        module (torch.nn.Module): 要搜索的模块
        layers (list): 要查找的层类型列表
        name (str): 模块名称（用于递归调用）

    Returns:
        dict: 包含找到的层的字典，键为层名称，值为层对象
    """
    # 如果当前模块类型在查找列表中，返回该模块
    if type(module) in layers:
        return {name: module}
    # 初始化结果字典
    res = {}
    # 递归遍历子模块
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def model_open_calibrate(model, args):
    """
    开启模型校准模式

    将模型中所有ActQuantWrapper层的量化器设置为校准模式。

    Args:
        model (torch.nn.Module): 模型
        args (argparse.Namespace): 命令行参数

    Returns:
        torch.nn.Module: 配置后的模型
    """
    # 查找模型中的所有ActQuantWrapper层
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    # 遍历所有量化层
    for name in qlayers:
        # 跳过指定名称的层
        if any(p_name in name for p_name in args.skip_names):
            continue
        # 开启校准模式
        qlayers[name].quantizer.calibrate = True
    return model


def model_open_last_calibrate(model, args):
    """
    开启模型最后一次校准

    将模型中所有ActQuantWrapper层的量化器设置为最后一次校准模式。

    Args:
        model (torch.nn.Module): 模型
        args (argparse.Namespace): 命令行参数

    Returns:
        torch.nn.Module: 配置后的模型
    """
    # 查找模型中的所有ActQuantWrapper层
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    # 遍历所有量化层
    for name in qlayers:
        # 跳过指定名称的层
        if any(p_name in name for p_name in args.skip_names):
            continue
        # 开启最后一次校准模式
        qlayers[name].quantizer.last_calibrate = True
    return model


def model_close_calibrate(model, args):
    """
    关闭模型校准模式

    将模型中所有ActQuantWrapper层的量化器关闭校准模式。

    Args:
        model (torch.nn.Module): 模型
        args (argparse.Namespace): 命令行参数

    Returns:
        torch.nn.Module: 配置后的模型
    """
    # 查找模型中的所有ActQuantWrapper层
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    # 遍历所有量化层
    for name in qlayers:
        # 跳过指定名称的层
        if any(p_name in name for p_name in args.skip_names):
            continue
        # 关闭校准模式
        qlayers[name].quantizer.calibrate = False
    return model


def model_quant(model, args):
    """
    开启模型量化模式

    将模型中所有ActQuantWrapper层的量化器设置为量化模式。

    Args:
        model (torch.nn.Module): 模型
        args (argparse.Namespace): 命令行参数

    Returns:
        torch.nn.Module: 配置后的模型
    """
    # 查找模型中的所有ActQuantWrapper层
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    # 遍历所有量化层
    for name in qlayers:
        # 跳过指定名称的层
        if any(p_name in name for p_name in args.skip_names):
            continue
        # 开启量化模式
        qlayers[name].quantizer.quant = True
    return model


def model_no_quant(model, args):
    """
    关闭模型量化模式

    将模型中所有ActQuantWrapper层的量化器关闭量化模式。

    Args:
        model (torch.nn.Module): 模型
        args (argparse.Namespace): 命令行参数

    Returns:
        torch.nn.Module: 配置后的模型
    """
    # 查找模型中的所有ActQuantWrapper层
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    # 遍历所有量化层
    for name in qlayers:
        # 跳过指定名称的层
        if any(p_name in name for p_name in args.skip_names):
            continue
        # 关闭量化模式
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
