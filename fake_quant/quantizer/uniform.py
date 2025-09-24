# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
均匀量化器模块

该模块实现了均匀量化器，通过均匀分布的量化级别对浮点数进行量化。
支持对称和非对称量化。

依赖项:
    torch: PyTorch深度学习框架
    torch.nn (nn): PyTorch神经网络模块
    .base.BaseQuantizer: 量化器基类
"""

import torch
import torch.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):
    """
    均匀量化器，实现标准的均匀量化算法

    该类继承自BaseQuantizer，实现了具体的均匀量化和反量化逻辑。
    使用观察器计算的缩放因子和零点进行量化操作。

    依赖项:
        torch: PyTorch深度学习框架
        BaseQuantizer: 量化器基类

    属性:
        scale (torch.Tensor or None): 缩放因子，用于浮点数和整数之间的转换
        zero_point (torch.Tensor or None): 零点，量化后的整数值对应浮点0的位置
    """

    def __init__(self, bit_type, observer, module_type):
        """
        初始化UniformQuantizer实例

        Args:
            bit_type (BitType): 位类型对象
            observer (BaseObserver): 观察器对象
            module_type (str): 模块类型
        """
        # 调用父类初始化方法
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        # 初始化缩放因子
        self.scale = None
        # 初始化零点
        self.zero_point = None

    def update_quantization_params(self, *args, **kwargs):
        """
        更新量化参数（缩放因子和零点）

        调用观察器的get_quantization_params方法获取最新的量化参数。

        Args:
            *args: 可变位置参数，传递给观察器
            **kwargs: 可变关键字参数，传递给观察器
        """
        # 从观察器获取量化参数
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs
        )

    def quant(self, inputs, scale=None, zero_point=None):
        """
        将浮点输入均匀量化为整数

        使用公式：quantized = round(input / scale + zero_point)
        结果被钳位到量化位类型的合法范围内。

        Args:
            inputs (torch.Tensor): 浮点输入张量
            scale (torch.Tensor, optional): 缩放因子，如未提供则使用实例变量
            zero_point (torch.Tensor, optional): 零点，如未提供则使用实例变量

        Returns:
            torch.Tensor: 量化后的整数张量
        """
        # 如果未提供缩放因子则使用实例变量
        if scale is None:
            scale = self.scale
        # 如果未提供零点则使用实例变量
        if zero_point is None:
            zero_point = self.zero_point
        # 获取参数重塑形状
        range_shape = self.get_reshape_range(inputs)
        # 重塑缩放因子和零点以匹配输入形状
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        # 执行均匀量化：输入除以缩放因子加上零点
        outputs = inputs / scale + zero_point
        # 四舍五入并钳位到量化范围
        outputs = outputs.round().clamp(
            self.bit_type.lower_bound, self.bit_type.upper_bound
        )
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        """
        将整数量化值均匀反量化为浮点数

        使用公式：float = (quantized - zero_point) * scale

        Args:
            inputs (torch.Tensor): 整数量化张量
            scale (torch.Tensor, optional): 缩放因子，如未提供则使用实例变量
            zero_point (torch.Tensor, optional): 零点，如未提供则使用实例变量

        Returns:
            torch.Tensor: 反量化后的浮点张量
        """
        # 如果未提供缩放因子则使用实例变量
        if scale is None:
            scale = self.scale
        # 如果未提供零点则使用实例变量
        if zero_point is None:
            zero_point = self.zero_point
        # 获取参数重塑形状
        range_shape = self.get_reshape_range(inputs)
        # 重塑缩放因子和零点以匹配输入形状
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        # 执行均匀反量化：(量化值 - 零点) * 缩放因子
        outputs = (inputs - zero_point) * scale
        return outputs
