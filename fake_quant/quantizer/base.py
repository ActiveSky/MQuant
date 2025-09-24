# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
量化器基类模块

该模块定义了量化器的基类，提供了量化和反量化的基本接口和通用功能。
继承自PyTorch的nn.Module，可直接集成到神经网络中。

依赖项:
    torch: PyTorch深度学习框架
    torch.nn (nn): PyTorch神经网络模块
"""

import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):
    """
    量化器基类，继承自PyTorch的nn.Module

    该类定义了量化器的基本接口和通用功能，包括形状重塑、参数更新、量化和反量化方法。
    子类需要实现quant和dequantize方法。

    依赖项:
        torch: PyTorch深度学习框架
        torch.nn (nn): PyTorch神经网络模块

    属性:
        bit_type (BitType): 位类型对象，定义量化位宽和符号性
        observer (BaseObserver): 观察器对象，用于收集统计信息
        module_type (str): 模块类型，如"conv_weight"、"linear_weight"、"activation"等
    """

    def __init__(self, bit_type, observer, module_type):
        """
        初始化BaseQuantizer实例

        Args:
            bit_type (BitType): 位类型对象
            observer (BaseObserver): 观察器对象
            module_type (str): 模块类型
        """
        # 调用父类初始化方法
        super(BaseQuantizer, self).__init__()
        # 存储位类型信息
        self.bit_type = bit_type
        # 存储观察器对象
        self.observer = observer
        # 存储模块类型
        self.module_type = module_type

    def get_reshape_range(self, inputs):
        """
        根据模块类型获取重塑范围形状

        不同类型的模块需要不同的形状重塑策略以正确应用量化参数：
        - 卷积权重：(-1, 1, 1, 1) - 每个输出通道一个参数
        - 线性权重：(-1, 1) - 每个输出特征一个参数
        - 激活值：根据输入维度确定参数形状

        Args:
            inputs (torch.Tensor): 输入张量，用于确定维度信息

        Returns:
            tuple: 重塑后的参数形状

        Raises:
            NotImplementedError: 不支持的模块类型或维度
        """
        # 初始化重塑形状
        range_shape = None
        # 卷积层权重参数形状：每个输出通道一个参数
        if self.module_type == "conv_weight":
            range_shape = (-1, 1, 1, 1)
        # 线性层权重参数形状：每个输出特征一个参数
        elif self.module_type == "linear_weight":
            range_shape = (-1, 1)
        # 激活值参数形状：根据输入维度确定
        elif self.module_type == "activation":
            # 2D张量（如批次，特征）
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            # 3D张量（如批次，序列，特征）
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            # 4D张量（如批次，通道，高，宽）
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            # 5D张量（如批次，通道，时间，高，宽）
            elif len(inputs.shape) == 5:
                range_shape = (1, -1, 1, 1, 1)
            # 不支持的维度
            else:
                raise NotImplementedError
        # 不支持的模块类型
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        """
        更新量化参数

        该方法调用观察器来更新量化参数，可在子类中重写以实现特定逻辑。

        Args:
            *args: 可变位置参数，传递给观察器
            **kwargs: 可变关键字参数，传递给观察器
        """
        # 基类实现为空，子类可重写
        pass

    def quant(self, inputs, scale=None, zero_point=None):
        """
        将浮点输入量化为整数

        该方法需要在子类中实现具体的量化逻辑。

        Args:
            inputs (torch.Tensor): 浮点输入张量
            scale (torch.Tensor, optional): 缩放因子
            zero_point (torch.Tensor, optional): 零点

        Returns:
            torch.Tensor: 量化后的整数张量

        Raises:
            NotImplementedError: 基类方法未实现，需要子类重写
        """
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        """
        将整数量化值反量化为浮点数

        该方法需要在子类中实现具体的反量化逻辑。

        Args:
            inputs (torch.Tensor): 整数量化张量
            scale (torch.Tensor, optional): 缩放因子
            zero_point (torch.Tensor, optional): 零点

        Returns:
            torch.Tensor: 反量化后的浮点张量

        Raises:
            NotImplementedError: 基类方法未实现，需要子类重写
        """
        raise NotImplementedError

    def forward(self, inputs):
        """
        前向传播方法，执行量化-反量化过程

        该方法在神经网络前向传播时自动调用，实现模拟量化效果。

        Args:
            inputs (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 量化-反量化后的输出张量（保持原始数据类型）
        """
        # 保存原始数据类型
        dtype = inputs.dtype
        # 转换为浮点数进行计算
        inputs = inputs.float()
        # 执行量化操作
        outputs = self.quant(inputs)
        # 执行反量化操作
        outputs = self.dequantize(outputs)
        # 恢复原始数据类型
        outputs = outputs.to(dtype)
        return outputs
