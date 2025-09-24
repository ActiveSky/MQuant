# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
观察器基类模块

该模块定义了量化观察器的基类，用于收集张量的统计信息以计算量化参数。
提供了张量重塑、值更新和量化参数获取的接口。

依赖项:
    torch: PyTorch深度学习框架
"""

import torch


class BaseObserver:
    """
    观察器基类，用于收集张量统计信息以计算量化参数

    该类定义了量化观察器的基本接口和通用功能，包括张量重塑、值跟踪和参数计算。
    子类需要实现update和get_quantization_params方法。

    依赖项:
        torch: PyTorch深度学习框架

    属性:
        module_type (str): 模块类型，如"conv_weight"、"linear_weight"、"activation"等
        bit_type (BitType): 位类型对象，定义量化位宽和符号性
        calibration_mode (str): 校准模式，如"layer_wise"等
        max_val (torch.Tensor or None): 跟踪的最大值
        min_val (torch.Tensor or None): 跟踪的最小值
        eps (float): 用于数值稳定性的小值
    """

    def __init__(self, module_type, bit_type, calibration_mode):
        """
        初始化BaseObserver实例

        Args:
            module_type (str): 模块类型，决定张量重塑方式
            bit_type (BitType): 位类型对象
            calibration_mode (str): 校准模式
        """
        # 存储模块类型，用于确定张量重塑策略
        self.module_type = module_type
        # 存储位类型信息
        self.bit_type = bit_type
        # 存储校准模式
        self.calibration_mode = calibration_mode
        # 初始化最大值跟踪器
        self.max_val = None
        # 初始化最小值跟踪器
        self.min_val = None
        # 设置数值稳定性常数，防止除零错误
        self.eps = torch.finfo(torch.float32).eps

    def reshape_tensor(self, v):
        """
        根据模块类型重塑张量形状

        不同类型的模块需要不同的张量重塑策略以正确计算量化参数：
        - 卷积和线性层权重：重塑为(out_features, -1)
        - Softmax层：保持原形状
        - 其他层：根据维度进行特定重塑

        Args:
            v (torch.Tensor or array-like): 输入张量

        Returns:
            torch.Tensor: 重塑后的张量
        """
        # 如果输入不是张量则转换为张量
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        # 分离张量以避免影响计算图
        v = v.detach()
        # 卷积层和线性层权重重塑：(out_features, in_features)
        if self.module_type in ["conv_weight", "linear_weight"]:
            v = v.reshape(v.shape[0], -1)
        # Softmax层保持原形状
        elif self.module_type in ["softmax"]:
            return v
        # 其他类型张量的重塑
        else:
            # 4维张量（如卷积特征图）的维度重排
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1)
            # 重塑为(-1, last_dim)形式
            v = v.reshape(-1, v.shape[-1])
            # 转置以适应通道维度处理
            v = v.transpose(0, 1)
        return v

    def update(self, v):
        """
        更新观察器的统计信息（最大值和最小值）

        该方法需要在子类中实现，用于根据输入张量更新max_val和min_val。

        Args:
            v (torch.Tensor): 输入张量

        Raises:
            NotImplementedError: 基类方法未实现，需要子类重写
        """
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        """
        计算并返回量化参数（如缩放因子和零点）

        该方法需要在子类中实现，用于基于收集的统计信息计算量化参数。

        Args:
            *args: 可变位置参数
            **kwargs: 可变关键字参数

        Returns:
            tuple: 量化参数（通常包括scale和zero_point）

        Raises:
            NotImplementedError: 基类方法未实现，需要子类重写
        """
        raise NotImplementedError
