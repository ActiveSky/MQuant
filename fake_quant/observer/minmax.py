# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
最小最大观察器模块

该模块实现了基于最小最大值的量化参数观察器，通过跟踪张量的最小值和最大值来计算量化参数。
支持对称和非对称量化。

依赖项:
    torch: PyTorch深度学习框架
    .base.BaseObserver: 观察器基类
"""

import torch

from .base import BaseObserver


class MinmaxObserver(BaseObserver):
    """
    最小最大观察器，基于张量的最小值和最大值计算量化参数

    该类继承自BaseObserver，实现了具体的最小最大值跟踪和量化参数计算逻辑。
    支持对称量化（有符号）和非对称量化（无符号）。

    依赖项:
        torch: PyTorch深度学习框架
        BaseObserver: 观察器基类

    属性:
        symmetric (bool): 是否使用对称量化，基于位类型的符号性确定
    """

    def __init__(self, module_type, bit_type, calibration_mode):
        """
        初始化MinmaxObserver实例

        Args:
            module_type (str): 模块类型
            bit_type (BitType): 位类型对象
            calibration_mode (str): 校准模式
        """
        # 调用父类初始化方法
        super(MinmaxObserver, self).__init__(module_type, bit_type, calibration_mode)
        # 根据位类型确定是否使用对称量化
        self.symmetric = self.bit_type.signed

    def update(self, v):
        """
        更新最小值和最大值统计信息

        通过跟踪输入张量的最小值和最大值来更新内部状态。
        在layer_wise模式下，会对所有通道取全局最大最小值。

        Args:
            v (torch.Tensor): 输入张量
        """
        # 根据模块类型重塑张量
        v = self.reshape_tensor(v)
        # 计算当前批次的最大值（沿最后一个维度）
        cur_max = v.max(axis=-1).values
        # 更新全局最大值跟踪器
        if self.max_val is None:
            # 初始化最大值为当前最大值和0中的较大者
            self.max_val = torch.max(cur_max, torch.zeros_like(cur_max))
        else:
            # 更新最大值为历史最大值和当前最大值中的较大者
            self.max_val = torch.max(cur_max, self.max_val)
        # 计算当前批次的最小值（沿最后一个维度）
        cur_min = v.min(axis=-1).values
        # 更新全局最小值跟踪器
        if self.min_val is None:
            # 初始化最小值为当前最小值和0中的较小者
            self.min_val = torch.min(cur_min, torch.zeros_like(cur_min))
        else:
            # 更新最小值为历史最小值和当前最小值中的较小者
            self.min_val = torch.min(cur_min, self.min_val)

        # 如果是层级校准模式，则取所有通道的全局最大最小值
        if self.calibration_mode == "layer_wise":
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, *args, **kwargs):
        """
        基于收集的最小最大值计算量化参数

        根据位类型的符号性选择对称或非对称量化参数计算方法：
        - 对称量化：使用最大绝对值计算缩放因子
        - 非对称量化：使用最小最大值范围计算缩放因子和零点

        Args:
            *args: 可变位置参数
            **kwargs: 可变关键字参数

        Returns:
            tuple: (scale, zero_point) 量化参数元组
                - scale (torch.Tensor): 缩放因子
                - zero_point (torch.Tensor): 零点（量化后的整数值对应浮点0的位置）
        """
        # 获取跟踪的最小值和最大值
        max_val = self.max_val
        min_val = self.min_val

        # 获取位类型的上下界
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        # 初始化缩放因子和零点
        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        # 对称量化参数计算（适用于有符号量化）
        if self.symmetric:
            # 缩放因子为最小值和最大值绝对值中的较大者除以量化范围
            scale = torch.max(
                torch.abs(min_val / qmin),
                torch.abs(max_val / qmax),
            )
            # 钳位到eps以保证数值稳定性
            scale.clamp_(self.eps)
            # 对称量化零点始终为0
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        # 非对称量化参数计算（适用于无符号量化）
        else:
            # 缩放因子为值范围除以量化范围
            scale = (max_val - min_val) / float(qmax - qmin)
            # 钳位到eps以保证数值稳定性
            scale.clamp_(self.eps)
            # 零点计算：量化范围的最小值减去浮点范围最小值除以缩放因子
            zero_point = qmin - torch.round(min_val / scale)
            # 钳位零点到合法范围内
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
