# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
EMA观察器模块

该模块实现了基于指数移动平均(EMA)的量化参数观察器，通过EMA来跟踪张量的统计信息。
EMA观察器能够平滑地更新最大值和最小值估计，对动态范围变化更加适应。

依赖项:
    torch: PyTorch深度学习框架
    .base.BaseObserver: 观察器基类
"""

import torch

from .base import BaseObserver


class EmaObserver(BaseObserver):
    """
    EMA观察器，基于指数移动平均计算量化参数

    该类继承自BaseObserver，使用指数移动平均方法来跟踪张量的最大值和最小值。
    EMA方法能够平滑地适应输入数据的动态范围变化。

    依赖项:
        torch: PyTorch深度学习框架
        BaseObserver: 观察器基类

    属性:
        ema_sigma (float): 指数移动平均的衰减因子
        symmetric (bool): 是否使用对称量化，基于位类型的符号性确定
    """

    def __init__(self, module_type, bit_type, calibration_mode, ema_sigma=0.01):
        """
        初始化EmaObserver实例

        Args:
            module_type (str): 模块类型
            bit_type (BitType): 位类型对象
            calibration_mode (str): 校准模式
            ema_sigma (float): 指数移动平均的衰减因子，默认为0.01
        """
        # 调用父类初始化方法
        super(EmaObserver, self).__init__(module_type, bit_type, calibration_mode)
        # 设置指数移动平均的衰减因子
        self.ema_sigma = ema_sigma
        # 根据位类型确定是否使用对称量化
        self.symmetric = self.bit_type.signed

    def update(self, v):
        """
        使用指数移动平均更新统计信息

        通过EMA来更新最大值和最小值估计，能够平滑地适应输入数据的变化。
        在layer_wise模式下，会对所有通道取全局最大最小值。

        Args:
            v (torch.Tensor): 输入张量
        """
        # 根据模块类型重塑张量
        v = self.reshape_tensor(v)
        # 计算当前批次的最大值（沿第一个维度）
        cur_max = v.max(axis=1).values
        # 更新全局最大值跟踪器（使用指数移动平均）
        if self.max_val is None:
            # 初始化最大值
            self.max_val = cur_max
        else:
            # 使用指数移动平均更新最大值
            self.max_val = self.max_val + self.ema_sigma * (cur_max - self.max_val)
        # 计算当前批次的最小值（沿第一个维度）
        cur_min = v.min(axis=1).values
        # 更新全局最小值跟踪器（使用指数移动平均）
        if self.min_val is None:
            # 初始化最小值
            self.min_val = cur_min
        else:
            # 使用指数移动平均更新最小值
            self.min_val = self.min_val + self.ema_sigma * (cur_min - self.min_val)

        # 如果是层级校准模式，则取所有通道的全局最大最小值
        if self.calibration_mode == "layer_wise":
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, *args, **kwargs):
        """
        基于EMA收集的统计信息计算量化参数

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
            # 使用最大绝对值作为范围
            max_val = torch.max(-min_val, max_val)
            # 缩放因子为最大值除以量化范围的一半
            scale = max_val / (float(qmax - qmin) / 2)
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
