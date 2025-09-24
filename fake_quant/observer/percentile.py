# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
百分位观察器模块

该模块实现了基于百分位数的量化参数观察器，通过跟踪张量的百分位数值来计算量化参数。
相比最小最大观察器，百分位观察器对异常值更加鲁棒。

依赖项:
    numpy (np): 数值计算库
    torch: PyTorch深度学习框架
    torch.nn (nn): PyTorch神经网络模块
    .base.BaseObserver: 观察器基类
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseObserver


class PercentileObserver(BaseObserver):
    """
    百分位观察器，基于张量的百分位数值计算量化参数

    该类继承自BaseObserver，使用百分位数方法来估计张量的范围，对异常值更加鲁棒。
    使用指数移动平均来更新最大值和最小值估计。

    依赖项:
        torch: PyTorch深度学习框架
        numpy (np): 数值计算库
        BaseObserver: 观察器基类

    属性:
        percentile_sigma (float): 指数移动平均的衰减因子
        percentile_alpha (float): 百分位数参数，通常接近1.0
        symmetric (bool): 是否使用对称量化，基于位类型的符号性确定
    """

    def __init__(
        self,
        module_type,
        bit_type,
        calibration_mode,
        percentile_sigma=0.01,
        percentile_alpha=0.99999,
    ):
        """
        初始化PercentileObserver实例

        Args:
            module_type (str): 模块类型
            bit_type (BitType): 位类型对象
            calibration_mode (str): 校准模式
            percentile_sigma (float): 指数移动平均的衰减因子，默认为0.01
            percentile_alpha (float): 百分位数参数，通常接近1.0，默认为0.99999
        """
        # 调用父类初始化方法
        super(PercentileObserver, self).__init__(
            module_type, bit_type, calibration_mode
        )
        # 设置指数移动平均的衰减因子
        self.percentile_sigma = percentile_sigma
        # 设置百分位数参数
        self.percentile_alpha = percentile_alpha
        # 根据位类型确定是否使用对称量化
        self.symmetric = self.bit_type.signed

    def update(self, v):
        """
        更新百分位数统计信息

        使用指数移动平均来更新最大值和最小值估计，对异常值更加鲁棒。
        仅支持layer_wise校准模式。

        Args:
            v (torch.Tensor): 输入张量

        Raises:
            AssertionError: 如果校准模式不是layer_wise
        """
        # channel-wise需要太多时间，因此只支持layer_wise模式
        assert self.calibration_mode == "layer_wise"
        # 根据模块类型重塑张量
        v = self.reshape_tensor(v)
        # 尝试使用PyTorch的quantile函数计算百分位数
        try:
            # 计算最大值百分位数
            cur_max = torch.quantile(v.reshape(-1).float(), self.percentile_alpha)
            # 计算最小值百分位数
            cur_min = torch.quantile(v.reshape(-1).float(), 1.0 - self.percentile_alpha)
        # 如果PyTorch的quantile函数失败，则使用numpy的percentile函数
        except:
            # 使用numpy计算最大值百分位数
            cur_max = torch.tensor(
                np.percentile(v.reshape(-1).cpu(), self.percentile_alpha * 100),
                device=v.device,
                dtype=torch.float32,
            )
            # 使用numpy计算最小值百分位数
            cur_min = torch.tensor(
                np.percentile(v.reshape(-1).cpu(), (1 - self.percentile_alpha) * 100),
                device=v.device,
                dtype=torch.float32,
            )
        # 更新全局最大值跟踪器（使用指数移动平均）
        if self.max_val is None:
            # 初始化最大值为当前最大值和0中的较大者
            self.max_val = torch.max(cur_max, torch.zeros_like(cur_max))
        else:
            # 使用指数移动平均更新最大值
            self.max_val = self.max_val + self.percentile_sigma * (
                cur_max - self.max_val
            )
        # 更新全局最小值跟踪器（使用指数移动平均）
        if self.min_val is None:
            # 初始化最小值为当前最小值和0中的较小者
            self.min_val = torch.min(cur_min, torch.zeros_like(cur_min))
        else:
            # 使用指数移动平均更新最小值
            self.min_val = self.min_val + self.percentile_sigma * (
                cur_min - self.min_val
            )

    def get_quantization_params(self, *args, **kwargs):
        """
        基于收集的百分位数计算量化参数

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
