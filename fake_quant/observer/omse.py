# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
OMSE观察器模块

该模块实现了基于最优均方误差(OMSE)的量化参数观察器，通过最小化量化误差来优化量化参数。
OMSE观察器使用网格搜索方法找到最优的缩放因子和零点。

依赖项:
    torch: PyTorch深度学习框架
    .base.BaseObserver: 观察器基类
    .utils.lp_loss: Lp范数损失函数
"""

import torch

from .base import BaseObserver
from .utils import lp_loss


class OmseObserver(BaseObserver):
    """
    OMSE观察器，基于最优均方误差计算量化参数

    该类继承自BaseObserver，通过最小化L2范数损失来优化量化参数。
    使用网格搜索方法在一定范围内搜索最优的缩放因子和零点。

    依赖项:
        torch: PyTorch深度学习框架
        BaseObserver: 观察器基类
        lp_loss: Lp范数损失函数
    """

    def __init__(self, module_type, bit_type, calibration_mode):
        """
        初始化OmseObserver实例

        Args:
            module_type (str): 模块类型
            bit_type (BitType): 位类型对象
            calibration_mode (str): 校准模式
        """
        # 调用父类初始化方法
        super(OmseObserver, self).__init__(module_type, bit_type, calibration_mode)

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
        # 计算当前批次的最大值（沿第一个维度）
        cur_max = v.max(axis=1).values
        # 更新全局最大值跟踪器
        if self.max_val is None:
            # 初始化最大值
            self.max_val = cur_max
        else:
            # 更新最大值为历史最大值和当前最大值中的较大者
            self.max_val = torch.max(cur_max, self.max_val)
        # 计算当前批次的最小值（沿第一个维度）
        cur_min = v.min(axis=1).values
        # 更新全局最小值跟踪器
        if self.min_val is None:
            # 初始化最小值
            self.min_val = cur_min
        else:
            # 更新最小值为历史最小值和当前最小值中的较小者
            self.min_val = torch.min(cur_min, self.min_val)

        # 如果是层级校准模式，则取所有通道的全局最大最小值
        if self.calibration_mode == "layer_wise":
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs):
        """
        基于最优均方误差计算量化参数

        使用网格搜索方法在一定范围内搜索最优的缩放因子和零点，
        使得量化后的张量与原始张量之间的L2范数损失最小。

        Args:
            inputs (torch.Tensor): 输入张量，用于计算量化误差

        Returns:
            tuple: (scale, zero_point) 最优量化参数元组
                - scale (torch.Tensor): 最优缩放因子
                - zero_point (torch.Tensor): 最优零点
        """
        # 获取跟踪的最小值和最大值
        max_val = self.max_val
        min_val = self.min_val
        # 获取位类型的上下界
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        # 初始化最佳评分
        best_score = 1e10
        # 网格搜索最优参数（90个搜索点）
        for i in range(90):
            # 计算新的最大值和最小值（逐步收缩范围）
            new_max = max_val * (1.0 - (i * 0.01))
            new_min = min_val * (1.0 - (i * 0.01))
            # 计算新的缩放因子
            new_scale = (new_max - new_min) / float(qmax - qmin)
            # 钳位到eps以保证数值稳定性
            new_scale.clamp_(self.eps)
            # 计算新的零点
            new_zero_point = qmin - torch.round(new_min / new_scale)
            # 钳位零点到合法范围内
            new_zero_point.clamp_(qmin, qmax)
            # 执行量化操作
            inputs_q = (
                (inputs / new_scale + new_zero_point).round().clamp(qmin, qmax)
                - new_zero_point
            ) * new_scale
            # L_p范数最小化，如LAPQ中所述
            # https://arxiv.org/abs/1911.07190
            # 计算L2范数损失
            score = lp_loss(inputs, inputs_q, p=2.0, reduction="all")
            # 更新最优参数
            if score < best_score:
                best_score = score
                self.max_val = new_max
                self.min_val = new_min
                scale = new_scale
                zero_point = new_zero_point
        return scale, zero_point
