# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
观察器构建工具模块

该模块提供了观察器的工厂函数，用于根据字符串标识符创建相应的观察器实例。
支持多种类型的观察器，包括最小最大、EMA、OMSE、百分位和PTF观察器。

依赖项:
    .ema.EmaObserver: EMA观察器
    .minmax.MinmaxObserver: 最小最大观察器
    .omse.OmseObserver: OMSE观察器
    .percentile.PercentileObserver: 百分位观察器
    .ptf.PtfObserver: PTF观察器
"""

from .ema import EmaObserver
from .minmax import MinmaxObserver
from .omse import OmseObserver
from .percentile import PercentileObserver
from .ptf import PtfObserver

# 观察器类型映射字典，将字符串标识符映射到对应的观察器类
str2observer = {
    "minmax": MinmaxObserver,      # 最小最大观察器
    "ema": EmaObserver,            # 指数移动平均观察器
    "omse": OmseObserver,          # 最优均方误差观察器
    "percentile": PercentileObserver,  # 百分位观察器
    "ptf": PtfObserver,            # PTF观察器
}


def build_observer(observer_str, module_type, bit_type, calibration_mode):
    """
    构建观察器实例的工厂函数

    根据提供的观察器类型字符串创建相应的观察器实例。

    Args:
        observer_str (str): 观察器类型字符串，如"minmax"、"ema"等
        module_type (str): 模块类型
        bit_type (BitType): 位类型对象
        calibration_mode (str): 校准模式

    Returns:
        BaseObserver: 对应类型的观察器实例

    Raises:
        KeyError: 如果observer_str不在str2observer字典中
    """
    # 根据字符串标识符获取对应的观察器类
    observer = str2observer[observer_str]
    # 创建并返回观察器实例
    return observer(module_type, bit_type, calibration_mode)
