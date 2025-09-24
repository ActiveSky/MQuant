# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
"""
位类型定义模块

该模块定义了量化中使用的位类型，包括位宽、符号性等属性，以及相关的边界计算方法。
用于支持不同精度的量化操作。

依赖项:
    numpy (np): 数值计算库
    torch: PyTorch深度学习框架
    torch.nn (nn): PyTorch神经网络模块
"""

import numpy as np
import torch
import torch.nn as nn


class BitType:
    """
    位类型类，用于定义量化中的位宽和符号属性

    该类封装了量化位类型的相关信息，包括位数、是否为有符号数以及对应的数值范围。
    提供了计算上下界和数值范围的方法。

    依赖项:
        无外部依赖

    属性:
        bits (int): 位宽，表示量化使用的位数
        signed (bool): 符号性，True表示有符号数，False表示无符号数
        name (str): 位类型名称，如"int8"、"uint4"等
    """

    def __init__(self, bits, signed, name=None):
        """
        初始化BitType实例

        Args:
            bits (int): 位宽，表示量化使用的位数
            signed (bool): 符号性，True表示有符号数，False表示无符号数
            name (str, optional): 位类型名称，如未提供则自动生成
        """
        # 存储位宽信息
        self.bits = bits
        # 存储符号性信息
        self.signed = signed
        # 如果提供了名称则使用，否则根据位宽和符号性生成名称
        if name is not None:
            self.name = name
        else:
            self.update_name()

    @property
    def upper_bound(self):
        """
        计算该位类型的上界值

        对于无符号数，上界为2^bits - 1
        对于有符号数，上界为2^(bits-1) - 1

        Returns:
            int: 上界值
        """
        # 无符号数的上界计算：2^bits - 1
        if not self.signed:
            return 2**self.bits - 1
        # 有符号数的上界计算：2^(bits-1) - 1
        return 2 ** (self.bits - 1) - 1

    @property
    def lower_bound(self):
        """
        计算该位类型的下界值

        对于无符号数，下界为0
        对于有符号数，下界为-(2^(bits-1))

        Returns:
            int: 下界值
        """
        # 无符号数的下界为0
        if not self.signed:
            return 0
        # 有符号数的下界计算：-(2^(bits-1))
        return -(2 ** (self.bits - 1))

    @property
    def range(self):
        """
        计算该位类型的数值范围

        范围为2^bits，表示该位类型能表示的不同数值个数

        Returns:
            int: 数值范围大小
        """
        # 返回该位类型能表示的数值总数：2^bits
        return 2**self.bits

    def update_name(self):
        """
        根据位宽和符号性更新位类型名称

        无符号数前缀为"uint"，有符号数前缀为"int"，后接位数
        例如：uint4, int8等
        """
        # 初始化名称为空字符串
        self.name = ""
        # 根据符号性添加前缀：无符号为"uint"，有符号为"int"
        if not self.signed:
            self.name += "uint"
        else:
            self.name += "int"
        # 添加位数后缀
        self.name += "{}".format(self.bits)


# 预定义的位类型列表，包含常用的量化位类型
BIT_TYPE_LIST = [
    BitType(4, False, "uint4"),   # 4位无符号整数
    BitType(8, True, "int8"),     # 8位有符号整数
    BitType(8, False, "uint8"),   # 8位无符号整数
    BitType(16, True, "int16"),   # 16位有符号整数
    BitType(20, True, "int20"),   # 20位有符号整数
    BitType(18, True, "int18"),   # 18位有符号整数
]
# 位类型字典，通过名称快速查找对应的BitType对象
# 键为位类型名称，值为对应的BitType实例
BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}
