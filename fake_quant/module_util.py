"""
模块工具函数模块

该模块提供了模块操作的实用工具函数，包括模块替换和RMSN层实现。
用于在神经网络模块层次结构中查找和替换特定类型的模块。

依赖项:
    torch: PyTorch深度学习框架
    typing: 类型提示支持
    transformers: Hugging Face Transformers库
    os: 操作系统接口
    logging: 日志记录模块
"""

import torch
import typing
import transformers
import os
import logging


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """使用提供的模块工厂替换给定类型的模块

    从根模块开始对模块层次结构执行深度优先搜索，
    并将所有type_to_replace类型的实例替换为new_module_factory创建的模块。
    被替换模块的子模块不会被处理。

    Args:
        root (torch.nn.Module): 模块层次结构的根节点，从这里开始替换模块
        type_to_replace (type): 需要被替换的模块类型
        new_module_factory (callable): 给定应被替换的模块，
            产生用于替换它的新模块的函数
        replace_layers (bool): 是否替换层，用于区分不同的替换场景
            - True: 用于替换transformer层的场景
            - False: 用于融合layernorms的场景
    """
    # 遍历根模块的子模块
    for name, module in root.named_children():
        new_module = None
        # 检查当前模块是否为要替换的类型
        if isinstance(module, type_to_replace):
            # 根据replace_layers标志决定如何创建新模块
            if (
                replace_layers
            ):  # layernorm_fusion.replace_layers case where transformer layers are replaced
                # 为替换transformer层的情况创建新模块
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                # 为融合layernorms的情况创建新模块
                new_module = new_module_factory(module)
        # 如果当前模块有子模块，则递归处理
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        # 如果创建了新模块，则替换原模块
        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    均方根归一化层(RMSN)实现类

    该类实现了均方根归一化层，用于替代标准的LayerNorm。
    使用来自LLAMARMSNorm的实现：
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75

    依赖项:
        torch.nn.Module: PyTorch神经网络模块基类

    属性:
        eps (float): 用于数值稳定性的极小值
        mean_dim (int): 计算均值的维度
        weight (torch.nn.Parameter): 权重参数
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        """
        初始化RMSN层

        Args:
            mean_dim (int): 计算均值的维度
            eps (float): 用于数值稳定性的极小值，默认为1e-5
        """
        # 调用父类初始化方法
        super().__init__()
        # 设置数值稳定性常数
        self.eps = eps
        # 设置计算均值的维度
        self.mean_dim = mean_dim
        # 初始化权重参数
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，执行均方根归一化

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 归一化后的输出张量
        """
        # 保存输入数据类型
        input_dtype = x.dtype
        # 如果输入为float16，则转换为float32进行计算
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        # 计算方差：平方和除以维度
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        # 执行均方根归一化：输入乘以方差的倒数平方根
        x = x * torch.rsqrt(variance + self.eps)
        # 将结果转换回原始数据类型
        return x.to(input_dtype)
