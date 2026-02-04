"""
exam/internvl_quant_config.py

该模块定义了一个配置类 `InternVLQuantConfig`，用于为 InternVL2-1B 模型的量化和评估脚本
`exam/my_quant_internvl.py` 中使用的 `argparse` 参数提供类型声明和默认值。
这有助于提高代码的可读性、可维护性，并支持更好的IDE自动补全和静态类型检查。
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    """
    Config 类用于封装模型量化脚本
    `exam/my_quant_internvl.py` 的所有命令行参数。
    每个属性都直接对应于 `argparse` 定义的参数，并带有相应的类型提示和默认值。
    """
    # 通用参数
    name: Optional[str] = field(default=None, metadata={"help": "当前运行的名称，用于日志文件命名。"})
    demo: bool = field(default=False, metadata={"help": "是否运行演示模式。"})
    quant: bool = field(default=False, metadata={"help": "是否启用模型量化。"})

    # 旋转参数 (Rotation Arguments)
    rotate: bool = field(default=False, metadata={"help": "是否对模型进行旋转优化。"})
    analysis: bool = field(default=False, metadata={"help": "是否进行激活分析。"})
    analysis_c_proj: bool = field(default=False, metadata={"help": "是否对 'c_proj' 层进行激活分析。"})
    draw_save_path: str = field(default="output/qwenvl_base", metadata={"help": "激活分析结果的保存路径。"})
    rotate_visual_clip: bool = field(default=False, metadata={"help": "是否旋转视觉编码器（Visual CLIP）部分。"})
    rotate_visual_cross_attn: bool = field(default=False, metadata={"help": "是否旋转视觉交叉注意力层。"})
    rotate_llm: bool = field(default=False, metadata={"help": "是否旋转语言模型（LLM）部分。"})
    rotate_mode: str = field(default="hadamard", metadata={"help": "旋转模式，可选 'hadamard' 或 'random'。"})

    # 激活量化参数 (Activation Quantization Arguments)
    visual_a_bits: int = field(default=8, metadata={"help": "视觉模块线性层输入的比特数。"})
    llm_a_bits: int = field(default=8, metadata={"help": "语言模型线性层输入的比特数。"})
    a_groupsize: int = field(default=-1, metadata={"help": "激活量化的 group size。应与权重量化的 group size 相同。"})
    a_asym: bool = field(default=False, metadata={"help": "激活量化是否使用非对称量化（默认：False）。"})
    a_clip_ratio: float = field(default=1.0, metadata={"help": "激活量化的裁剪比例。新的最大值 = 原始最大值 * 裁剪比例。"})

    # 权重量化参数 (Weight Quantization Arguments)
    visual_w_bits: int = field(default=4, metadata={"help": "视觉模块线性层权重的比特数。"})
    llm_w_bits: int = field(default=4, metadata={"help": "语言模型线性层权重的比特数。"})
    w_groupsize: int = field(default=-1, metadata={"help": "权重量化的 group size。应与激活量化的 group size 相同。"})
    w_asym: bool = field(default=False, metadata={"help": "权重量化是否使用非对称量化（默认：False）。"})
    visual_w_rtn: bool = field(default=False, metadata={"help": "是否对视觉模块权重使用 RtN (Round-to-Nearest) 量化。如果权重比特数小于16且此标志未设置，则使用 GPTQ。"})
    llm_w_rtn: bool = field(default=False, metadata={"help": "是否对语言模型权重使用 RtN 量化。如果权重比特数小于16且此标志未设置，则使用 GPTQ。"})
    visual_w_clip: bool = field(default=False, metadata={"help": "是否对视觉模块权重量化进行裁剪。脚本在权重量化期间会自动寻找最佳裁剪比例。"})
    llm_w_clip: bool = field(default=False, metadata={"help": "是否对语言模型权重量化进行裁剪。脚本在权重量化期间会自动寻找最佳裁剪比例。"})
    percdamp: float = field(default=0.01, metadata={"help": "GPTQ 中用于阻尼的 Hessian 对角线平均值的百分比。"})
    act_order: bool = field(default=False, metadata={"help": "GPTQ 中是否使用 act-order。"})
    seed: int = field(default=42, metadata={"help": "随机种子，用于复现实验结果。"})

    # 通用量化参数 (General Quantization Arguments)
    int8_down_proj: bool = field(default=False, metadata={"help": "是否对下投影层使用 INT8 量化。如果启用，该层的权重和激活都将是 INT8。"})
    quant_llm: bool = field(default=False, metadata={"help": "是否量化 InternVL2-1B 语言模型部分。"})
    quant_visual_clip: bool = field(default=False, metadata={"help": "是否量化视觉特征模型部分。"})
    quant_cross_attention: bool = field(default=False, metadata={"help": "是否量化交叉注意力模型部分。"})
    act_per_tensor: bool = field(default=False, metadata={"help": "是否对激活进行 per-tensor 量化。"})
    nsamples: int = field(default=8, metadata={"help": "GPTQ 校准数据样本的数量。"})
    skip_names: List[str] = field(default_factory=list, metadata={"help": "跳过这些指定名称的层的量化。"})
    no_fuse_visual_clip: bool = field(default=False, metadata={"help": "是否不融合视觉编码器（Visual CLIP）相关的层。"})
    no_fuse_visual_cross_attn: bool = field(default=False, metadata={"help": "是否不融合视觉交叉注意力相关的层。"})
    no_fuse_llm: bool = field(default=False, metadata={"help": "是否不融合语言模型相关的层。"})
    not_fuse_layer_norms: bool = field(default=False, metadata={"help": "是否不融合 Layer Normalization 层。"})
    llm_static: bool = field(default=False, metadata={"help": "是否对语言模型激活使用静态（static）尺度和零点进行量化。"})
    visual_static: bool = field(default=False, metadata={"help": "是否对视觉模块激活使用静态（static）尺度和零点进行量化。"})
    calib_num: int = field(default=32, metadata={"help": "校准样本的数量。"})
    eval_num: int = field(default=32, metadata={"help": "评估样本的数量。"})
    calib_mode: str = field(default="v2", metadata={"help": "校准模式，可选 'v1' 或 'v2'。"})
    analysis_num: int = field(default=32, metadata={"help": "分析样本的数量。"})
    analysis_mode: str = field(default="v1", metadata={"help": "分析模式，可选 'v1' 或 'v2'。" })
    dataset_name: str = field(default="TextVQA_VAL", metadata={"help": "用于校准和评估的数据集名称。"})
    analysis_text: bool = field(default=False, metadata={"help": "是否进行文本分析。"})
    online_visual_hadamard: bool = field(default=False, metadata={"help": "是否对视觉模块使用在线 Hadamard 旋转。"})
    online_llm_hadamard: bool = field(default=False, metadata={"help": "是否对语言模型使用在线 Hadamard 旋转。"})
    fp32_had: bool = field(default=False, metadata={"help": "Hadamard 旋转是否在 FP32 精度下应用（默认：False）。"})
    dump_gptq: Optional[str] = field(default=None, metadata={"help": "将 GPTQ 量化后的模型保存到此路径。"})
    load_gptq: Optional[str] = field(default=None, metadata={"help": "从指定路径加载 GPTQ 量化后的模型。"})
    visual_split: bool = field(default=False, metadata={"help": "是否对视觉模块进行权重拆分（用于在线 Hadamard 旋转）。"})
    llm_split: bool = field(default=False, metadata={"help": "是否对语言模型进行权重拆分（用于在线 Hadamard 旋转）。"})

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> "Config":
        """
        从 argparse.Namespace 对象创建一个 Config 实例。

        Args:
            args (argparse.Namespace): 包含所有命令行解析参数的命名空间对象。

        Returns:
            Config: 一个填充了 argparse 参数值的 Config 实例。
        """
        # 构建一个字典，只包含 InternVLQuantConfig 中定义的属性，
        # 并从 argparse.Namespace 中获取对应的值
        config_kwargs = {
            key: getattr(args, key)
            for key in cls.__annotations__.keys()
            if hasattr(args, key)
        }
        return cls(**config_kwargs)

def get_parser() -> argparse.ArgumentParser:
    """
    获取 Config 对应的 argparse.ArgumentParser 实例。
    这个函数是为了方便快速查阅参数定义，但实际使用中，
    建议直接定义 Config 类并结合 from_argparse_args 方法。

    Returns:
        argparse.ArgumentParser: 包含了所有 InternVLQuantConfig 成员的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(description="InternVL2-1B 模型量化和评估脚本。")
    # 遍历 Config 的所有字段，并添加到 parser 中
    for field_name, field_obj in Config.__dataclass_fields__.items():
        field_type = field_obj.type
        default_value = field_obj.default
        help_text = field_obj.metadata.get("help", "")

        # 处理 List[str] 类型
        if field_type == List[str]:
            parser.add_argument(
                f"--{field_name}",
                nargs="+",
                default=default_value,
                help=help_text
            )
        # 处理 bool 类型 (action="store_true")
        elif field_type == bool:
            # 对于 bool 类型的参数，如果默认值是 False，则使用 action='store_true'
            # 如果默认值是 True，则使用 action='store_false'
            # 但通常命令行参数的 bool 型都是以 --flag 开启某个功能，默认关闭 (False)
            if default_value is False:
                parser.add_argument(
                    f"--{field_name}",
                    action="store_true",
                    default=default_value,
                    help=help_text
                )
            else: # 如果有 default=True 的情况，可以用 action='store_false' 来关闭
                parser.add_argument(
                    f"--{field_name}",
                    action="store_false",
                    default=default_value,
                    help=help_text
                )
        # 处理 Optional[str] 或其他类型
        elif field_type == Optional[str]:
            parser.add_argument(
                f"--{field_name}",
                type=str,
                default=default_value,
                help=help_text
            )
        else: # 其他常规类型
            parser.add_argument(
                f"--{field_name}",
                type=field_type,
                default=default_value,
                help=help_text
            )
    return parser

# 示例用法 (可以在 `if __name__ == "__main__":` 块中测试)
if __name__ == "__main__":
    # 创建一个模拟的 argparse Namespace 对象，或使用实际的 ArgumentParser 来解析
    # 为了演示，我们手动创建一个 Namespace
    mock_args = argparse.Namespace(
        name="test_run",
        quant=True,
        llm_w_bits=4,
        skip_names=["layer1", "layer2"],
        rotate=True,
        visual_a_bits=8,
        dataset_name="OCRBench",
        online_llm_hadamard=True,
        visual_split=False
    )
    # 假设 get_parser() 返回的是原始脚本的 parser
    # actual_parser = get_parser()
    # actual_args = actual_parser.parse_args()

    # 从命名空间创建配置实例
    config_instance = Config.from_argparse_args(mock_args)

    print("Config 实例创建成功:")
    print(f"Name: {config_instance.name}")
    print(f"Quantization enabled: {config_instance.quant}")
    print(f"LLM Weight Bits: {config_instance.llm_w_bits}")
    print(f"Skipped Layers: {config_instance.skip_names}")
    print(f"Rotate: {config_instance.rotate}")
    print(f"Visual Activation Bits: {config_instance.visual_a_bits}")
    print(f"Dataset Name: {config_instance.dataset_name}")
    print(f"Online LLM Hadamard: {config_instance.online_llm_hadamard}")
    print(f"Visual Split: {config_instance.visual_split}")

    # 尝试访问一个未在 mock_args 中设置但 Config 中有默认值的属性
    print(f"Demo mode (default): {config_instance.demo}")

    # 可以通过 get_parser() 来验证参数解析
    # parser = get_parser()
    # parsed_args = parser.parse_args(['--quant', '--llm_w_bits', '4', '--skip_names', 'layer1', 'layer2'])
    # config_from_parsed = InternVLQuantConfig.from_argparse_args(parsed_args)
    # print("\nConfig 实例从实际解析的参数创建:")
    # print(f"Quantization enabled: {config_from_parsed.quant}")
    # print(f"LLM Weight Bits: {config_from_parsed.llm_w_bits}")
    # print(f"Skipped Layers: {config_from_parsed.skip_names}")
