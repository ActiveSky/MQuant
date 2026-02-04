"""
exam/my_quant_internvl.py

该脚本用于对 InternVL2-1B 模型进行量化和评估。
它支持多种量化技术，包括基于 GPTQ 的权重截断、Hadamard 旋转以及激活量化，
旨在优化模型的性能和部署效率。

主要功能:
1. 加载 InternVL2-1B 预训练模型。
2. 可选地融合模型中的 Layer Normalization 层。
3. 可选地对模型进行旋转（如 Hadamard 旋转），以优化量化效果。
4. 配置并应用激活量化，可选择在线 Hadamard 旋转。
5. 执行 GPTQ 权重量化（如果未加载预量化模型）。
6. 进行激活校准（如果启用静态量化）。
7. 在指定数据集上评估量化后模型的性能。

命令行参数允许用户细粒度地控制量化的各个方面，包括比特数、group size、
对称性、裁剪比例、校准样本数、旋转模式等。
"""

import torch, torch.nn as nn, torch.nn.functional as F, argparse, datetime, os
from loguru import logger
from evaluation.eval import eval_dataset
from fake_quant import quant_utils
from fake_quant import gptq
from fake_quant import utils
from fake_quant import hadamard_utils
from fake_quant.internvl_rotation import fuse_internvl_layer_norms, rotate_internvl2_model
from vlmeval.config import supported_VLM
from exam.my_config import Config
# 禁用梯度计算，因为该脚本主要用于模型量化和评估，不需要训练
torch.set_grad_enabled(False)



def init_logger(args:Config):
    """
    初始化日志记录器，将日志输出到文件和控制台。

    Args:
        args (argparse.Namespace): 包含脚本参数的对象，其中可能包含 'name' 用于命名日志文件。
    """
    # 生成日志文件名，格式为 月-日 时:分:秒.log
    logger_file = str(datetime.datetime.now().strftime("%m-%d %H:%M:%S")) + ".log"
    # 创建日志目录，如果不存在则创建
    os.makedirs("log", exist_ok=True)
    # 如果指定了名称参数，则将其添加到日志文件名前
    if args.name is not None:
        logger_file = args.name + "_" + logger_file
    # 构造完整的日志文件路径
    logger_file = "log/" + logger_file
    # 添加日志处理器，将日志写入文件
    logger.add(logger_file)


def main(args:Config):
    """
    脚本的主要执行函数，负责模型的加载、量化、校准和评估。

    Args:
        args (argparse.Namespace): 包含所有命令行参数的对象。
    """
    # 定义要加载的模型名称和路径
    model_name = "InternVL2-1B"
    # 从 vlmeval 配置中加载 InternVL2-1B 模型实例
    model = supported_VLM[model_name](model_path="weights/InternVL2-1B")
    # 融合 InternVL 模型中的特定层，以适应量化需求
    quant_utils.fuse_internvl(model)

    # 设置随机种子以保证实验的可复现性
    utils.seed_everything(args.seed)
    # 如果未禁用层归一化融合，则执行融合操作
    if not args.not_fuse_layer_norms:
        fuse_internvl_layer_norms(model, args)
    # 如果指定了旋转参数，则对模型进行旋转
    if args.rotate:
        rotate_internvl2_model(model.model, args)

    # 在不进行量化但启用在线LLM Hadamard旋转的场景
    if not args.quant and args.online_llm_hadamard:
        # 如果LLM需要旋转，则标记为量化LLM
        if args.rotate_llm:
            args.quant_llm = True
        # 为模型添加激活量化包装器
        quant_utils.internvl_add_act_qaunt(model, args)
        # 查找语言模型中所有ActQuantWrapper层
        qlayers = quant_utils.find_qlayers(
            model.model.language_model, layers=[quant_utils.ActQuantWrapper]
        )
        # 遍历这些层，并为其配置Hadamard旋转
        for name in qlayers:
            if "feed_forward.w2" in name: # 针对特定的层
                # 获取Hadamard旋转矩阵K及其逆K
                had_K, K = hadamard_utils.get_hadK(
                    model.model.config.llm_config.intermediate_size
                )
                qlayers[name].online_full_had = True  # 启用在线全Hadamard旋转
                qlayers[name].had_K = had_K            # 设置Hadamard旋转矩阵
                qlayers[name].K = K                    # 设置Hadamard逆矩阵
                qlayers[name].fp32_had = args.fp32_had # 设置是否在FP32进行Hadamard旋转

    # 在不进行量化但启用在线视觉Hadamard旋转的场景
    if not args.quant and args.online_visual_hadamard:
        # 如果视觉裁剪需要旋转，则标记为量化视觉裁剪
        if args.rotate_visual_clip:
            args.quant_visual_clip = True
        # 为模型添加激活量化包装器
        quant_utils.internvl_add_act_qaunt(model, args)
        # 查找视觉模型中所有ActQuantWrapper层
        qlayers = quant_utils.find_qlayers(
            model.model.vision_model, layers=[quant_utils.ActQuantWrapper]
        )
        # 遍历这些层，并为其配置Hadamard旋转
        for name in qlayers:
            if "mlp.fc2" in name: # 针对特定的层
                # 获取Hadamard旋转矩阵K及其逆K
                had_K, K = hadamard_utils.get_hadK(
                    int(model.model.config.vision_config.intermediate_size)
                )
                qlayers[name].online_full_had = True  # 启用在线全Hadamard旋转
                qlayers[name].had_K = had_K            # 设置Hadamard旋转矩阵
                qlayers[name].K = K                    # 设置Hadamard逆矩阵
                qlayers[name].fp32_had = args.fp32_had # 设置是否在FP32进行Hadamard旋转

    # 如果启用了量化 (args.quant 为 True)
    if args.quant:
        # 根据在线LLM Hadamard旋转和LLM旋转参数，标记量化LLM
        if args.online_llm_hadamard:
            if args.rotate_llm:
                args.quant_llm = True
        # 根据在线视觉Hadamard旋转和视觉裁剪旋转参数，标记量化视觉裁剪
        if args.online_visual_hadamard:
            if args.rotate_visual_clip:
                args.quant_visual_clip = True
        # 为模型添加激活量化包装器
        quant_utils.internvl_add_act_qaunt(model, args)

        # 如果同时启用了在线LLM Hadamard旋转和LLM旋转
        if args.online_llm_hadamard and args.rotate_llm:
            print("adding online hadamard rotation for LLM")
            # 查找语言模型中所有ActQuantWrapper层
            qlayers = quant_utils.find_qlayers(
                model.model.language_model, layers=[quant_utils.ActQuantWrapper]
            )
            # 遍历这些层，并为其配置Hadamard旋转
            for name in qlayers:
                if "feed_forward.w2" in name: # 针对特定的层
                    # 获取Hadamard旋转矩阵K及其逆K
                    had_K, K = hadamard_utils.get_hadK(
                        model.model.config.llm_config.intermediate_size
                    )
                    qlayers[name].online_full_had = True # 启用在线全Hadamard旋转
                    qlayers[name].had_K = had_K           # 设置Hadamard旋转矩阵
                    qlayers[name].K = K                   # 设置Hadamard逆矩阵
                    qlayers[name].fp32_had = args.fp32_had # 设置是否在FP32进行Hadamard旋转
                    qlayers[name].split = args.llm_split  # 设置是否进行权重拆分
                    if args.llm_split:
                        qlayers[name].split_weights()     # 执行权重拆分

        # 如果同时启用了在线视觉Hadamard旋转和视觉裁剪旋转
        if args.online_visual_hadamard and args.rotate_visual_clip:
            print("adding online hadamard rotation for visual clip")
            # 查找视觉模型中所有ActQuantWrapper层
            qlayers = quant_utils.find_qlayers(
                model.model.vision_model, layers=[quant_utils.ActQuantWrapper]
            )
            # 遍历这些层，并为其配置Hadamard旋转
            for name in qlayers:
                if "mlp.fc2" in name: # 针对特定的层
                    # 获取Hadamard旋转矩阵K及其逆K
                    had_K, K = hadamard_utils.get_hadK(
                        int(model.model.config.vision_config.intermediate_size)
                    )
                    qlayers[name].online_full_had = True # 启用在线全Hadamard旋转
                    qlayers[name].had_K = had_K           # 设置Hadamard旋转矩阵
                    qlayers[name].K = K                   # 设置Hadamard逆矩阵
                    qlayers[name].fp32_had = args.fp32_had # 设置是否在FP32进行Hadamard旋转
                    qlayers[name].split = args.visual_split # 设置是否进行权重拆分
                    if args.visual_split:
                        qlayers[name].split_weights()     # 执行权重拆分

        # 将模型移动到指定的设备 (GPU)
        model.model.to(utils.DEV)

        # 如果指定了加载 GPTQ 模型路径
        if args.load_gptq:
            print("Loading GPTQ model from: ", args.load_gptq)
            model.model = torch.load(args.load_gptq) # 加载预训练的 GPTQ 模型
        else:
            # 动态导入 build_dataset 函数
            from vlmeval.dataset import build_dataset

            # 构建数据集用于校准
            dataset = build_dataset(args.dataset_name)
            # 如果模型没有图像转储函数，则设置它
            if not hasattr(model, "dump_image_func"):
                model.set_dump_image(dataset.dump_image)

            # 执行 InternVL GPTQ 量化
            gptq.internvl_rtn_gptq_fwrd_plus(
                model, dataset, utils.DEV, args.dataset_name, args
            )
            # 如果指定了保存 GPTQ 模型路径
            if args.dump_gptq:
                torch.save(model.model, args.dump_gptq) # 保存量化后的模型
                print("Dumped the GPTQ model to: ", args.dump_gptq)

        # 视觉模块激活量化配置
        if args.visual_a_bits < 16 or args.visual_static:
            if args.visual_static and args.visual_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            # 查找视觉模型和 mlp1 中所有 ActQuantWrapper 层
            qlayers = quant_utils.find_qlayers(
                model.model.vision_model, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers.update(
                quant_utils.find_qlayers(
                    model.model.mlp1, layers=[quant_utils.ActQuantWrapper]
                )
            )
            # 遍历并配置这些层的量化器
            for name in qlayers:
                # 跳过指定名称的层
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.visual_a_bits # 激活比特数
                layer_groupsize = args.a_groupsize     # 激活 group size
                layer_a_sym = not (args.a_asym)        # 激活是否对称
                layer_a_clip = args.a_clip_ratio       # 激活裁剪比例

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.visual_static,
                    observer_type="minmax", # 观察器类型为 minmax
                )

        # LLM 激活量化配置
        if args.llm_a_bits < 16 or args.llm_static:
            if args.llm_static and args.llm_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            # 查找语言模型中所有 ActQuantWrapper 层
            qlayers = quant_utils.find_qlayers(
                model.model.language_model, layers=[quant_utils.ActQuantWrapper]
            )
            # 遍历并配置这些层的量化器
            for name in qlayers:
                # 跳过指定名称的层
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.llm_a_bits # 激活比特数
                layer_groupsize = args.a_groupsize # 激活 group size
                layer_a_sym = not (args.a_asym)    # 激活是否对称
                layer_a_clip = args.a_clip_ratio   # 激活裁剪比例

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.llm_static,
                    observer_type="minmax", # 观察器类型为 minmax
                )

    # 动态导入 build_dataset 函数
    from vlmeval.dataset import build_dataset

    # 构建数据集用于校准和评估
    dataset = build_dataset(args.dataset_name)

    # 如果模型没有图像转储函数，则设置它
    if not hasattr(model, "dump_image_func"):
        model.set_dump_image(dataset.dump_image)

    # 如果启用了 LLM 或视觉静态量化，则执行 VQA 校准
    if args.llm_static or args.visual_static:
        quant_utils.calib_vqa_plus(model, args, dataset, args.calib_num)

    # 评估量化后模型在指定数据集上的性能
    eval_dataset(
        model,
        dataset,
        args.dataset_name,
        model_name="InternVL-2B", # 评估时使用的模型名称
        verbose=False,              # 不显示详细输出
    )


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="InternVL2-1B 模型量化和评估脚本。")
    parser.add_argument("--name", type=str, default=None, help="当前运行的名称，用于日志文件命名。")
    parser.add_argument("--demo", action="store_true", help="是否运行演示模式。")
    parser.add_argument("--quant", action="store_true", help="是否启用模型量化。")

    # 旋转参数 (Rotation Arguments)
    parser.add_argument(
        "--rotate", action="store_true", default=False, help="是否对模型进行旋转优化。"
    )
    parser.add_argument(
        "--analysis", action="store_true", default=False, help="是否进行激活分析。"
    )
    parser.add_argument(
        "--analysis_c_proj",
        action="store_true",
        default=False,
        help="是否对 'c_proj' 层进行激活分析。",
    )
    parser.add_argument(
        "--draw_save_path",
        type=str,
        default="output/qwenvl_base",
        help="激活分析结果的保存路径。",
    )
    parser.add_argument(
        "--rotate_visual_clip",
        action="store_true",
        default=False,
        help="是否旋转视觉编码器（Visual CLIP）部分。",
    )
    parser.add_argument(
        "--rotate_visual_cross_attn",
        action="store_true",
        default=False,
        help="是否旋转视觉交叉注意力层。",
    )
    parser.add_argument(
        "--rotate_llm",
        action="store_true",
        default=False,
        help="是否旋转语言模型（LLM）部分。",
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"],
        help="旋转模式，可选 'hadamard' 或 'random'。"
    )

    # 激活量化参数 (Activation Quantization Arguments)
    parser.add_argument(
        "--visual_a_bits",
        type=int,
        default=8,
        help="视觉模块线性层输入的比特数。"
    )
    parser.add_argument(
        "--llm_a_bits",
        type=int,
        default=8,
        help="语言模型线性层输入的比特数。"
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="激活量化的 group size。应与权重量化的 group size 相同。"
    )
    parser.add_argument(
        "--a_asym",
        action="store_true",
        default=False,
        help="激活量化是否使用非对称量化（默认：False）。"
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="激活量化的裁剪比例。新的最大值 = 原始最大值 * 裁剪比例。"
    )

    # 权重量化参数 (Weight Quantization Arguments)
    parser.add_argument(
        "--visual_w_bits",
        type=int,
        default=4,
        help="视觉模块线性层权重的比特数。"
    )
    parser.add_argument(
        "--llm_w_bits",
        type=int,
        default=4,
        help="语言模型线性层权重的比特数。"
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="权重量化的 group size。应与激活量化的 group size 相同。"
    )
    parser.add_argument(
        "--w_asym",
        action="store_true",
        default=False,
        help="权重量化是否使用非对称量化（默认：False）。"
    )
    parser.add_argument(
        "--visual_w_rtn",
        action="store_true",
        default=False,
        help="是否对视觉模块权重使用 RtN (Round-to-Nearest) 量化。如果权重比特数小于16且此标志未设置，则使用 GPTQ。"
    )
    parser.add_argument(
        "--llm_w_rtn",
        action="store_true",
        default=False,
        help="是否对语言模型权重使用 RtN 量化。如果权重比特数小于16且此标志未设置，则使用 GPTQ。"
    )
    parser.add_argument(
        "--visual_w_clip",
        action="store_true",
        default=False,
        help="是否对视觉模块权重量化进行裁剪。脚本在权重量化期间会自动寻找最佳裁剪比例。"
    )
    parser.add_argument(
        "--llm_w_clip",
        action="store_true",
        default=False,
        help="是否对语言模型权重量化进行裁剪。脚本在权重量化期间会自动寻找最佳裁剪比例。"
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="GPTQ 中用于阻尼的 Hessian 对角线平均值的百分比。"
    )
    parser.add_argument(
        "--act_order", action="store_true", default=False, help="GPTQ 中是否使用 act-order。"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于复现实验结果。")

    # 通用量化参数 (General Quantization Arguments)
    parser.add_argument(
        "--int8_down_proj",
        action="store_true",
        default=False,
        help="是否对下投影层使用 INT8 量化。如果启用，该层的权重和激活都将是 INT8。"
    )
    parser.add_argument(
        "--quant_llm",
        action="store_true",
        default=False,
        help="是否量化 InternVL2-1B 语言模型部分。"
    )
    parser.add_argument(
        "--quant_visual_clip",
        action="store_true",
        default=False,
        help="是否量化视觉特征模型部分。"
    )
    parser.add_argument(
        "--quant_cross_attention",
        action="store_true",
        default=False,
        help="是否量化交叉注意力模型部分。"
    )
    parser.add_argument(
        "--act_per_tensor",
        action="store_true",
        default=False,
        help="是否对激活进行 per-tensor 量化。"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=8,
        help="GPTQ 校准数据样本的数量。"
    )
    parser.add_argument(
        "--skip_names",
        nargs="+",
        default=[],
        help="跳过这些指定名称的层的量化。"
    )
    parser.add_argument(
        "--no_fuse_visual_clip",
        action="store_true",
        default=False,
        help="是否不融合视觉编码器（Visual CLIP）相关的层。"
    )
    parser.add_argument(
        "--no_fuse_visual_cross_attn",
        action="store_true",
        default=False,
        help="是否不融合视觉交叉注意力相关的层。"
    )
    parser.add_argument(
        "--no_fuse_llm",
        action="store_true",
        default=False,
        help="是否不融合语言模型相关的层。"
    )
    parser.add_argument(
        "--not_fuse_layer_norms",
        action="store_true",
        default=False,
        help="是否不融合 Layer Normalization 层。"
    )
    parser.add_argument(
        "--llm_static",
        action="store_true",
        default=False,
        help="是否对语言模型激活使用静态（static）尺度和零点进行量化。"
    )
    parser.add_argument(
        "--visual_static",
        action="store_true",
        default=False,
        help="是否对视觉模块激活使用静态（static）尺度和零点进行量化。"
    )
    parser.add_argument(
        "--calib_num",
        type=int,
        default=32,
        help="校准样本的数量。"
    )
    parser.add_argument(
        "--eval_num",
        type=int,
        default=32,
        help="评估样本的数量。"
    )
    parser.add_argument(
        "--calib_mode",
        type=str,
        default="v2",
        help="校准模式，可选 'v1' 或 'v2'。"
    )
    parser.add_argument(
        "--analysis_num",
        type=int,
        default=32,
        help="分析样本的数量。"
    )
    parser.add_argument(
        "--analysis_mode",
        type=str,
        default="v1",
        help="分析模式，可选 'v1' 或 'v2'。"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TextVQA_VAL",
        help="用于校准和评估的数据集名称。"
    )
    parser.add_argument(
        "--analysis_text",
        action="store_true",
        default=False,
        help="是否进行文本分析。"
    )
    parser.add_argument(
        "--online_visual_hadamard",
        action="store_true",
        default=False,
        help="是否对视觉模块使用在线 Hadamard 旋转。"
    )
    parser.add_argument(
        "--online_llm_hadamard",
        action="store_true",
        default=False,
        help="是否对语言模型使用在线 Hadamard 旋转。"
    )
    parser.add_argument(
        "--fp32_had",
        action="store_true",
        default=False,
        help="Hadamard 旋转是否在 FP32 精度下应用（默认：False）。"
    )
    parser.add_argument(
        "--dump_gptq",
        type=str,
        default=None,
        help="将 GPTQ 量化后的模型保存到此路径。"
    )
    parser.add_argument(
        "--load_gptq",
        type=str,
        default=None,
        help="从指定路径加载 GPTQ 量化后的模型。"
    )
    parser.add_argument(
        "--visual_split",
        action="store_true",
        default=False,
        help="是否对视觉模块进行权重拆分（用于在线 Hadamard 旋转）。"
    )
    parser.add_argument(
        "--llm_split",
        action="store_true",
        default=False,
        help="是否对语言模型进行权重拆分（用于在线 Hadamard 旋转）。"
    )
    args = parser.parse_args() # 解析命令行参数
    init_logger(args)          # 初始化日志
    main(args)                 # 执行主函数
