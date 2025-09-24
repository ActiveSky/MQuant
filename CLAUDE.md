# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MQuant is the official code for the ACM MM2025 paper "MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization". It provides quantization solutions for 5 mainstream Multimodal Large Language Models (MLLMs) including Qwen-VL, Intern-VL2, MiniCPM-V, Qwen2-VL, and GLM-4V.

The project implements Modality-Specific Static Quantization (MSQ) to significantly reduce Time-to-First-Token (TTFT) and Rotation Magnitude Suppression (RMS) to mitigate weight outliers, achieving near-floating-point accuracy (<1% degradation) while reducing inference latency by up to 30% under W4A8 setting.

## Repository Structure

- `fake_quant/` - Core quantization implementation including quantizers, observers, and utilities
- `model/` - Model implementations for Qwen-VL variants
- `exam/` - Quantization examples and scripts for different models
- `evaluation/` - Evaluation scripts for benchmarking quantized models
- `docs/` - Documentation for installation and model-specific quantization
- `third/` - Third-party dependencies including VLMEvalKit

## Development Environment

### Setup Commands

1. Create conda environment:
```bash
conda env create -f environment.yml
```

2. Install fast-hadamard-transform:
```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .
```

3. Install VLMEvalKit:
```bash
cd third/VLMEvalKit
pip install -r requirements.txt
```

### Dependencies

Key dependencies include:
- Python 3.8
- PyTorch 2.4.1
- CUDA 12.1
- Transformers 4.46.3
- Accelerate 0.34.0
- Datasets 2.19.1
- NumPy 1.24.4

## Code Architecture

### Quantization Framework

The quantization framework consists of:

1. **Bit Types** (`fake_quant/bit_type.py`) - Defines quantization bit widths and signed/unsigned types
2. **Observers** (`fake_quant/observer/`) - Collect statistics for quantization parameter calculation
   - MinmaxObserver: Uses min/max values for scaling
   - PercentileObserver: Uses percentile values for more robust scaling
3. **Quantizers** (`fake_quant/quantizer/`) - Apply quantization using parameters from observers
   - UniformQuantizer: Standard uniform quantization
4. **Quantization Utilities** (`fake_quant/quant_utils.py`) - Core quantization functions and wrappers

### Rotation Techniques

The framework implements Rotation Magnitude Suppression (RMS) through:
- Hadamard transformations in `fake_quant/hadamard_utils.py`
- Layer norm fusion in `fake_quant/rotation_utils.py`

### Model Support

1. **Qwen-VL** - Implemented in `model/` and quantized via `exam/quant_qwenvl.py`
2. **Intern-VL2** - Quantized via `exam/quant_internvl.py`
3. **MiniCPM-V** - Quantized via `exam/quant_minicpmv.py`
4. **Qwen2-VL** - Quantized via `exam/quant_qwen2vl.py`

## Common Development Tasks

### Running Quantization

For Qwen-VL with W4A8 quantization on OCRBench:
```bash
export PYTHONPATH=.
python exam/quant_qwenvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name OCRBench --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
```

### Evaluation

Evaluation is performed using the VLMEvalKit framework through scripts in the `evaluation/` directory.

## Testing

Tests are conducted using standard benchmarks:
- OCRBench
- MME
- TextVQA
- DocVQA

Evaluation is performed through the `evaluation/eval.py` script which integrates with the VLMEvalKit framework.