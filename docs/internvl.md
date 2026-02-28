# 1. internvl2

## Downloads Weights

    ```shell
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/OpenGVLab/InternVL2-8B
    cd InternVL2-8B
    git lfs pull
    ln -s path weights
    ```

## Quant

### OCRBench

#### OCRBench w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name OCRBench --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

#### OCRBench w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name OCRBench --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

### MME

#### MME w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name MME --nsamples 256 --calib_num 512 --online_visual_hadamard --visual_split
    ```

#### MME w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name MME --nsamples 256 --calib_num 512 --online_visual_hadamard --visual_split
    ```

### TextVQA

#### TextVQA w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name TextVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
    ```

#### TextVQA w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name TextVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
    ```

### DocVQA

#### DocVQA w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name DocVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
    ```

#### DocVQA w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name DocVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
    ```

### MMMU (新增实验)

#### MMMU w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name MMMU_DEV_VAL --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

#### MMMU w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name MMMU_DEV_VAL --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

### ScienceQA (新增实验)

#### ScienceQA w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name ScienceQA_VAL --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

#### ScienceQA w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name ScienceQA_VAL --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

### 说明
- 使用 `MMMU_DEV_VAL` 作为 MMMU 数据集名称（VLMEvalKit 支持 MMMU_DEV_VAL 和 MMMU_TEST）
- 使用 `ScienceQA_VAL` 作为 ScienceQA 数据集名称（VLMEvalKit 支持 ScienceQA_VAL 和 ScienceQA_TEST）
- 使用 `SEEDBench_IMG` 作为 SEEDBench 数据集名称（VLMEvalKit 支持 SEEDBench_IMG、SEEDBench2、SEEDBench2_Plus）
- 使用 `VizWiz` 作为 VizWiz 数据集名称
- 其他参数与现有实验保持一致

### SEEDBench (新增实验)

#### SEEDBench w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name SEEDBench_IMG --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

#### SEEDBench w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name SEEDBench_IMG --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

### VizWiz (新增实验)

#### VizWiz w8a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name VizWiz --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```

#### VizWiz w4a8 + w4a8

    ```shell
    export PYTHONPATH=.
    python exam/quant_internvl.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name VizWiz --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
    ```
