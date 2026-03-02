# MQuant

## Environment Configuration

### 1. create env

```shell
conda env create -f environment.yml
```

### 2. hardmard transform

```shell
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install  .
# 原先的安装命令，使用当前的目录进行安装
# pip install -e .
```

### 3. vlmevalit install

```shell
cd third/VLMEvalKit
pip install -r requirements.txt
```

## VLLMs Quant

### QwenVL Chat

 [View QwenVL Chat's quantitative documentation](qwenvl.md)
 