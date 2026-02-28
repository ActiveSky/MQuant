# VLMEvalKit 升级指南

## 备份符号链接

在删除符号链接之前，先记录其指向的目标：

```bash
# 查看当前符号链接指向的目标
readlink vlmeval

# 输出应该是：third/VLMEvalKit/vlmeval
# 记录下这个路径备用
```

## 升级步骤

### 方式一：从 GitHub 克隆新版本（推荐）

由于 PyPI 上没有 `vlmeval` 包，需要从 GitHub 克隆：

```bash
# 1. 删除现有符号链接
rm vlmeval

# 2. 备份旧版本目录（可选，建议保留）
# cp -r third/VLMEvalKit third/VLMEvalKit_backup

# 3. 克隆新版本到临时目录
git clone https://github.com/open-compass/VLMEvalKit.git third/VLMEvalKit_new

# 4. 切换到稳定版本
cd third/VLMEvalKit_new
git checkout v0.2

# 5. 安装新版本依赖
pip install -r requirements.txt

# 6. 使用新版本（替换旧版本）
cd ../..
rm -rf third/VLMEvalKit
mv third/VLMEvalKit_new third/VLMEvalKit

# 7. 重新创建符号链接
ln -s third/VLMEvalKit/vlmeval vlmeval
```

### 方式二：使用 ms-vlmeval（ModelScope 版本）

```bash
# 安装 ModelScope 版本的 vlmeval
pip install ms-vlmeval
```

注意：`ms-vlmeval` 是 ModelScope 维护的版本，可能与原版有差异。

## 验证安装

```bash
# 检查版本
python -c "import vlmeval; print(vlmeval.__version__)"

# 验证数据集支持
python -c "from vlmeval.dataset import build_dataset; print('OK')"
```

## 回滚方案

如果需要恢复到原来的本地版本：

```bash
# 删除符号链接
rm vlmeval

# 恢复备份的目录
# cp -r third/VLMEvalKit_backup third/VLMEvalKit

# 或者从 git 历史恢复
cd third/VLMEvalKit
git checkout <原版本tag或commit>

# 重新创建符号链接
ln -s third/VLMEvalKit/vlmeval vlmeval
```
