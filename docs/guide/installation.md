# 安装指南

本指南将帮助你在各种环境中安装和配置 Yihuier。

## 系统要求

### Python 版本

Yihuier 需要 **Python 3.13 或更高版本**。

```bash
# 检查 Python 版本
python --version
```

如果版本低于 3.13，请先升级 Python：

```bash
# macOS 使用 Homebrew
brew install python@3.13

# Ubuntu/Debian
sudo apt update
sudo apt install python3.13 python3.13-venv

# Windows
# 访问 https://www.python.org/downloads/ 下载安装程序
```

### 依赖包

Yihuier 依赖以下核心包：

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| pandas | 2.1.4 | 数据处理 |
| numpy | 1.26.0 | 数值计算 |
| scikit-learn | 1.3.2 | 机器学习 |
| xgboost | 2.0.3 | 梯度提升 |
| matplotlib | 3.8.2 | 数据可视化 |
| seaborn | 0.12.2 | 统计图表 |

## 安装方法

### 方法一：使用 uv 安装（推荐）

[uv](https://github.com/astral-sh/uv) 是一个快速的 Python 包管理器。

```bash
# 安装 uv（如果还没安装）
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 使用 uv 安装 Yihuier
uv pip install yihuier
```

### 方法二：使用 pip 安装

```bash
# 标准安装
pip install yihuier

# 使用国内镜像源（加速）
pip install yihuier -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方法三：从源码安装

```bash
# 克隆仓库
git clone https://github.com/encyc/yihuier.git
cd yihuier

# 使用 uv
uv pip install -e .

# 或使用 pip
pip install -e .
```

## 虚拟环境设置

强烈建议使用虚拟环境来隔离项目依赖。

### venv

```bash
# 创建虚拟环境
python3.13 -m venv venv

# 激活虚拟环境
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 安装 Yihuier
pip install yihuier
```

### conda

```bash
# 创建 Python 3.13 环境
conda create -n yihuier-env python=3.13

# 激活环境
conda activate yihuier-env

# 安装 Yihuier
pip install yihuier
```

## 验证安装

```python
# 测试导入
from yihuier import Yihuier
import pandas as pd

# 创建测试数据
data = pd.DataFrame({
    'v1': [1, 2, 3, 4, 5],
    'v2': [10, 20, 30, 40, 50],
    'dlq_flag': [0, 0, 1, 0, 1]
})

# 初始化
yh = Yihuier(data, target='dlq_flag')
print("Yihuier 安装成功！")
print(f"数据形状: {yh.data.shape}")
```

## 可选依赖

### 开发依赖

如果你计划贡献代码，安装开发依赖：

```bash
# 从源码安装开发版本
git clone https://github.com/encyc/yihuier.git
cd yihuier

# 安装开发依赖
uv pip install -e ".[dev]"
```

开发依赖包括：
- `pytest` - 测试框架
- `ruff` - 代码检查和格式化
- `mypy` - 静态类型检查

### Jupyter 支持

```bash
pip install jupyter
```

## 常见问题

### ImportError: No module named 'yihuier'

**原因**：包未正确安装或虚拟环境未激活

**解决方案**：
```bash
# 检查包是否已安装
pip list | grep yihuier

# 如果未安装，重新安装
pip install yihuier

# 确保使用正确的 Python
which python
python --version
```

### NumPy 版本冲突

**原因**：系统中存在多个 Python 环境

**解决方案**：
```bash
# 卸载旧版本
pip uninstall numpy

# 重新安装兼容版本
pip install "numpy>=1.26.0"
```

### XGBoost 安装失败

**原因**：缺少编译依赖

**解决方案**：
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt install build-essential

# Windows
# 安装 Visual Studio Build Tools

# 然后重新安装
pip install xgboost
```

### 权限错误

**原因**：系统级 Python 安装需要管理员权限

**解决方案**：
```bash
# 使用用户安装
pip install --user yihuier

# 或使用虚拟环境（推荐）
python -m venv venv
source venv/bin/activate
pip install yihuier
```

## 升级 Yihuier

```bash
# 使用 uv
uv pip install --upgrade yihuier

# 使用 pip
pip install --upgrade yihuier
```

## 卸载

```bash
pip uninstall yihuier
```

## 下一步

安装完成后，继续阅读：
- [快速开始](quick-start.md) - 运行第一个示例
- [项目简介](intro.md) - 了解设计理念
- [API 文档](api.md) - 查看完整 API 参考
