#!/bin/bash
# 本地测试安装脚本

set -e

echo "🧪 本地安装测试"
echo "=========================="

# 清理旧的构建产物
echo ""
echo "🧹 清理旧的构建产物..."
rm -rf dist/ build/ *.egg-info

# 构建包
echo ""
echo "🔨 构建包..."
python -m build --outdir dist/

# 创建虚拟环境测试
echo ""
echo "📦 创建虚拟环境进行测试..."
TEST_VENV="test_venv_$$"

python -m venv "/tmp/$TEST_VENV"
source "/tmp/$TEST_VENV/bin/activate"

# 安装包
echo ""
echo "📥 安装包..."
pip install --upgrade pip
pip install dist/yihuier-*.whl

# 测试导入
echo ""
echo "✅ 测试导入..."
python -c "
from yihuier import Yihuier
import pandas as pd

# 测试版本
print('✅ Yihuier 导入成功')
print(f'版本: {Yihuier.__version__}')

# 测试基本功能
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 4, 6, 8, 10],
    'target': [0, 0, 1, 1, 1]
})

yh = Yihuier(data, target='target')
print('✅ Yihuier 初始化成功')
print('✅ 所有测试通过！')
"

# 清理
echo ""
echo "🧹 清理测试环境..."
deactivate
rm -rf "/tmp/$TEST_VENV"

echo ""
echo "✅ 本地安装测试完成！"
echo "📝 包已准备好发布到 PyPI"
