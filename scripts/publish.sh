#!/bin/bash
# PyPI 发布脚本

set -e

echo "🚀 Yihuier PyPI 发布脚本"
echo "=========================="

# 检查版本号
VERSION=$(grep '^version' pyproject.toml | head -1 | awk -F'"' '{print $2}')
echo "📦 当前版本: $VERSION"

# 清理旧的构建产物
echo ""
echo "🧹 清理旧的构建产物..."
rm -rf dist/ build/ *.egg-info

# 构建包
echo ""
echo "🔨 构建发布包..."
python -m build --outdir dist/

# 验证包
echo ""
echo "✅ 验证包..."
twine check dist/*

# 询问是否继续
echo ""
read -p "是否继续发布到 PyPI? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ 发布已取消"
    echo ""
    echo "💡 提示：你可以手动测试发布到 TestPyPI："
    echo "   python -m twine upload --repository testpypi dist/*"
    exit 0
fi

# 发布到 PyPI
echo ""
echo "📤 发布到 PyPI..."
python -m twine upload dist/*

echo ""
echo "✅ 发布成功！"
echo ""
echo "📝 后续步骤："
echo "1. 等待 1-2 分钟"
echo "2. 验证安装: pip install yihuier"
echo "3. 创建 Git 标签: git tag v$VERSION && git push origin v$VERSION"
echo "4. 在 GitHub 创建 Release"
