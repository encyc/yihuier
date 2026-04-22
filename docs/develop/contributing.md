# 贡献指南

感谢您对 Yihuier 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议：

1. **检查现有 Issues**：先搜索是否已有类似问题
2. **创建新 Issue**：使用清晰的标题和详细的描述
3. **提供示例**：如果可能，提供可复现的代码示例

**Issue 模板**：

```markdown
### 问题描述
简要描述问题

### 复现步骤
1. 步骤一
2. 步骤二
3. 步骤三

### 期望行为
描述期望的行为

### 实际行为
描述实际发生的行为

### 环境信息
- Python 版本：
- Yihuier 版本：
- 操作系统：

### 附加信息
日志、截图等
```

### 提交代码

#### 准备工作

1. **Fork 仓库**：在 GitHub 上 fork yihuier 仓库
2. **克隆本地**：
```bash
git clone https://github.com/YOUR_USERNAME/yihuier.git
cd yihuier
```

3. **安装开发依赖**：
```bash
# 使用 uv
uv pip install -e ".[dev]"

# 或使用 pip
pip install -e ".[dev]"
```

4. **创建分支**：
```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

#### 开发流程

1. **编写代码**：
   - 遵循 [代码风格](#代码风格)
   - 添加类型提示
   - 添加文档字符串

2. **编写测试**：
   - 为新功能添加单元测试
   - 确保测试覆盖率不下降

3. **运行测试**：
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_binning.py -v

# 查看覆盖率
pytest tests/ --cov=yihuier --cov-report=html
```

4. **代码检查**：
```bash
# 代码格式化
ruff format yihuier/

# 代码检查
ruff check yihuier/

# 类型检查
mypy yihuier/
```

5. **提交代码**：
```bash
git add .
git commit -m "feat: add new binning method"
```

6. **推送到 GitHub**：
```bash
git push origin feature/your-feature-name
```

7. **创建 Pull Request**：在 GitHub 上创建 PR

#### Commit 消息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>: <description>

[optional body]
```

**类型（type）**：
- `feat`：新功能
- `fix`：Bug 修复
- `docs`：文档更新
- `style`：代码格式（不影响功能）
- `refactor`：重构
- `test`：测试相关
- `chore`：构建/工具相关

**示例**：
```
feat: add monotonic binning method

Implement a new binning method that ensures monotonic
WOE values across bins. This is useful for variables
with business monotonicity requirements.
```

## 代码风格

### Python 风格指南

遵循 [PEP 8](https://pep8.org/) 规范：

```bash
# 自动格式化
ruff format yihuier/

# 检查代码风格
ruff check yihuier/
```

### 类型提示

为所有公共函数添加类型提示：

```python
from typing import List, Optional

def binning_num(
    self,
    col_list: List[str],
    max_bin: Optional[int] = None
) -> tuple:
    pass
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def binning_num(
    self,
    col_list: List[str],
    max_bin: Optional[int] = None
) -> tuple:
    """数值型变量分箱。

    Args:
        col_list: 变量列表
        max_bin: 最大分箱数

    Returns:
        (分箱结果列表, IV 值列表)

    Raises:
        ValueError: 如果变量不存在于数据中
    """
    pass
```

### 命名规范

- **类名**：PascalCase（如 `BinningModule`）
- **函数名**：snake_case（如 `binning_num`）
- **常量**：UPPER_SNAKE_CASE（如 `MAX_BIN`）
- **私有方法**：前缀双下划线（如 `__assign_bin`）

## 项目结构

```
yihuier/
├── yihuier/              # 源代码
│   ├── __init__.py
│   ├── yihuier.py        # 主类
│   ├── eda.py            # EDA 模块
│   ├── data_processing.py
│   ├── binning.py
│   ├── var_select.py
│   ├── model_evaluation.py
│   ├── scorecard_implement.py
│   ├── scorecard_monitor.py
│   ├── cluster.py
│   ├── pipeline.py
│   ├── binning_function.py
│   ├── constants.py
│   └── frame.py
├── tests/                # 测试
│   ├── conftest.py
│   ├── test_binning.py
│   ├── test_data_processing.py
│   └── ...
├── examples/             # 示例
│   ├── basic_usage.py
│   └── advanced_pipeline.py
├── docs/                 # 文档
│   ├── README.md
│   ├── guide/
│   └── develop/
├── pyproject.toml        # 项目配置
├── README.md
└── LICENSE
```

## 添加新功能

### 添加新分箱方法

1. 在 `yihuier/binning_function.py` 中添加辅助函数
2. 在 `BinningModule` 中添加包装方法
3. 在 `tests/test_binning.py` 中添加测试
4. 在 `docs/guide/modules/binning.md` 中更新文档

### 添加新模块

1. 创建 `yihuier/new_module.py`
2. 在 `yihuier/frame.py` 中添加类定义
3. 在 `Yihuier.__init__` 中初始化模块
4. 创建测试文件 `tests/test_new_module.py`
5. 添加文档 `docs/guide/modules/new_module.md`

## 测试要求

### 单元测试

- 每个新功能必须有单元测试
- 使用 pytest 框架
- 测试文件命名：`test_<module>.py`

```python
# tests/test_binning.py
def test_binning_num():
    yh = Yihuier(sample_data, target='dlq_flag')
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=['v1', 'v2'],
        max_bin=5
    )
    assert len(bin_df) == 2
    assert all(iv > 0 for iv in iv_value)
```

### 测试覆盖率

- 新代码测试覆盖率应 ≥ 80%
- 核心功能测试覆盖率应 ≥ 90%

```bash
# 查看覆盖率报告
pytest tests/ --cov=yihuier --cov-report=html
open htmlcov/index.html
```

### 集成测试

确保新功能与现有功能兼容：

```python
def test_integration():
    # 完整建模流程
    yh = Yihuier(data, target='dlq_flag')
    # ... 使用新功能 ...
    assert final_result is not None
```

## 文档要求

### API 文档

为所有公共函数添加文档字符串：

```python
def new_function(self, param1: str, param2: int) -> bool:
    """函数功能简述。

    详细描述函数功能和使用场景。

    Args:
        param1: 参数1说明
        param2: 参数2说明

    Returns:
        返回值说明

    Raises:
        ValueError: 异常情况说明

    Examples:
        >>> yh = Yihuier(data, target='dlq_flag')
        >>> result = yh.new_function('test', 10)
        True
    """
    pass
```

### 更新用户文档

在 `docs/guide/` 中更新相关文档：

- 添加新功能的说明
- 添加使用示例
- 更新 API 参考

### 更新 CHANGELOG

在 `docs/develop/changelog.md` 中记录变更：

```markdown
## [Unreleased]

### Added
- 新分箱方法：monotonic binning
- 新评估指标：custom metric

### Fixed
- 修复 ChiMerge 分箱的 KeyError 问题

### Changed
- 优化 WOE 转换性能
```

## Pull Request 检查清单

提交 PR 前请确认：

- [ ] 代码通过所有测试
- [ ] 代码覆盖率不下降
- [ ] 代码通过 ruff 检查
- [ ] 添加了类型提示
- [ ] 添加了文档字符串
- [ ] 更新了相关文档
- [ ] 更新了 CHANGELOG
- [ ] Commit 消息遵循规范

## 发布流程

### 版本号规范

遵循 [Semantic Versioning](https://semver.org/)：

- **MAJOR**：不兼容的 API 变更
- **MINOR**：向后兼容的功能新增
- **PATCH**：向后兼容的 Bug 修复

示例：`1.2.3`
- MAJOR: 1
- MINOR: 2
- PATCH: 3

### 发布步骤

1. **更新版本号**：
```bash
# 在 pyproject.toml 中更新
version = "1.2.3"
```

2. **更新 CHANGELOG**：
```markdown
## [1.2.3] - 2024-01-01

### Added
- 新功能

### Fixed
- 修复问题
```

3. **创建 Git 标签**：
```bash
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3
```

4. **发布到 PyPI**：
```bash
# 构建
python -m build

# 发布（需要 PyPI token）
twine upload dist/*
```

## 社区规范

### 行为准则

- 尊重所有贡献者
- 欢迎新手提问
- 建设性反馈
- 专注于解决问题

### 沟通渠道

- **GitHub Issues**：Bug 报告和功能请求
- **GitHub Discussions**：一般讨论和问题
- **Pull Requests**：代码审查

## 许可证

贡献的代码将采用 MIT 许可证发布。

## 致谢

感谢所有贡献者！

## 参考资源

- [系统架构](architecture.md) - 了解系统设计
- [更新日志](changelog.md) - 版本更新记录
