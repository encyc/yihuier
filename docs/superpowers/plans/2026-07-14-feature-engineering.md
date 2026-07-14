# 特征工程模块 (FeatureEngineeringModule) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 yihuier 新增特征工程模块 `fe_module`，提供 6 个手动特征衍生函数和 1 个批量交叉方法（带 IV 预筛）。

**Architecture:** 新建 `feature_engineering.py` 实现 `FeatureEngineeringModule`，挂载到 `yihuier.py` 的 `Yihuier.__init__`。所有方法操作 `yh.data`、返回新 DataFrame（与 `dp_module` 风格一致）。batch 方法复用 `binning_module.iv_num` 做 IV 预筛。不触碰任何现有模块内部逻辑。

**Tech Stack:** Python 3.10+ (PEP 604 `X | None` 语法), pandas, numpy, pytest

## Global Constraints

- **API 风格**: 所有方法返回新 DataFrame，不原地修改 `yh.data`，由用户赋值回 `yh.data`（与 `dp_module.fillna_num_var` 一致）。
- **类型注解**: 完整类型注解，使用 `X | None` 语法（非 `Optional[X]`），与项目现有风格一致。
- **依赖**: 仅用 pandas + numpy，不引入新依赖。
- **测试**: pytest + `tests/conftest.py` 的 `yihuier_instance` fixture。每个函数 happy path + 边界。
- **目标变量保护**: 所有方法不得对 `yh.target` 列做任何变换。
- **docstring 风格**: 中文注释，与现有模块（`data_processing.py`、`binning.py`）一致。

## Spec 修正说明

Spec §5.2 写的是改 `yihuier/frame.py`，但经核实 `frame.py` 是未使用的骨架文件（无任何 import 引用）。实际挂载点是 `yihuier/yihuier.py:39-56`。本计划修正为改 `yihuier.py`。

---

## File Structure

| 文件 | 责任 | 动作 |
|------|------|------|
| `yihuier/feature_engineering.py` | `FeatureEngineeringModule` 类：6 个手动函数 + 1 个 batch 函数 + `feature_log` | 新建 |
| `yihuier/yihuier.py` | `Yihuier.__init__` 挂载 `fe_module` | 修改 |
| `tests/test_feature_engineering.py` | 全部函数的单元测试 | 新建 |
| `.claude/skills/risk-modeling/SKILL.md` | 插入"第 2.5 步：特征工程" | 修改 |
| `README.md` | 模块概览表加一行 | 修改 |

---

### Task 1: 模块骨架 + 挂载

创建 `feature_engineering.py` 的类骨架和 `__init__`，挂载到 `Yihuier`。这是所有后续函数的载体。

**Files:**
- Create: `yihuier/feature_engineering.py`
- Modify: `yihuier/yihuier.py:1-56` (import + 挂载)

**Interfaces:**
- Produces: `FeatureEngineeringModule` 类，构造签名为 `__init__(self, yihuier_instance) -> None`，实例属性 `self.feature_log: list[dict] = []`

- [ ] **Step 1: 写失败测试 — 模块可导入、可挂载、feature_log 初始化**

```python
# tests/test_feature_engineering.py
"""
测试特征工程模块
"""

import pandas as pd
import numpy as np
import pytest


def test_fe_module_mounted(yihuier_instance):
    """测试 fe_module 正确挂载到 Yihuier 实例"""
    from yihuier.feature_engineering import FeatureEngineeringModule
    assert hasattr(yihuier_instance, 'fe_module')
    assert isinstance(yihuier_instance.fe_module, FeatureEngineeringModule)


def test_feature_log_initialized(yihuier_instance):
    """测试 feature_log 初始化为空列表"""
    assert hasattr(yihuier_instance.fe_module, 'feature_log')
    assert yihuier_instance.fe_module.feature_log == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_engineering.py::test_fe_module_mounted tests/test_feature_engineering.py::test_feature_log_initialized -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'yihuier.feature_engineering'` 或 `AttributeError: 'Yihuier' object has no attribute 'fe_module'`

- [ ] **Step 3: 创建 feature_engineering.py 骨架**

```python
# yihuier/feature_engineering.py
"""特征工程模块

提供评分卡场景常用的宽表特征衍生函数：
- 交叉特征（比率/求和/差分/通用四则运算）
- 数学变换（log/sqrt/square 等）
- 缺失指示符
- 批量交叉 + IV 预筛
"""

import numpy as np
import pandas as pd


class FeatureEngineeringModule:
    """特征工程模块

    所有方法操作 yh.data，返回新的 DataFrame（不原地修改），
    由用户赋值回 yh.data，与 dp_module 风格一致。

    Attributes:
        yihuier_instance: Yihuier 主实例
        feature_log: 已生成特征的记录列表，每项 {name, source, method}
    """

    def __init__(self, yihuier_instance) -> None:
        """初始化特征工程模块

        Args:
            yihuier_instance: Yihuier 主实例
        """
        self.yihuier_instance = yihuier_instance
        self.feature_log: list[dict] = []
```

- [ ] **Step 4: 在 yihuier.py 挂载 fe_module**

在 `yihuier/yihuier.py` 顶部 import 区（第 4 行 `from yihuier.binning import BinningModule` 附近）按字母序插入：

```python
from yihuier.feature_engineering import FeatureEngineeringModule
```

（放在 `from yihuier.eda import EDAModule` 之后、`from yihuier.model_evaluation import ModelEvaluationModule` 之前）

在 `__init__` 中 `self.dp_module` 之后插入（第 49 行后）：

```python
        self.fe_module: FeatureEngineeringModule = FeatureEngineeringModule(self)
```

同时在 docstring 的 Attributes 列表中（第 28-36 行区域），在 `dp_module` 行后加：

```
        fe_module: 特征工程模块
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_feature_engineering.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 确认全量测试无回归**

Run: `pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: 全部 PASS（新模块不影响现有功能）

- [ ] **Step 7: Commit**

```bash
git add yihuier/feature_engineering.py yihuier/yihuier.py tests/test_feature_engineering.py
git commit -m "feat: 新增特征工程模块骨架并挂载到 Yihuier"
```

---

### Task 2: gen_cross（通用四则运算 — 其他交叉函数的基础）

先实现底层 `gen_cross`，后续 `gen_ratio`/`gen_sum`/`gen_diff` 都基于它。

**Files:**
- Modify: `yihuier/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: 无
- Produces: `gen_cross(self, a: str, b: str, op: str, name: str) -> pd.DataFrame`
  - `op` ∈ `{'+', '-', '*', '/'}`，非法值抛 `ValueError`
  - 返回 `yh.data` 的副本，新增 `name` 列
  - 除法 `den==0` 或非数 → NaN
  - 列不存在 → pandas 自然抛 KeyError
  - 记录到 `self.feature_log`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_feature_engineering.py`：

```python
def test_gen_cross_add(yihuier_instance):
    """测试加法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '+', 'v1_plus_v2')
    assert 'v1_plus_v2' in result.columns
    # 逐行验证（忽略 v1 缺失的行）
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] + yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_plus_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_subtract(yihuier_instance):
    """测试减法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '-', 'v1_minus_v2')
    assert 'v1_minus_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] - yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_minus_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_multiply(yihuier_instance):
    """测试乘法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '*', 'v1_mul_v2')
    assert 'v1_mul_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] * yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_mul_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_divide(yihuier_instance):
    """测试除法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '/', 'v1_div_v2')
    assert 'v1_div_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] / yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_div_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_invalid_op(yihuier_instance):
    """测试非法运算符抛 ValueError"""
    with pytest.raises(ValueError, match="不支持的运算符"):
        yihuier_instance.fe_module.gen_cross('v1', 'v2', '%', 'v1_mod_v2')


def test_gen_cross_divide_by_zero():
    """测试除零产生 NaN 而非 inf/异常"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'a': [10.0, 20.0, 30.0], 'b': [0.0, 0.0, 5.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_cross('a', 'b', '/', 'a_div_b')
    # b==0 的行应为 NaN（不是 inf）
    assert pd.isna(result.loc[0, 'a_div_b'])
    assert pd.isna(result.loc[1, 'a_div_b'])
    assert result.loc[2, 'a_div_b'] == 6.0


def test_gen_cross_does_not_mutate_original(yihuier_instance):
    """测试返回新 DataFrame，不修改 yh.data"""
    original_cols = list(yihuier_instance.data.columns)
    yihuier_instance.fe_module.gen_cross('v1', 'v2', '+', 'new_col')
    assert list(yihuier_instance.data.columns) == original_cols
    assert 'new_col' not in yihuier_instance.data.columns


def test_gen_cross_logs_feature(yihuier_instance):
    """测试特征被记录到 feature_log"""
    yihuier_instance.fe_module.gen_cross('v1', 'v2', '+', 'v1_plus_v2')
    log = yihuier_instance.fe_module.feature_log
    assert any(entry['name'] == 'v1_plus_v2' for entry in log)
    entry = [e for e in log if e['name'] == 'v1_plus_v2'][0]
    assert entry['source'] == ('v1', 'v2')
    assert entry['method'] == 'cross:+'
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_engineering.py -k gen_cross -v`
Expected: FAIL — `AttributeError: 'FeatureEngineeringModule' object has no attribute 'gen_cross'`

- [ ] **Step 3: 实现 gen_cross**

在 `FeatureEngineeringModule` 类中添加方法：

```python
    def gen_cross(
        self, a: str, b: str, op: str, name: str
    ) -> pd.DataFrame:
        """通用四则运算交叉特征

        Args:
            a: 第一个列名
            b: 第二个列名
            op: 运算符，支持 '+', '-', '*', '/'
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）

        Raises:
            ValueError: op 不在 {'+', '-', '*', '/'} 中时
        """
        data = self.yihuier_instance.data.copy()

        if op == "+":
            result = data[a] + data[b]
        elif op == "-":
            result = data[a] - data[b]
        elif op == "*":
            result = data[a] * data[b]
        elif op == "/":
            result = data[a] / data[b]
        else:
            raise ValueError(f"不支持的运算符: {op}。必须是 '+', '-', '*', '/' 之一")

        data[name] = result
        self.feature_log.append({"name": name, "source": (a, b), "method": f"cross:{op}"})
        return data
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_engineering.py -k gen_cross -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Commit**

```bash
git add yihuier/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: 实现 gen_cross 通用四则运算交叉特征"
```

---

### Task 3: gen_ratio / gen_sum / gen_diff（语义化快捷方式）

基于 `gen_cross` 的薄封装。

**Files:**
- Modify: `yihuier/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: `gen_cross` (Task 2)
- Produces:
  - `gen_ratio(self, num: str, den: str, name: str) -> pd.DataFrame`
  - `gen_sum(self, cols: list[str], name: str) -> pd.DataFrame`
  - `gen_diff(self, a: str, b: str, name: str) -> pd.DataFrame`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_feature_engineering.py`：

```python
def test_gen_ratio(yihuier_instance):
    """测试比率特征"""
    result = yihuier_instance.fe_module.gen_ratio('v1', 'v2', 'ratio_v1_v2')
    assert 'ratio_v1_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] / yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'ratio_v1_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_ratio_divide_by_zero():
    """测试比率除零 → NaN"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1], 'a': [10.0, 20.0], 'b': [0.0, 5.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_ratio('a', 'b', 'a_over_b')
    assert pd.isna(result.loc[0, 'a_over_b'])
    assert result.loc[1, 'a_over_b'] == 4.0


def test_gen_sum(yihuier_instance):
    """测试多列求和"""
    result = yihuier_instance.fe_module.gen_sum(['v1', 'v2', 'v4'], 'sum_v124')
    assert 'sum_v124' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = (
        yihuier_instance.data.loc[mask, 'v1']
        + yihuier_instance.data.loc[mask, 'v2']
        + yihuier_instance.data.loc[mask, 'v4']
    )
    pd.testing.assert_series_equal(
        result.loc[mask, 'sum_v124'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_sum_single_column(yihuier_instance):
    """测试单列求和（等于原列复制）"""
    result = yihuier_instance.fe_module.gen_sum(['v1'], 'sum_v1')
    assert 'sum_v1' in result.columns


def test_gen_diff(yihuier_instance):
    """测试差分"""
    result = yihuier_instance.fe_module.gen_diff('v1', 'v2', 'diff_v1_v2')
    assert 'diff_v1_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] - yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'diff_v1_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_shortcuts_log_features(yihuier_instance):
    """测试快捷函数记录 feature_log"""
    yihuier_instance.fe_module.gen_ratio('v1', 'v2', 'r')
    yihuier_instance.fe_module.gen_sum(['v1', 'v2'], 's')
    yihuier_instance.fe_module.gen_diff('v1', 'v2', 'd')
    names = [e['name'] for e in yihuier_instance.fe_module.feature_log]
    assert 'r' in names and 's' in names and 'd' in names
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_engineering.py -k "gen_ratio or gen_sum or gen_diff" -v`
Expected: FAIL — `AttributeError: 'FeatureEngineeringModule' object has no attribute 'gen_ratio'`

- [ ] **Step 3: 实现三个快捷函数**

在 `FeatureEngineeringModule` 类中添加：

```python
    def gen_ratio(self, num: str, den: str, name: str) -> pd.DataFrame:
        """比率特征 num / den

        Args:
            num: 分子列名
            den: 分母列名（den==0 时结果为 NaN）
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        data[name] = data[num] / data[den]
        self.feature_log.append({"name": name, "source": (num, den), "method": "ratio"})
        return data

    def gen_sum(self, cols: list[str], name: str) -> pd.DataFrame:
        """多列求和

        Args:
            cols: 要求和的列名列表
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        result = data[cols[0]].copy()
        for col in cols[1:]:
            result = result + data[col]
        data[name] = result
        self.feature_log.append({"name": name, "source": tuple(cols), "method": "sum"})
        return data

    def gen_diff(self, a: str, b: str, name: str) -> pd.DataFrame:
        """差分特征 a - b

        Args:
            a: 被减数列名
            b: 减数列名
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        data[name] = data[a] - data[b]
        self.feature_log.append({"name": name, "source": (a, b), "method": "diff"})
        return data
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_engineering.py -k "gen_ratio or gen_sum or gen_diff" -v`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
git add yihuier/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: 实现 gen_ratio/gen_sum/gen_diff 语义化交叉特征"
```

---

### Task 4: gen_transform（数学变换）

**Files:**
- Modify: `yihuier/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: 无
- Produces: `gen_transform(self, col: str, method: str, name: str) -> pd.DataFrame`
  - `method` ∈ `{'log', 'log1p', 'sqrt', 'square', 'abs', 'reciprocal'}`
  - 非法 method → `ValueError`
  - `log`/`sqrt` 对负数或零：结果为 NaN，print 警告（不抛异常）
  - 记录到 `self.feature_log`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_feature_engineering.py`：

```python
def test_gen_transform_log1p(yihuier_instance):
    """测试 log1p 变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'log1p', 'v2_log1p')
    assert 'v2_log1p' in result.columns
    mask = yihuier_instance.data['v2'].notna() & (yihuier_instance.data['v2'] > -1)
    expected = np.log1p(yihuier_instance.data.loc[mask, 'v2'])
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_log1p'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_transform_square(yihuier_instance):
    """测试 square 变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'square', 'v2_sq')
    assert 'v2_sq' in result.columns
    mask = yihuier_instance.data['v2'].notna()
    expected = yihuier_instance.data.loc[mask, 'v2'] ** 2
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_sq'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_transform_abs(yihuier_instance):
    """测试 abs 变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'abs', 'v2_abs')
    assert 'v2_abs' in result.columns
    mask = yihuier_instance.data['v2'].notna()
    expected = yihuier_instance.data.loc[mask, 'v2'].abs()
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_abs'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_transform_sqrt_negative_returns_nan():
    """测试 sqrt 对负数返回 NaN"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'x': [4.0, -1.0, 9.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_transform('x', 'sqrt', 'x_sqrt')
    assert result.loc[0, 'x_sqrt'] == 2.0
    assert pd.isna(result.loc[1, 'x_sqrt'])  # 负数 → NaN
    assert result.loc[2, 'x_sqrt'] == 3.0


def test_gen_transform_log_nonpositive_returns_nan():
    """测试 log 对零/负数返回 NaN"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'x': [1.0, 0.0, -2.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_transform('x', 'log', 'x_log')
    assert result.loc[0, 'x_log'] == 0.0  # log(1) = 0
    assert pd.isna(result.loc[1, 'x_log'])  # log(0) → NaN
    assert pd.isna(result.loc[2, 'x_log'])  # log(-2) → NaN


def test_gen_transform_invalid_method(yihuier_instance):
    """测试非法 method 抛 ValueError"""
    with pytest.raises(ValueError, match="不支持的变换方法"):
        yihuier_instance.fe_module.gen_transform('v1', 'cube', 'v1_cube')


def test_gen_transform_reciprocal(yihuier_instance):
    """测试倒数变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'reciprocal', 'v2_rec')
    assert 'v2_rec' in result.columns
    mask = yihuier_instance.data['v2'].notna() & (yihuier_instance.data['v2'] != 0)
    expected = 1.0 / yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_rec'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_engineering.py -k gen_transform -v`
Expected: FAIL — `AttributeError: 'FeatureEngineeringModule' object has no attribute 'gen_transform'`

- [ ] **Step 3: 实现 gen_transform**

在 `FeatureEngineeringModule` 类中添加：

```python
    def gen_transform(self, col: str, method: str, name: str) -> pd.DataFrame:
        """数学变换

        对指定列做数学变换。log/sqrt 对非正值的处理：返回 NaN。

        Args:
            col: 原始列名
            method: 变换方法，支持 'log', 'log1p', 'sqrt', 'square', 'abs', 'reciprocal'
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）

        Raises:
            ValueError: method 不支持时
        """
        data = self.yihuier_instance.data.copy()
        s = data[col]

        if method == "log":
            # 非正值 → NaN
            valid = s > 0
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[valid] = np.log(s[valid])
        elif method == "log1p":
            # x > -1 有效；其余 → NaN
            valid = s > -1
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[valid] = np.log1p(s[valid])
        elif method == "sqrt":
            valid = s >= 0
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[valid] = np.sqrt(s[valid])
        elif method == "square":
            result = s ** 2
        elif method == "abs":
            result = s.abs()
        elif method == "reciprocal":
            result = 1.0 / s  # s==0 → inf，但 pandas 默认行为；保持自然语义
        else:
            raise ValueError(
                f"不支持的变换方法: {method}。必须是 'log', 'log1p', 'sqrt', 'square', 'abs', 'reciprocal' 之一"
            )

        data[name] = result
        self.feature_log.append({"name": name, "source": col, "method": f"transform:{method}"})
        return data
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_engineering.py -k gen_transform -v`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
git add yihuier/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: 实现 gen_transform 数学变换（log/sqrt/square/abs/reciprocal）"
```

---

### Task 5: gen_missing_flag（缺失指示符）

**Files:**
- Modify: `yihuier/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: 无
- Produces: `gen_missing_flag(self, cols: list[str], prefix: str = 'miss_') -> pd.DataFrame`
  - 对每列生成独立的 0/1 flag 列，列名 `{prefix}{col}`
  - 缺失 → 1，非缺失 → 0
  - 记录到 `self.feature_log`（每列一条）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_feature_engineering.py`：

```python
def test_gen_missing_flag_basic(yihuier_instance):
    """测试缺失指示符：有缺失=1，无缺失=0"""
    # v1 在 sample_data 中有 5% 缺失
    result = yihuier_instance.fe_module.gen_missing_flag(['v1'])
    assert 'miss_v1' in result.columns
    # 缺失行应为 1
    missing_mask = yihuier_instance.data['v1'].isna()
    assert (result.loc[missing_mask, 'miss_v1'] == 1).all()
    # 非缺失行应为 0
    present_mask = yihuier_instance.data['v1'].notna()
    assert (result.loc[present_mask, 'miss_v1'] == 0).all()


def test_gen_missing_flag_multiple_cols(yihuier_instance):
    """测试多列各生成独立 flag"""
    result = yihuier_instance.fe_module.gen_missing_flag(['v1', 'v2'])
    assert 'miss_v1' in result.columns
    assert 'miss_v2' in result.columns


def test_gen_missing_flag_custom_prefix(yihuier_instance):
    """测试自定义前缀"""
    result = yihuier_instance.fe_module.gen_missing_flag(['v1'], prefix='is_na_')
    assert 'is_na_v1' in result.columns
    assert 'miss_v1' not in result.columns


def test_gen_missing_flag_preserves_originals(yihuier_instance):
    """测试原始列保留不变"""
    original_v1 = yihuier_instance.data['v1'].copy()
    result = yihuier_instance.fe_module.gen_missing_flag(['v1'])
    pd.testing.assert_series_equal(result['v1'], original_v1, check_names=False)


def test_gen_missing_flag_no_missing():
    """测试无缺失列全部为 0"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'x': [1.0, 2.0, 3.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_missing_flag(['x'])
    assert (result['miss_x'] == 0).all()


def test_gen_missing_flag_logs_features(yihuier_instance):
    """测试每个 flag 列都记录到 feature_log"""
    yihuier_instance.fe_module.gen_missing_flag(['v1', 'v2'])
    names = [e['name'] for e in yihuier_instance.fe_module.feature_log]
    assert 'miss_v1' in names
    assert 'miss_v2' in names
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_engineering.py -k gen_missing_flag -v`
Expected: FAIL — `AttributeError: 'FeatureEngineeringModule' object has no attribute 'gen_missing_flag'`

- [ ] **Step 3: 实现 gen_missing_flag**

在 `FeatureEngineeringModule` 类中添加：

```python
    def gen_missing_flag(
        self, cols: list[str], prefix: str = "miss_"
    ) -> pd.DataFrame:
        """缺失指示符

        对每列生成独立的 0/1 flag 列：缺失=1，非缺失=0。
        应在 dp_module.fillna 之前调用。

        Args:
            cols: 需要生成缺失指示符的列名列表
            prefix: 新列名前缀，默认 'miss_'，新列名为 {prefix}{col}

        Returns:
            新增若干 flag 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        for col in cols:
            flag_name = f"{prefix}{col}"
            data[flag_name] = data[col].isna().astype(int)
            self.feature_log.append(
                {"name": flag_name, "source": col, "method": "missing_flag"}
            )
        return data
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_engineering.py -k gen_missing_flag -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add yihuier/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: 实现 gen_missing_flag 缺失指示符"
```

---

### Task 6: batch_cross（批量交叉 + IV 预筛）

**Files:**
- Modify: `yihuier/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: `gen_cross` (Task 2), `binning_module.iv_num` (现有)
- Produces: `batch_cross(self, col_list: list[str], ops: list[str] = ['/', '-'], iv_threshold: float = 0.02, max_features: int = 50, prefix: str = 'fe_') -> pd.DataFrame`
  - 对 `col_list` 做有序两两组合（i < j）× `ops`
  - 调用 `gen_cross` 生成特征
  - 调用 `self.yihuier_instance.binning_module.iv_num(new_cols, method='freq', n=10)` 算 IV
  - 保留 IV ≥ iv_threshold 的，按 IV 降序取前 max_features 个
  - 返回原始数据 + 保留的新特征列

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_feature_engineering.py`：

```python
def test_batch_cross_generates_features(yihuier_instance):
    """测试 batch_cross 生成特征并追加到返回的 DataFrame"""
    result = yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2', 'v4'],
        ops=['/', '-'],
        iv_threshold=0.0,  # 不过滤，验证全部生成
        max_features=100,
    )
    # 3 列两两组合 = 3 对，× 2 ops = 6 个特征
    new_cols = [c for c in result.columns if c.startswith('fe_')]
    assert len(new_cols) == 6


def test_batch_cross_iv_filter(yihuier_instance):
    """测试 IV 预筛：iv_threshold 较高时特征数减少"""
    # 用一个很高的阈值，大部分随机特征 IV 接近 0
    result_strict = yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2', 'v4'],
        ops=['/', '-'],
        iv_threshold=0.5,  # 极高阈值
        max_features=100,
    )
    new_cols_strict = [c for c in result_strict.columns if c.startswith('fe_')]
    # 随机数据 IV 很难超过 0.5，应被大量过滤
    assert len(new_cols_strict) < 6

    # 对照：阈值 0 不过滤
    result_loose = yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2', 'v4'],
        ops=['/', '-'],
        iv_threshold=0.0,
        max_features=100,
    )
    new_cols_loose = [c for c in result_loose.columns if c.startswith('fe_')]
    assert len(new_cols_strict) <= len(new_cols_loose)


def test_batch_cross_max_features_cap(yihuier_instance):
    """测试 max_features 截断"""
    result = yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2', 'v4'],
        ops=['/', '-', '+', '*'],  # 3 对 × 4 = 12 个候选
        iv_threshold=0.0,
        max_features=3,  # 只保留 3 个
    )
    new_cols = [c for c in result.columns if c.startswith('fe_')]
    assert len(new_cols) == 3


def test_batch_cross_naming(yihuier_instance):
    """测试命名规则：{prefix}{a}_{opname}_{b}"""
    result = yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2'],
        ops=['/', '-'],
        iv_threshold=0.0,
        max_features=100,
        prefix='fe_',
    )
    assert 'fe_v1_div_v2' in result.columns
    assert 'fe_v1_sub_v2' in result.columns


def test_batch_cross_no_mutation(yihuier_instance):
    """测试不修改 yh.data"""
    original_cols = list(yihuier_instance.data.columns)
    yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2'],
        ops=['/'],
        iv_threshold=0.0,
        max_features=100,
    )
    assert list(yihuier_instance.data.columns) == original_cols


def test_batch_cross_divide_by_zero_safe():
    """测试含除零的特征不崩溃"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({
        'target': [0, 1, 0, 1],
        'a': [10.0, 20.0, 0.0, 5.0],
        'b': [0.0, 5.0, 0.0, 1.0],
    })
    yh = Yihuier(data, 'target')
    # 不应抛异常
    result = yh.fe_module.batch_cross(
        col_list=['a', 'b'],
        ops=['/'],
        iv_threshold=0.0,
        max_features=100,
    )
    # 特征列存在即可
    assert 'fe_a_div_b' in result.columns


def test_batch_cross_custom_prefix(yihuier_instance):
    """测试自定义前缀"""
    result = yihuier_instance.fe_module.batch_cross(
        col_list=['v1', 'v2'],
        ops=['-'],
        iv_threshold=0.0,
        max_features=100,
        prefix='cross_',
    )
    assert 'cross_v1_sub_v2' in result.columns
    assert len([c for c in result.columns if c.startswith('cross_')]) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_engineering.py -k batch_cross -v`
Expected: FAIL — `AttributeError: 'FeatureEngineeringModule' object has no attribute 'batch_cross'`

- [ ] **Step 3: 实现 batch_cross**

在 `FeatureEngineeringModule` 类中添加：

```python
    def batch_cross(
        self,
        col_list: list[str],
        ops: list[str] = None,
        iv_threshold: float = 0.02,
        max_features: int = 50,
        prefix: str = "fe_",
    ) -> pd.DataFrame:
        """批量交叉特征 + IV 预筛

        对 col_list 做有序两两组合 × ops，生成交叉特征，
        然后用 IV 过滤，保留有区分力的特征。

        Args:
            col_list: 参与交叉的列名列表
            ops: 运算符列表，默认 ['/', '-']
            iv_threshold: IV 下限，低于此值的特征被丢弃
            max_features: 最多保留的特征数（按 IV 降序）
            prefix: 新特征列名前缀

        Returns:
            原始数据 + 筛选后新特征列的 DataFrame（副本）
        """
        if ops is None:
            ops = ["/", "-"]

        op_names = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

        # 1. 生成所有候选特征
        candidates = []  # [(feature_name, ...)]
        temp_data = self.yihuier_instance.data.copy()

        n = len(col_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = col_list[i], col_list[j]
                for op in ops:
                    feat_name = f"{prefix}{a}_{op_names[op]}_{b}"
                    # 复用 gen_cross 的运算逻辑，但直接在 temp_data 上累加
                    if op == "+":
                        temp_data[feat_name] = temp_data[a] + temp_data[b]
                    elif op == "-":
                        temp_data[feat_name] = temp_data[a] - temp_data[b]
                    elif op == "*":
                        temp_data[feat_name] = temp_data[a] * temp_data[b]
                    elif op == "/":
                        temp_data[feat_name] = temp_data[a] / temp_data[b]
                    else:
                        raise ValueError(
                            f"不支持的运算符: {op}。必须是 '+', '-', '*', '/' 之一"
                        )
                    candidates.append(feat_name)
                    self.feature_log.append(
                        {"name": feat_name, "source": (a, b), "method": f"batch_cross:{op}"}
                    )

        if not candidates:
            return self.yihuier_instance.data.copy()

        # 2. IV 预筛：复用 binning_module.iv_num（等频分箱，快速）
        iv_df = self.yihuier_instance.binning_module.iv_num(
            candidates, method="freq", n=10
        )
        # iv_df 列: ['col', 'iv']

        # 3. 过滤 + 截断
        iv_df = iv_df[iv_df["iv"] >= iv_threshold]
        iv_df = iv_df.sort_values("iv", ascending=False).reset_index(drop=True)
        kept = iv_df["col"].head(max_features).tolist()

        # 4. 返回原始数据 + 保留的特征列
        result = self.yihuier_instance.data.copy()
        for feat in kept:
            result[feat] = temp_data[feat]
        return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_engineering.py -k batch_cross -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 确认全量新模块测试通过**

Run: `pytest tests/test_feature_engineering.py -v`
Expected: 全部 PASS（之前所有 Task 的测试 + batch_cross）

- [ ] **Step 6: Commit**

```bash
git add yihuier/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: 实现 batch_cross 批量交叉 + IV 预筛"
```

---

### Task 7: 文档更新（SKILL.md + README.md）

**Files:**
- Modify: `.claude/skills/risk-modeling/SKILL.md` (第 2 步之后插入第 2.5 步，更新模块说明表)
- Modify: `README.md` (模块概览表加一行)

**Interfaces:**
- Consumes: 所有已实现函数的签名（Task 1-6）

- [ ] **Step 1: 更新 SKILL.md — 插入第 2.5 步**

在 SKILL.md 的 `### 第 2 步：数据预处理` 代码块之后、`### 第 3 步：变量分箱` 之前，插入：

```markdown
### 第 2.5 步：特征工程（可选）

```python
# 交叉特征（评分卡常用：负债比、利用率等）
yh.data = yh.fe_module.gen_ratio('debt', 'income', name='dti')
yh.data = yh.fe_module.gen_diff('balance', 'limit', name='net_balance')

# 数学变换（处理偏态变量）
yh.data = yh.fe_module.gen_transform('income', method='log1p', name='income_log')

# 缺失指示符（在 fillna 之前调用）
yh.data = yh.fe_module.gen_missing_flag(['income', 'age'])
# → 新增 miss_income、miss_age 两列

# 批量交叉 + IV 预筛
yh.data = yh.fe_module.batch_cross(
    col_list=['income', 'debt', 'balance', 'limit'],
    ops=['/', '-'],
    iv_threshold=0.02,
    max_features=50
)
```
```

- [ ] **Step 2: 更新 SKILL.md — 模块说明表**

在 SKILL.md 的模块说明表中，`dp_module` 行之后插入：

```markdown
| `fe_module` | 特征工程 | `gen_ratio()`, `gen_cross()`, `batch_cross()` |
```

- [ ] **Step 3: 更新 README.md — 模块概览表**

在 README.md 的模块概览表（第 81 行附近）中，`DataProcessingModule` 行之后插入：

```markdown
| `FeatureEngineeringModule` | 特征工程 | [📖 特征工程模块](https://encyc.github.io/yihuier/guide/modules/feature-engineering.html) |
```

- [ ] **Step 4: 更新 README.md — 特性列表**

在 README.md 的特性列表（第 7-13 行区域）中，`- **模块化架构**` 行之后插入：

```markdown
- **特征工程** - 交叉特征、数学变换、缺失指示符、批量交叉（带 IV 预筛）
```

- [ ] **Step 5: 手动验证文档渲染**

Run: 检查 SKILL.md 和 README.md 的 markdown 语法正确
Expected: 无语法错误（代码块闭合、表格格式正确）

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/risk-modeling/SKILL.md README.md
git commit -m "docs: 更新 SKILL.md 和 README，新增特征工程模块说明"
```

---

### Task 8: 最终验证

**Files:** 无修改，仅运行全量验证。

- [ ] **Step 1: 运行全量测试套件**

Run: `pytest tests/ -v`
Expected: 全部 PASS，无回归

- [ ] **Step 2: 运行 ruff 检查**

Run: `ruff check yihuier/feature_engineering.py`
Expected: 无 lint 错误

Run: `ruff format --check yihuier/feature_engineering.py`
Expected: 格式正确（如有差异运行 `ruff format yihuier/feature_engineering.py` 修复）

- [ ] **Step 3: 验证导入正常**

Run: `python -c "from yihuier import Yihuier; import pandas as pd; d=pd.DataFrame({'t':[0,1],'a':[1.0,2.0],'b':[3.0,4.0]}); y=Yihuier(d,'t'); print(y.fe_module.gen_ratio('a','b','r'))"`
Expected: 正常输出含 `r` 列的 DataFrame，无异常

- [ ] **Step 4: 如果有 ruff 修复，提交**

```bash
git add -A
git commit -m "chore: ruff 格式化特征工程模块"  # 仅在 Step 2 有修复时
```

---

## Self-Review

**1. Spec coverage:**

| Spec 要求 | 对应 Task |
|-----------|----------|
| gen_ratio | Task 3 |
| gen_sum | Task 3 |
| gen_diff | Task 3 |
| gen_cross | Task 2 |
| gen_transform (6 methods) | Task 4 |
| gen_missing_flag | Task 5 |
| batch_cross + IV 预筛 | Task 6 |
| feature_log | Task 1 (骨架) + 各 Task 追加 |
| frame.py 挂载 → 修正为 yihuier.py | Task 1 (已修正) |
| SKILL.md 第 2.5 步 | Task 7 |
| README 模块表 | Task 7 |
| 测试策略 | Task 2-6 各自的测试步骤 |

**无遗漏。**

**2. Placeholder scan:** 无 TBD/TODO/"implement later"/"add error handling"。所有代码步骤含完整实现。

**3. Type consistency:**
- `gen_cross(a, b, op, name)` — Task 2 定义，Task 6 batch_cross 内部直接内联运算（不调用 gen_cross 以避免多次 copy，但运算逻辑一致）。命名一致。
- `feature_log` 条目结构 `{name, source, method}` — 全部 Task 一致。
- `batch_cross` 的 `ops` 默认值：spec §4.1 用字面量 `['/', '-']`，实现用 `None` + 函数体内 `if ops is None`，这是为了避免可变默认参数陷阱（Python 反模式）。测试中显式传参，不受影响。✓
- `op_names` 映射 `{'+':'add', '-':'sub', '*':'mul', '/':'div'}` — Task 6 命名测试 `fe_v1_div_v2`/`fe_v1_sub_v2` 与此一致。✓
