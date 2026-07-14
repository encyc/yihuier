# 特征工程模块设计 (FeatureEngineeringModule)

**日期**: 2026-07-14
**状态**: 已批准（待实现）
**范围**: MVP — 扁平宽表场景的特征衍生

---

## 1. 背景与目标

yihuier 当前的建模流程假设输入数据已是扁平宽表，但缺少特征衍生能力。评估报告指出"特征工程深度不足"，本模块在不破坏现有架构的前提下，补齐评分卡场景最常用的宽表衍生特征。

### 设计约束

- **数据形态**: 仅扁平宽表（一行一客户）。不处理交易/行为明细数据，不做 RFM/滑动窗口。
- **API 风格**: 与现有模块（`dp_module` 等）一致——返回新 DataFrame，由用户赋值回 `yh.data`。
- **自动化程度**: 先暴露手动精细函数，再封装 batch 方法。
- **模块定位**: 独立新模块 `fe_module`，位于流程中 `dp_module` 之后、`binning_module` 之前。

### 非目标 (YAGNI)

- 不做需要明细数据的时序/聚合特征（RFM、滑动窗口、趋势特征）。
- 不做 WOE 代理特征（`gen_woe_proxy`）——与 `binning_module` 职责重叠，明确排除。
- 不做自动数学变换批量生成、不做自动缺失指示符批量生成（适用性因变量而异，应人工判断）。
- 不触碰任何现有模块的内部逻辑。

---

## 2. 模块定位与 API 风格

新模块 `FeatureEngineeringModule`，挂载为 `yh.fe_module`。

流程位置：

```
EDA → 数据处理(dp) → 【特征工程(fe)】 → 分箱(binning) → 变量选择(var_select) → ...
```

### API 风格

每个方法操作 `yh.data`，返回新的 DataFrame，用户赋值回 `yh.data`：

```python
yh.data = yh.fe_module.gen_ratio('income', 'debt', name='dti')  # 新增 dti 列
```

### 特征日志

模块实例维护 `self.feature_log: list[dict]`，记录每个生成特征的 `{name, source, method}`，便于追溯。batch 方法返回前会更新此日志。

---

## 3. 手动精细函数（核心 API）

### 3.1 交叉特征（Cross Features）

评分卡最常用，业务可解释性最高。

| 函数 | 签名 | 说明 |
|------|------|------|
| `gen_ratio` | `(num: str, den: str, name: str) -> pd.DataFrame` | 比率 `num/den`，`den==0` → NaN |
| `gen_sum` | `(cols: list[str], name: str) -> pd.DataFrame` | 多列求和 |
| `gen_diff` | `(a: str, b: str, name: str) -> pd.DataFrame` | 差分 `a - b` |
| `gen_cross` | `(a: str, b: str, op: str, name: str) -> pd.DataFrame` | 通用四则运算，`op` ∈ {`+`, `-`, `*`, `/`} |

`gen_ratio`/`gen_sum`/`gen_diff` 是 `gen_cross` 的语义化快捷方式，单独提供是因为在评分卡中出现频率高（负债比、利用率、净收入），明确命名更易读。

### 3.2 数学变换（Math Transforms）

| 函数 | 签名 | 说明 |
|------|------|------|
| `gen_transform` | `(col: str, method: str, name: str) -> pd.DataFrame` | `method` ∈ {`log`, `log1p`, `sqrt`, `square`, `abs`, `reciprocal`} |

处理偏态变量（收入、金额）。`log1p` 对 0 值安全；`sqrt`/`log` 对负数/零的处理：返回 NaN 并 print 警告。

### 3.3 缺失指示符（Missing Indicators）

| 函数 | 签名 | 说明 |
|------|------|------|
| `gen_missing_flag` | `(cols: list[str], prefix: str = 'miss_') -> pd.DataFrame` | 对每列生成独立的 `is_missing` 的 0/1 列，列名 `{prefix}{col}` |

应在 `dp_module.fillna` **之前**调用——填充会抹掉缺失信息，而"是否缺失"本身常是强信号。每列独立生成（而非组合成一个 flag 列），因为"收入缺失"和"年龄缺失"是不同的业务信号。

---

## 4. Batch 方法 + IV 预筛

封装层，在手动函数之上。

### 4.1 签名

```python
def batch_cross(
    self,
    col_list: list[str],
    ops: list[str] = ['/', '-'],
    iv_threshold: float = 0.02,
    max_features: int = 50,
    prefix: str = 'fe_',
) -> pd.DataFrame:
```

### 4.2 流程

1. 对 `col_list` 做两两组合（有序，i < j）× `ops`，调用 `gen_cross` 生成特征，追加到返回的 DataFrame。
2. 用 `self.yihuier_instance.binning_module.iv_num(new_cols, method='freq', n=10)` 对新生成特征算 IV。
3. 按 IV 降序排序，保留 `iv_threshold` 以上、且不超过 `max_features` 个。
4. 只保留通过筛选的特征列，连同原始数据一起返回。
5. 更新 `self.feature_log`。

### 4.3 关键设计决策

- **IV 预筛复用现有能力**: 直接调 `binning_module.iv_num`，不重新实现分箱/IV。
- **除零/NaN 处理**: 继承自 `gen_cross`（除零 → NaN）；IV 计算时 NaN 落入独立箱，不会崩溃。
- **命名规则**: `{prefix}{a}_{op}_{b}`，如 `fe_income_div_debt`。`op` 映射：`/` → `div`, `*` → `mul`, `+` → `add`, `-` → `sub`。保证可追溯、不撞名。
- **`max_features` 安全阀**: 防止特征爆炸淹没下游变量选择。

### 4.4 不做什么

- 不做自动数学变换批量（log/sqrt 适用性因变量而异）。
- 不做自动缺失指示符批量（哪些列需要缺失指示是业务判断）。

---

## 5. 模块集成与文件改动

### 5.1 新增文件

| 文件 | 内容 | 行数估算 |
|------|------|---------|
| `yihuier/feature_engineering.py` | `FeatureEngineeringModule` 类：6 个手动函数 + 1 个 batch 函数 + `feature_log` | ~350-400 行 |
| `tests/test_feature_engineering.py` | 每个函数的 happy path + 边界测试 | ~200 行 |

### 5.2 修改文件（极小改动）

| 文件 | 改动 |
|------|------|
| `yihuier/frame.py` | `Yihuier.__init__` 加 `self.fe_module = FeatureEngineeringModule(self)`；加 `FeatureEngineeringModule` 骨架类 |
| `.claude/skills/risk-modeling/SKILL.md` | 第 2 步后插入"第 2.5 步：特征工程"，并补充模块说明表 |
| `README.md` | 模块概览表加 `FeatureEngineeringModule` 一行 |

### 5.3 不触碰的部分

- `binning.py` / `var_select.py` / `model_evaluation.py` / `scorecard_implement.py` / `data_processing.py` 的内部逻辑——零改动。
- `pipeline.py`（可选后续，不在 MVP）。
- `yihuier/__init__.py`（只导出 `Yihuier`，无需改）。

---

## 6. 测试策略

测试遵循现有 `tests/` 风格（pytest + `conftest.py` 的 fixture）。

| 函数 | 测试点 |
|------|--------|
| `gen_ratio` | 正常除法；`den==0` → NaN；`den` 含负数；列不存在 → KeyError |
| `gen_sum` | 正常求和；单列；列不存在 |
| `gen_diff` | 正常差分；列不存在 |
| `gen_cross` | 各 op 正确性；非法 `op` → ValueError；结果列名正确 |
| `gen_transform` | 各 method 正确性（`log`/`sqrt` 对 0/负数 → NaN + 警告）；非法 method → ValueError |
| `gen_missing_flag` | 有缺失 → 1；无缺失 → 0；多列各生成独立 flag 列；列名格式 `{prefix}{col}`；原始列保留 |
| `batch_cross` | 生成数量正确；IV 预筛生效（构造 IV≈0 的特征验证被丢弃）；`max_features` 截断；除零特征不崩溃；命名规则正确 |

---

## 7. SKILL.md 流程更新（第 2.5 步）

在现有 SKILL.md 第 2 步（数据预处理）之后、第 3 步（变量分箱）之前插入：

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
# → 新增 miss_income、miss_age 两列，各为 0/1

# 批量交叉 + IV 预筛
yh.data = yh.fe_module.batch_cross(
    col_list=['income', 'debt', 'balance', 'limit'],
    ops=['/', '-'],
    iv_threshold=0.02,
    max_features=50
)
```
```

> 注：`gen_missing_flag` 对每列生成独立的 flag 列（`{prefix}{col}`），而非组合成一个列，因为不同变量的缺失信号不同。

---

## 8. 工作量总结

| 部分 | 估算 |
|------|------|
| 手动精细函数（6 个） | ~300-400 行 |
| Batch 方法（1 个） | ~100-150 行 |
| 框架接入（frame.py） | ~10 行 |
| 测试 | ~200 行 |
| 文档（SKILL.md + README） | 中 |
| **合计** | **1 新模块文件 + 1 测试文件 + 3 文件小改，中等偏小工作量** |

风险低——核心是一个新模块，不触碰任何现有模块内部逻辑，可独立合并。
