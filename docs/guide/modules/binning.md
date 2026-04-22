# 变量分箱模块 (BinningModule)

## 概述

`BinningModule` 是评分卡建模的核心模块，提供变量分箱、WOE转换和IV值计算功能。

### 主要功能

- **变量分箱** - 支持类别型和数值型变量的多种分箱方法
- **IV值计算** - 评估变量的预测能力
- **WOE转换** - 将变量转换为WOE值
- **分箱质量检验** - 检验WOE单调性、异常值等

---

## 核心概念

### 什么是分箱？

分箱（Binning）是将连续变量离散化为有限个区间（箱）的过程。

**为什么要分箱？**
1. **缓解异常值影响** - 异常值会被分配到边缘箱体
2. **非线性关系线性化** - 通过WOE转换
3. **业务可解释性** - 每个箱体有明确的业务含义
4. **模型稳定性** - 减少过拟合风险

### IV值（Information Value）

衡量变量预测能力的指标：

| IV值范围 | 预测能力 |
|---------|---------|
| < 0.02 | 无预测能力 |
| 0.02 - 0.1 | 弱预测能力 |
| 0.1 - 0.3 | 中等预测能力 |
| > 0.3 | 强预测能力 |

### WOE值（Weight of Evidence）

衡量每个箱体内好坏样本的比值：

```
WOE = ln(好坏比_当前箱 / 好坏比_总体)
```

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

binning = yh.binning_module
```

---

## API 参考

### 1. binning_cate() - 类别型变量分箱

对类别型变量进行分箱并计算WOE和IV值。

#### 语法

```python
binning.binning_cate(
    col_list: List[str]
) -> Tuple[list, list, list]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 类别型变量列表 |

#### 返回值

返回一个元组 `(bin_df, iv_value, ks_value)`：

- `bin_df` - 分箱结果列表，每个元素是一个DataFrame
- `iv_value` - IV值列表
- `ks_value` - KS值列表

#### 使用示例

```python
# 对类别型变量分箱
cate_vars = ['education', 'marriage', 'house_type']
bin_df, iv_value, ks_value = binning.binning_cate(cate_vars)

# 查看结果
for i, col in enumerate(cate_vars):
    print(f"\n=== {col} ===")
    print(f"IV值: {iv_value[i]}")
    print(f"KS值: {ks_value[i]}")
    print(bin_df[i])
```

#### 输出格式

每个 bin_df 包含以下列：

| 列名 | 说明 |
|------|------|
| `min_bin` / `max_bin` | 区间边界 |
| `total` | 样本数 |
| `totalrate` | 样本占比 |
| `bad` | 坏样本数 |
| `badrate` | 坏样本率 |
| `good` | 好样本数 |
| `goodrate` | 好样本率 |
| `badattr` | 坏样本占比（整体） |
| `goodattr` | 好样本占比（整体） |
| `woe` | WOE值 |
| `bin_iv` | 箱体IV贡献 |
| `IV` | 总IV值 |
| `GB_index` | 好坏指数（如150G表示好比为1.5） |

---

### 2. binning_num() - 数值型变量分箱

对数值型变量进行分箱，支持多种分箱方法。

#### 语法

```python
binning.binning_num(
    col_list: List[str],
    max_bin: Optional[int] = None,
    min_binpct: Optional[float] = None,
    method: str = 'ChiMerge',
    n: int = 10,
    leaf_stop_percent: float = 0.05
) -> Tuple[list, list]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 数值型变量列表 |
| `max_bin` | int | None | 最大分箱数（ChiMerge方法） |
| `min_binpct` | float | None | 最小箱体占比（ChiMerge方法） |
| `method` | str | 'ChiMerge' | 分箱方法 |
| `n` | int | 10 | 分箱数（freq/count方法） |
| `leaf_stop_percent` | float | 0.05 | 叶子节点占比（CART方法） |

#### 分箱方法

**1. ChiMerge（推荐）**

卡方分箱，基于卡方检验的合并策略。

```python
bin_df, iv_value = binning.binning_num(
    col_list=['age', 'income'],
    max_bin=5,           # 最多5个箱
    min_binpct=0.05,     # 每箱至少5%样本
    method='ChiMerge'
)
```

**2. 等频分箱（freq）**

每个箱体内的样本数大致相等。

```python
bin_df, iv_value = binning.binning_num(
    col_list=['age', 'income'],
    n=10,                # 分为10个箱
    method='freq'
)
```

**3. 等距分箱（count）**

每个箱体的宽度相等。

```python
bin_df, iv_value = binning.binning_num(
    col_list=['age', 'income'],
    n=10,
    method='count'
)
```

#### 返回值

返回一个元组 `(bin_df, iv_value)`：

- `bin_df` - 分箱结果列表
- `iv_value` - IV值列表

#### 使用示例

```python
# ChiMerge分箱
num_vars = ['age', 'income', 'debt_ratio']
bin_df, iv_value = binning.binning_num(
    col_list=num_vars,
    max_bin=5,
    method='ChiMerge'
)

# 筛选高IV值变量
iv_df = pd.DataFrame({
    'var': num_vars,
    'iv': iv_value
})
high_iv_vars = iv_df[iv_df['iv'] > 0.1]['var'].tolist()
print(f"高IV值变量: {high_iv_vars}")
```

---

### 3. iv_num() - 数值型变量IV明细表

快速计算数值型变量的IV值。

#### 语法

```python
binning.iv_num(
    col_list: List[str],
    **kwargs
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 数值型变量列表 |
| `max_bin` | int | None | 最大分箱数 |
| `min_binpct` | float | None | 最小箱体占比 |
| `method` | str | None | 分箱方法 |

#### 使用示例

```python
# 快速计算所有数值型变量的IV值
num_vars = yh.get_numeric_variables()
iv_df = binning.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='freq'
)

# 按IV值排序
iv_df = iv_df.sort_values('iv', ascending=False)
print(iv_df)

# 选择IV值大于0.1的变量
selected_vars = iv_df[iv_df['iv'] > 0.1]['col'].tolist()
print(f"\n选择的变量: {selected_vars}")
```

---

### 4. binning_self() - 自定义分箱

使用自定义的分箱切点进行分箱。

#### 语法

```python
binning.binning_self(
    col: str,
    cut: Optional[list] = None,
    right_border: bool = True
) -> Tuple[pd.DataFrame, float]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col` | str | 必填 | 变量名 |
| `cut` | list | None | 分箱切点列表 |
| `right_border` | bool | True | 是否左闭右开 |

#### 使用示例

```python
# 自定义年龄分箱
bin_df, iv_value = binning.binning_self(
    col='age',
    cut=[20, 30, 40, 50, 60],  # 切点: (-inf, 20], (20, 30], ..., (60, inf)
    right_border=True
)

print(f"IV值: {iv_value}")
print(bin_df)
```

---

### 5. woe_df_concat() - 合并WOE结果

将所有变量的分箱结果合并为一个DataFrame。

#### 语法

```python
binning.woe_df_concat() -> pd.DataFrame
```

#### 使用示例

```python
# 先进行分箱
binning.binning_cate(['education', 'marriage'])
binning.binning_num(['age', 'income'], max_bin=5)

# 合并WOE结果
woe_df = binning.woe_df_concat()
print(woe_df.head())

# 保存WOE结果
woe_df.to_csv('woe_result.csv', index=False)
```

#### 输出格式

WOE结果表包含以下列：

| 列名 | 说明 |
|------|------|
| `col` | 变量名 |
| `bin` | 分箱区间 |
| `min_bin` | 区间下界 |
| `max_bin` | 区间上界 |
| `woe` | WOE值 |
| `bin_iv` | 箱体IV贡献 |
| ... | 其他统计列 |

---

### 6. woe_transform() - WOE转换

将原始数据转换为WOE值。

#### 语法

```python
binning.woe_transform() -> pd.DataFrame
```

#### 使用示例

```python
# 1. 先进行分箱
binning.binning_cate(['education', 'marriage'])
binning.binning_num(['age', 'income'], max_bin=5)

# 2. 生成WOE结果表
binning.woe_df_concat()

# 3. WOE转换
data_woe = binning.woe_transform()

# 4. 查看转换后的数据
print(data_woe.head())

# 5. 保存WOE数据
data_woe.to_csv('data_woe.csv', index=False)
```

⚠️ **注意**：
- 必须先调用 `woe_df_concat()` 生成WOE结果表
- 目标变量不会被转换
- 原始数据会被覆盖

---

### 7. plot_woe() - WOE可视化

绘制变量WOE值的柱状图。

#### 语法

```python
binning.plot_woe(
    hspace: float = 0.4,
    wspace: float = 0.4,
    plt_size: Optional[Tuple[int, int]] = None,
    plt_num: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None
) -> None
```

#### 使用示例

```python
# 先进行分箱
binning.binning_num(['age', 'income', 'debt_ratio'], max_bin=5)

# 绘制WOE图
binning.plot_woe(
    plt_size=(12, 8),
    plt_num=3,
    x=2,
    y=2
)
```

---

### 8. woe_monoton() - WOE单调性检验

检验WOE值是否呈单调变化。

#### 语法

```python
binning.woe_monoton() -> Tuple[list, pd.DataFrame]
```

#### 返回值

返回一个元组 `(woe_notmonoton_col, woe_judge_df)`：

- `woe_notmonoton_col` - 非单调变化的变量列表
- `woe_judge_df` - 检验结果DataFrame

#### 使用示例

```python
# 检验WOE单调性
not_monoton, judge_df = binning.woe_monoton()

print("非单调变量:", not_monoton)
print(judge_df)

# 处理非单调变量
if not_monoton:
    print("以下变量WOE不单调，建议调整分箱:")
    for var in not_monoton:
        print(f"  - {var}")
```

---

### 9. woe_large() - WOE异常值检验

检验是否存在WOE值大于1的箱体。

#### 语法

```python
binning.woe_large() -> Tuple[list, pd.DataFrame]
```

#### 使用示例

```python
# 检验WOE异常值
large_vars, judge_df = binning.woe_large()

print("WOE值大于1的变量:", large_vars)
print(judge_df)

# 处理异常WOE值
if large_vars:
    print("以下变量存在WOE>1的箱体，建议合并相邻箱体:")
    for var in large_vars:
        print(f"  - {var}")
```

---

## 完整分箱流程

### 标准分箱流程

```python
from yihuier import Yihuier
import pandas as pd

# 1. 初始化
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')
binning = yh.binning_module

# 2. 类别型变量分箱
print("=== 类别型变量分箱 ===")
cate_vars = yh.get_categorical_variables()
if cate_vars:
    bin_df_cate, iv_cate, ks_cate = binning.binning_cate(cate_vars)

    # 筛选高IV值变量
    iv_df_cate = pd.DataFrame({
        'var': cate_vars,
        'iv': iv_cate
    })
    high_iv_cate = iv_df_cate[iv_df_cate['iv'] > 0.1]['var'].tolist()
    print(f"高IV类别型变量: {high_iv_cate}")

# 3. 数值型变量分箱
print("\n=== 数值型变量分箱 ===")
num_vars = yh.get_numeric_variables()

# 快速计算IV值
iv_df_num = binning.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='ChiMerge'
)

# 筛选高IV值变量
high_iv_num = iv_df_num[iv_df_num['iv'] > 0.1]['col'].tolist()
print(f"高IV数值型变量:")
print(iv_df_num.sort_values('iv', ascending=False).head(10))

# 对高IV值变量进行详细分箱
selected_vars = high_iv_num[:10]  # 选择前10个
bin_df_num, iv_value = binning.binning_num(
    col_list=selected_vars,
    max_bin=5,
    min_binpct=0.05,
    method='ChiMerge'
)

# 4. WOE转换
print("\n=== WOE转换 ===")
# 重新对所有选择的变量分箱
all_vars = high_iv_cate + selected_vars
binning.binning_cate(high_iv_cate)
binning.binning_num(selected_vars, max_bin=5)

# 生成WOE结果表
woe_df = binning.woe_df_concat()
print(f"WOE结果表形状: {woe_df.shape}")

# WOE转换
data_woe = binning.woe_transform()
print(f"WOE数据形状: {data_woe.shape}")

# 5. 检验分箱质量
print("\n=== 分箱质量检验 ===")

# WOE单调性
not_monoton, judge_df = binning.woe_monoton()
if not_monoton:
    print(f"非单调变量: {not_monoton}")

# WOE异常值
large_vars, judge_df = binning.woe_large()
if large_vars:
    print(f"WOE异常变量: {large_vars}")

# 6. 可视化WOE
print("\n=== WOE可视化 ===")
binning.plot_woe(
    plt_size=(12, 8),
    plt_num=len(selected_vars),
    x=3,
    y=3
)
```

---

## 分箱方法选择

### 方法对比

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **ChiMerge** | 推荐默认方法 | 考虑目标变量，分箱质量高 | 速度较慢 |
| **等频分箱** | 快速分析 | 速度快，每个箱样本数均衡 | 可能切分到同一类 |
| **等距分箱** | 均匀分布的变量 | 保持变量分布特征 | 对异常值敏感 |
| **自定义分箱** | 有业务知识 | 符合业务逻辑 | 需要领域知识 |

### 选择建议

```python
# 1. 默认使用ChiMerge
binning.binning_num(
    col_list=['age', 'income'],
    max_bin=5,
    method='ChiMerge'
)

# 2. 大数据集快速分析使用等频分箱
binning.binning_num(
    col_list=num_vars,
    n=10,
    method='freq'
)

# 3. 有明确业务阈值使用自定义分箱
binning.binning_self(
    col='age',
    cut=[18, 25, 35, 45, 55, 65]
)

# 4. 多种方法对比
for method in ['ChiMerge', 'freq', 'count']:
    _, iv_value = binning.binning_num(
        col_list=['income'],
        max_bin=5,
        method=method
    )
    print(f"{method} IV值: {iv_value[0]}")
```

---

## 分箱调优

### 参数调优

```python
# 调整分箱数
for max_bin in [3, 5, 7, 10]:
    _, iv_value = binning.binning_num(
        col_list=['income'],
        max_bin=max_bin,
        method='ChiMerge'
    )
    print(f"max_bin={max_bin}, IV={iv_value[0]}")

# 调整最小箱体占比
for min_pct in [0.01, 0.03, 0.05, 0.1]:
    _, iv_value = binning.binning_num(
        col_list=['income'],
        max_bin=5,
        min_binpct=min_pct,
        method='ChiMerge'
    )
    print(f"min_binpct={min_pct}, IV={iv_value[0]}")
```

### 处理非单调WOE

```python
# 检测非单调变量
not_monoton, _ = binning.woe_monoton()

# 对非单调变量重新分箱
for var in not_monoton:
    print(f"调整 {var} 的分箱...")

    # 尝试减少分箱数
    for max_bin in [3, 4, 5]:
        _, iv_value = binning.binning_num(
            col_list=[var],
            max_bin=max_bin,
            method='ChiMerge'
        )

        # 重新检验单调性
        not_mono, _ = binning.woe_monoton()

        if var not in not_mono:
            print(f"  max_bin={max_bin} 时WOE单调")
            break
```

---

## 注意事项

### 1. 分箱前数据准备

```python
# 确保数据已经清洗
yh.data = yh.dp_module.delete_missing_var(threshold=0.15)
yh.data = yh.dp_module.const_delete(threshold=0.9)

# 检查目标变量
assert yh.data[yh.target].notnull().all(), "目标变量有缺失值"
assert yh.data[yh.target].nunique() == 2, "目标变量应该是二分类"
```

### 2. 分箱数选择

```python
# 经验法则
# - 样本数 < 1000: max_bin = 3-4
# - 样本数 1000-10000: max_bin = 5-7
# - 样本数 > 10000: max_bin = 7-10

n_samples = len(yh.data)
if n_samples < 1000:
    max_bin = 3
elif n_samples < 10000:
    max_bin = 5
else:
    max_bin = 7
```

### 3. 保存分箱结果

```python
# 保存WOE结果表
woe_df = binning.woe_df_concat()
woe_df.to_csv('output/woe_result.csv', index=False)

# 保存WOE数据
data_woe = binning.woe_transform()
data_woe.to_csv('output/data_woe.csv', index=False)

# 保存分箱配置
import pickle
with open('output/binning_config.pkl', 'wb') as f:
    pickle.dump(binning.bin_df, f)
```

---

## 常见问题

### Q1: ChiMerge分箱很慢怎么办？

```python
# 1. 减少变量数量，先计算IV值筛选
iv_df = binning.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='freq'
)
high_iv_vars = iv_df[iv_df['iv'] > 0.1]['col'].tolist()

# 2. 只对高IV值变量使用ChiMerge
binning.binning_num(
    col_list=high_iv_vars,
    max_bin=5,
    method='ChiMerge'
)

# 3. 或使用等频分箱作为替代
binning.binning_num(
    col_list=num_vars,
    n=10,
    method='freq'
)
```

### Q2: 某个变量分箱失败？

```python
# 检查变量
col = 'problem_var'

# 检查缺失率
missing_pct = yh.data[col].isnull().mean()
print(f"缺失率: {missing_pct:.1%}")

# 检查唯一值数量
unique_cnt = yh.data[col].nunique()
print(f"唯一值数量: {unique_cnt}")

# 如果唯一值太少，考虑作为类别型变量处理
if unique_cnt < 10:
    # 使用类别型分箱
    binning.binning_cate([col])
else:
    # 尝试等频分箱
    binning.binning_num([col], n=5, method='freq')
```

### Q3: 如何应用分箱到新数据？

```python
# 保存分箱配置
import pickle

# 训练时保存
with open('binning_config.pkl', 'wb') as f:
    config = {
        'woe_df': binning.woe_result_df,
        'bin_df': binning.bin_df
    }
    pickle.dump(config, f)

# 预测时加载
with open('binning_config.pkl', 'rb') as f:
    config = pickle.load(f)

# 应用到新数据
def apply_woe(new_data, woe_df):
    """应用WOE转换到新数据"""
    for col in new_data.columns:
        if col == yh.target:
            continue

        # 获取该变量的WOE映射
        var_woe = woe_df[woe_df['col'] == col]

        # 应用WOE转换
        for _, row in var_woe.iterrows():
            lower = row['min_bin']
            upper = row['max_bin']

            # 找到落在该箱体的样本
            if lower == upper:
                mask = new_data[col] == lower
            else:
                mask = (new_data[col] >= lower) & (new_data[col] <= upper)

            # 赋值WOE
            new_data.loc[mask, col] = row['woe']

    return new_data

# 使用
new_data = pd.read_csv('new_data.csv')
new_data_woe = apply_woe(new_data, config['woe_df'])
```

---

## 相关文档

- [数据处理模块](data-processing.md) - 分箱前的数据清洗
- [变量选择模块](var-select.md) - 基于IV值的变量筛选
- [模型评估模块](model-evaluation.md) - 评估分箱后的模型效果
