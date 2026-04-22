# 数据处理模块 (DataProcessingModule)

## 概述

`DataProcessingModule` 提供了全面的数据预处理功能，是建模前的重要准备步骤。

### 主要功能

- **缺失值分析** - 可视化缺失值分布
- **缺失值处理** - 多种填充和删除策略
- **常变量删除** - 识别和删除同值化严重的变量
- **目标变量处理** - 删除目标变量缺失的样本
- **日期变量处理** - 转换为二进制变量

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

dp = yh.dp_module
```

---

## API 参考

### 1. plot_bar_missing_var() - 缺失值分布可视化

绘制所有变量的缺失率柱状图。

#### 语法

```python
dp.plot_bar_missing_var(
    plt_size: Optional[Tuple[int, int]] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `plt_size` | Tuple[int, int] | None | 图表尺寸，如 (12, 6) |

#### 使用示例

```python
# 查看缺失值分布
dp.plot_bar_missing_var(plt_size=(12, 6))

# 打印详细缺失值信息
missing_data = dp._DataProcessingModule__missing_var_cal()
print(missing_data)
```

#### 输出说明

- 图中每个柱子代表一个变量
- 柱子高度表示该变量的缺失率
- 可以快速识别高缺失率变量

---

### 2. fillna_cate_var() - 类别型变量缺失值填充

填充类别型变量的缺失值。

#### 语法

```python
dp.fillna_cate_var(
    col_list: List[str],
    fill_type: Optional[str] = None,
    fill_str: Optional[str] = None
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 需要填充的变量列表 |
| `fill_type` | str | None | 填充方式：'class' 或 'mode' |
| `fill_str` | str | None | 当 fill_type='class' 时，填充的字符串 |

#### 填充方式

**class（作为一个类别）：**
```python
data_filled = dp.fillna_cate_var(
    col_list=['education', 'marriage'],
    fill_type='class',
    fill_str='Unknown'
)
# 缺失值会被填充为 'Unknown'
```

**mode（众数）：**
```python
data_filled = dp.fillna_cate_var(
    col_list=['education', 'marriage'],
    fill_type='mode'
)
# 缺失值会被填充为该变量的众数
```

---

### 3. fillna_num_var() - 数值型变量缺失值填充

填充数值型变量的缺失值，支持多种策略。

#### 语法

```python
dp.fillna_num_var(
    col_list: List[str],
    fill_type: Optional[str] = None,
    fill_class_num: Optional[float] = None,
    filled_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 需要填充的变量列表 |
| `fill_type` | str | None | 填充方式：'0', 'median', 'class', 'rf' |
| `fill_class_num` | float | None | 当 fill_type='class' 时，填充的数值 |
| `filled_data` | pd.DataFrame | None | 当 fill_type='rf' 时，已填充的其他变量数据 |

#### 填充方式

**1. 填充为 0：**
```python
data_filled = dp.fillna_num_var(
    col_list=['debt_amount'],
    fill_type='0'
)
```

**2. 中位数填充（推荐，缺失率 < 5%）：**
```python
data_filled = dp.fillna_num_var(
    col_list=['income', 'age'],
    fill_type='median'
)
```

**3. 作为特殊类别填充（缺失率 > 15%）：**
```python
from yihuier.constants import MISSING_VALUE_NEG_999

data_filled = dp.fillna_num_var(
    col_list=['income'],
    fill_type='class',
    fill_class_num=MISSING_VALUE_NEG_999  # -999
)
```

**4. 随机森林填充（缺失率 5%-15%）：**

⚠️ **注意**：需要先填充缺失率较低的变量

```python
# 第一步：填充低缺失率变量
data_filled = dp.fillna_num_var(
    col_list=['age', 'work_years'],
    fill_type='median'
)

# 第二步：用随机森林填充高缺失率变量
data_filled = dp.fillna_num_var(
    col_list=['income'],
    fill_type='rf',
    filled_data=data_filled
)
```

---

### 4. delete_missing_var() - 删除高缺失率变量

删除缺失率超过阈值的变量。

#### 语法

```python
dp.delete_missing_var(
    threshold: float
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 必填 | 缺失率阈值（0-1），如 0.15 表示15% |

#### 使用示例

```python
# 删除缺失率超过15%的变量
data_clean = dp.delete_missing_var(threshold=0.15)
# 输出: 缺失率超过0.15的变量个数为3
```

---

### 5. delete_missing_obs() - 删除高缺失样本

删除包含过多缺失值的观测（样本）。

#### 语法

```python
dp.delete_missing_obs(
    threshold: Union[float, int]
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float/int | 必填 | 阈值。≥1表示变量数量，<1表示百分比 |

#### 使用示例

```python
# 删除缺失变量数超过5个的样本
data_clean = dp.delete_missing_obs(threshold=5)

# 或删除缺失变量数超过30%的样本
data_clean = dp.delete_missing_obs(threshold=0.3)
```

---

### 6. const_delete() - 删除常变量

删除唯一值比例过低的变量（常变量/同值化严重）。

#### 语法

```python
dp.const_delete(
    threshold: float = 0.9
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.9 | 唯一值比例阈值 |

#### 使用示例

```python
# 删除90%以上都是同一个值的变量
data_clean = dp.const_delete(threshold=0.9)
# 输出: 删除常变量/同值化处理后的变量个数为2
```

#### 说明

- 如果某个变量中某个值的占比超过 90%，则认为是常变量
- 常变量对模型没有区分能力，应该删除

---

### 7. target_missing_delete() - 删除目标变量缺失的样本

删除目标变量为空的观测。

#### 语法

```python
dp.target_missing_delete() -> Optional[pd.DataFrame]
```

#### 使用示例

```python
data_clean = dp.target_missing_delete()
# 输出: 删除目标变量缺失的观测数: 15
```

⚠️ **注意**：如果没有指定目标变量，此方法不会执行任何操作。

---

### 8. date_var_shift_binary() - 日期变量转二进制

将日期型变量转换为二进制变量（表示日期是否存在）。

#### 语法

```python
dp.date_var_shift_binary(
    col_list: List[str],
    replace: bool = False
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 日期型变量列表 |
| `replace` | bool | False | 是否保留原始日期变量 |

#### 使用示例

```python
# 获取日期变量
date_vars = yh.get_date_variables()

# 转换为二进制变量，不保留原始日期列
data_binary = dp.date_var_shift_binary(
    col_list=date_vars,
    replace=False
)

# 结果会创建新变量: {原变量名}_binary
# 1 = 日期存在，0 = 日期缺失
```

---

## 完整处理流程

### 推荐的数据预处理流程

```python
from yihuier import Yihuier
import pandas as pd

# 1. 初始化
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')
dp = yh.dp_module

# 2. 查看缺失值分布
print("=== 缺失值分析 ===")
dp.plot_bar_missing_var(plt_size=(12, 6))

# 3. 删除目标变量缺失的样本
print("\n=== 删除目标变量缺失样本 ===")
yh.data = dp.target_missing_delete()

# 4. 删除高缺失率变量（>15%）
print("\n=== 删除高缺失率变量 ===")
yh.data = dp.delete_missing_var(threshold=0.15)

# 5. 删除常变量
print("\n=== 删除常变量 ===")
yh.data = dp.const_delete(threshold=0.9)

# 6. 填充类别型变量
print("\n=== 填充类别型变量 ===")
cate_vars = yh.get_categorical_variables()
if cate_vars:
    yh.data = dp.fillna_cate_var(
        col_list=cate_vars,
        fill_type='mode'
    )

# 7. 填充数值型变量
print("\n=== 填充数值型变量 ===")
num_vars = yh.get_numeric_variables()

# 获取每个变量的缺失率
missing_pct = yh.data[num_vars].isnull().mean()

# 按缺失率分层处理
low_missing = missing_pct[missing_pct < 0.05].index.tolist()
mid_missing = missing_pct[(missing_pct >= 0.05) & (missing_pct < 0.15)].index.tolist()
high_missing = missing_pct[missing_pct >= 0.15].index.tolist()

# 低缺失率：中位数填充
if low_missing:
    yh.data = dp.fillna_num_var(
        col_list=low_missing,
        fill_type='median'
    )
    print(f"  中位数填充: {low_missing}")

# 中缺失率：随机森林填充
if mid_missing and low_missing:
    yh.data = dp.fillna_num_var(
        col_list=mid_missing,
        fill_type='rf',
        filled_data=yh.data
    )
    print(f"  随机森林填充: {mid_missing}")

# 高缺失率：作为特殊类别
if high_missing:
    from yihuier.constants import MISSING_VALUE_NEG_999
    yh.data = dp.fillna_num_var(
        col_list=high_missing,
        fill_type='class',
        fill_class_num=MISSING_VALUE_NEG_999
    )
    print(f"  特殊类别填充: {high_missing}")

# 8. 删除高缺失样本（可选）
print("\n=== 删除高缺失样本 ===")
yh.data = dp.delete_missing_obs(threshold=5)

# 9. 处理日期变量
print("\n=== 处理日期变量 ===")
date_vars = yh.get_date_variables()
if date_vars:
    yh.data = dp.date_var_shift_binary(
        col_list=date_vars,
        replace=False
    )
    print(f"  日期变量已转换: {date_vars}")

print("\n=== 数据预处理完成 ===")
print(f"处理后数据形状: {yh.data.shape}")
```

---

## 缺失值处理策略

### 策略选择指南

| 缺失率 | 推荐策略 | 方法 |
|---------|---------|------|
| < 5% | 中位数填充 | `fillna_num_var(..., fill_type='median')` |
| 5% - 15% | 随机森林填充 | `fillna_num_var(..., fill_type='rf')` |
| > 15% | 作为特殊类别 | `fillna_num_var(..., fill_type='class')` |
| > 50% | 删除变量 | `delete_missing_var(threshold=0.5)` |

### 自动化处理函数

```python
def auto_fill_missing(yh):
    """自动选择最佳策略填充缺失值"""
    dp = yh.dp_module
    num_vars = yh.get_numeric_variables()

    # 计算缺失率
    missing_pct = yh.data[num_vars].isnull().mean()

    # 分层处理
    for var, pct in missing_pct.items():
        if pct == 0:
            continue
        elif pct < 0.05:
            print(f"{var} ({pct:.1%}): 中位数填充")
            yh.data = dp.fillna_num_var(
                col_list=[var],
                fill_type='median'
            )
        elif pct < 0.15:
            print(f"{var} ({pct:.1%}): 保留或特殊值填充")
            yh.data = dp.fillna_num_var(
                col_list=[var],
                fill_type='class',
                fill_class_num=-999
            )
        else:
            print(f"{var} ({pct:.1%}): 建议删除或保留")

# 使用
auto_fill_missing(yh)
```

---

## 注意事项

### 1. 数据泄露风险

⚠️ **重要**：填充缺失值时，应该只使用训练集的统计量。

```python
# 错误做法
from sklearn.model_selection import train_test_split

# 先填充再分割
data_filled = dp.fillna_num_var(['income'], fill_type='median')
X_train, X_test = train_test_split(data_filled, test_size=0.3)

# 正确做法
# 先分割
X_train, X_test = train_test_split(data, test_size=0.3)

# 用训练集的中位数填充测试集
train_median = X_train['income'].median()
X_train['income'] = X_train['income'].fillna(train_median)
X_test['income'] = X_test['income'].fillna(train_median)
```

### 2. 日期变量识别

`Yihuier` 会自动识别以下日期格式：
- `%Y%m%d` - 20231205
- `%Y%m` - 202312
- `%m%d` - 1205
- `%Y%m%d%H%M%S` - 20231205123000
- `%Y-%m-%d` - 2023-12-05
- `%Y-%m` - 2023-12
- `%m-%d` - 12-05

```python
# 查看自动识别的日期变量
date_vars = yh.get_date_variables()
print("日期变量:", date_vars)

# 手动检查
for col in data.columns:
    try:
        pd.to_datetime(data[col], errors='raise')
        print(f"{col} 可能是日期变量")
    except:
        pass
```

### 3. 常量定义

建议使用 `constants.py` 中定义的常量：

```python
from yihuier.constants import MISSING_VALUE_NEG_999, MISSING_VALUE_NEG_1111

# 使用常量而非魔法数字
data_filled = dp.fillna_num_var(
    col_list=['income'],
    fill_type='class',
    fill_class_num=MISSING_VALUE_NEG_999  # 而非直接写 -999
)
```

---

## 常见问题

### Q1: 如何查看填充前后的差异？

```python
# 记录原始数据
original_missing = yh.data.isnull().sum()

# 填充数据
yh.data = dp.fillna_num_var(['income'], fill_type='median')

# 对比
filled_missing = yh.data.isnull().sum()
print("填充前后缺失值变化:")
print((original_missing - filled_missing).to_frame('filled_count'))
```

### Q2: 随机森林填充很慢怎么办？

```python
# 1. 减少填充变量数量
# 2. 使用快速填充替代
yh.data = dp.fillna_num_var(
    col_list=['income'],
    fill_type='class',
    fill_class_num=-999
)

# 3. 或使用中位数快速填充所有变量
yh.data = dp.fillna_num_var(
    col_list=num_vars,
    fill_type='median'
)
```

### Q3: 如何处理类别型变量的缺失值？

```python
# 选项1: 填充为 'Unknown'
yh.data = dp.fillna_cate_var(
    col_list=['education'],
    fill_type='class',
    fill_str='Unknown'
)

# 选项2: 填充为众数
yh.data = dp.fillna_cate_var(
    col_list=['education'],
    fill_type='mode'
)
```

---

## 相关文档

- [EDA模块](01-eda.md) - 数据处理前的探索分析
- [分箱模块](03-binning.md) - 数据处理后的分箱操作
- [常量定义](../yihuier/constants.py) - 使用的特殊值常量
