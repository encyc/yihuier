# 探索性数据分析模块 (EDAModule)

## 概述

`EDAModule` 提供了全面的探索性数据分析（EDA）功能，帮助您在建模前深入了解数据特征。

### 主要功能

- **变量分布可视化** - 类别型和数值型变量的分布分析
- **违约率分析** - 按变量分组分析违约率
- **自动EDA** - 快速生成数据集统计摘要

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# EDA 模块会自动识别变量类型
eda = yh.eda_module
```

### 自动变量分类

模块初始化时会自动将变量分类：

- `category_variables` - 类别型变量（object, category 类型）
- `numeric_variables` - 数值型变量（int, float 类型，排除日期型）
- `variables` - 所有变量名

---

## API 参考

### 1. plot_cate_var() - 类别型变量分布

绘制类别型变量的分布柱状图。

#### 语法

```python
eda.plot_cate_var(
    col_list: List[str],
    hspace: float = 0.4,
    wspace: float = 0.4,
    plt_size: Optional[Tuple[int, int]] = None,
    plt_num: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 要绘制的变量列表 |
| `hspace` | float | 0.4 | 子图之间的垂直间隔 |
| `wspace` | float | 0.4 | 子图之间的水平间隔 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸，如 (10, 6) |
| `plt_num` | int | None | 子图的数量 |
| `x` | int | None | 子图矩阵的行数 |
| `y` | int | None | 子图矩阵的列数 |

#### 使用示例

```python
# 绘制单个变量
eda.plot_cate_var(
    col_list=['education'],
    plt_size=(6, 4),
    plt_num=1,
    x=1,
    y=1
)

# 绘制多个变量
eda.plot_cate_var(
    col_list=['education', 'marriage', 'house_type'],
    plt_size=(12, 8),
    plt_num=3,
    x=2,
    y=2
)
```

#### 输出

生成类别型变量的分布柱状图，显示每个类别值的样本数量。

---

### 2. plot_num_col() - 数值型变量分布

绘制数值型变量的分布图，支持直方图、箱线图和散点图。

#### 语法

```python
eda.plot_num_col(
    col_list: List[str],
    plt_type: str = 'hist',
    hspace: float = 0.4,
    wspace: float = 0.4,
    plt_size: Optional[Tuple[int, int]] = None,
    plt_num: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 要绘制的变量列表 |
| `plt_type` | str | 'hist' | 图表类型：'hist'(直方图), 'box'(箱线图), 'stripplot'(散点图) |
| `hspace` | float | 0.4 | 子图之间的垂直间隔 |
| `wspace` | float | 0.4 | 子图之间的水平间隔 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |
| `plt_num` | int | None | 子图的数量 |
| `x` | int | None | 子图矩阵的行数 |
| `y` | int | None | 子图矩阵的列数 |

#### 使用示例

```python
# 直方图
eda.plot_num_col(
    col_list=['age', 'income', 'debt_ratio'],
    plt_type='hist',
    plt_size=(12, 8),
    plt_num=3,
    x=2,
    y=2
)

# 箱线图 - 查看异常值
eda.plot_num_col(
    col_list=['income', 'debt_ratio'],
    plt_type='box',
    plt_size=(10, 4),
    plt_num=2,
    x=1,
    y=2
)

# 散点图
eda.plot_num_col(
    col_list=['age'],
    plt_type='stripplot',
    plt_size=(8, 4)
)
```

---

### 3. plot_default_cate() - 类别型变量违约率分析

按类别型变量的各个值分组，分析每组的违约率。

#### 语法

```python
eda.plot_default_cate(
    col_list: List[str],
    hspace: float = 0.4,
    wspace: float = 0.4,
    plt_size: Optional[Tuple[int, int]] = None,
    plt_num: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 要分析的变量列表 |
| `hspace` | float | 0.4 | 子图之间的垂直间隔 |
| `wspace` | float | 0.4 | 子图之间的水平间隔 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |
| `plt_num` | int | None | 子图的数量 |
| `x` | int | None | 子图矩阵的行数 |
| `y` | int | None | 子图矩阵的列数 |

#### 使用示例

```python
# 分析不同教育程度的违约率
eda.plot_default_cate(
    col_list=['education', 'marriage'],
    plt_size=(12, 6),
    plt_num=2,
    x=1,
    y=2
)
```

#### 输出说明

- 图中会显示一条垂直线，表示总体违约率
- 每个柱子表示该类别值的违约率
- 高于总体违约率的类别表示风险较高

---

### 4. plot_default_num() - 数值型变量违约率分析

将数值型变量等深分箱后，分析每个箱的违约率变化趋势。

#### 语法

```python
eda.plot_default_num(
    col_list: List[str],
    hspace: float = 0.4,
    wspace: float = 0.4,
    q: Optional[int] = None,
    plt_size: Optional[Tuple[int, int]] = None,
    plt_num: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 要分析的变量列表 |
| `hspace` | float | 0.4 | 子图之间的垂直间隔 |
| `wspace` | float | 0.4 | 子图之间的水平间隔 |
| `q` | int | None | 等深分箱的箱数 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |
| `plt_num` | int | None | 子图的数量 |
| `x` | int | None | 子图矩阵的行数 |
| `y` | int | None | 子图矩阵的列数 |

#### 使用示例

```python
# 分析年龄与违约率的关系
eda.plot_default_num(
    col_list=['age', 'income', 'debt_ratio'],
    q=10,  # 分为10个箱
    plt_size=(12, 8),
    plt_num=3,
    x=2,
    y=2
)
```

#### 输出说明

- 图中会显示一条水平线，表示总体违约率
- 折线图显示违约率随变量值的变化趋势
- 可以识别变量的单调性（风险评分的重要特征）

---

### 5. auto_eda_profiling() - 自动EDA（有图）

使用 `ydata_profiling` 自动生成详细的数据分析报告。

⚠️ **注意**：当数据维度较高时，此方法速度较慢。

#### 语法

```python
eda.auto_eda_profiling() -> None
```

#### 使用示例

```python
# 生成自动EDA报告
eda.auto_eda_profiling()
# 报告将保存为 Data/output.html
```

#### 前置条件

需要安装 `ydata-profiling`：

```bash
pip install ydata-profiling
```

---

### 6. auto_eda_simple() - 快速自动EDA（无图）

快速计算变量的统计摘要，不生成图形。

#### 语法

```python
eda.auto_eda_simple() -> pd.DataFrame
```

#### 返回值

返回一个 DataFrame，包含以下统计信息：

**类别型变量：**
- `unique_count` - 唯一值数量
- `entropy` - 熵值（衡量分布的不确定性）
- `missing_pct` - 缺失值百分比

**数值型变量：**
- `mean` - 均值
- `min` - 最小值
- `q1` - 25分位数
- `median` - 中位数
- `q3` - 75分位数
- `max` - 最大值
- `missing_pct` - 缺失值百分比

#### 使用示例

```python
# 快速查看所有变量的统计摘要
stats_df = eda.auto_eda_simple()
print(stats_df)

# 筛选高缺失率变量
high_missing = stats_df[stats_df['missing_pct'] > 30]
print("高缺失率变量：", high_missing.index.tolist())

# 筛选高熵值变量（分布较分散）
high_entropy = stats_df[stats_df['entropy'] > 2]
print("高熵值变量：", high_entropy.index.tolist())
```

---

## 使用流程

### 典型的 EDA 工作流程

```python
from yihuier import Yihuier
import pandas as pd

# 1. 初始化
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 2. 快速了解数据
print("数据形状:", yh.data.shape)
print("\n变量类型分布:")
print("  类别型变量:", yh.get_categorical_variables())
print("  数值型变量:", yh.get_numeric_variables())

# 3. 快速统计摘要
stats = yh.eda_module.auto_eda_simple()
print("\n数据统计摘要:")
print(stats)

# 4. 分析类别型变量分布
if yh.get_categorical_variables():
    yh.eda_module.plot_cate_var(
        col_list=yh.get_categorical_variables()[:4],
        plt_size=(12, 8),
        plt_num=4,
        x=2,
        y=2
    )

# 5. 分析数值型变量分布
if yh.get_numeric_variables():
    yh.eda_module.plot_num_col(
        col_list=yh.get_numeric_variables()[:4],
        plt_type='hist',
        plt_size=(12, 8),
        plt_num=4,
        x=2,
        y=2
    )

# 6. 分析违约率
# 类别型变量
if yh.get_categorical_variables():
    yh.eda_module.plot_default_cate(
        col_list=yh.get_categorical_variables()[:3],
        plt_size=(12, 8),
        plt_num=3,
        x=1,
        y=3
    )

# 数值型变量
if yh.get_numeric_variables():
    yh.eda_module.plot_default_num(
        col_list=yh.get_numeric_variables()[:3],
        q=10,
        plt_size=(12, 8),
        plt_num=3,
        x=1,
        y=3
    )
```

---

## 注意事项

### 1. 数据要求

- 目标变量必须指定（`target` 参数）
- 建议在分析前处理缺失值和异常值

### 2. 可视化最佳实践

```python
# 根据变量数量动态计算子图布局
import math

def plot_multiple_vars(eda, col_list, plot_type='hist'):
    n = len(col_list)
    # 计算最佳行列数
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)

    if plot_type == 'cate':
        eda.plot_cate_var(
            col_list=col_list,
            plt_size=(4*cols, 4*rows),
            plt_num=n,
            x=rows,
            y=cols
        )
    else:
        eda.plot_num_col(
            col_list=col_list,
            plt_type=plot_type,
            plt_size=(4*cols, 4*rows),
            plt_num=n,
            x=rows,
            y=cols
        )

# 使用示例
plot_multiple_vars(eda, ['var1', 'var2', 'var3', 'var4'], 'cate')
```

### 3. 大数据集处理

```python
# 对于大数据集，先采样再可视化
sample_size = 10000
if len(yh.data) > sample_size:
    sample = yh.data.sample(n=sample_size, random_state=42)
    yh_sample = Yihuier(sample, target='dlq_flag')
    yh_sample.eda_module.plot_num_col(
        col_list=['income'],
        plt_type='hist'
    )
```

---

## 常见问题

### Q1: 图表中文字显示为方框怎么办？

这是中文字体问题，解决方案：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### Q2: 如何保存图表？

```python
import matplotlib.pyplot as plt

# 绘制图表
eda.plot_cate_var(col_list=['education'])

# 保存当前图表
plt.savefig('eda_education.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Q3: auto_eda_profiling 报错怎么办？

确保安装了 ydata-profiling：

```bash
pip install ydata-profiling
```

或者使用 auto_eda_simple() 作为替代方案。

---

## 相关文档

- [数据处理模块](data-processing.md) - EDA 前的数据清洗
- [变量分箱模块](binning.md) - 分箱后的 WOE 分析
- [变量选择模块](var-select.md) - 基于 EDA 结果的特征选择
