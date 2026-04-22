# 评分卡实现模块 (ScorecardImplementModule)

## 概述

`ScorecardImplementModule` 将逻辑回归模型转换为可解释的信用评分卡，并生成最终的客户评分。

### 主要功能

- **评分卡刻度** - 设置评分卡基准和PDO
- **变量得分表** - 计算每个变量各箱的得分
- **分数转换** - 将客户数据转换为评分
- **评分分析** - KS曲线、PR曲线、得分分布等
- **阈值验证** - 设定cut-off点并评估

---

## 核心概念

### 什么是评分卡？

评分卡是一种将复杂模型转换为简单加法公式的工具：

```
总分 = 基础分 + ∑(变量得分)
变量得分 = 回归系数 × WOE值 × 缩放因子
```

### 评分卡参数

- **基础分** - 当所有变量都取平均水平时的分数
- **PDO (Point to Double Odds)** - 好坏比翻倍所需分数
- **Odds** - 在特定分数下的好坏比

### 计算公式

```
A = PDO / ln(2)
B = Score - A × ln(Odds)
总分 = A × ln(Odds) + B
```

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 准备数据
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 数据预处理和分箱
yh.data = yh.dp_module.delete_missing_var(threshold=0.15)
yh.binning_module.binning_num(
    col_list=yh.get_numeric_variables(),
    max_bin=5
)
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 训练模型
X = data_woe.drop([yh.target], axis=1)
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# 初始化评分卡模块
si = yh.si_module
```

---

## API 参考

### 1. cal_scale() - 计算评分卡刻度

计算评分卡的基础分和缩放因子。

#### 语法

```python
si.cal_scale(
    score: float,
    odds: float,
    PDO: float,
    model: BaseEstimator
) -> Tuple[float, float, float]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `score` | float | 必填 | 特定odds下的分数 |
| `odds` | float | 必填 | 好坏比（好样本/坏样本） |
| `PDO` | float | 必填 | 好坏比翻倍的分数差 |
| `model` | BaseEstimator | 必填 | 训练好的逻辑回归模型 |

#### 常用参数组合

```python
# 组合1: 600分时好坏比为20:1，PDO=20
A, B, base_score = si.cal_scale(
    score=600,
    odds=20,
    PDO=20,
    model=model
)

# 组合2: 500分时好坏比为10:1，PDO=15
A, B, base_score = si.cal_scale(
    score=500,
    odds=10,
    PDO=15,
    model=model
)
```

#### 返回值

返回一个元组 `(A, B, base_score)`：

- `A` - 缩放因子A
- `B` - 缩放因子B
- `base_score` - 基础分

---

### 2. score_df_concat() - 生成变量得分表

将WOE结果表转换为得分表。

#### 语法

```python
si.score_df_concat(
    woe_df: pd.DataFrame,
    model: BaseEstimator,
    B: float
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `woe_df` | pd.DataFrame | 必填 | WOE结果表 |
| `model` | BaseEstimator | 必填 | 训练好的逻辑回归模型 |
| `B` | float | 必填 | 缩放因子B（从cal_scale获得） |

#### 使用示例

```python
# 1. 计算评分卡刻度
A, B, base_score = si.cal_scale(
    score=600,
    odds=20,
    PDO=20,
    model=model
)

# 2. 获取WOE结果表
woe_df = yh.binning_module.woe_result_df

# 3. 生成得分表
score_df = si.score_df_concat(
    woe_df=woe_df,
    model=model,
    B=B
)

# 4. 查看得分表
print(score_df.head(20))

# 保存得分表
score_df.to_csv('output/score_table.csv', index=False)
```

#### 得分表格式

| 列名 | 说明 |
|------|------|
| `col` | 变量名 |
| `bin` | 分箱区间 |
| `woe` | WOE值 |
| `score` | 该箱的得分 |
| ... | 其他统计列 |

---

### 3. score_transform() - 分数转换

将客户数据转换为评分。

#### 语法

```python
si.score_transform(
    df: pd.DataFrame,
    target: str,
    df_score: pd.DataFrame
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 客户数据（原始数据） |
| `target` | str | 必填 | 目标变量名 |
| `df_score` | pd.DataFrame | 必填 | 得分表 |

#### 使用示例

```python
# 1. 准备得分表
A, B, base_score = si.cal_scale(600, 20, 20, model)
woe_df = yh.binning_module.woe_result_df
score_df = si.score_df_concat(woe_df, model, B)

# 2. 转换训练集
train_scored = si.score_transform(
    df=yh.data,
    target=yh.target,
    df_score=score_df
)

# 3. 查看得分
print(train_scored.head())

# 4. 查看得分统计
print(f"最小分: {train_scored.drop([yh.target], axis=1).min().min():.0f}")
print(f"最大分: {train_scored.drop([yh.target], axis=1).max().max():.0f}")
print(f"平均分: {train_scored.drop([yh.target], axis=1).mean().mean():.0f}")
```

⚠️ **注意**：转换后，原始变量列会被分数替换。如需保留原始数据，请先复制。

---

### 4. plot_score_ks() - 评分KS曲线

绘制评分的KS曲线。

#### 语法

```python
si.plot_score_ks(
    df: pd.DataFrame,
    score_col: str,
    target: str
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 包含评分的数据 |
| `score_col` | str | 必填 | 评分列名 |
| `target` | str | 必填 | 目标变量名 |

#### 使用示例

```python
# 假设评分列名为 'score'
# 如果有多列评分，需要先求和
scored_data = train_scored.copy()
scored_data['score'] = scored_data.drop([yh.target], axis=1).sum(axis=1)

# 绘制KS曲线
si.plot_score_ks(
    df=scored_data,
    score_col='score',
    target=yh.target
)
```

---

### 5. plot_PR() - PR曲线

绘制精确率-召回率曲线。

#### 语法

```python
si.plot_PR(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    plt_size: Optional[Tuple[int, int]] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 包含评分的数据 |
| `score_col` | str | 必填 | 评分列名 |
| `target` | str | 必填 | 目标变量名 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |

#### 使用示例

```python
si.plot_PR(
    df=scored_data,
    score_col='score',
    target=yh.target,
    plt_size=(8, 6)
)
```

---

### 6. plot_score_hist() - 得分分布图

绘制好坏用户的得分分布。

#### 语法

```python
si.plot_score_hist(
    df: pd.DataFrame,
    target: str,
    score_col: str,
    plt_size: Optional[Tuple[int, int]] = None,
    cutoff: Optional[float] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 包含评分的数据 |
| `target` | str | 必填 | 目标变量名 |
| `score_col` | str | 必填 | 评分列名 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |
| `cutoff` | float | None | cut-off分界线 |

#### 使用示例

```python
# 绘制得分分布（无cut-off线）
si.plot_score_hist(
    df=scored_data,
    target=yh.target,
    score_col='score',
    plt_size=(10, 6)
)

# 绘制得分分布（带cut-off线）
si.plot_score_hist(
    df=scored_data,
    target=yh.target,
    score_col='score',
    cutoff=500,  # 在500分处画线
    plt_size=(10, 6)
)
```

---

### 7. score_info() - 得分明细表

生成评分的详细统计表。

#### 语法

```python
si.score_info(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    step: Optional[float] = None
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 包含评分的数据 |
| `score_col` | str | 必填 | 评分列名 |
| `target` | str | 必填 | 目标变量名 |
| `x` | float | None | 最小区间左值 |
| `y` | float | None | 最大区间右值 |
| `step` | float | None | 区间分数间隔 |

#### 使用示例

```python
# 生成得分明细表
info_df = si.score_info(
    df=scored_data,
    score_col='score',
    target=yh.target,
    x=200,  # 从200分开始
    y=800,  # 到800分结束
    step=20  # 每20分一个区间
)

print(info_df)

# 保存明细表
info_df.to_csv('output/score_info.csv', index=False)
```

#### 输出格式

| 列名 | 说明 |
|------|------|
| `score_bin` | 分数区间 |
| `用户数` | 该区间用户数 |
| `坏用户` | 该区间坏用户数 |
| `好用户` | 该区间好用户数 |
| `违约占比` | 该区间违约率 |
| `累计用户` | 累计用户数 |
| `坏用户累计` | 累计坏用户数 |
| `好用户累计` | 累计好用户数 |
| `坏用户累计占比` | 累计坏用户占比 |
| `好用户累计占比` | 累计好用户占比 |

---

### 8. plot_lifting() - 提升图和洛伦兹曲线

绘制模型的提升效果。

#### 语法

```python
si.plot_lifting(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    bins: int = 10,
    plt_size: Optional[Tuple[int, int]] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 包含评分的数据 |
| `score_col` | str | 必填 | 评分列名 |
| `target` | str | 必填 | 目标变量名 |
| `bins` | int | 10 | 分数等份数 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |

#### 使用示例

```python
si.plot_lifting(
    df=scored_data,
    score_col='score',
    target=yh.target,
    bins=10,
    plt_size=(12, 5)
)
```

#### 输出说明

- **提升图**：展示模型相比随机选择的优势
- **洛伦兹曲线**：展示累积捕获效果

---

### 9. rule_verify() - Cut-off点验证

设定cut-off点并评估效果。

#### 语法

```python
si.rule_verify(
    df: pd.DataFrame,
    col_score: str,
    target: str,
    cutoff: float
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 包含评分的数据 |
| `col_score` | str | 必填 | 评分列名 |
| `target` | str | 必填 | 目标变量名 |
| `cutoff` | float | 必填 | cut-off分数 |

#### 使用示例

```python
# 设定cut-off点为500分
matrix_df = si.rule_verify(
    df=scored_data,
    col_score='score',
    target=yh.target,
    cutoff=500
)

# 输出:
# 精确率: 0.85
# 查全率: 0.72
# 误伤率: 0.15
# 规则拒绝率: 0.35

print(matrix_df)
```

#### 输出指标

- **精确率**：被拒绝用户中实际坏用户的比例
- **查全率**：捕获的坏用户占所有坏用户的比例
- **误伤率**：被拒绝的好用户占所有好用户的比例
- **规则拒绝率**：被拒绝用户占所有用户的比例

---

## 完整评分卡实现流程

### 标准流程

```python
from yihuier import Yihuier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 数据准备
print("=== 数据准备 ===")
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 2. 数据预处理
print("\n=== 数据预处理 ===")
yh.data = yh.dp_module.delete_missing_var(threshold=0.15)
yh.data = yh.dp_module.const_delete(threshold=0.9)

# 3. 变量分箱
print("\n=== 变量分箱 ===")
num_vars = yh.get_numeric_variables()
yh.binning_module.binning_num(
    col_list=num_vars,
    max_bin=5,
    method='ChiMerge'
)

# 4. WOE转换
print("\n=== WOE转换 ===")
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 5. 训练模型
print("\n=== 训练模型 ===")
X = data_woe.drop([yh.target], axis=1)
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# 6. 模型评估
print("\n=== 模型评估 ===")
y_pred = model.predict_proba(X_test)[:, 1]
from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)
ks = yh.me_module.model_ks(y_test, y_pred)
print(f"AUC: {auc:.3f}, KS: {ks:.3f}")

# 7. 计算评分卡刻度
print("\n=== 计算评分卡刻度 ===")
A, B, base_score = yh.si_module.cal_scale(
    score=600,
    odds=20,
    PDO=20,
    model=model
)
print(f"缩放因子A: {A:.2f}")
print(f"缩放因子B: {B:.2f}")
print(f"基础分: {base_score:.2f}")

# 8. 生成得分表
print("\n=== 生成得分表 ===")
woe_df = yh.binning_module.woe_result_df
score_df = yh.si_module.score_df_concat(
    woe_df=woe_df,
    model=model,
    B=B
)

# 保存得分表
score_df.to_csv('output/score_table.csv', index=False)
print(f"得分表已保存，共 {len(score_df)} 行")

# 9. 分数转换
print("\n=== 分数转换 ===")
train_scored = yh.si_module.score_transform(
    df=yh.data,
    target=yh.target,
    df_score=score_df
)

# 计算总分
train_scored['total_score'] = train_scored.drop([yh.target], axis=1).sum(axis=1)
print(f"总分范围: {train_scored['total_score'].min():.0f} - {train_scored['total_score'].max():.0f}")

# 10. 评分分析
print("\n=== 评分分析 ===")

# KS曲线
yh.si_module.plot_score_ks(
    df=train_scored,
    score_col='total_score',
    target=yh.target
)

# 得分分布
yh.si_module.plot_score_hist(
    df=train_scored,
    target=yh.target,
    score_col='total_score',
    plt_size=(10, 6)
)

# 提升图
yh.si_module.plot_lifting(
    df=train_scored,
    score_col='total_score',
    target=yh.target,
    bins=10
)

# 11. 得分明细表
print("\n=== 得分明细表 ===")
info_df = yh.si_module.score_info(
    df=train_scored,
    score_col='total_score',
    target=yh.target,
    x=200,
    y=800,
    step=20
)
print(info_df)
info_df.to_csv('output/score_info.csv', index=False)

# 12. Cut-off点验证
print("\n=== Cut-off点验证 ===")
cutoff = 500
yh.si_module.rule_verify(
    df=train_scored,
    col_score='total_score',
    target=yh.target,
    cutoff=cutoff
)

# 13. 应用到新客户
print("\n=== 应用到新客户 ===")
new_customer = pd.DataFrame({
    'age': [35],
    'income': [50000],
    'debt_ratio': [0.3],
    # ... 其他变量
})

# 转换为评分
new_customer_scored = yh.si_module.score_transform(
    df=new_customer,
    target=yh.target,
    df_score=score_df
)

# 计算总分
new_customer_scored['total_score'] = new_customer_scored.sum(axis=1)
print(f"新客户评分: {new_customer_scored['total_score'].iloc[0]:.0f}")

# 判断是否通过
if new_customer_scored['total_score'].iloc[0] >= cutoff:
    print("建议：通过")
else:
    print("建议：拒绝")
```

---

## Cut-off点选择

### 基于业务目标选择

```python
def find_optimal_cutoff(scored_data, score_col, target, metric='f1'):
    """寻找最优cut-off点"""
    from sklearn.metrics import f1_score, precision_score, recall_score

    best_cutoff = None
    best_score = 0

    for cutoff in range(int(scored_data[score_col].min()),
                      int(scored_data[score_col].max()),
                      10):
        y_pred = (scored_data[score_col] >= cutoff).astype(int)

        if metric == 'f1':
            score = f1_score(scored_data[target], y_pred)
        elif metric == 'precision':
            score = precision_score(scored_data[target], y_pred)
        elif metric == 'recall':
            score = recall_score(scored_data[target], y_pred)
        else:
            raise ValueError(f"未知的metric: {metric}")

        if score > best_score:
            best_score = score
            best_cutoff = cutoff

    return best_cutoff, best_score

# 使用
best_cutoff, best_f1 = find_optimal_cutoff(
    scored_data=train_scored,
    score_col='total_score',
    target=yh.target,
    metric='f1'
)

print(f"最优cut-off点: {best_cutoff} (F1={best_f1:.3f})")
```

### 基于误损成本选择

```python
def find_cutoff_by_cost(scored_data, score_col, target,
                       cost_fp=1000, cost_fn=5000):
    """基于成本寻找最优cut-off点

    Args:
        cost_fp: 误伤好客户的成本（如失去的利润）
        cost_fn: 漏掉坏客户的成本（如坏账损失）
    """
    best_cutoff = None
    min_cost = float('inf')

    for cutoff in range(int(scored_data[score_col].min()),
                      int(scored_data[score_col].max()),
                      10):
        y_pred = (scored_data[score_col] >= cutoff).astype(int)
        y_true = scored_data[target]

        # 计算混淆矩阵
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        # 计算总成本
        total_cost = fp * cost_fp + fn * cost_fn

        if total_cost < min_cost:
            min_cost = total_cost
            best_cutoff = cutoff

    return best_cutoff, min_cost

# 使用
# 假设误伤一个好客户损失1000元利润
# 漏掉一个坏客户损失5000元
best_cutoff, min_cost = find_cutoff_by_cost(
    scored_data=train_scored,
    score_col='total_score',
    target=yh.target,
    cost_fp=1000,
    cost_fn=5000
)

print(f"最优cut-off点: {best_cutoff}")
print(f"最小预期成本: {min_cost:,.0f}元")
```

---

## 评分卡监控

### 评分分布监控

```python
def monitor_score_distribution(scored_data, score_col, period='month'):
    """监控评分分布变化"""

    # 按时间分组统计
    if period == 'month':
        scored_data['period'] = pd.to_datetime(scored_data['date']).dt.to_period('M')
    else:
        scored_data['period'] = pd.to_datetime(scored_data['date']).dt.to_period('W')

    # 计算每个时期的评分统计
    score_stats = scored_data.groupby('period').agg({
        score_col: ['mean', 'std', 'min', 'max', 'count']
    })

    return score_stats

# 使用
score_stats = monitor_score_distribution(train_scored, 'total_score')
print(score_stats)
```

---

## 注意事项

### 1. 分数有效性

```python
# 检查分数是否合理
def validate_scores(scored_data, score_col):
    """验证分数的有效性"""

    # 1. 检查是否有缺失值
    missing = scored_data[score_col].isnull().sum()
    if missing > 0:
        print(f"⚠️  警告: {missing} 个样本的分数为缺失值")

    # 2. 检查分数范围
    min_score = scored_data[score_col].min()
    max_score = scored_data[score_col].max()
    print(f"分数范围: {min_score:.0f} - {max_score:.0f}")

    # 3. 检查分数分布
    mean_score = scored_data[score_col].mean()
    std_score = scored_data[score_col].std()
    print(f"平均分: {mean_score:.0f}, 标准差: {std_score:.0f}")

    # 4. 检查是否有异常分数
    if std_score > 200:
        print("⚠️  警告: 分数标准差过大，可能存在异常值")

# 使用
validate_scores(train_scored, 'total_score')
```

### 2. 多变量处理

```python
# 当有多个评分变量时，需要计算总分
score_vars = [col for col in train_scored.columns if col != yh.target]

# 方法1: 简单求和
train_scored['total_score'] = train_scored[score_vars].sum(axis=1)

# 方法2: 加权求和（如果有权重）
weights = {var: 1.0 for var in score_vars}
# weights['age'] = 1.2  # 年龄权重更高

weighted_sum = sum(train_scored[var] * weights.get(var, 1.0)
                  for var in score_vars)
train_scored['weighted_score'] = weighted_sum
```

### 3. 评分卡保存

```python
# 保存完整的评分卡配置
import pickle
import json

scorecard_config = {
    'model': model,
    'score_df': score_df,
    'A': A,
    'B': B,
    'base_score': base_score,
    'woe_df': woe_df,
    'bin_df': yh.binning_module.bin_df
}

# 保存为pickle
with open('output/scorecard_config.pkl', 'wb') as f:
    pickle.dump(scorecard_config, f)

# 保存关键参数为JSON
key_params = {
    'A': A,
    'B': B,
    'base_score': base_score,
    'cutoff': 500,
    'variables': list(X.columns)
}

with open('output/scorecard_params.json', 'w') as f:
    json.dump(key_params, f, indent=2)
```

---

## 常见问题

### Q1: 所有客户的分数都一样？

```python
# 原因：可能所有变量都落在同一个箱体

# 解决：检查分箱结果
for var in X.columns:
    print(f"{var}: {yh.data[var].nunique()} 个唯一值")
    # 如果唯一值太少，检查分箱是否合理
```

### Q2: 分数为负数？

```python
# 原因：基础分设置过低或变量得分为负

# 解决：调整评分卡刻度
A, B, base_score = si.cal_scale(
    score=600,    # 提高基准分
    odds=20,
    PDO=20,
    model=model
)
```

### Q3: 如何应用到生产环境？

```python
# 创建评分函数
def calculate_score(customer_data, score_df, base_score):
    """计算单个客户的评分"""

    total_score = base_score

    for var in customer_data.columns:
        if var == yh.target:
            continue

        # 查找该变量对应的箱体得分
        var_score = score_df[score_df['col'] == var]

        for _, row in var_score.iterrows():
            lower = row['min_bin']
            upper = row['max_bin']

            # 判断客户落在哪个箱体
            if lower == upper:
                if customer_data[var].iloc[0] == lower:
                    total_score += row['score']
                    break
            else:
                if lower <= customer_data[var].iloc[0] <= upper:
                    total_score += row['score']
                    break

    return total_score

# 使用API部署
# from fastapi import FastAPI
# app = FastAPI()
#
# @app.post("/score")
# async def score_customer(customer: dict):
#     customer_df = pd.DataFrame([customer])
#     score = calculate_score(customer_df, score_df, base_score)
#     return {"score": score}
```

---

## 相关文档

- [变量分箱模块](03-binning.md) - 评分卡的基础
- [变量选择模块](04-var-select.md) - 选择评分卡变量
- [模型评估模块](05-model-evaluation.md) - 评估评分卡效果
- [评分卡监控模块](07-scorecard-monitor.md) - 监控评分卡表现
