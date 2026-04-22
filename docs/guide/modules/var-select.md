# 变量选择模块 (VarSelectModule)

## 概述

`VarSelectModule` 提供了多种变量选择方法，帮助从大量特征中筛选出最有预测能力的变量子集。

### 主要功能

- **特征重要性筛选** - XGBoost、随机森林
- **相关性分析** - 识别和剔除高度相关的变量
- **统计显著性筛选** - 基于p值和回归系数
- **组合策略** - 考虑IV值或重要性的相关性剔除

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

vs = yh.var_select_module
```

---

## API 参考

### 1. select_xgboost() - XGBoost特征选择

使用XGBoost模型评估特征重要性。

#### 语法

```python
vs.select_xgboost(
    col_list: List[str],
    imp_num: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 候选变量列表 |
| `imp_num` | int | None | 选择的特征数量 |

#### 返回值

返回一个元组 `(xg_fea_imp, xg_fea_imp_rank, xg_select_col)`：

- `xg_fea_imp` - 所有变量的重要性表
- `xg_fea_imp_rank` - 按重要性排序的前N个变量
- `xg_select_col` - 选中的变量名列表

#### 使用示例

```python
# 选择重要性前10的变量
xg_imp, xg_rank, xg_cols = vs.select_xgboost(
    col_list=num_vars,
    imp_num=10
)

# 查看重要性
print("特征重要性:")
print(xg_imp.sort_values('imp', ascending=False).head(10))

# 查看选中的变量
print(f"\n选中的变量: {xg_cols}")

# 可视化
import matplotlib.pyplot as plt
xg_rank.sort_values('imp').plot.barh(x='col', y='imp')
plt.title('XGBoost特征重要性')
plt.show()
```

---

### 2. select_rf() - 随机森林特征选择

使用随机森林模型评估特征重要性。

#### 语法

```python
vs.select_rf(
    col_list: List[str],
    imp_num: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 候选变量列表 |
| `imp_num` | int | None | 选择的特征数量 |

#### 返回值

返回一个元组 `(rf_fea_imp, rf_select_col)`：

- `rf_fea_imp` - 按重要性排序的特征表
- `rf_select_col` - 选中的变量名列表

#### 使用示例

```python
# 选择重要性前15的变量
rf_imp, rf_cols = vs.select_rf(
    col_list=num_vars,
    imp_num=15
)

print("随机森林特征重要性:")
print(rf_imp)
```

---

### 3. plot_corr() - 相关性热力图

绘制变量相关性热力图。

#### 语法

```python
vs.plot_corr(
    col_list: List[str],
    threshold: Optional[float] = None,
    mask_direction: str = 'lt',
    plt_size: Optional[Tuple[int, int]] = None,
    is_annot: bool = False
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 变量列表 |
| `threshold` | float | None | 相关性阈值 |
| `mask_direction` | str | 'lt' | 掩码方向：'lt'(小于) 或 'gt'(大于) |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |
| `is_annot` | bool | False | 是否显示相关系数值 |

#### 使用示例

```python
# 绘制完整相关性矩阵
vs.plot_corr(
    col_list=num_vars[:20],
    plt_size=(12, 10),
    is_annot=True
)

# 只显示高相关性（>0.7）
vs.plot_corr(
    col_list=num_vars[:20],
    threshold=0.7,
    mask_direction='gt',
    plt_size=(12, 10)
)
```

---

### 4. corr_mapping() - 相关性映射表

生成变量间的相关性映射表。

#### 语法

```python
vs.corr_mapping(
    col_list: List[str],
    threshold: Optional[float] = None,
    direction: str = 'lt'
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 变量列表 |
| `threshold` | float | None | 相关性阈值 |
| `direction` | str | 'lt' | 比较方向 |

#### 使用示例

```python
# 找出高相关性变量对（相关系数 > 0.7）
high_corr = vs.corr_mapping(
    col_list=num_vars,
    threshold=0.7,
    direction='gt'
)

print("高相关性变量对:")
print(high_corr)

# 筛选极强相关（>0.9）
very_high_corr = high_corr[high_corr['corr'].abs() > 0.9]
print(f"\n极强相关变量对: {len(very_high_corr)}")
print(very_high_corr)
```

---

### 5. forward_delete_corr() - 相关性剔除

逐个剔除高相关性变量。

#### 语法

```python
vs.forward_delete_corr(
    col_list: List[str],
    threshold: Optional[float] = None
) -> List[str]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 候选变量列表 |
| `threshold` | float | None | 相关性阈值 |

#### 使用示例

```python
# 剔除相关系数 > 0.7 的变量
selected_vars = vs.forward_delete_corr(
    col_list=num_vars,
    threshold=0.7
)

print(f"原始变量数: {len(num_vars)}")
print(f"剔除后变量数: {len(selected_vars)}")
print(f"剔除的变量数: {len(num_vars) - len(selected_vars)}")
```

⚠️ **注意**：此方法会按顺序剔除，可能不是最优策略。

---

### 6. forward_delete_corr_ivfirst() - 考虑IV的相关性剔除

剔除高相关性变量时，优先保留高IV值的变量。

#### 语法

```python
vs.forward_delete_corr_ivfirst(
    col_list: List[str],
    threshold: float = 0.5
) -> List[str]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 候选变量列表 |
| `threshold` | float | 0.5 | 相关性阈值（0-1） |

#### 使用示例

```python
# 先计算IV值
iv_df = yh.binning_module.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='freq'
)

# 剔除相关性，保留高IV值变量
selected_vars = vs.forward_delete_corr_ivfirst(
    col_list=num_vars,
    threshold=0.7
)

print(f"选中的变量: {selected_vars}")
```

---

### 7. forward_delete_corr_impfirst() - 考虑重要性的相关性剔除

剔除高相关性变量时，优先保留高特征重要性的变量。

#### 语法

```python
vs.forward_delete_corr_impfirst(
    col_list: List[str],
    type: str,
    threshold: float = 0.5
) -> List[str]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 候选变量列表 |
| `type` | str | 必填 | 重要性类型：'xgboost' 或 'rf' |
| `threshold` | float | 0.5 | 相关性阈值 |

#### 使用示例

```python
# 使用XGBoost重要性进行相关性剔除
selected_vars = vs.forward_delete_corr_impfirst(
    col_list=num_vars,
    type='xgboost',
    threshold=0.7
)

# 或使用随机森林重要性
selected_vars = vs.forward_delete_corr_impfirst(
    col_list=num_vars,
    type='rf',
    threshold=0.7
)

print(f"选中的变量: {selected_vars}")
```

---

### 8. forward_delete_pvalue() - 显著性筛选

使用向前选择法，基于p值筛选变量。

#### 语法

```python
vs.forward_delete_pvalue(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[List[str], str]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `x_train` | pd.DataFrame | 必填 | 训练集特征（需WOE转换） |
| `y_train` | pd.Series | 必填 | 训练集目标 |

#### 返回值

返回一个元组 `(selected_vars, summary)`：

- `selected_vars` - 选中的变量列表
- `summary` - 模型摘要

#### 使用示例

```python
from sklearn.model_selection import train_test_split

# 准备WOE数据
data_woe = yh.binning_module.woe_transform()
X = data_woe.drop([yh.target], axis=1)
y = data_woe[yh.target]

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 显著性筛选
selected_vars, summary = vs.forward_delete_pvalue(
    x_train=X_train,
    y_train=y_train
)

print(f"选中的变量: {selected_vars}")
print(f"\n模型摘要:\n{summary}")
```

---

### 9. forward_delete_coef() - 系数符号筛选

确保逻辑回归系数符号一致。

#### 语法

```python
vs.forward_delete_coef(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[List[str], pd.DataFrame]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `x_train` | pd.DataFrame | 必填 | 训练集特征（需WOE转换） |
| `y_train` | pd.Series | 必填 | 训练集目标 |

#### 返回值

返回一个元组 `(selected_vars, coef_df)`：

- `selected_vars` - 系数符号一致的变量列表
- `coef_df` - 变量系数表

#### 使用示例

```python
# 系数符号筛选
selected_vars, coef_df = vs.forward_delete_coef(
    x_train=X_train,
    y_train=y_train
)

print(f"选中的变量: {selected_vars}")
print(f"\n系数信息:")
print(coef_df)
```

---

## 完整变量选择流程

### 推荐流程

```python
from yihuier import Yihuier
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 初始化
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 2. 数据预处理
print("=== 数据预处理 ===")
yh.data = yh.dp_module.delete_missing_var(threshold=0.15)
yh.data = yh.dp_module.const_delete(threshold=0.9)

# 3. 分箱
print("\n=== 变量分箱 ===")
num_vars = yh.get_numeric_variables()
_, iv_value = yh.binning_module.binning_num(
    col_list=num_vars,
    max_bin=5,
    method='ChiMerge'
)

# 4. WOE转换
print("\n=== WOE转换 ===")
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()
print(f"WOE数据形状: {data_woe.shape}")

# 5. 准备建模数据
X = data_woe.drop([yh.target], axis=1)
y = data_woe[yh.target]

# 6. 第一步：特征重要性筛选
print("\n=== 特征重要性筛选 ===")
xg_imp, xg_rank, xg_cols = yh.var_select_module.select_xgboost(
    col_list=X.columns.tolist(),
    imp_num=30  # 选择前30个
)
print(f"选中变量: {len(xg_cols)}")

# 7. 第二步：相关性剔除（考虑IV）
print("\n=== 相关性剔除 ===")
selected_vars = yh.var_select_module.forward_delete_corr_ivfirst(
    col_list=xg_cols,
    threshold=0.7
)
print(f"剔除后变量: {len(selected_vars)}")

# 8. 第三步：统计显著性筛选
print("\n=== 统计显著性筛选 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_vars], y,
    test_size=0.3,
    random_state=42
)

sig_vars, summary = yh.var_select_module.forward_delete_pvalue(
    x_train=X_train,
    y_train=y_train
)
print(f"显著性筛选后: {len(sig_vars)}")

# 9. 第四步：系数符号筛选
print("\n=== 系数符号筛选 ===")
final_vars, coef_df = yh.var_select_module.forward_delete_coef(
    x_train=X_train[sig_vars],
    y_train=y_train
)
print(f"最终变量: {len(final_vars)}")
print(f"变量列表: {final_vars}")

# 10. 结果汇总
print("\n=== 变量选择汇总 ===")
print(f"原始变量数: {len(num_vars)}")
print(f"特征重要性筛选: {len(xg_cols)}")
print(f"相关性剔除: {len(selected_vars)}")
print(f"显著性筛选: {len(sig_vars)}")
print(f"系数符号筛选: {len(final_vars)}")
print(f"\n最终变量列表:\n{final_vars}")
```

---

## 变量选择策略

### 策略1: 快速筛选（适合大数据）

```python
# 1. IV值初筛
iv_df = yh.binning_module.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='freq'
)
high_iv_vars = iv_df[iv_df['iv'] > 0.1]['col'].tolist()

# 2. XGBoost筛选
_, _, selected_vars = yh.var_select_module.select_xgboost(
    col_list=high_iv_vars,
    imp_num=20
)

print(f"快速筛选结果: {selected_vars}")
```

### 策略2: 精细筛选（适合建模）

```python
# 1. IV值筛选
iv_df = yh.binning_module.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='ChiMerge'
)
high_iv_vars = iv_df[iv_df['iv'] > 0.1]['col'].tolist()

# 2. XGBoost筛选
_, _, xg_vars = yh.var_select_module.select_xgboost(
    col_list=high_iv_vars,
    imp_num=30
)

# 3. 相关性剔除（考虑XGBoost重要性）
corr_vars = yh.var_select_module.forward_delete_corr_impfirst(
    col_list=xg_vars,
    type='xgboost',
    threshold=0.7
)

# 4. 显著性筛选
sig_vars, _ = yh.var_select_module.forward_delete_pvalue(
    x_train=X_train[corr_vars],
    y_train=y_train
)

# 5. 系数符号筛选
final_vars, _ = yh.var_select_module.forward_delete_coef(
    x_train=X_train[sig_vars],
    y_train=y_train
)

print(f"精细筛选结果: {final_vars}")
```

### 策略3: 业务导向筛选

```python
# 1. 业务专家选择核心变量
core_vars = ['age', 'income', 'debt_ratio', 'employment_years']

# 2. 数据驱动补充变量
iv_df = yh.binning_module.iv_num(
    col_list=num_vars,
    max_bin=5,
    method='ChiMerge'
)
high_iv_vars = iv_df[iv_df['iv'] > 0.1]['col'].tolist()

# 3. 合并
candidate_vars = list(set(core_vars + high_iv_vars))

# 4. 相关性剔除
final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
    col_list=candidate_vars,
    threshold=0.7
)

print(f"业务导向筛选: {final_vars}")
```

---

## 相关性分析

### 相关性阈值选择

```python
# 测试不同阈值
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    selected = yh.var_select_module.forward_delete_corr(
        col_list=num_vars,
        threshold=threshold
    )
    print(f"阈值={threshold}: {len(selected)}个变量")
```

### 相关性矩阵分析

```python
# 计算相关性矩阵
corr_matrix = yh.data[num_vars].corr()

# 找出高相关性变量对
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'corr': corr_val
            })

high_corr_df = pd.DataFrame(high_corr)
print("高相关性变量对:")
print(high_corr_df.sort_values('corr', key=abs, ascending=False))
```

---

## 注意事项

### 1. 数据泄露

⚠️ **重要**：变量选择应该在训练集上进行，避免数据泄露。

```python
# 错误做法
# 在全数据集上进行变量选择
selected_vars = vs.select_xgboost(col_list=all_vars, imp_num=10)
X_train, X_test = train_test_split(X[selected_vars])

# 正确做法
# 先分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 在训练集上进行变量选择
selected_vars = vs.select_xgboost(
    col_list=X_train.columns.tolist(),
    imp_num=10
)

# 应用到测试集
X_train_selected = X_train[selected_vars]
X_test_selected = X_test[selected_vars]
```

### 2. 变量数量

经验法则：
- 样本数 < 1000: 选择 5-10 个变量
- 样本数 1000-10000: 选择 10-20 个变量
- 样本数 > 10000: 选择 20-30 个变量

```python
n_samples = len(yh.data)
if n_samples < 1000:
    max_vars = 10
elif n_samples < 10000:
    max_vars = 20
else:
    max_vars = 30

_, _, selected_vars = vs.select_xgboost(
    col_list=num_vars,
    imp_num=max_vars
)
```

### 3. WOE转换要求

统计显著性筛选和系数符号筛选需要WOE转换后的数据。

```python
# 确保已进行WOE转换
if yh.binning_module.woe_result_df is None:
    # 先分箱
    yh.binning_module.binning_num(num_vars, max_bin=5)
    # 再转换
    yh.binning_module.woe_df_concat()
    data_woe = yh.binning_module.woe_transform()
```

---

## 常见问题

### Q1: 如何选择合适的特征数量？

```python
# 方法1: 肘部法则
iv_list = []
for n in range(5, 30, 5):
    _, iv_value = yh.binning_module.binning_num(
        col_list=num_vars[:n],
        max_bin=5
    )
    iv_list.append(sum(iv_value))

import matplotlib.pyplot as plt
plt.plot(range(5, 30, 5), iv_list)
plt.xlabel('特征数量')
plt.ylabel('总IV值')
plt.show()

# 方法2: 交叉验证
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

for n in [10, 15, 20, 25]:
    _, _, vars_n = vs.select_xgboost(num_vars, imp_num=n)
    X_selected = X[vars_n]

    scores = cross_val_score(
        LogisticRegression(),
        X_selected, y,
        cv=5,
        scoring='roc_auc'
    )
    print(f"特征数={n}, AUC={scores.mean():.3f}")
```

### Q2: 特征重要性为0？

```python
# 检查数据
xg_imp, _, _ = vs.select_xgboost(num_vars, imp_num=10)

# 查看重要性为0的变量
zero_imp = xg_imp[xg_imp['imp'] == 0]['col'].tolist()
print(f"重要性为0的变量: {zero_imp}")

# 可能原因：
# 1. 变量是常数
# 2. 变量与目标变量无关
# 3. 变量有太多缺失值
```

### Q3: 相关性剔除后变量太少？

```python
# 降低相关性阈值
selected = vs.forward_delete_corr_ivfirst(
    col_list=num_vars,
    threshold=0.8  # 从0.7提高到0.8
)

# 或使用相关性剔除的简单版本
selected = vs.forward_delete_corr(
    col_list=num_vars,
    threshold=0.8
)
```

---

## 相关文档

- [分箱模块](binning.md) - 变量选择前的WOE转换
- [模型评估模块](model-evaluation.md) - 评估变量选择效果
- [评分卡实现模块](scorecard-implement.md) - 使用选定变量建模
