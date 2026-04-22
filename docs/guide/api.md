# API 文档

Yihuier 提供简洁易用的 Python API，涵盖评分卡建模的完整流程。

## 核心类

### Yihuier

主类，用于初始化评分卡建模项目并管理所有功能模块。

#### 初始化

```python
from yihuier import Yihuier

yh = Yihuier(data, target='dlq_flag')
```

**参数**:
- `data` (pd.DataFrame): 输入数据集
- `target` (str): 目标变量列名

**属性**:
- `data` (pd.DataFrame): 数据集
- `target` (str): 目标变量名
- `eda_module` (EDAModule): 数据探索模块
- `dp_module` (DataProcessingModule): 数据预处理模块
- `cluster_module` (ClusterModule): 聚类分析模块
- `binning_module` (BinningModule): 分箱模块
- `var_select_module` (VarSelectModule): 变量选择模块
- `me_module` (ModelEvaluationModule): 模型评估模块
- `si_module` (ScorecardImplementModule): 评分卡实现模块
- `pipeline_module` (PipelineModule): 流水线模块

#### 方法

##### get_numeric_variables()

获取数值型变量列表。

```python
numeric_vars = yh.get_numeric_variables()
# ['v1', 'v2', 'v3', ...]
```

**返回**: `list[str]` - 数值型变量列名列表

##### get_categorical_variables()

获取类别型变量列表。

```python
categorical_vars = yh.get_categorical_variables()
# ['category1', 'category2', ...]
```

**返回**: `list[str]` - 类别型变量列名列表

##### get_date_variables()

获取日期型变量列表。

```python
date_vars = yh.get_date_variables()
# ['date1', 'date2', ...]
```

**返回**: `list[str]` - 日期型变量列名列表

---

## 数据预处理模块

### DataProcessingModule

数据预处理模块，提供缺失值处理、常变量删除等功能。

#### fillna_num_var()

填充数值型变量的缺失值。

```python
yh.data = yh.dp_module.fillna_num_var(
    col_list=['v1', 'v2', 'v3'],
    fill_type='0'
)
```

**参数**:
- `col_list` (list[str]): 变量列表
- `fill_type` (str): 填充类型
  - `'0'`: 填充 0
  - `'mean'`: 填充均值
  - `'median'`: 填充中位数
  - `'class'`: 填充特殊类别（-999）

**返回**: `pd.DataFrame` - 填充后的数据集

#### fillna_cate_var()

填充类别型变量的缺失值。

```python
yh.data = yh.dp_module.fillna_cate_var(
    col_list=['category1', 'category2'],
    fill_type='mode'
)
```

**参数**:
- `col_list` (list[str]): 变量列表
- `fill_type` (str): 填充类型
  - `'mode'`: 填充众数
  - `'class'`: 填充特殊类别（'Unknown'）

**返回**: `pd.DataFrame` - 填充后的数据集

#### delete_missing_var()

删除高缺失率变量。

```python
yh.data = yh.dp_module.delete_missing_var(threshold=0.2)
```

**参数**:
- `threshold` (float): 缺失率阈值（0-1）

**返回**: `pd.DataFrame` - 删除后的数据集

#### const_delete()

删除常变量/同值化严重的变量。

```python
yh.data = yh.dp_module.const_delete(threshold=0.9)
```

**参数**:
- `threshold` (float): 同值化阈值（0-1）

**返回**: `pd.DataFrame` - 删除后的数据集

#### target_missing_delete()

删除目标变量缺失的样本。

```python
yh.data = yh.dp_module.target_missing_delete()
```

**返回**: `pd.DataFrame` - 删除后的数据集

---

## 分箱模块

### BinningModule

变量分箱模块，支持多种分箱方法和 WOE 转换。

#### binning_num()

数值型变量分箱。

```python
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=['v1', 'v2', 'v3'],
    max_bin=5,
    method='ChiMerge',
    min_binpct=0.05
)
```

**参数**:
- `col_list` (list[str]): 变量列表
- `max_bin` (int): 最大分箱数（默认 10）
- `min_binpct` (float): 最小分箱占比（默认 0）
- `method` (str): 分箱方法
  - `'ChiMerge'`: 卡方分箱（默认）
  - `'freq'`: 等频分箱
  - `'count'`: 等距分箱
  - `'monotonic'`: 单调性分箱

**返回**:
- `bin_df` (list): 分箱结果列表
- `iv_value` (list): IV 值列表

#### binning_cate()

类别型变量分箱。

```python
bin_df, iv_value, ks_value = yh.binning_module.binning_cate(
    col_list=['category1', 'category2']
)
```

**参数**:
- `col_list` (list[str]): 变量列表

**返回**:
- `bin_df` (list): 分箱结果列表
- `iv_value` (list): IV 值列表
- `ks_value` (list): KS 值列表

#### woe_df_concat()

拼接所有 WOE 结果表。

```python
woe_df = yh.binning_module.woe_df_concat()
```

**返回**: `pd.DataFrame` - WOE 结果表

#### woe_transform()

WOE 转换。

```python
data_woe = yh.binning_module.woe_transform()
```

**返回**: `pd.DataFrame` - WOE 转换后的数据集

---

## 变量选择模块

### VarSelectModule

变量选择模块，提供多种变量选择策略。

#### select_xgboost()

基于 XGBoost 特征重要性选择变量。

```python
xg_imp, xg_rank, xg_cols = yh.var_select_module.select_xgboost(
    col_list=['v1', 'v2', 'v3', 'v4', 'v5'],
    imp_num=3
)
```

**参数**:
- `col_list` (list[str]): 候选变量列表
- `imp_num` (int): 选择的变量数量

**返回**:
- `xg_imp` (pd.DataFrame): 所有变量的特征重要性
- `xg_rank` (pd.DataFrame): 选中变量的特征重要性
- `xg_cols` (list[str]): 选中的变量列表

#### select_rf()

基于随机森林特征重要性选择变量。

```python
rf_imp, rf_cols = yh.var_select_module.select_rf(
    col_list=['v1', 'v2', 'v3', 'v4', 'v5'],
    imp_num=3
)
```

**参数**:
- `col_list` (list[str]): 候选变量列表
- `imp_num` (int): 选择的变量数量

**返回**:
- `rf_imp` (pd.DataFrame): 所有变量的特征重要性
- `rf_cols` (list[str]): 选中的变量列表

#### forward_delete_corr_ivfirst()

考虑 IV 值的相关变量删除。

```python
final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
    col_list=['v1', 'v2', 'v3', 'v4'],
    threshold=0.6
)
```

**参数**:
- `col_list` (list[str]): 候选变量列表
- `threshold` (float): 相关系数阈值

**返回**: `list[str]` - 筛选后的变量列表

#### forward_delete_corr_impfirst()

考虑特征重要性的相关变量删除。

```python
final_vars = yh.var_select_module.forward_delete_corr_impfirst(
    col_list=['v1', 'v2', 'v3', 'v4'],
    type='xgboost',
    threshold=0.6
)
```

**参数**:
- `col_list` (list[str]): 候选变量列表
- `type` (str): 特征重要性类型（'xgboost' 或 'rf'）
- `threshold` (float): 相关系数阈值

**返回**: `list[str]` - 筛选后的变量列表

#### plot_corr()

绘制相关性热力图。

```python
yh.var_select_module.plot_corr(
    col_list=['v1', 'v2', 'v3'],
    threshold=0.6,
    plt_size=(10, 8)
)
```

---

## 模型评估模块

### ModelEvaluationModule

模型评估模块，提供全面的模型性能评估。

#### plot_roc()

绘制 ROC 曲线。

```python
yh.me_module.plot_roc(y_test, y_pred)
```

**参数**:
- `y_label` (np.ndarray): 真实标签
- `y_pred` (np.ndarray): 预测概率

#### plot_model_ks()

绘制 KS 曲线。

```python
yh.me_module.plot_model_ks(y_test, y_pred)
```

**参数**:
- `y_label` (np.ndarray): 真实标签
- `y_pred` (np.ndarray): 预测概率

#### model_ks()

计算 KS 值。

```python
ks_value = yh.me_module.model_ks(y_test, y_pred)
```

**参数**:
- `y_label` (np.ndarray): 真实标签
- `y_pred` (np.ndarray): 预测概率

**返回**: `float` - KS 值

#### cross_verify()

交叉验证。

```python
yh.me_module.cross_verify(
    X_train, y_train,
    estimator=LogisticRegression(),
    fold=5,
    scoring='roc_auc'
)
```

**参数**:
- `x` (np.ndarray): 特征矩阵
- `y` (np.ndarray): 目标变量
- `estimator`: 评估器模型
- `fold` (int): 折数
- `scoring` (str): 评分指标

---

## 评分卡实现模块

### ScorecardImplementModule

评分卡实现模块，提供评分卡刻度计算和分数转换。

#### cal_scale()

计算评分卡刻度参数。

```python
A, B, base_score = yh.si_module.cal_scale(
    score=600,
    odds=50,
    PDO=20,
    model=lr_model
)
```

**参数**:
- `score` (float): 指定 odds 时的分数
- `odds` (float): 好坏比
- `PDO` (float): odds 翻倍时分数的减少量
- `model`: 训练好的逻辑回归模型

**返回**:
- `A` (float): 参数 A
- `B` (float): 参数 B
- `base_score` (float): 基础分

#### score_df_concat()

生成变量得分表。

```python
score_df = yh.si_module.score_df_concat(
    woe_df=woe_df,
    model=lr_model,
    B=B
)
```

**参数**:
- `woe_df` (pd.DataFrame): WOE 结果表
- `model`: 逻辑回归模型
- `B` (float): 参数 B

**返回**: `pd.DataFrame` - 变量得分表

#### rule_verify()

验证 cutoff 有效性。

```python
matrix = yh.si_module.rule_verify(
    df=test_scores,
    col_score='score',
    target='dlq_flag',
    cutoff=650
)
```

**参数**:
- `df` (pd.DataFrame): 包含分数和目标变量的数据
- `col_score` (str): 分数列名
- `target` (str): 目标变量列名
- `cutoff` (float): cutoff 阈值

**返回**: `pd.DataFrame` - 混淆矩阵

---

## 评分卡监控模块

### ScorecardMonitorModule

评分卡监控模块，提供模型稳定性监控。

#### score_psi()

计算 PSI（Population Stability Index）。

```python
psi_df = yh.sm_module.score_psi(
    df1=model_data,
    df2=online_data,
    id_col='customer_id',
    score_col='score',
    x=200,
    y=800,
    step=20
)
```

**参数**:
- `df1` (pd.DataFrame): 建模样本数据
- `df2` (pd.DataFrame): 上线样本数据
- `id_col` (str): ID 列名
- `score_col` (str): 分数列名
- `x` (float): 分数区间左端点
- `y` (float): 分数区间右端点
- `step` (float): 分数区间步长

**返回**: `pd.DataFrame` - PSI 计算结果

---

## 完整示例

```python
import pandas as pd
from yihuier import Yihuier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 加载数据
data = pd.read_csv('data.csv')

# 2. 初始化
yh = Yihuier(data, target='dlq_flag')

# 3. 数据预处理
numeric_vars = yh.get_numeric_variables()
yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type='0')
yh.data = yh.dp_module.target_missing_delete()

# 4. 变量分箱
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=numeric_vars[:10],
    max_bin=5,
    method='ChiMerge'
)

# 5. WOE 转换
data_woe = yh.binning_module.woe_transform()

# 6. 变量选择
feature_cols = [col for col in data_woe.columns if col != yh.target]
xg_imp, _, xg_cols = yh.var_select_module.select_xgboost(
    col_list=feature_cols,
    imp_num=10
)

# 7. 模型训练
X = data_woe[xg_cols]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. 模型评估
y_pred = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)
ks = yh.me_module.model_ks(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"KS: {ks:.4f}")

yh.me_module.plot_roc(y_test, y_pred)
yh.me_module.plot_model_ks(y_test, y_pred)

# 9. 评分卡实现
A, B, base_score = yh.si_module.cal_scale(
    score=600, odds=50, PDO=20, model=model
)

test_scores = base_score + X_test.dot(model.coef_[0]) * B

print(f"基础分: {base_score:.2f}")
print(f"分数范围: {test_scores.min():.2f} - {test_scores.max():.2f}")
```

---

更多详情请参考：
- [模块索引](/guide/modules/index)
- [示例集合](/guide/examples)
- [最佳实践](/guide/best-practices)
