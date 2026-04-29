---
name: risk-modeling
description: 信用评分卡建模全流程，使用 yihuier 工具包完成从数据探索到模型部署。适用于：信用评分卡、反欺诈模型、风险评级模型。产出评分卡（含分箱、WOE、得分）、KS Bucket 表格、模型检测报告。质量标准：AUC≥0.65, KS≥0.15, PSI<0.25。
---

# 信用评分卡建模

使用 `yihuier` 工具包完成标准化建模流程。

## 快速开始

直接使用模板脚本 [scripts/risk_modeling_template.py](scripts/risk_modeling_template.py)。

## 前置要求

```python
pip install yihuier
```

数据要求：
- 目标变量：二分类（0=正常，1=违约）
- 样本量：≥ 1000（推荐 ≥ 5000）
- 正样本占比：≥ 5%

## 核心流程

### 第 1 步：初始化

```python
import pandas as pd
from yihuier import Yihuier

data = pd.read_excel('data.xlsx')
yh = Yihuier(data, target='target_column')
```

### 第 2 步：数据预处理

```python
numeric_vars = yh.get_numeric_variables()
yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type='0')
yh.data = yh.dp_module.target_missing_delete()
```

### 第 3 步：变量分箱

```python
binning_vars = [col for col in numeric_vars if col != yh.target]

# ChiMerge 分箱（推荐）
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=binning_vars,
    max_bin=5,
    method='ChiMerge',
    min_binpct=0.05
)

iv_dict = dict(zip(binning_vars, iv_value))
```

**注意**：`bin_df` 是 DataFrame 列表，每个元素对应一个变量。使用 `woe_result_df` 获取合并的分箱详情。

### 第 4 步：WOE 转换

```python
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()
```

`woe_result_df` 列名：`col`, `index`, `bin`, `min_bin`, `max_bin`, `total`, `totalrate`, `bad`, `badrate`, `woe`, `bin_iv`, `IV`

### 第 5 步：变量选择

```python
# IV 筛选
high_iv_vars = [var for var, iv in iv_dict.items() if iv > 0.05]

# 相关性去重
final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
    col_list=high_iv_vars,
    threshold=0.6
)
```

### 第 6 步：模型训练

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = data_woe[final_vars]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
```

### 第 7 步：模型评估

```python
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
ks = yh.me_module.model_ks(y_test, y_pred)

# 交叉验证
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    X, y, cv=5, scoring='roc_auc'
)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 第 8 步：评分卡

```python
A, B, base_score = yh.si_module.cal_scale(
    score=600, odds=50, PDO=20, model=model
)

scores = base_score + X_test.dot(model.coef_[0]) * B
```

### 第 9 步：PSI 稳定性

```python
# 使用 sm_module.calculate_psi（推荐）
train_scores = base_score + X_train.dot(model.coef_[0]) * B
test_scores = base_score + X_test.dot(model.coef_[0]) * B

psi = yh.sm_module.calculate_psi(train_scores, test_scores)
print(f"PSI: {psi:.4f}")

# 或获取详细表格
psi_detail = yh.sm_module.calculate_psi(train_scores, test_scores, return_detail=True)
```

### 第 10 步：评分卡详情

```python
woe_result_df = yh.binning_module.woe_result_df

scorecard = []
for var in final_vars:
    var_df = woe_result_df[woe_result_df['col'] == var]
    coef = model.coef_[0][final_vars.index(var)]
    
    for _, row in var_df.iterrows():
        scorecard.append({
            '变量名': var,
            '分箱': row['bin'],
            'WOE': row['woe'],
            '得分': round(-coef * row['woe'] * B, 2)
        })
```

## 模块说明

| 模块 | 功能 | 关键方法 |
|------|------|----------|
| `eda_module` | 数据探索 | `auto_eda_simple()` |
| `dp_module` | 数据处理 | `fillna_num_var()`, `target_missing_delete()` |
| `binning_module` | 分箱 | `binning_num()`, `woe_df_concat()`, `woe_transform()` |
| `var_select_module` | 变量选择 | `forward_delete_corr_ivfirst()` |
| `me_module` | 模型评估 | `model_ks()`, `cross_verify()` |
| `si_module` | 评分卡实现 | `cal_scale()`, `rule_verify()` |
| `sm_module` | 评分卡监控 | `calculate_psi()`, `score_psi()` |

## 输出要求

1. **评分卡 Excel**：变量名、分箱、WOE、得分
2. **KS Bucket 表**：分数范围、样本数、坏样本率、KS
3. **模型检测报告**：AUC、KS、PSI、入模变量

## 质量标准

- ✅ AUC ≥ 0.65
- ✅ KS ≥ 0.15
- ✅ PSI < 0.25
- ✅ 变量数 7-15 个
- ✅ 交叉验证标准差 < 0.05