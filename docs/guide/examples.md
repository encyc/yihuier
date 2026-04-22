# 示例集合

本文档提供了 Yihuier 的各种使用示例，从基础用法到高级场景。

## 示例数据说明

所有示例使用相同的测试数据集：

```python
import pandas as pd

# 加载数据
data = pd.read_csv('data.csv')

# 数据信息
print(f"样本数: {data.shape[0]}")
print(f"特征数: {data.shape[1]}")
print(f"目标变量: dlq_flag")
print(f"正样本占比: {data['dlq_flag'].mean():.2%}")
```

## 基础示例

### 示例 1：快速建模流程

```python
from yihuier import Yihuier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 初始化
yh = Yihuier(data, target='dlq_flag')

# 数据预处理
numeric_vars = yh.get_numeric_variables()
yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type='0')

# 变量分箱
binning_vars = numeric_vars[:5]
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=binning_vars,
    max_bin=5,
    method='freq'
)

# WOE 转换
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 变量选择
feature_cols = [col for col in data_woe.columns if col != yh.target]
_, _, xg_cols = yh.var_select_module.select_xgboost(
    col_list=feature_cols,
    imp_num=5
)

# 模型训练
X = data_woe[xg_cols]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}")
```

### 示例 2：探索性数据分析

```python
# 自动 EDA 报告
eda_stats = yh.eda_module.auto_eda_simple()

# 数值型变量分布
yh.eda_module.plot_num_col(
    col_list=['v1', 'v2', 'v3'],
    plot_type='hist',  # 'hist' 或 'box'
    plt_size=(12, 4)
)

# 类别型变量分布
yh.eda_module.plot_cate_var(
    col_list=['category1', 'category2'],
    plt_size=(12, 4)
)

# 违约率分析
yh.eda_module.plot_default_num(
    col_list=['v1', 'v2'],
    plt_size=(12, 4)
)
```

## 高级示例

### 示例 3：多种分箱方法对比

```python
import pandas as pd

methods = ['ChiMerge', 'freq', 'count', 'monotonic']
results = []

for method in methods:
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=['v1', 'v2', 'v3'],
        max_bin=5,
        method=method
    )
    avg_iv = sum(iv_value) / len(iv_value)
    results.append({'method': method, 'avg_iv': avg_iv})

results_df = pd.DataFrame(results)
print(results_df)
```

### 示例 4：变量选择策略

```python
# 准备数据
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()
feature_cols = [col for col in data_woe.columns if col != yh.target]

# 策略 1：XGBoost 重要性
xg_imp, _, xg_cols = yh.var_select_module.select_xgboost(
    col_list=feature_cols,
    imp_num=10
)

# 策略 2：随机森林重要性
rf_imp, rf_cols = yh.var_select_module.select_rf(
    col_list=feature_cols,
    imp_num=10
)

# 策略 3：IV 筛选 + 相关性去重
# 先按 IV 选择
iv_vars = [var for var, iv in zip(feature_cols, yh.binning_module.iv_value) if iv > 0.1]

# 再考虑相关性去重（IV 优先）
final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
    col_list=iv_vars,
    threshold=0.6
)

print(f"XGBoost 选中: {len(xg_cols)} 个变量")
print(f"随机森林选中: {len(rf_cols)} 个变量")
print(f"IV+相关性选中: {len(final_vars)} 个变量")
```

### 示例 5：完整的评分卡建模

```python
from yihuier import Yihuier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 数据加载和探索
yh = Yihuier(data, target='dlq_flag')
yh.eda_module.auto_eda_simple()

# 2. 数据预处理
numeric_vars = yh.get_numeric_variables()
categorical_vars = yh.get_categorical_variables()

yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type='0')
yh.data = yh.dp_module.fillna_cate_var(categorical_vars, fill_type='mode')
yh.data = yh.dp_module.target_missing_delete()

# 3. 变量分箱
bin_df_num, iv_value_num = yh.binning_module.binning_num(
    col_list=numeric_vars[:10],
    max_bin=5,
    method='freq'
)

# 4. WOE 转换
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 5. 变量选择
feature_cols = [col for col in data_woe.columns if col != yh.target]
xg_imp, _, xg_cols = yh.var_select_module.select_xgboost(
    col_list=feature_cols,
    imp_num=10
)

# 6. 模型训练
X = data_woe[xg_cols]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. 模型评估
y_pred = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)
ks = yh.me_module.model_ks(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"KS: {ks:.4f}")

yh.me_module.plot_roc(y_test, y_pred)
yh.me_module.plot_model_ks(y_test, y_pred)

# 8. 评分卡实现
A, B, base_score = yh.si_module.cal_scale(
    score=600,
    odds=50,
    PDO=20,
    model=model
)

# 计算测试集分数
test_scores = pd.DataFrame({
    'score': base_score + X_test.dot(model.coef_[0]) * B,
    'dlq_flag': y_test.values
})

print(f"分数范围: {test_scores['score'].min():.2f} - {test_scores['score'].max():.2f}")

# 9. Cutoff 验证
matrix = yh.si_module.rule_verify(
    df=test_scores,
    col_score='score',
    target='dlq_flag',
    cutoff=600
)

print("混淆矩阵:")
print(matrix)

# 10. 模型监控模拟
psi_df = yh.sm_module.score_psi(
    df1=X_train.join(y_train),
    df2=X_test.join(y_test),
    id_col=None,
    score_col=None,
    x=base_score - 100,
    y=base_score + 100,
    step=20
)

print(f"训练集 vs 测试集 PSI: {psi_df['PSI'].sum():.4f}")
```

## 特殊场景示例

### 示例 6：处理类别不平衡

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# 计算类别权重
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = dict(enumerate(class_weights))

# 使用类别权重训练
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # 或使用 weight_dict
)
model.fit(X_train, y_train)
```

### 示例 7：时间序列验证

```python
# 假设数据有时间列
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# 按时间分割
train = data[data['date'] < '2024-01-01']
test = data[data['date'] >= '2024-01-01']

# 分别建模
yh_train = Yihuier(train, target='dlq_flag')
# ... 完整建模流程 ...

# 在测试集上验证
# ... 评估模型性能 ...
```

### 示例 8：特征工程集成

```python
# 派生新特征
data['debt_to_income'] = data['total_debt'] / data['income']
data['payment_to_income'] = data['monthly_payment'] / data['income']
data['credit_utilization'] = data['credit_balance'] / data['credit_limit']

# 处理无穷值
data = data.replace([np.inf, -np.inf], np.nan)

# 使用新特征建模
yh = Yihuier(data, target='dlq_flag')
# ... 正常建模流程 ...
```

### 示例 9：多模型比较

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(n_estimators=100)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    ks = yh.me_module.model_ks(y_test, y_pred)
    results.append({'model': name, 'auc': auc, 'ks': ks})

results_df = pd.DataFrame(results)
print(results_df)
```

### 示例 10：自动化建模流程

```python
def auto_scorecard_modeling(
    data,
    target='dlq_flag',
    max_bins=5,
    binning_method='freq',
    num_features=10,
    test_size=0.3
):
    """
    自动化评分卡建模流程
    """
    # 初始化
    yh = Yihuier(data, target=target)

    # 数据预处理
    numeric_vars = yh.get_numeric_variables()
    yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type='0')

    # 变量分箱
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=numeric_vars,
        max_bin=max_bins,
        method=binning_method
    )

    # WOE 转换
    yh.binning_module.woe_df_concat()
    data_woe = yh.binning_module.woe_transform()

    # 变量选择
    feature_cols = [col for col in data_woe.columns if col != target]
    _, _, selected_features = yh.var_select_module.select_xgboost(
        col_list=feature_cols,
        imp_num=num_features
    )

    # 模型训练
    X = data_woe[selected_features]
    y = data_woe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    ks = yh.me_module.model_ks(y_test, y_pred)

    return {
        'yh': yh,
        'model': model,
        'features': selected_features,
        'auc': auc,
        'ks': ks,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# 使用自动化流程
result = auto_scorecard_modeling(
    data=data,
    target='dlq_flag',
    max_bins=5,
    binning_method='freq',
    num_features=10
)

print(f"AUC: {result['auc']:.4f}")
print(f"KS: {result['ks']:.4f}")
print(f"Selected Features: {result['features']}")
```

## 运行示例

所有示例代码都可以直接运行。确保已安装 Yihuier：

```bash
pip install yihuier
```

然后运行任意示例：

```bash
python examples/basic_usage.py
python examples/advanced_pipeline.py
```

## 参考资源

- [快速开始](quick-start.md) - 快速上手指南
- [API 文档](api.md) - 完整 API 参考
- [最佳实践](best-practices.md) - 行业最佳实践
