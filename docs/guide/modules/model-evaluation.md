# 模型评估模块 (ModelEvaluationModule)

## 概述

`ModelEvaluationModule` 提供了全面的模型评估功能，包括ROC曲线、KS曲线、学习曲线、交叉验证等。

### 主要功能

- **ROC曲线分析** - 评估模型区分能力
- **KS分析** - 评估模型最优切分点
- **学习曲线** - 诊断模型拟合情况
- **交叉验证** - 评估模型稳定性
- **混淆矩阵** - 详细的分类性能分析

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 准备数据
data_woe = yh.binning_module.woe_transform()
X = data_woe.drop([yh.target], axis=1)
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测概率
y_pred = model.predict_proba(X_test)[:, 1]

# 初始化评估模块
me = yh.me_module
```

---

## API 参考

### 1. plot_roc() - ROC曲线

绘制ROC曲线并计算AUC值。

#### 语法

```python
me.plot_roc(
    y_label: np.ndarray,
    y_pred: np.ndarray
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `y_label` | np.ndarray | 必填 | 真实标签（0/1） |
| `y_pred` | np.ndarray | 必填 | 预测概率 |

#### 使用示例

```python
# 绘制ROC曲线
me.plot_roc(y_test, y_pred)

# 手动计算AUC
from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.3f}")
```

#### 输出说明

- ROC曲线显示不同阈值下的TPR和FPR
- AUC值越接近1，模型区分能力越强
- 对角线表示随机分类器（AUC=0.5）

---

### 2. plot_model_ks() - KS曲线

绘制KS曲线，展示好坏样本的累积分布差异。

#### 语法

```python
me.plot_model_ks(
    y_label: np.ndarray,
    y_pred: np.ndarray
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `y_label` | np.ndarray | 必填 | 真实标签（0/1） |
| `y_pred` | np.ndarray | 必填 | 预测概率 |

#### 使用示例

```python
# 绘制KS曲线
me.plot_model_ks(y_test, y_pred)

# 计算KS值
ks_value = me.model_ks(y_test, y_pred)
print(f"KS值: {ks_value:.3f}")
```

#### 输出说明

- 绿色曲线：好样本累积分布
- 红色曲线：坏样本累积分布
- 蓝色曲线：KS值（两条曲线的垂直距离）
- KS值越大，模型区分能力越强
- 一般要求KS > 0.2

---

### 3. model_ks() - 计算KS值

计算模型的最大KS值。

#### 语法

```python
me.model_ks(
    y_label: np.ndarray,
    y_pred: np.ndarray
) -> float
```

#### 返回值

返回最大KS值（0-1之间）。

#### 使用示例

```python
# 计算KS值
ks_value = me.model_ks(y_test, y_pred)

# 评估KS值
if ks_value < 0.1:
    print("模型区分能力较差")
elif ks_value < 0.2:
    print("模型区分能力一般")
elif ks_value < 0.3:
    print("模型区分能力良好")
else:
    print("模型区分能力优秀")
```

---

### 4. plot_learning_curve() - 学习曲线

绘制学习曲线，诊断模型是否过拟合或欠拟合。

#### 语法

```python
me.plot_learning_curve(
    estimator: BaseEstimator,
    x: np.ndarray,
    y: np.ndarray,
    cv: Optional[int] = None,
    train_size: np.ndarray = np.linspace(0.1, 1.0, 5),
    plt_size: Optional[Tuple[int, int]] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator` | BaseEstimator | 必填 | 模型对象 |
| `x` | np.ndarray | 必填 | 特征数据 |
| `y` | np.ndarray | 必填 | 目标变量 |
| `cv` | int | None | 交叉验证折数 |
| `train_size` | np.ndarray | linspace(0.1, 1.0, 5) | 训练集比例序列 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |

#### 使用示例

```python
from sklearn.linear_model import LogisticRegression

# 绘制学习曲线
me.plot_learning_curve(
    estimator=LogisticRegression(),
    x=X_train,
    y=y_train,
    cv=5,
    train_size=np.linspace(0.1, 1.0, 10),
    plt_size=(10, 6)
)
```

#### 输出说明

- 红色曲线：训练集得分
- 绿色曲线：交叉验证得分
- 阴影区域：标准差范围

**诊断：**
- 训练得分和验证得分都低 → 欠拟合
- 训练得分高，验证得分低 → 过拟合
- 两者都高且接近 → 拟合良好

---

### 5. cross_verify() - 交叉验证

使用交叉验证评估模型稳定性。

#### 语法

```python
me.cross_verify(
    x: np.ndarray,
    y: np.ndarray,
    estimators: BaseEstimator,
    fold: int,
    scoring: str = 'roc_auc'
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `x` | np.ndarray | 必填 | 特征数据 |
| `y` | np.ndarray | 必填 | 目标变量 |
| `estimators` | BaseEstimator | 必填 | 模型对象 |
| `fold` | int | 必填 | 交叉验证折数 |
| `scoring` | str | 'roc_auc' | 评估指标 |

#### 使用示例

```python
from sklearn.linear_model import LogisticRegression

# 5折交叉验证
me.cross_verify(
    x=X_train,
    y=y_train,
    estimators=LogisticRegression(),
    fold=5,
    scoring='roc_auc'
)
```

#### 输出说明

- 控制台输出：最大、最小、平均AUC值
- 箱线图：展示交叉验证结果的分布

---

### 6. plot_matrix_report() - 混淆矩阵和分类报告

绘制混淆矩阵并打印分类报告。

#### 语法

```python
me.plot_matrix_report(
    y_label: np.ndarray,
    y_pred: np.ndarray
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `y_label` | np.ndarray | 必填 | 真实标签 |
| `y_pred` | np.ndarray | 必填 | 预测标签（0/1） |

#### 使用示例

```python
# 将概率转换为类别
y_pred_class = (y_pred > 0.5).astype(int)

# 绘制混淆矩阵
me.plot_matrix_report(y_test, y_pred_class)
```

#### 输出说明

- 混淆矩阵热力图
- 分类报告（精确率、召回率、F1值）

---

## 完整评估流程

### 标准评估流程

```python
from yihuier import Yihuier
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# 1. 准备数据
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 数据预处理
yh.data = yh.dp_module.delete_missing_var(threshold=0.15)

# 分箱和WOE转换
yh.binning_module.binning_num(
    col_list=yh.get_numeric_variables(),
    max_bin=5
)
yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 2. 分割数据
X = data_woe.drop([yh.target], axis=1)
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 训练模型
print("=== 训练模型 ===")
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred > 0.5).astype(int)

# 5. ROC曲线分析
print("\n=== ROC曲线分析 ===")
yh.me_module.plot_roc(y_test, y_pred)

from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.3f}")

# 6. KS分析
print("\n=== KS分析 ===")
yh.me_module.plot_model_ks(y_test, y_pred)

ks_value = yh.me_module.model_ks(y_test, y_pred)
print(f"KS值: {ks_value:.3f}")

# 7. 交叉验证
print("\n=== 交叉验证 ===")
yh.me_module.cross_verify(
    x=X_train,
    y=y_train,
    estimators=LogisticRegression(),
    fold=5,
    scoring='roc_auc'
)

# 8. 学习曲线
print("\n=== 学习曲线 ===")
yh.me_module.plot_learning_curve(
    estimator=LogisticRegression(),
    x=X_train,
    y=y_train,
    cv=5,
    plt_size=(10, 6)
)

# 9. 混淆矩阵
print("\n=== 混淆矩阵和分类报告 ===")
yh.me_module.plot_matrix_report(y_test, y_pred_class)

# 10. 汇总报告
print("\n=== 模型评估汇总 ===")
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"AUC: {auc:.3f}")
print(f"KS: {ks_value:.3f}")

# 交叉验证AUC
cv_scores = cross_val_score(
    LogisticRegression(),
    X_train, y_train,
    cv=5,
    scoring='roc_auc'
)
print(f"交叉验证AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 分类指标
from sklearn.metrics import classification_report
print("\n分类报告:")
print(classification_report(y_test, y_pred_class))
```

---

## 模型对比

### 对比多个模型

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 定义模型
models = {
    '逻辑回归': LogisticRegression(),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# 评估每个模型
results = []
for name, model in models.items():
    print(f"\n=== 评估 {name} ===")

    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict_proba(X_test)[:, 1]

    # 评估
    auc = yh.me_module.plot_roc(y_test, y_pred)
    ks = yh.me_module.model_ks(y_test, y_pred)

    results.append({
        'model': name,
        'auc': auc,
        'ks': ks
    })

# 汇总结果
results_df = pd.DataFrame(results)
print("\n=== 模型对比 ===")
print(results_df.sort_values('auc', ascending=False))
```

---

## 阈值优化

### 选择最优阈值

```python
from sklearn.metrics import roc_curve

# 计算不同阈值下的指标
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# 计算KS值
ks_values = tpr - fpr
best_idx = np.argmax(ks_values)
best_threshold = thresholds[best_idx]

print(f"最优阈值: {best_threshold:.3f}")
print(f"对应KS值: {ks_values[best_idx]:.3f}")

# 应用最优阈值
y_pred_optimal = (y_pred >= best_threshold).astype(int)

# 评估
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_optimal))
```

### 业务导向的阈值选择

```python
# 根据业务要求选择阈值
# 例如：要求查全率（召回率）达到80%

from sklearn.metrics import recall_score

target_recall = 0.8
best_threshold = None

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_pred >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_thresh)

    if recall >= target_recall:
        best_threshold = threshold
        break

if best_threshold:
    print(f"满足{target_recall:.0%}查全率的最优阈值: {best_threshold:.2f}")

    y_pred_final = (y_pred >= best_threshold).astype(int)

    # 计算其他指标
    from sklearn.metrics import precision_score, f1_score
    precision = precision_score(y_test, y_pred_final)
    f1 = f1_score(y_test, y_pred_final)

    print(f"精确率: {precision:.3f}")
    print(f"F1值: {f1:.3f}")
```

---

## 注意事项

### 1. 数据泄露

⚠️ **重要**：确保评估使用的是测试集，不是训练集。

```python
# 错误做法
y_pred_train = model.predict_proba(X_train)[:, 1]
me.plot_roc(y_train, y_pred_train)  # 评估训练集

# 正确做法
y_pred_test = model.predict_proba(X_test)[:, 1]
me.plot_roc(y_test, y_pred_test)  # 评估测试集
```

### 2. 概率 vs 类别

```python
# ROC曲线和KS分析使用概率
y_pred_proba = model.predict_proba(X_test)[:, 1]
me.plot_roc(y_test, y_pred_proba)
me.plot_model_ks(y_test, y_pred_proba)

# 混淆矩阵使用类别
y_pred_class = model.predict(X_test)
me.plot_matrix_report(y_test, y_pred_class)
```

### 3. 样本不平衡

```python
# 检查样本比例
print(yh.data[yh.target].value_counts(normalize=True))

# 如果样本不平衡（如坏样本 < 10%），使用分层抽样
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X_train, y_train,
    cv=skf,
    scoring='roc_auc'
)
```

---

## 常见问题

### Q1: AUC值高但KS值低？

```python
# 可能原因：模型在某些区间区分能力强，但整体最优切分点不明显

# 解决方案：
# 1. 查看完整的ROC和KS曲线
me.plot_roc(y_test, y_pred)
me.plot_model_ks(y_test, y_pred)

# 2. 分析不同阈值下的性能
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

for i, thresh in enumerate(thresholds[::10]):  # 每10个点取样
    y_pred_temp = (y_pred >= thresh).astype(int)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred_temp)
    print(f"阈值={thresh:.2f}, 准确率={acc:.3f}")
```

### Q2: 学习曲线显示过拟合怎么办？

```python
# 方案1: 增加训练数据
# 方案2: 减少特征数量
# 方案3: 使用正则化

from sklearn.linear_model import LogisticRegression

# L1正则化（Lasso）
model_l1 = LogisticRegression(penalty='l1', solver='saga')
model_l1.fit(X_train, y_train)

# L2正则化（Ridge）
model_l2 = LogisticRegression(penalty='l2', C=0.1)  # C越小，正则化越强
model_l2.fit(X_train, y_train)

# 对比
print(f"L1正则化AUC: {me.model_ks(y_test, model_l1.predict_proba(X_test)[:, 1]):.3f}")
print(f"L2正则化AUC: {me.model_ks(y_test, model_l2.predict_proba(X_test)[:, 1]):.3f}")
```

### Q3: 交叉验证结果波动大？

```python
# 原因：模型不稳定或数据分布不均

# 解决方案：
# 1. 增加交叉验证折数
cv_scores = cross_val_score(model, X, y, cv=10)

# 2. 使用分层K折
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)

# 3. 重复交叉验证
from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=rskf)

print(f"平均AUC: {cv_scores.mean():.3f}")
print(f"标准差: {cv_scores.std():.3f}")
```

---

## 相关文档

- [变量分箱模块](binning.md) - WOE转换是建模的基础
- [变量选择模块](var-select.md) - 选择最优变量组合
- [评分卡实现模块](scorecard-implement.md) - 将模型转换为评分卡
