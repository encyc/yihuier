# 模型评估指标

本文介绍评分卡建模中常用的评估指标，包括区分度指标、稳定性指标和业务指标。

## 区分度指标

### AUC（Area Under ROC Curve）

#### 定义

AUC 是 ROC 曲线下的面积，衡量模型区分好坏用户的能力。

**ROC 曲线**：横轴为 FPR（假阳性率），纵轴为 TPR（真阳性率）。

$$\begin{aligned}
\text{TPR} &= \frac{TP}{TP + FN} \quad \text{(召回率)} \\
\text{FPR} &= \frac{FP}{FP + TN}
\end{aligned}$$

#### 取值范围

| AUC 值 | 区分能力 | 评价 |
|--------|---------|------|
| 0.5 | 无区分能力 | 模型无效 |
| 0.5 - 0.6 | 很弱 | 需要改进 |
| 0.6 - 0.7 | 弱 | 可用但不理想 |
| 0.7 - 0.8 | 中等 | 基本满足要求 |
| 0.8 - 0.9 | 强 | 良好模型 |
| > 0.9 | 很强 | 优秀模型（注意过拟合） |

#### 优点

- 不受阈值选择影响
- 对类别不平衡不敏感
- 可解释性强：随机挑选一个好用户和一个坏用户，模型给好用户更高分数的概率

#### 计算示例

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 计算 AUC
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### KS（Kolmogorov-Smirnov）

#### 定义

KS 值衡量好坏用户分数分布的最大距离。

**公式**：

$$\text{KS} = \max_{s} |F_{Good}(s) - F_{Bad}(s)|$$

其中：
- $F_{Good}(s)$：好用户中分数 ≤ s 的累计比例
- $F_{Bad}(s)$：坏用户中分数 ≤ s 的累计比例

#### 取值范围

| KS 值 | 区分能力 | 评价 |
|-------|---------|------|
| < 0.1 | 很弱 | 模型无效 |
| 0.1 - 0.2 | 弱 | 需要改进 |
| 0.2 - 0.3 | 中等 | 基本满足要求 |
| > 0.3 | 强 | 良好模型 |

#### 优点

- 直观反映模型区分能力
- 可以确定最佳 cutoff 点（KS 最大的分数）
- 对业务人员易于理解

#### 计算示例

```python
# 计算 KS 值
ks_value = yh.me_module.model_ks(y_test, y_pred)
print(f"KS: {ks_value:.4f}")

# 绘制 KS 曲线
yh.me_module.plot_model_ks(y_test, y_pred)
```

### 其他区分度指标

#### 精确率（Precision）

$$\text{Precision} = \frac{TP}{TP + FP}$$

表示预测为坏用户的样本中，真正坏用户的比例。

#### 召回率（Recall）

$$\text{Recall} = \frac{TP}{TP + FN}$$

表示所有坏用户中，被正确识别的比例。

#### F1-Score

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

精确率和召回率的调和平均。

## 稳定性指标

### PSI（Population Stability Index）

#### 定义

PSI 衡量两个分布（如训练样本和上线样本）的差异程度。

**公式**：

$$\text{PSI} = \sum_{i=1}^{n} (\text{Actual}_i - \text{Expected}_i) \times \ln\left(\frac{\text{Actual}_i}{\text{Expected}_i}\right)$$

其中：
- $\text{Expected}_i$：训练样本在第 $i$ 个分箱的占比
- $\text{Actual}_i$：上线样本在第 $i$ 个分箱的占比

#### 取值范围

| PSI 值 | 稳定性 | 建议 |
|--------|--------|------|
| < 0.1 | 稳定 | 无需动作 |
| 0.1 - 0.2 | 轻微变化 | 监控 |
| > 0.25 | 显著变化 | **需要重新训练** |

#### 应用场景

1. **分数 PSI**：监控分数分布变化
2. **特征 PSI**：监控各变量分布变化
3. **时间对比**：不同时期样本对比
4. **样本对比**：开发样本 vs 验证样本 vs 测试样本

#### 计算示例

```python
# 计算分数 PSI
psi_df = yh.sm_module.score_psi(
    df1=model_data,
    df2=online_data,
    id_col='customer_id',
    score_col='score',
    x=200,
    y=800,
    step=20
)

print(f"总 PSI: {psi_df['PSI'].sum():.4f}")

# 可视化对比
yh.sm_module.plot_score_compare(
    df1=model_data,
    df2=online_data,
    score_col='score'
)
```

### 交叉验证（Cross-Validation）

#### 定义

将数据集分成 K 份，每次用 K-1 份训练，1 份验证，重复 K 次。

**目的**：评估模型的泛化能力，避免过拟合。

#### K 折交叉验证

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

# 5 折交叉验证
cv_scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='roc_auc'
)

print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 使用 Yihuier
yh.me_module.cross_verify(
    x=X, y=y,
    estimator=LogisticRegression(max_iter=1000),
    fold=5,
    scoring='roc_auc'
)
```

### 学习曲线（Learning Curve）

#### 定义

学习曲线展示模型性能随训练样本量增加的变化趋势。

**用途**：
- 判断是否需要更多数据
- 检测过拟合/欠拟合
- 评估模型收敛情况

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    estimator=LogisticRegression(max_iter=1000),
    X=X, y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='roc_auc'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training AUC')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation AUC')
plt.xlabel('Training Size')
plt.ylabel('AUC')
plt.title('Learning Curve')
plt.legend()
plt.show()
```

## 业务指标

### 提升度（Lift）

#### 定义

提升度衡量评分卡相对于随机选择的改进程度。

**公式**：

$$\text{Lift} = \frac{\text{评分卡捕获率}}{\text{随机选择捕获率}}$$

#### 提升图

```python
# 绘制提升图和洛伦兹曲线
yh.si_module.plot_lifting(
    df=test_scores,
    score_col='score',
    target='dlq_flag'
)
```

### 通过率和坏账率

```python
# 计算不同 cutoff 下的通过率和坏账率
cutoffs = range(400, 800, 50)
results = []

for cutoff in cutoffs:
    passed = test_scores['score'] >= cutoff
    pass_rate = passed.mean()
    bad_rate = test_scores.loc[passed, 'dlq_flag'].mean()
    results.append({'cutoff': cutoff, 'pass_rate': pass_rate, 'bad_rate': bad_rate})

results_df = pd.DataFrame(results)
print(results_df)
```

## 指标选择建议

### 模型开发阶段

**核心指标**：
- AUC：整体区分能力
- KS：最佳 cutoff 确定
- 交叉验证 AUC：泛化能力

**关注点**：模型的预测能力

### 模型验证阶段

**核心指标**：
- 验证集 AUC/KS
- 训练集 vs 验证集差异
- 学习曲线分析

**关注点**：模型稳定性和过拟合

### 模型监控阶段

**核心指标**：
- PSI：分数分布稳定性
- 特征 PSI：变量分布变化
- 实际 AUC/KS：在线性能

**关注点**：模型衰减和重训触发

### 业务评估阶段

**核心指标**：
- 通过率
- 坏账率
- 提升度
- 混淆矩阵

**关注点**：业务影响和 ROI

## 常见问题

### Q1: AUC 高但 KS 低，可能吗？

**可能**。AUC 衡量整体性能，KS 衡量最大差异。如果模型在某些分数段区分很好，但整体提升有限，会出现这种情况。

### Q2: PSI 高但 AUC 没下降，需要重训吗？

**需要**。PSI 高说明分布已经变化，即使当前 AUC 还正常，未来性能很可能下降。预防性重训是明智的。

### Q3: 如何平衡精确率和召回率？

根据业务损失：
- **漏掉坏用户损失高**：提高召回率（降低 cutoff）
- **误伤好用户损失高**：提高精确率（提高 cutoff）

### Q4: 交叉验证方差大怎么办？

可能原因：
1. 样本量不足
2. 数据不稳定
3. 模型过于复杂

解决方案：
1. 增加数据
2. 使用分层采样
3. 简化模型
4. 增加正则化

## 参考资源

- [模型评估模块](../modules/model-evaluation.md) - 评估指标计算
- [评分卡监控模块](../modules/scorecard-monitor.md) - PSI 分析
- [评分卡实现模块](../modules/scorecard-implement.md) - cutoff 验证
