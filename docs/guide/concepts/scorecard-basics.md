# 评分卡基础

评分卡是一种广泛应用于信贷风控的评分模型，通过将客户的多个特征转化为一个综合分数，来评估其信用风险。

## 评分卡原理

### 数学基础

评分卡基于逻辑回归模型，将预测概率转换为可解释的分数。

**逻辑回归公式**：

$$P(Default) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}$$

其中：
- $\beta_0$：截距项
- $\beta_1, \beta_2, ..., \beta_n$：各变量系数
- $x_1, x_2, ..., x_n$：各变量值（WOE 转换后）

### 评分转换

将逻辑回归的对数几率（log-odds）转换为分数：

$$\text{Score} = A - B \times \ln(\text{odds})$$

其中：
- $\text{odds} = \frac{P(\text{Good})}{P(\text{Bad})} = \frac{1 - P(\text{Default})}{P(\text{Default})}$
- $A, B$：刻度参数

展开后：

$$\text{Score} = A - B \times (\beta_0 + \beta_1 \text{WOE}_1 + \beta_2 \text{WOE}_2 + ... + \beta_n \text{WOE}_n)$$

### PDO（Points to Double Odds）

PDO 表示当 odds 翻倍时，分数的变化量。

**公式推导**：

设分数 $S_1$ 对应 odds $o_1$，分数 $S_2$ 对应 odds $o_2 = 2o_1$：

$$\begin{cases}
S_1 = A - B \times \ln(o_1) \\
S_2 = A - B \times \ln(2o_1)
\end{cases}$$

则：

$$\text{PDO} = S_1 - S_2 = B \times \ln(2)$$

因此：

$$B = \frac{\text{PDO}}{\ln(2)}$$

### 参数确定

根据业务需求设定三个参数：
1. **基础分（Base Score）**：某个特定 odds 时的分数
2. **Odds**：基础分对应的好坏比（如 50:1）
3. **PDO**：odds 翻倍时的分数变化（如 20 分）

**计算公式**：

$$\begin{cases}
B = \frac{\text{PDO}}{\ln(2)} \\
A = \text{Base Score} + B \times \ln(\text{odds})
\end{cases}$$

## 示例计算

### 场景设定

- 基础分：600 分对应 50:1 的 odds
- PDO：20 分（odds 翻倍时减少 20 分）

### 参数计算

$$\begin{aligned}
B &= \frac{20}{\ln(2)} = \frac{20}{0.693} = 28.85 \\
A &= 600 + 28.85 \times \ln(50) \\
  &= 600 + 28.85 \times 3.912 \\
  &= 600 + 112.84 \\
  &= 712.84
\end{aligned}$$

### 分数计算

假设训练好的逻辑回归模型为：

$$\ln(\text{odds}) = -0.5 + 0.3 \times \text{WOE}_1 + 0.5 \times \text{WOE}_2$$

某客户的 WOE 值为 $\text{WOE}_1 = 0.2$, $\text{WOE}_2 = -0.3$：

$$\begin{aligned}
\ln(\text{odds}) &= -0.5 + 0.3 \times 0.2 + 0.5 \times (-0.3) \\
&= -0.5 + 0.06 - 0.15 \\
&= -0.59
\end{aligned}$$

$$\text{Score} = 712.84 - 28.85 \times (-0.59) = 712.84 + 17.02 = 729.86$$

## 变量得分分解

### 单个变量的得分

每个变量在不同分箱的得分：

$$\text{Score}_i = -B \times \beta_i \times \text{WOE}_{ij}$$

其中：
- $\beta_i$：变量 $i$ 的系数
- $\text{WOE}_{ij}$：变量 $i$ 在第 $j$ 箱的 WOE 值

### 总分

$$\text{Total Score} = \text{Base Score} + \sum_i \text{Score}_i$$

### 得分表示例

| 变量 | 分箱 | WOE | 系数 | 得分 |
|------|------|-----|------|------|
| 年龄 | 20-30 | -0.5 | 0.3 | -4.33 |
| 年龄 | 30-40 | 0.2 | 0.3 | 1.73 |
| 年龄 | 40-50 | 0.5 | 0.3 | 4.33 |
| 收入 | 低 | -0.8 | 0.5 | -11.54 |
| 收入 | 中 | 0.1 | 0.5 | 1.44 |
| 收入 | 高 | 0.6 | 0.5 | 8.66 |

基础分：712.84

某客户（年龄 35，收入中）：
$$712.84 + 1.73 + 1.44 = 716.01$$

## 评分卡刻度选择

### 常见刻度标准

| 行业 | 基础分 | Odds | PDO | 分数范围 |
|------|--------|------|-----|---------|
| 传统银行 | 600 | 50:1 | 20 | 300-850 |
| 消费金融 | 500 | 20:1 | 15 | 200-800 |
| 信用卡 | 650 | 30:1 | 25 | 400-900 |

### 刻度选择原则

1. **业务可解释性**：分数应该容易被业务人员理解
2. **区分度**：分数分布应该有合理的分散度
3. **稳定性**：刻度参数变化不应过于敏感

### Yihuier 实现

```python
from yihuier import Yihuier
from sklearn.linear_model import LogisticRegression

# 训练模型
yh = Yihuier(data, target='dlq_flag')
# ... 预处理和分箱 ...

model = LogisticRegression()
model.fit(X_train, y_train)

# 计算评分卡刻度
A, B, base_score = yh.si_module.cal_scale(
    score=600,    # 基础分
    odds=50,      # 好坏比 50:1
    PDO=20,       # odds 翻倍减少 20 分
    model=model
)

print(f"A = {A:.2f}")
print(f"B = {B:.2f}")
print(f"基础分 = {base_score:.2f}")
```

## Cutoff 设置

### 定义

Cutoff（截断点）是决定是否批准申请的分数阈值：
- **分数 ≥ cutoff**：批准（低风险）
- **分数 < cutoff**：拒绝（高风险）

### 选择方法

#### 方法一：基于业务目标

$$\text{Cutoff} = A - B \times \ln\left(\frac{\text{批准率}}{1 - \text{批准率}}\right)$$

例如，目标批准率 70%：

$$\text{Cutoff} = A - B \times \ln\left(\frac{0.7}{0.3}\right)$$

#### 方法二：基于 KS 值

选择 KS 最大的分数点作为 cutoff：

```python
# 计算每个分数的 KS
ks_df = yh.si_module.score_info(test_scores, 'score', 'dlq_flag')

# 选择 KS 最大的点
cutoff = ks_df.loc[ks_df['KS'].idxmax(), 'score']
```

#### 方法三：基于损失矩阵

考虑批准坏用户和拒绝好用户的损失：

$$\text{最优 Cutoff} = \arg\min_s [C_{FN} \times P(Bad|s) \times P(s) + C_{FP} \times P(Good|s) \times P(s)]$$

其中：
- $C_{FN}$：漏掉坏用户的损失（批准坏用户）
- $C_{FP}$：误伤好用户的损失（拒绝好用户）

### 验证 Cutoff

```python
# 验证 cutoff 有效性
matrix = yh.si_module.rule_verify(
    df=test_scores,
    col_score='score',
    target='dlq_flag',
    cutoff=650
)

print("混淆矩阵：")
print(matrix)

# 计算通过率
pass_rate = (test_scores['score'] >= 650).mean()
print(f"通过率: {pass_rate:.2%}")
```

## 评分卡评估

### 区分度指标

1. **AUC**：ROC 曲线下面积
2. **KS**：好坏用户分布的最大距离
3. **提升度**：评分卡相对于随机选择的提升

### 稳定性指标

1. **PSI**：分数分布稳定性
2. **特征稳定性**：各变量分布变化
3. **互换验证**：不同时间样本的验证

### 业务指标

1. **通过率**：申请被批准的比例
2. **坏账率**：批准用户中的违约比例
3. **覆盖率**：捕捉到的好用户比例

## 常见问题

### Q1: 为什么评分卡分数可以出现负数？

**原因**：如果 A - B × ln(odds) 计算结果为负。

**解决方案**：
1. 调整刻度参数（增大 A 或减小 B）
2. 确保所有变量得分合理

### Q2: 如何处理新出现的类别？

**方案**：
1. 将新类别分配到最相似的现有分箱
2. 使用该变量的平均 WOE 值
3. 标记为缺失值

### Q3: 评分卡多久需要重新训练？

**建议**：
1. **定期验证**：每月监控 PSI 和模型性能
2. **触发重训**：
   - PSI > 0.2（分数分布显著变化）
   - AUC 下降超过 5%
   - KS 下降超过 20%
   - 业务规则变化

## 参考资源

- [WOE 和 IV 介绍](woe-iv.md) - 理解 WOE 编码
- [评分卡实现模块](../modules/scorecard-implement.md) - 刻度计算和分数转换
- [评分卡监控模块](../modules/scorecard-monitor.md) - PSI 分析和稳定性监控
