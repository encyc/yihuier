# WOE 和 IV 介绍

WOE（Weight of Evidence）和 IV（Information Value）是评分卡建模中最核心的两个概念，用于变量编码和特征选择。

## WOE（证据权重）

### 定义

WOE 是一种编码方式，用于将类别型变量转换为数值型变量，同时保留其预测能力。

**数学公式**：

$$WOE_i = \ln\left(\frac{P(Good_i)}{P(Bad_i)}\right) = \ln\left(\frac{Good_i / Good_{total}}{Bad_i / Bad_{total}}\right)$$

其中：
- $Good_i$：第 $i$ 箱的好用户数量
- $Bad_i$：第 $i$ 箱的坏用户数量
- $Good_{total}$：总好用户数量
- $Bad_{total}$：总坏用户数量

### 直观理解

WOE 表示该分箱中好坏用户的相对比例与整体好坏比例的差异：

- **WOE > 0**：该箱好用户占比高于整体，风险较低
- **WOE < 0**：该箱坏用户占比高于整体，风险较高
- **WOE = 0**：该箱好坏比例与整体相同，无区分能力

### 示例

```python
import pandas as pd

# 示例数据
data = pd.DataFrame({
    'age_range': ['20-30', '20-30', '30-40', '30-40', '40-50'],
    'dlq_flag': [1, 0, 0, 0, 0]  # 1=违约，0=正常
})

# 计算各箱的 WOE
# 20-30岁：好用户1个，坏用户1个
# 30-40岁：好用户2个，坏用户0个
# 40-50岁：好用户1个，坏用户0个

# 总体：好用户4个，坏用户1个
```

**WOE 计算结果**：
- `20-30岁`：$\ln((1/4)/(1/1)) = \ln(0.25) = -1.39$
- `30-40岁`：$\ln((2/4)/(0/1)) = \ln(\infty) = +\infty$（需要平滑处理）
- `40-50岁`：$\ln((1/4)/(0/1)) = \ln(\infty) = +\infty$（需要平滑处理）

### WOE 的优势

1. **处理非线性关系**：将复杂的非线性关系转换为线性
2. **标准化不同尺度**：不同变量都在同一数量级
3. **处理缺失值**：缺失值可以作为一个独立的 WOE 值
4. **业务可解释性**：每个 WOE 值都有明确的业务含义

### 平滑处理

当某些分箱没有坏用户（或好用户）时，WOE 会趋向无穷大。需要添加平滑：

$$WOE_i = \ln\left(\frac{Good_i + \alpha}{Bad_i + \beta}\right)$$

其中 $\alpha, \beta$ 是小常数（如 0.5, 1.0）。

## IV（信息价值）

### 定义

IV 用于衡量变量的预测能力，是特征选择的核心指标。

**数学公式**：

$$IV = \sum_{i=1}^{n} (P(Good_i) - P(Bad_i)) \times WOE_i$$

展开后：

$$IV = \sum_{i=1}^{n} \left(\frac{Good_i}{Good_{total}} - \frac{Bad_i}{Bad_{total}}\right) \times \ln\left(\frac{Good_i / Good_{total}}{Bad_i / Bad_{total}}\right)$$

### 直观理解

IV 衡量了变量的整体区分能力：
- 每个分箱的 $(P(Good) - P(Bad))$ 表示该箱的差异性
- 乘以 WOE 得到该箱的贡献
- 所有分箱的贡献相加得到总 IV

### IV 值判断标准

| IV 值 | 预测能力 | 使用建议 |
|-------|---------|---------|
| < 0.02 | 无预测能力 | 不使用 |
| 0.02 - 0.1 | 弱预测能力 | 谨慎使用 |
| 0.1 - 0.3 | 中等预测能力 | 可以使用 |
| > 0.3 | 强预测能力 | **强烈推荐** |

### 示例计算

```python
# 假设某变量的分箱结果
# 箱1：好用户100，坏用户20
# 箱2：好用户200，坏用户50
# 箱3：好用户150，坏用户80

# 总体：好用户450，坏用户150

# 箱1 IV: (100/450 - 20/150) * ln((100/450)/(20/150))
#       = (0.222 - 0.133) * ln(1.667)
#       = 0.089 * 0.511 = 0.045

# 箱2 IV: (200/450 - 50/150) * ln((200/450)/(50/150))
#       = (0.444 - 0.333) * ln(1.333)
#       = 0.111 * 0.288 = 0.032

# 箱3 IV: (150/450 - 80/150) * ln((150/450)/(80/150))
#       = (0.333 - 0.533) * ln(0.625)
#       = -0.200 * -0.470 = 0.094

# 总 IV = 0.045 + 0.032 + 0.094 = 0.171 (中等预测能力)
```

## WOE 与 IV 的关系

### 核心区别

| 特性 | WOE | IV |
|------|-----|-----|
| 作用 | 变量编码 | 特征选择 |
| 粒度 | 每个分箱一个值 | 整个变量一个值 |
| 用途 | 转换为模型输入 | 评估变量重要性 |
| 可加性 | 无 | 各分箱 IV 可相加 |

### 计算流程

1. **分箱**：将连续变量离散化
2. **计算 WOE**：为每个分箱计算 WOE 值
3. **计算 IV**：将各分箱的贡献相加得到总 IV
4. **变量筛选**：根据 IV 值选择变量
5. **模型训练**：使用 WOE 转换后的数据

### 完整示例

```python
from yihuier import Yihuier

# 初始化
yh = Yihuier(data, target='dlq_flag')

# 变量分箱
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=['age', 'income', 'debt_ratio'],
    max_bin=5,
    method='ChiMerge'
)

# 查看 IV 值
print(f"age IV: {iv_value[0]:.4f}")
print(f"income IV: {iv_value[1]:.4f}")
print(f"debt_ratio IV: {iv_value[2]:.4f}")

# 选择 IV > 0.1 的变量
selected_vars = [
    var for var, iv in zip(['age', 'income', 'debt_ratio'], iv_value)
    if iv > 0.1
]

# WOE 转换
woe_df = yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 查看转换后的数据
print(data_woe[selected_vars].head())
```

## 最佳实践

### 1. 分箱数选择

- **太少**：损失信息，IV 偏低
- **太多**：过拟合，IV 虚高
- **推荐**：5-10 箱（根据样本量和业务）

### 2. 缺失值处理

将缺失值作为独立分箱：
```python
# Yihuier 自动处理缺失值
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=['income'],
    max_bin=5
)
# 缺失值会自动成为一个分箱
```

### 3. 单调性检查

对于有业务含义的变量（如年龄、收入），检查 WOE 单调性：
```python
# 检查 WOE 是否单调
# 如果不单调，考虑：
# 1. 减少分箱数
# 2. 使用 monotonic 分箱方法
# 3. 手动调整分箱边界
```

### 4. 特殊值处理

对于特殊值（如 -999, -1111），先替换为缺失值：
```python
# 替换特殊值
data['income'] = data['income'].replace(-999, np.nan)

# 然后正常分箱
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=['income'],
    max_bin=5
)
```

## 常见问题

### Q1: WOE 转换后还需要标准化吗？

**不需要**。WOE 本身已经是对数变换后的值，量级相近。

### Q2: IV 值是否越大越好？

**不是**。IV 过大（> 0.5）可能意味着：
- 变量与目标变量存在完美关系（可能是数据泄露）
- 分箱过细，过拟合
- 需要业务审查

### Q3: 为什么我的变量 IV 很低？

可能原因：
1. 变量与目标变量确实无关
2. 分箱不合理
3. 数据质量问题
4. 缺失值过多

解决方案：
```python
# 尝试不同分箱方法
for method in ['ChiMerge', 'freq', 'count']:
    _, iv = yh.binning_module.binning_num(
        col_list=['var1'],
        max_bin=5,
        method=method
    )
    print(f"{method} IV: {iv[0]:.4f}")
```

## 参考资源

- [评分卡基础](scorecard-basics.md) - 评分卡的数学原理
- [变量分箱](../modules/binning.md) - 分箱模块使用指南
- [变量选择](../modules/var-select.md) - 基于 IV 的变量选择
