# 最佳实践

本文总结了评分卡建模的行业最佳实践，帮助您构建高质量、可投产的评分卡模型。

## 数据质量

### 1.1 数据探索

**必查项**：
- ✅ 目标变量分布（正样本占比）
- ✅ 特征缺失率统计
- ✅ 特征唯一值数量（识别常变量）
- ✅ 特征相关性矩阵
- ✅ 异常值检测

```python
# 自动化数据探索
yh = Yihuier(data, target='dlq_flag')
eda_stats = yh.eda_module.auto_eda_simple()

# 检查目标变量分布
print(f"正样本占比: {data['dlq_flag'].mean():.2%}")

# 检查缺失率
missing_rate = data.isnull().sum() / len(data)
print(f"最大缺失率: {missing_rate.max():.2%}")
```

### 1.2 数据清洗原则

**缺失值处理**：
- 缺失率 > 40%：考虑删除变量
- 缺失率 20-40%：填充特殊类别
- 缺失率 < 20%：填充均值/中位数

**常变量处理**：
- 同值化 > 90%：删除变量
- 同值化 70-90%：谨慎使用

```python
# 删除高缺失率变量
yh.data = yh.dp_module.delete_missing_var(threshold=0.2)

# 删除常变量（谨慎使用）
# yh.data = yh.dp_module.const_delete(threshold=0.9)
```

### 1.3 样本量要求

**最低要求**：
- 总样本数 ≥ 1000
- 正样本数 ≥ 100
- 特征数 ≤ 样本数 / 10

**推荐配置**：
- 总样本数 ≥ 5000
- 正样本数 ≥ 500
- 正负比 ≥ 1:5

## 变量分箱

### 2.1 分箱方法选择

| 方法 | 适用场景 | 注意事项 |
|------|---------|---------|
| ChiMerge | 追求最优预测能力 | 可能过拟合，计算慢 |
| 等频分箱 | 样本量大，分布不均 | 可能打破单调性 |
| 等距分箱 | 数据分布均匀 | 对异常值敏感 |
| 单调性分箱 | 需要单调趋势 | 可能损失信息 |

**推荐策略**：
```python
# 优先尝试 ChiMerge
try:
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=vars,
        max_bin=5,
        method='ChiMerge'
    )
except Exception as e:
    # 失败则使用等频分箱
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=vars,
        max_bin=5,
        method='freq'
    )
```

### 2.2 分箱数选择

**一般原则**：
- 最少：3 箱（避免过度拟合）
- 推荐：5-7 箱
- 最多：10 箱（除非样本量很大）

**样本量与分箱数**：
- < 1000 样本：3-5 箱
- 1000-5000 样本：5-7 箱
- > 5000 样本：7-10 箱

### 2.3 分箱验证

**必须检查**：
1. ✅ 单调性（对于有业务含义的变量）
2. ✅ 每箱样本数 ≥ 50
3. ✅ 最小分箱占比 ≥ 5%
4. ✅ WOE 值在合理范围 (-3 到 3)

```python
# 检查分箱结果
for df in bin_df:
    print(df)
    # 查看每箱样本数
    # 查看 WOE 值
    # 检查单调性
```

## 变量选择

### 3.1 IV 筛选标准

| IV 值 | 决策 |
|-------|------|
| < 0.02 | 不使用 |
| 0.02 - 0.1 | 谨慎使用 |
| 0.1 - 0.3 | 可以使用 |
| > 0.3 | 强烈推荐 |

**注意**：IV > 0.5 时要警惕数据泄露。

### 3.2 相关性处理

**策略**：IV 优先 + 相关性去重

```python
# 先按 IV 筛选
high_iv_vars = [var for var, iv in zip(vars, iv_values) if iv > 0.1]

# 再考虑相关性去重
final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
    col_list=high_iv_vars,
    threshold=0.6
)
```

**相关性阈值建议**：
- 严格：0.5（减少共线性）
- 标准：0.6（平衡性能和多样性）
- 宽松：0.7（保留更多信息）

### 3.3 变量数量控制

**推荐变量数**：
- 最少：5 个
- 推荐：10-15 个
- 最多：20 个

**变量过多的问题**：
- 过拟合风险增加
- 可解释性下降
- 部署成本增加

## 模型训练

### 4.1 数据集划分

**标准划分**：
- 训练集：70%
- 测试集：30%

**三数据集划分**（推荐）：
- 训练集：60%
- 验证集：20%
- 测试集：20%

**注意事项**：
- ✅ 使用分层采样（stratify）
- ✅ 固定随机种子（random_state）
- ✅ 时间序列数据按时间划分

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # 分层采样
)
```

### 4.2 正则化选择

| 场景 | 推荐配置 |
|------|---------|
| 特征少（< 10） | penalty='l2', C=1.0 |
| 特征中等（10-20） | penalty='l2', C=0.1 |
| 特征多（> 20） | penalty='l1', C=0.01 |

```python
model = LogisticRegression(
    penalty='l2',
    C=0.1,
    max_iter=1000,
    random_state=42
)
```

### 4.3 类别不平衡处理

**方法选择**：
- 轻度不平衡（正样本 10-20%）：class_weight='balanced'
- 中度不平衡（正样本 5-10%）：增加正样本权重
- 重度不平衡（正样本 < 5%）：考虑采样方法

```python
# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = dict(enumerate(class_weights))

model = LogisticRegression(
    class_weight=weight_dict,
    max_iter=1000
)
```

## 模型评估

### 5.1 评估指标优先级

**开发阶段**：
1. AUC：整体区分能力
2. KS：最大区分点
3. 交叉验证 AUC：泛化能力

**验证阶段**：
1. 验证集 AUC/KS
2. 训练集 vs 验证集差异
3. 学习曲线

**投产阶段**：
1. PSI：分数稳定性
2. 实际坏账率
3. 通过率

### 5.2 性能基准

**最低要求**：
- AUC ≥ 0.65
- KS ≥ 0.15
- 训练集和验证集 AUC 差异 ≤ 0.05

**优秀模型**：
- AUC ≥ 0.75
- KS ≥ 0.30
- 交叉验证 AUC 标准差 ≤ 0.02

### 5.3 过拟合检测

**红色警报**：
- 训练集 AUC ≥ 0.95
- 训练集和验证集 AUC 差异 > 0.10
- 学习曲线中训练线和验证线不收敛

**解决方案**：
1. 减少特征数量
2. 增加正则化（减小 C）
3. 增加训练样本
4. 减少分箱数

## 评分卡实现

### 6.1 刻度参数选择

**行业标准**：
- 基础分：600 分对应 50:1 odds
- PDO：20 分（odds 翻倍减少 20 分）

**自定义原则**：
- 分数范围：300-850 分
- 基础分：接近分数中位数
- PDO：使分数分布合理

```python
A, B, base_score = yh.si_module.cal_scale(
    score=600,    # 基础分
    odds=50,      # 好坏比 50:1
    PDO=20,       # odds 翻倍减少 20 分
    model=model
)
```

### 6.2 Cutoff 设置

**方法选择**：
1. **KS 最大点**：平衡精确率和召回率
2. **业务目标**：根据目标通过率计算
3. **损失矩阵**：最小化总损失

**验证要点**：
- ✅ 通过率是否合理（30-70%）
- ✅ 坏账率是否可接受
- ✅ 误伤率是否过高

```python
# 验证 cutoff
matrix = yh.si_module.rule_verify(
    df=test_scores,
    col_score='score',
    target='dlq_flag',
    cutoff=600
)

pass_rate = (test_scores['score'] >= 600).mean()
bad_rate = test_scores[test_scores['score'] >= 600]['dlq_flag'].mean()

print(f"通过率: {pass_rate:.2%}")
print(f"坏账率: {bad_rate:.2%}")
```

### 6.3 分数分布检查

**健康分布**：
- 均值：500-650
- 标准差：80-120
- 范围：300-850
- 无极端聚集

**异常信号**：
- 大量分数聚集在某一区间
- 分数出现负数或过高
- 分布严重偏斜

## 模型监控

### 7.1 PSI 监控阈值

| PSI 值 | 行动 |
|--------|------|
| < 0.1 | 继续监控 |
| 0.1 - 0.2 | 增加监控频率，分析原因 |
| 0.2 - 0.25 | 准备重训 |
| > 0.25 | **立即重训** |

### 7.2 性能衰减监控

**触发重训的条件**：
- AUC 下降 ≥ 5%
- KS 下降 ≥ 20%
- PSI > 0.25
- 实际坏账率超过预期 20%

### 7.3 特征监控

**监控内容**：
1. 特征缺失率变化
2. 特征分布变化（特征 PSI）
3. 异常值比例
4. 新类别出现

**频率建议**：
- 上线后 1 个月：每周监控
- 稳定期：每月监控
- 异常期：每日监控

## 部署建议

### 8.1 模型导出

**推荐格式**：
1. **Pickle**：完整模型对象
2. **PMML**：跨平台部署
3. **SQL**：规则引擎部署

```python
# Pickle 导出
import pickle

with open('scorecard_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 导出评分卡规则
score_df = yh.si_module.score_df_concat(
    woe_df=woe_df,
    model=model,
    B=B
)
score_df.to_csv('scorecard_rules.csv', index=False)
```

### 8.2 批量评分

```python
# 批量计算分数
def batch_score(model, data, A, B, base_score):
    """
    批量计算评分卡分数
    """
    scores = base_score + data.dot(model.coef_[0]) * B
    return scores
```

### 8.3 实时评分

```python
# 实时评分接口
def real_time_score(customer_features, model, woe_rules, A, B, base_score):
    """
    实时计算单个客户分数
    """
    # 1. WOE 转换
    woe_features = apply_woe(customer_features, woe_rules)

    # 2. 计算分数
    score = base_score + sum(woe_features * model.coef_[0]) * B

    return score
```

## 常见陷阱

### 陷阱 1：数据泄露

**表现**：训练集 AUC ≥ 0.95，测试集 AUC ≤ 0.65

**原因**：使用了未来的信息

**预防**：
- 仔细检查每个特征的定义
- 确保特征计算不使用未来数据
- 时间序列验证

### 陷阱 2：过度拟合

**表现**：训练集和测试集性能差异很大

**原因**：模型过于复杂

**预防**：
- 增加正则化
- 减少特征数量
- 交叉验证
- 简化分箱

### 陷阱 3：样本不足

**表现**：性能不稳定，不同随机种子结果差异大

**原因**：样本量不足

**预防**：
- 增加数据收集
- 使用简单模型
- 减少特征数量
- 使用预训练模型

### 陷阱 4：忽视业务

**表现**：AUC 很高，但业务效果差

**原因**：模型优化目标与业务目标不一致

**预防**：
- 理解业务需求
- 使用业务指标评估
- 与业务人员沟通
- A/B 测试

## 参考资源

- [项目简介](intro.md) - 了解 Yihuier 设计理念
- [核心概念](concepts/) - WOE、IV、评分卡基础
- [模块文档](modules/) - 各模块详细用法
