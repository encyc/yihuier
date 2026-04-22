# 高级示例

本文档提供 Yihuier 的高级使用场景和完整案例，帮助您解决实际项目中的复杂问题。

## 完整建模流程

### 案例：信用评分卡端到端建模

```python
import pandas as pd
import numpy as np
from yihuier import Yihuier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# 1. 数据加载
data = pd.read_csv('credit_data.csv')
print(f"原始数据: {data.shape}")
print(f"目标变量分布:\n{data['dlq_flag'].value_counts()}")

# 2. 初始化
yh = Yihuier(data, target='dlq_flag')

# 3. 数据探索
print("\n=== 数据探索 ===")
eda_stats = yh.eda_module.auto_eda_simple()
print(f"数据质量报告:\n{eda_stats}")

# 检查变量类型
numeric_vars = yh.get_numeric_variables()
categorical_vars = yh.get_categorical_variables()
print(f"数值型变量: {len(numeric_vars)} 个")
print(f"类别型变量: {len(categorical_vars)} 个")

# 4. 数据预处理
print("\n=== 数据预处理 ===")

# 处理数值型变量缺失值
yh.data = yh.dp_module.fillna_num_var(
    col_list=numeric_vars,
    fill_type='0'  # 缺失值填充为 0
)

# 处理类别型变量缺失值
if categorical_vars:
    yh.data = yh.dp_module.fillna_cate_var(
        col_list=categorical_vars,
        fill_type='mode'  # 缺失值填充为众数
    )

# 删除目标变量缺失的样本
yh.data = yh.dp_module.target_missing_delete()

print(f"预处理后数据: {yh.data.shape}")

# 5. 变量分箱
print("\n=== 变量分箱 ===")

# 选择部分变量进行分箱（演示用）
binning_vars = numeric_vars[:10]

try:
    # 尝试 ChiMerge 分箱
    bin_df_num, iv_value_num = yh.binning_module.binning_num(
        col_list=binning_vars,
        max_bin=5,
        method='ChiMerge',
        min_binpct=0.05
    )
    print(f"使用 ChiMerge 分箱")
except Exception as e:
    # ChiMerge 失败则使用等频分箱
    print(f"ChiMerge 失败，改用等频分箱: {e}")
    bin_df_num, iv_value_num = yh.binning_module.binning_num(
        col_list=binning_vars,
        max_bin=5,
        method='freq'
    )

# 显示 IV 值
iv_df = pd.DataFrame({
    'variable': binning_vars,
    'iv_value': iv_value_num
}).sort_values('iv_value', ascending=False)

print(f"\n变量 IV 值（前10）:\n{iv_df}")

# 6. WOE 转换
print("\n=== WOE 转换 ===")

# 拼接 WOE 结果表
woe_df = yh.binning_module.woe_df_concat()
print(f"WOE 结果表: {woe_df.shape if woe_df is not None else 'None'}")

# WOE 转换
data_woe = yh.binning_module.woe_transform()
print(f"WOE 转换后数据: {data_woe.shape}")

# 7. 变量选择
print("\n=== 变量选择 ===")

# 准备特征列表
feature_cols = [col for col in data_woe.columns if col != yh.target]

# 方法 1: XGBoost 特征重要性
print("方法 1: XGBoost 特征重要性")
xg_imp, xg_rank, xg_cols = yh.var_select_module.select_xgboost(
    col_list=feature_cols,
    imp_num=10
)
print(f"选中的变量 ({len(xg_cols)} 个): {xg_cols}")

# 方法 2: 随机森林特征重要性
print("\n方法 2: 随机森林特征重要性")
rf_imp, rf_cols = yh.var_select_module.select_rf(
    col_list=feature_cols,
    imp_num=10
)
print(f"选中的变量 ({len(rf_cols)} 个): {rf_cols}")

# 方法 3: IV + 相关性去重
print("\n方法 3: IV 筛选 + 相关性去重")
# 筛选 IV > 0.1 的变量
high_iv_vars = iv_df[iv_df['iv_value'] > 0.1]['variable'].tolist()
print(f"IV > 0.1 的变量: {len(high_iv_vars)} 个")

if len(high_iv_vars) > 0:
    final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
        col_list=high_iv_vars,
        threshold=0.6
    )
    print(f"相关性去重后: {len(final_vars)} 个")
else:
    final_vars = xg_cols  # 如果没有高 IV 变量，使用 XGBoost 选择的变量

# 8. 模型训练
print("\n=== 模型训练 ===")

# 使用最终选定的变量
selected_vars = final_vars if len(final_vars) > 0 else xg_cols

X = data_woe[selected_vars]
y = data_woe[yh.target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 训练逻辑回归模型
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # 处理类别不平衡
)
model.fit(X_train, y_train)

# 9. 模型评估
print("\n=== 模型评估 ===")

# 预测
y_pred = model.predict_proba(X_test)[:, 1]

# 计算 AUC
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}")

# 计算 KS
ks_value = yh.me_module.model_ks(y_test, y_pred)
print(f"KS: {ks_value:.4f}")

# 交叉验证
print("\n交叉验证:")
yh.me_module.cross_verify(
    x=X, y=y,
    estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    fold=5,
    scoring='roc_auc'
)

# 绘制 ROC 曲线
yh.me_module.plot_roc(y_test, y_pred)

# 绘制 KS 曲线
yh.me_module.plot_model_ks(y_test, y_pred)

# 10. 评分卡实现
print("\n=== 评分卡实现 ===")

# 计算刻度参数
A, B, base_score = yh.si_module.cal_scale(
    score=600,    # 基础分 600 分对应 50:1 odds
    odds=50,
    PDO=20,
    model=model
)

print(f"A = {A:.2f}")
print(f"B = {B:.2f}")
print(f"基础分 = {base_score:.2f}")

# 计算测试集分数
test_scores = pd.DataFrame({
    'score': base_score + X_test.dot(model.coef_[0]) * B,
    'dlq_flag': y_test.values
})

print(f"\n分数统计:")
print(test_scores['score'].describe())

# 11. Cutoff 验证
print("\n=== Cutoff 验证 ===")

# 测试不同的 cutoff
cutoffs = [500, 550, 600, 650, 700]

for cutoff in cutoffs:
    matrix = yh.si_module.rule_verify(
        df=test_scores,
        col_score='score',
        target=yh.target,
        cutoff=cutoff
    )
    pass_rate = (test_scores['score'] >= cutoff).mean()
    print(f"\nCutoff = {cutoff}:")
    print(f"  通过率: {pass_rate:.2%}")
    print(f"  混淆矩阵:\n{matrix}")

# 12. 模型监控（模拟）
print("\n=== 模型监控 ===")

# 计算训练集和测试集的 PSI
train_scores = pd.DataFrame({
    'score': base_score + X_train.dot(model.coef_[0]) * B
})

# 合并数据用于 PSI 计算
train_data = X_train.copy()
train_data['score'] = train_scores['score']

test_data = X_test.copy()
test_data['score'] = test_scores['score']

try:
    psi_df = yh.sm_module.score_psi(
        df1=train_data,
        df2=test_data,
        id_col=None,
        score_col='score',
        x=base_score - 100,
        y=base_score + 100,
        step=20
    )
    total_psi = psi_df['PSI'].sum()
    print(f"训练集 vs 测试集 PSI: {total_psi:.4f}")

    if total_psi < 0.1:
        print("✓ 模型稳定")
    elif total_psi < 0.2:
        print("⚠ 模型轻微变化，建议监控")
    else:
        print("✗ 模型显著变化，需要重新训练")
except Exception as e:
    print(f"PSI 计算失败: {e}")

print("\n=== 建模完成 ===")
```

## 高级分箱策略

### 自定义分箱边界

```python
from yihuier import Yihuier

yh = Yihuier(data, target='dlq_flag')

# 定义自定义分箱边界
custom_bins = {
    'age': [0, 25, 35, 45, 55, 100],
    'income': [0, 5000, 10000, 20000, 50000, 100000]
}

# 应用自定义分箱
for var, bins in custom_bins.items():
    # 手动创建分箱
    data[f'{var}_bin'] = pd.cut(data[var], bins=bins, include_lowest=True)

    # 计算每个箱的 WOE 和 IV
    # ... 继续建模流程
```

### 单调性分箱

```python
# 对有业务含义的变量使用单调性分箱
monotonic_vars = ['age', 'income', 'employment_length']

for var in monotonic_vars:
    try:
        bin_df, iv_value = yh.binning_module.binning_num(
            col_list=[var],
            max_bin=5,
            method='monotonic'  # 确保单调性
        )
        print(f"{var} IV: {iv_value[0]:.4f}")
    except Exception as e:
        print(f"{var} 单调性分箱失败: {e}")
```

## 处理类别不平衡

### 场景：正样本占比 < 5%

```python
# 检查目标变量分布
target_dist = data['dlq_flag'].value_counts(normalize=True)
print(f"正样本占比: {target_dist.get(1, 0):.2%}")

if target_dist.get(1, 0) < 0.05:
    print("检测到严重类别不平衡，采用以下策略:")

    # 策略 1: 使用类别权重
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = dict(enumerate(class_weights))

    model = LogisticRegression(
        max_iter=1000,
        class_weight=weight_dict
    )

    # 策略 2: 调整分箱数（减少）
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=vars,
        max_bin=3,  # 减少分箱数
        method='freq'
    )

    # 策略 3: 使用更严格的 cutoff
    # 在评分卡实现阶段调整 cutoff
```

## 时间序列验证

### 场景：按时间划分数据

```python
# 假设数据有日期列
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# 按时间划分
train_data = data[data['date'] < '2024-01-01']
test_data = data[data['date'] >= '2024-01-01']

print(f"训练集时间范围: {train_data['date'].min()} 到 {train_data['date'].max()}")
print(f"测试集时间范围: {test_data['date'].min()} 到 {test_data['date'].max()}")

# 分别建模
yh_train = Yihuier(train_data, target='dlq_flag')
# ... 完整建模流程 ...

# 在测试集上验证
yh_test = Yihuier(test_data, target='dlq_flag')
# ... 评估模型性能 ...

# 时间序列 PSI
psi_df = yh.sm_module.score_psi(
    df1=train_data,
    df2=test_data,
    id_col='customer_id',
    score_col='score',
    x=300, y=800, step=25
)
```

## 特征工程集成

### 派生新特征

```python
# 创建派生特征
def create_features(data):
    """创建派生特征"""
    df = data.copy()

    # 比率特征
    df['debt_to_income'] = df['total_debt'] / df['income'].replace(0, np.nan)
    df['payment_to_income'] = df['monthly_payment'] / df['income'].replace(0, np.nan)
    df['credit_utilization'] = df['credit_balance'] / df['credit_limit'].replace(0, np.nan)

    # 差值特征
    df['income_minus_payment'] = df['income'] - df['monthly_payment']
    df['limit_minus_balance'] = df['credit_limit'] - df['credit_balance']

    # 聚合特征
    df['total_debt_ratio'] = df[['debt_to_income', 'payment_to_income']].mean(axis=1)

    # 处理无穷值
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# 应用特征工程
data_enhanced = create_features(data)

# 使用增强后的数据建模
yh = Yihuier(data_enhanced, target='dlq_flag')
```

## 多模型比较

### 场景：比较不同算法

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# 准备数据
yh = Yihuier(data, target='dlq_flag')
# ... 预处理和分箱 ...
data_woe = yh.binning_module.woe_transform()

feature_cols = [col for col in data_woe.columns if col != yh.target]
X = data_woe[feature_cols]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

# 训练和评估
results = []

for name, model in models.items():
    print(f"\n训练 {name}...")

    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred > 0.5).astype(int)

    # 评估
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_class)
    ks = yh.me_module.model_ks(y_test, y_pred)

    results.append({
        'model': name,
        'auc': auc,
        'accuracy': acc,
        'ks': ks
    })

    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"KS: {ks:.4f}")

# 结果汇总
results_df = pd.DataFrame(results)
print("\n=== 模型比较 ===")
print(results_df.sort_values('auc', ascending=False))
```

## 自动化建模流程

### 封装建模函数

```python
def auto_scorecard(
    data,
    target='dlq_flag',
    num_features=10,
    test_size=0.3,
    random_state=42
):
    """
    自动化评分卡建模流程

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    target : str
        目标变量名
    num_features : int
        选择的特征数量
    test_size : float
        测试集比例
    random_state : int
        随机种子

    Returns:
    --------
    dict : 包含模型和评估结果的字典
    """
    print(f"\n{'='*50}")
    print(f"自动化评分卡建模")
    print(f"{'='*50}\n")

    # 1. 初始化
    yh = Yihuier(data, target=target)

    # 2. 数据预处理
    print("[1/7] 数据预处理...")
    numeric_vars = yh.get_numeric_variables()
    yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type='0')
    yh.data = yh.dp_module.target_missing_delete()

    # 3. 变量分箱
    print("[2/7] 变量分箱...")
    try:
        bin_df, iv_value = yh.binning_module.binning_num(
            col_list=numeric_vars,
            max_bin=5,
            method='ChiMerge'
        )
    except:
        bin_df, iv_value = yh.binning_module.binning_num(
            col_list=numeric_vars,
            max_bin=5,
            method='freq'
        )

    # 4. WOE 转换
    print("[3/7] WOE 转换...")
    yh.binning_module.woe_df_concat()
    data_woe = yh.binning_module.woe_transform()

    # 5. 变量选择
    print("[4/7] 变量选择...")
    feature_cols = [col for col in data_woe.columns if col != target]
    _, _, selected_vars = yh.var_select_module.select_xgboost(
        col_list=feature_cols,
        imp_num=num_features
    )

    # 6. 模型训练
    print("[5/7] 模型训练...")
    X = data_woe[selected_vars]
    y = data_woe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # 7. 模型评估
    print("[6/7] 模型评估...")
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    ks = yh.me_module.model_ks(y_test, y_pred)

    # 8. 评分卡实现
    print("[7/7] 评分卡实现...")
    A, B, base_score = yh.si_module.cal_scale(
        score=600, odds=50, PDO=20, model=model
    )

    # 返回结果
    return {
        'yh': yh,
        'model': model,
        'features': selected_vars,
        'auc': auc,
        'ks': ks,
        'A': A,
        'B': B,
        'base_score': base_score,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# 使用自动化流程
result = auto_scorecard(
    data=data,
    target='dlq_flag',
    num_features=10
)

print(f"\n建模完成!")
print(f"AUC: {result['auc']:.4f}")
print(f"KS: {result['ks']:.4f}")
print(f"选中变量: {len(result['features'])} 个")
```

## 批量建模

### 场景：为多个产品建立评分卡

```python
products = {
    'credit_card': 'credit_card_data.csv',
    'personal_loan': 'personal_loan_data.csv',
    'mortgage': 'mortgage_data.csv'
}

models = {}

for product, filename in products.items():
    print(f"\n{'='*50}")
    print(f"产品: {product}")
    print(f"{'='*50}")

    # 加载数据
    data = pd.read_csv(filename)

    # 自动化建模
    result = auto_scorecard(
        data=data,
        target='dlq_flag',
        num_features=10
    )

    # 保存模型
    models[product] = result

    print(f"{product} AUC: {result['auc']:.4f}")

# 比较各产品模型性能
print("\n=== 产品模型比较 ===")
for product, result in models.items():
    print(f"{product}: AUC={result['auc']:.4f}, KS={result['ks']:.4f}")
```

## 模型监控和重训

### 场景：定期监控模型性能

```python
import pandas as pd
from datetime import datetime, timedelta

def monitor_model(yh, model, new_data, A, B, base_score):
    """
    监控模型性能

    Parameters:
    -----------
    yh : Yihuier
        Yihuier 实例
    model : sklearn model
        训练好的模型
    new_data : pd.DataFrame
        新数据
    A, B, base_score : float
        评分卡参数

    Returns:
    --------
    dict : 监控结果
    """
    # 计算分数
    X_new = new_data[model.feature_names_in_]
    scores = base_score + X_new.dot(model.coef_[0]) * B

    # PSI 计算
    # 假设有基准数据
    baseline_scores = yh.binning_module.data_woe  # 建模时的数据

    # ... 计算 PSI ...

    # 返回监控结果
    return {
        'date': datetime.now(),
        'score_mean': scores.mean(),
        'score_std': scores.std(),
        'psi': 0.05,  # 示例值
        'status': 'stable'  # stable, warning, critical
    }

# 定期监控
def schedule_monitoring():
    """定期监控模型"""
    # 每周运行一次
    while True:
        # 获取本周数据
        week_data = get_week_data()

        # 监控
        result = monitor_model(yh, model, week_data, A, B, base_score)

        # 记录结果
        log_monitoring_result(result)

        # 判断是否需要重训
        if result['psi'] > 0.25:
            print("警告: PSI 过高，需要重新训练模型")
            retrain_model()

        # 等待下周
        time.sleep(7 * 24 * 3600)
```

## 最佳实践总结

1. **数据质量优先**
   - 充分的探索性数据分析
   - 仔细处理缺失值和异常值
   - 验证业务逻辑合理性

2. **分箱策略**
   - 优先使用 ChiMerge
   - 对有业务含义的变量使用单调性分箱
   - 控制分箱数（5-7 箱）

3. **变量选择**
   - 结合多种方法（XGBoost + 相关性）
   - 控制变量数量（10-15 个）
   - 保留业务可解释性

4. **模型评估**
   - 使用多种指标（AUC, KS, PSI）
   - 时间序列验证
   - 交叉验证

5. **监控和维护**
   - 定期监控 PSI
   - 设置自动重训触发器
   - 记录模型版本

## 参考资源

- [最佳实践](best-practices.md) - 行业最佳实践
- [API 文档](api.md) - 完整 API 参考
- [系统架构](/develop/architecture) - 架构设计
