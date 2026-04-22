# 流水线模块 (PipelineModule)

## 概述

`PipelineModule` 提供了一键式的数据分析和建模流程，适合快速评估第三方数据厂商的数据质量。

### 主要功能

- **自动变量识别** - 识别类别型、数值型、日期型变量
- **自动化数据处理** - 缺失值填充、变量删除
- **自动化分箱** - 变量分箱和IV值计算
- **特征重要性评估** - XGBoost和随机森林特征重要性

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

pipeline = yh.pipeline_module
```

---

## API 参考

### product_test() - 第三方数据评估

一键式评估第三方数据厂商的数据质量。

#### 语法

```python
pipeline.product_test() -> pd.DataFrame
```

#### 返回值

返回一个包含变量特征重要性的DataFrame：

| 列名 | 说明 |
|------|------|
| `col` | 变量名 |
| `iv` | IV值 |
| `imp_xgb` | XGBoost特征重要性 |
| `imp_rf` | 随机森林特征重要性 |

#### 执行流程

1. **变量识别**
   - 识别类别型、数值型、日期型变量

2. **日期变量处理**
   - 将日期变量转换为二进制变量

3. **类别型变量处理**
   - 缺失值填充为 'Unknown'

4. **数值型变量处理**
   - 缺失值填充为 -999
   - 删除高缺失率变量（>1%）

5. **变量分箱**
   - 对数值型变量进行等频分箱
   - 计算IV值

6. **特征重要性评估**
   - XGBoost特征重要性
   - 随机森林特征重要性

7. **结果合并**
   - 合并IV值和特征重要性

---

## 使用示例

### 完整流程

```python
from yihuier import Yihuier
import pandas as pd

# 1. 加载第三方数据
print("=== 加载数据 ===")
third_party_data = pd.read_csv('third_party_data.csv')
print(f"数据形状: {third_party_data.shape}")
print(f"数据预览:\n{third_party_data.head()}")

# 2. 初始化
yh = Yihuier(third_party_data, target='dlq_flag')

# 3. 执行自动化评估
print("\n=== 自动化评估 ===")
feature_importance = yh.pipeline_module.product_test()

# 4. 查看结果
print("\n=== 特征重要性 ===")
print(feature_importance)

# 5. 筛选高质量变量
print("\n=== 筛选高质量变量 ===")

# 条件1: IV值 > 0.02
high_iv = feature_importance[feature_importance['iv'] > 0.02]

# 条件2: XGBoost重要性 > 0（有预测能力）
high_xgb = feature_importance[feature_importance['imp_xgb'] > 0]

# 取交集
selected_vars = pd.merge(high_iv, high_xgb, on='col', how='inner')
print(f"满足条件的变量数: {len(selected_vars)}")
print(f"变量列表: {selected_vars['col'].tolist()}")

# 6. 保存结果
selected_vars.to_csv('output/selected_vars.csv', index=False)

# 7. 生成评估报告
print("\n=== 评估报告 ===")
report = {
    'data_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'total_vars': len(feature_importance),
    'valid_vars': len(selected_vars),
    'data_quality': 'Good' if len(selected_vars) > 5 else 'Poor'
}

print(pd.Series(report))
```

---

## 自定义流水线

### 创建自定义流程

```python
def custom_pipeline(yh, target_iv=0.1, target_imp=0.01):
    """自定义流水线"""

    # 1. 变量识别
    print("=== 1. 变量识别 ===")
    cate_vars = yh.get_categorical_variables()
    num_vars = yh.get_numeric_variables()
    date_vars = yh.get_date_variables()

    print(f"类别型变量: {len(cate_vars)}")
    print(f"数值型变量: {len(num_vars)}")
    print(f"日期型变量: {len(date_vars)}")

    # 2. 数据预处理
    print("\n=== 2. 数据预处理 ===")

    # 处理日期变量
    if date_vars:
        yh.data = yh.dp_module.date_var_shift_binary(
            col_list=date_vars,
            replace=False
        )
        print(f"日期变量已转换: {len(date_vars)}")

    # 处理类别型变量
    if cate_vars:
        yh.data = yh.dp_module.fillna_cate_var(
            col_list=cate_vars,
            fill_type='class',
            fill_str='Unknown'
        )
        print(f"类别型变量已填充: {len(cate_vars)}")

    # 处理数值型变量
    if num_vars:
        yh.data = yh.dp_module.fillna_num_var(
            col_list=num_vars,
            fill_type='class',
            fill_class_num=-999
        )
        print(f"数值型变量已填充: {len(num_vars)}")

    # 删除高缺失率变量
    yh.data = yh.dp_module.delete_missing_var(threshold=0.15)
    print(f"已删除高缺失率变量")

    # 3. 变量分箱
    print("\n=== 3. 变量分箱 ===")

    num_vars = yh.get_numeric_variables()
    iv_df = yh.binning_module.iv_num(
        col_list=num_vars,
        max_bin=5,
        method='freq'
    )

    print(f"分箱完成，变量数: {len(iv_df)}")
    print(f"平均IV值: {iv_df['iv'].mean():.3f}")

    # 4. 特征重要性
    print("\n=== 4. 特征重要性 ===")

    xg_imp, _, _ = yh.var_select_module.select_xgboost(
        col_list=num_vars,
        imp_num=len(num_vars)
    )

    rf_imp, _ = yh.var_select_module.select_rf(
        col_list=num_vars,
        imp_num=len(num_vars)
    )

    # 5. 结果合并
    print("\n=== 5. 结果合并 ===")

    result = iv_df.merge(xg_imp, on='col', how='outer')
    result = result.merge(rf_imp, on='col', how='outer', suffixes=('_xgb', '_rf'))

    # 填充缺失值
    result['iv'] = result['iv'].fillna(0)
    result['imp_xgb'] = result['imp_xgb'].fillna(0)
    result['imp_rf'] = result['imp_rf'].fillna(0)

    # 6. 变量筛选
    print("\n=== 6. 变量筛选 ===")

    selected = result[
        (result['iv'] > target_iv) &
        (result['imp_xgb'] > target_imp)
    ]

    print(f"筛选后变量数: {len(selected)}")

    return result, selected

# 使用
result, selected = custom_pipeline(yh, target_iv=0.05, target_imp=0.01)

print("\n筛选结果:")
print(selected)
```

---

## 批量评估

### 评估多个数据源

```python
# 多个第三方数据源
data_sources = {
    'data_provider_A': 'data/provider_a.csv',
    'data_provider_B': 'data/provider_b.csv',
    'data_provider_C': 'data/provider_c.csv'
}

results = {}

for provider, file_path in data_sources.items():
    print(f"\n{'='*50}")
    print(f"评估 {provider}")
    print(f"{'='*50}")

    # 加载数据
    data = pd.read_csv(file_path)
    yh = Yihuier(data, target='dlq_flag')

    # 执行评估
    try:
        result = yh.pipeline_module.product_test()
        results[provider] = result

        # 统计
        valid_vars = result[(result['iv'] > 0.02) &
                            (result['imp_xgb'] > 0)]

        print(f"总变量数: {len(result)}")
        print(f"有效变量数: {len(valid_vars)}")

    except Exception as e:
        print(f"评估失败: {e}")
        results[provider] = None

# 汇总结果
print("\n" + "="*50)
print("评估汇总")
print("="*50)

for provider, result in results.items():
    if result is not None:
        valid_vars = result[(result['iv'] > 0.02) &
                            (result['imp_xgb'] > 0)]

        print(f"\n{provider}:")
        print(f"  总变量: {len(result)}, 有效变量: {len(valid_vars)}")

        # 排序
        print(f"  Top 5 变量:")
        top5 = result.nlargest(5, 'iv')
        for _, row in top5.iterrows():
            print(f"    {row['col']}: IV={row['iv']:.3f}")
```

---

## 注意事项

### 1. 数据质量要求

```python
# 评估前检查数据质量

def check_data_quality(data, target):
    """检查数据质量"""

    checks = {}

    # 1. 样本量
    checks['sample_size'] = len(data)
    checks['sample_status'] = 'OK' if len(data) > 1000 else 'Small'

    # 2. 变量数量
    checks['var_count'] = len(data.columns) - 1
    checks['var_status'] = 'OK' if checks['var_count'] > 10 else 'Few'

    # 3. 目标变量分布
    target_rate = data[target].mean()
    checks['target_rate'] = target_rate
    checks['target_status'] = 'OK' if 0.01 < target_rate < 0.5 else 'Imbalanced'

    # 4. 缺失值
    missing_rate = data.isnull().mean().mean()
    checks['missing_rate'] = missing_rate
    checks['missing_status'] = 'OK' if missing_rate < 0.3 else 'High'

    return pd.Series(checks)

# 使用
quality = check_data_quality(yh.data, yh.target)
print(quality)
```

### 2. 处理异常情况

```python
# 添加异常处理

try:
    result = yh.pipeline_module.product_test()
except Exception as e:
    print(f"评估失败: {e}")

    # 尝试降级处理
    print("\n尝试降级处理...")

    # 1. 检查数据
    if yh.data[yh.target].isnull().any():
        print("删除目标变量缺失的样本")
        yh.data = yh.dp_module.target_missing_delete()

    # 2. 简单分箱
    num_vars = yh.get_numeric_variables()
    iv_df = yh.binning_module.iv_num(
        col_list=num_vars,
        max_bin=3,
        method='freq'
    )

    # 3. 简单评估
    xg_imp, _, _ = yh.var_select_module.select_xgboost(
        col_list=num_vars,
        imp_num=len(num_vars)
    )

    result = iv_df.merge(xg_imp, on='col', how='outer')
    print("降级处理完成")
```

### 3. 参数调整

```python
# 修改流水线参数

# 调整缺失值阈值
yh.data = yh.dp_module.delete_missing_var(threshold=0.2)  # 放宽到20%

# 调整分箱参数
iv_df = yh.binning_module.iv_num(
    col_list=num_vars,
    max_bin=3,      # 减少分箱数
    method='freq'
)

# 调整特征选择阈值
selected = result[result['imp_xgb'] > result['imp_xgb'].quantile(0.5)]
```

---

## 输出报告

### 生成HTML报告

```python
def generate_html_report(result, provider_name, output_path):
    """生成HTML评估报告"""

    html = f"""
    <html>
    <head>
        <title>{provider_name} 数据评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .good {{ color: green; }}
            .bad {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>{provider_name} 数据评估报告</h1>
        <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>总体情况</h2>
        <p>总变量数: {len(result)}</p>

        <h2>变量详情</h2>
        <table>
            <tr>
                <th>变量名</th>
                <th>IV值</th>
                <th>XGBoost重要性</th>
                <th>随机森林重要性</th>
            </tr>
    """

    # 添加变量行
    for _, row in result.iterrows():
        iv_class = 'good' if row['iv'] > 0.1 else 'bad' if row['iv'] < 0.02 else ''
        imp_class = 'good' if row['imp_xgb'] > 0.01 else ''

        html += f"""
            <tr>
                <td>{row['col']}</td>
                <td class="{iv_class}">{row['iv']:.3f}</td>
                <td class="{imp_class}">{row['imp_xgb']:.3f}</td>
                <td>{row['imp_rf']:.3f}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML报告已保存: {output_path}")

# 使用
generate_html_report(
    result=feature_importance,
    provider_name="XX数据公司",
    output_path="output/evaluation_report.html"
)
```

---

## 常见问题

### Q1: 评估时出现错误？

```python
# 常见错误及解决方案

# 错误1: 目标变量不存在
# 解决: 检查并设置目标变量
if yh.target is None:
    yh.target = 'dlq_flag'  # 设置正确的目标变量名

# 错误2: 变量全被删除
# 解决: 放宽缺失值阈值
yh.data = yh.dp_module.delete_missing_var(threshold=0.5)  # 提高到50%

# 错误3: 分箱失败
# 解决: 减少分箱数或使用等频分箱
iv_df = yh.binning_module.iv_num(
    col_list=num_vars,
    max_bin=3,       # 减少分箱数
    method='freq'    # 使用等频分箱
)
```

### Q2: 如何提高评估效率？

```python
# 对于大数据集，采样评估

if len(yh.data) > 100000:
    print("数据集过大，进行采样...")

    # 分层采样
    from sklearn.model_selection import train_test_split
    _, sample_data = train_test_split(
        yh.data,
        test_size=0.1,  # 采样10%
        stratify=yh.data[yh.target],
        random_state=42
    )

    yh_sample = Yihuier(sample_data, target=yh.target)
    result = yh_sample.pipeline_module.product_test()
```

### Q3: 结果对比不同数据源？

```python
# 合并多个数据源的评估结果

all_results = []

for provider, file_path in data_sources.items():
    data = pd.read_csv(file_path)
    yh = Yihuier(data, target='dlq_flag')

    result = yh.pipeline_module.product_test()
    result['provider'] = provider

    all_results.append(result)

# 合并
combined = pd.concat(all_results, ignore_index=True)

# 对比分析
pivot_iv = combined.pivot(index='col', columns='provider', values='iv')
print("不同数据源的IV值对比:")
print(pivot_iv)

# 找出各数据源的优势变量
for provider in data_sources.keys():
    provider_vars = combined[combined['provider'] == provider]
    top_vars = provider_vars.nlargest(5, 'iv')
    print(f"\n{provider} 优势变量:")
    print(top_vars[['col', 'iv']])
```

---

## 相关文档

- [数据处理模块](02-data-processing.md) - 流水线中的数据清洗
- [变量分箱模块](03-binning.md) - 流水线中的分箱操作
- [变量选择模块](04-var-select.md) - 流水线中的特征重要性评估
