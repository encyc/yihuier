# 评分卡监控模块 (ScorecardMonitorModule)

## 概述

`ScorecardMonitorModule` 提供评分卡上线后的监控功能，包括PSI计算、评分对比、变量稳定性分析等。

### 主要功能

- **PSI计算** - 评估评分分布稳定性
- **评分对比** - 建模样本vs上线样本
- **变量稳定性** - 监控各变量的表现变化
- **变量偏移** - 分析变量得分占比变化

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

# 建模样本（训练时的数据）
model_data = pd.read_csv('model_data.csv')
yh_model = Yihuier(model_data, target='dlq_flag')

# 上线样本（当前业务数据）
online_data = pd.read_csv('online_data.csv')
yh_online = Yihuier(online_data, target='dlq_flag')

sm = yh_model.si_module  # 使用建模样本的监控模块
```

---

## API 参考

### 1. score_psi() - 计算评分PSI

计算群体稳定性指数（PSI），评估评分分布的变化。

#### 语法

```python
sm.score_psi(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col: str,
    score_col: str,
    x: float,
    y: float,
    step: Optional[float] = None
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df1` | pd.DataFrame | 必填 | 建模样本数据 |
| `df2` | pd.DataFrame | 必填 | 上线样本数据 |
| `id_col` | str | 必填 | 用户ID字段名 |
| `score_col` | str | 必填 | 得分字段名 |
| `x` | float | 必填 | 区间左值 |
| `y` | float | 必填 | 区间右值 |
| `step` | float | None | 区间步长 |

#### PSI值解读

| PSI值 | 稳定性 | 建议 |
|--------|--------|------|
| < 0.1 | 稳定 | 无需 action |
| 0.1 - 0.25 | 轻微变化 | 监控即可 |
| > 0.25 | 显著变化 | 需要检查并重新训练模型 |

#### 使用示例

```python
# 计算PSI
psi_result = sm.score_psi(
    df1=model_data[['user_id', 'score']],
    df2=online_data[['user_id', 'score']],
    id_col='user_id',
    score_col='score',
    x=200,
    y=800,
    step=20
)

print("PSI分析结果:")
print(psi_result)

# 查看总PSI
total_psi = psi_result['PSI'].iloc[0]
print(f"\n总PSI值: {total_psi:.3f}")

# 评估稳定性
if total_psi < 0.1:
    print("✅ 评分分布稳定")
elif total_psi < 0.25:
    print("⚠️ 评分分布有轻微变化，建议监控")
else:
    print("❌评分分布显著变化，建议重新训练模型")
```

---

### 2. plot_score_compare() - 评分对比图

绘制建模样本和上线样本的评分分布对比图。

#### 语法

```python
sm.plot_score_compare(
    df: pd.DataFrame,
    plt_size: Optional[Tuple[int, int]] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | PSI结果表 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |

#### 使用示例

```python
# 使用PSI结果绘制对比图
psi_result = sm.score_psi(
    df1=model_data[['user_id', 'score']],
    df2=online_data[['user_id', 'score']],
    id_col='user_id',
    score_col='score',
    x=200,
    y=800,
    step=20
)

sm.plot_score_compare(
    df=psi_result,
    plt_size=(12, 6)
)
```

#### 输出说明

- 绿色柱子：建模样本各区间用户占比
- 粉色柱子：上线样本各区间用户占比
- 可直观对比两个样本的评分分布差异

---

### 3. var_stable() - 变量稳定性分析

分析单个变量的稳定性变化。

#### 语法

```python
sm.var_stable(
    score_result: pd.DataFrame,
    df: pd.DataFrame,
    var: str,
    id_col: str,
    score_col: str,
    bins: list
) -> pd.DataFrame
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `score_result` | pd.DataFrame | 必填 | 评分卡得分明细表 |
| `df` | pd.DataFrame | 必填 | 上线样本数据 |
| `var` | str | 必填 | 分析的变量名 |
| `id_col` | str | 必填 | 用户ID字段名 |
| `score_col` | str | 必填 | 得分字段名 |
| `bins` | list | 必填 | 变量分箱的区间 |

#### 使用示例

```python
# 获取评分卡得分明细表
score_result = sm.score_info(
    df=train_scored,
    score_col='total_score',
    target='dlq_flag',
    x=200,
    y=800,
    step=20
)

# 分析变量稳定性（以年龄为例）
var_stable_df = sm.var_stable(
    score_result=score_result,
    df=online_data,
    var='age',
    id_col='user_id',
    score_col='age_score',
    bins=[18, 25, 35, 45, 55, 65, 100]
)

print("年龄变量稳定性分析:")
print(var_stable_df)
```

---

### 4. plot_var_shift() - 变量偏移图

绘制变量得分占比随时间的变化趋势。

#### 语法

```python
sm.plot_var_shift(
    df: pd.DataFrame,
    day_col: str,
    score_col: str,
    plt_size: Optional[Tuple[int, int]] = None
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 变量得分数据 |
| `day_col` | str | 必填 | 日期字段名 |
| `score_col` | str | 必填 | 得分字段名 |
| `plt_size` | Tuple[int, int] | None | 图表尺寸 |

#### 使用示例

```python
# 准备数据：每个客户每天各变量的得分
# 需要包含：日期、客户ID、变量得分

# 示例数据格式：
# date       | user_id | var      | score
# 2024-01-01 | 001     | age      | 50
# 2024-01-01 | 001     | income   | 80
# ...

sm.plot_var_shift(
    df=var_score_data,
    day_col='date',
    score_col='score',
    plt_size=(12, 6)
)
```

#### 输出说明

- 展示不同得分段在客户中的占比随时间的变化
- 面积图和柱状图叠加，清晰展示趋势

---

## 完整监控流程

### 标准监控流程

```python
from yihuier import Yihuier
import pandas as pd

# 1. 准备数据
print("=== 准备数据 ===")

# 建模样本（基线）
model_data = pd.read_csv('model_data.csv')
model_data['score'] = calculate_score(model_data)  # 需实现评分函数

# 上线样本（当前）
online_data = pd.read_csv('online_data_202401.csv')
online_data['score'] = calculate_score(online_data)

# 2. PSI计算
print("\n=== PSI计算 ===")
psi_result = sm.score_psi(
    df1=model_data[['user_id', 'score']],
    df2=online_data[['user_id', 'score']],
    id_col='user_id',
    score_col='score',
    x=200,
    y=800,
    step=20
)

total_psi = psi_result['PSI'].iloc[0]
print(f"总PSI值: {total_psi:.3f}")

# 评估
if total_psi < 0.1:
    print("✅ 稳定")
elif total_psi < 0.25:
    print("⚠️ 轻微变化")
else:
    print("❌ 显著变化，需要重新训练")

# 3. 评分对比可视化
print("\n=== 评分对比 ===")
sm.plot_score_compare(
    df=psi_result,
    plt_size=(12, 6)
)

# 4. 变量稳定性监控
print("\n=== 变量稳定性监控 ===")

# 对每个关键变量进行稳定性分析
key_vars = ['age', 'income', 'debt_ratio', 'employment_years']

for var in key_vars:
    print(f"\n--- {var} ---")

    # 获取该变量的分箱区间
    var_bins = get_variable_bins(var)  # 需实现

    # 计算稳定性
    var_stable_df = sm.var_stable(
        score_result=score_result,
        df=online_data,
        var=var,
        id_col='user_id',
        score_col=f'{var}_score',
        bins=var_bins
    )

    print(var_stable_df)

    # 检查权重变化
    if '权重差距' in var_stable_df.columns:
        max_shift = var_stable_df['权重差距'].abs().max()
        if max_shift > 10:
            print(f"⚠️ {var} 变化较大: {max_shift:.1f}")

# 5. 生成监控报告
print("\n=== 监控报告 ===")

report = {
    'monitoring_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'model_version': 'v1.0',
    'psi_value': total_psi,
    'psi_status': 'stable' if total_psi < 0.1 else 'warning' if total_psi < 0.25 else 'critical',
    'online_samples': len(online_data),
    'avg_score': online_data['score'].mean(),
    'variables_monitored': len(key_vars)
}

print("监控报告:")
print(pd.Series(report))

# 保存报告
report_df = pd.DataFrame([report])
report_df.to_csv(f'output/monitoring_report_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
                  index=False)
```

---

## 定期监控脚本

### 自动化监控

```python
import schedule
import time
from datetime import datetime

def daily_monitoring():
    """每日监控任务"""

    print(f"开始监控: {datetime.now()}")

    # 1. 获取最新数据
    today = datetime.now().strftime('%Y%m%d')
    online_data = pd.read_csv(f'output/online_data_{today}.csv')
    online_data['score'] = calculate_score(online_data)

    # 2. 计算PSI
    psi_result = sm.score_psi(
        df1=model_data[['user_id', 'score']],
        df2=online_data[['user_id', 'score']],
        id_col='user_id',
        score_col='score',
        x=200,
        y=800,
        step=20
    )

    total_psi = psi_result['PSI'].iloc[0]

    # 3. 判断是否需要告警
    if total_psi > 0.25:
        send_alert(f"PSI值过高: {total_psi:.3f}，请检查模型！")

    # 4. 保存监控结果
    monitoring_log = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'psi': total_psi,
        'avg_score': online_data['score'].mean(),
        'sample_count': len(online_data)
    }

    log_df = pd.DataFrame([monitoring_log])
    log_df.to_csv('output/monitoring_log.csv',
                  mode='a',
                  header=False,
                  index=False)

    print(f"监控完成: PSI={total_psi:.3f}")

def send_alert(message):
    """发送告警"""
    # 实现邮件、短信或webhook告警
    print(f"🚨 告警: {message}")
    # import requests
    # requests.post(webhook_url, json={"text": message})

# 每天凌晨2点执行
schedule.every().day.at("02:00").do(daily_monitoring)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 监控指标

### 关键监控指标

```python
def calculate_monitoring_metrics(model_data, online_data):
    """计算监控指标"""

    metrics = {}

    # 1. PSI
    psi_result = sm.score_psi(
        df1=model_data[['user_id', 'score']],
        df2=online_data[['user_id', 'score']],
        id_col='user_id',
        score_col='score',
        x=200, y=800, step=20
    )
    metrics['psi'] = psi_result['PSI'].iloc[0]

    # 2. 平均分差异
    metrics['score_diff'] = online_data['score'].mean() - model_data['score'].mean()
    metrics['score_diff_pct'] = metrics['score_diff'] / model_data['score'].mean()

    # 3. 分数区间分布
    model_bins = pd.cut(model_data['score'], bins=5).value_counts(normalize=True)
    online_bins = pd.cut(online_data['score'], bins=5).value_counts(normalize=True)
    metrics['distribution_diff'] = (model_bins - online_bins).abs().sum()

    # 4. 样本量变化
    metrics['sample_ratio'] = len(online_data) / len(model_data)

    # 5. 坏样本率变化
    model_bad_rate = model_data['dlq_flag'].mean()
    online_bad_rate = online_data['dlq_flag'].mean()
    metrics['bad_rate_diff'] = online_bad_rate - model_bad_rate

    return pd.Series(metrics)

# 使用
metrics = calculate_monitoring_metrics(model_data, online_data)
print("监控指标:")
print(metrics)

# 判断
if metrics['psi'] > 0.25:
    print("❌ PSI过高，建议重新训练")
if abs(metrics['score_diff_pct']) > 0.1:
    print("⚠️ 平均分变化超过10%，请检查")
if metrics['bad_rate_diff'] > 0.05:
    print("❌ 坏样本率显著上升，请检查数据质量")
```

---

## 注意事项

### 1. 监控频率

```python
# 根据业务量确定监控频率

# 小业务（<1000笔/天）：每周监控
schedule.every().monday.at("09:00").do(weekly_monitoring)

# 中业务（1000-10000笔/天）：每日监控
schedule.every().day.at("02:00").do(daily_monitoring)

# 大业务（>10000笔/天）：每小时监控
schedule.every().hour.do(hourly_monitoring)
```

### 2. 数据版本管理

```python
# 保存不同版本的建模数据
import os

model_versions = {}

for file in os.listdir('data/'):
    if file.startswith('model_data_') and file.endswith('.csv'):
        version = file.split('_')[-1].replace('.csv', '')
        model_versions[version] = pd.read_csv(f'data/{file}')

# 比较不同版本
for version, data in model_versions.items():
    psi = calculate_psi(data, online_data)
    print(f"版本 {version}: PSI = {psi:.3f}")
```

### 3. 告警规则

```python
class MonitoringAlerts:
    """监控告警规则"""

    @staticmethod
    def check_psi(psi_value):
        """PSI告警"""
        if psi_value < 0.1:
            return "OK", "评分分布稳定"
        elif psi_value < 0.25:
            return "WARNING", "评分分布有轻微变化"
        else:
            return "CRITICAL", "评分分布显著变化，请重新训练"

    @staticmethod
    def check_score_diff(diff_pct):
        """分数差异告警"""
        if abs(diff_pct) < 0.05:
            return "OK", "分数正常"
        elif abs(diff_pct) < 0.1:
            return "WARNING", "分数有轻微变化"
        else:
            return "CRITICAL", "分数显著异常"

    @staticmethod
    def check_bad_rate(bad_rate_diff):
        """坏样本率告警"""
        if bad_rate_diff < 0.02:
            return "OK", "坏样本率正常"
        elif bad_rate_diff < 0.05:
            return "WARNING", "坏样本率轻微上升"
        else:
            return "CRITICAL", "坏样本率显著上升，请检查"

# 使用
alerts = MonitoringAlerts()

level, message = alerts.check_psi(total_psi)
print(f"[{level}] {message}")
```

---

## 常见问题

### Q1: PSI值过大怎么办？

```python
# PSI > 0.25时的处理步骤

# 1. 检查样本分布
print("建模样本分布:")
print(pd.cut(model_data['score'], bins=10).value_counts())

print("\n上线样本分布:")
print(pd.cut(online_data['score'], bins=10).value_counts())

# 2. 检查变量分布变化
for var in key_vars:
    model_dist = model_data[var].describe()
    online_dist = online_data[var].describe()

    print(f"\n{var}变化:")
    print(f"  均值: {model_dist['mean']:.1f} -> {online_dist['mean']:.1f}")
    print(f"  标准差: {model_dist['std']:.1f} -> {online_dist['std']:.1f}")

# 3. 考虑重新训练模型
# 如果客户群体发生了系统性变化，需要重新训练
```

### Q2: 如何设置监控阈值？

```python
# 根据业务容忍度设置阈值

# 宽松阈值（业务波动大）
PSI_WARNING = 0.3
PSI_CRITICAL = 0.5

# 标准阈值（推荐）
PSI_WARNING = 0.25
PSI_CRITICAL = 0.5

# 严格阈值（业务波动小）
PSI_WARNING = 0.15
PSI_CRITICAL = 0.25

# 使用
if total_psi > PSI_CRITICAL:
    trigger_model_retrain()
```

### Q3: 多模型监控？

```python
# 同时监控多个模型版本
models = {
    'v1.0': {'data': model_data_v1, 'score_col': 'score_v1'},
    'v1.1': {'data': model_data_v2, 'score_col': 'score_v2'},
    'v2.0': {'data': model_data_v3, 'score_col': 'score_v3'}
}

for version, config in models.items():
    psi = sm.score_psi(
        df1=config['data'][['user_id', config['score_col']]],
        df2=online_data[['user_id', config['score_col']]],
        id_col='user_id',
        score_col=config['score_col'],
        x=200, y=800, step=20
    )

    print(f"{version}: PSI = {psi['PSI'].iloc[0]:.3f}")
```

---

## 相关文档

- [评分卡实现模块](scorecard-implement.md) - 创建评分卡
- [模型评估模块](model-evaluation.md) - 评估模型效果
- [数据处理模块](data-processing.md) - 处理监控数据
