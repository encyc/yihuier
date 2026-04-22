# 聚类分析模块 (ClusterModule)

## 概述

`ClusterModule` 提供了多种无监督聚类算法，用于客户分群、异常检测等场景。

### 主要功能

- **10种聚类算法** - 从K-Means到高斯混合模型
- **自动可视化** - 聚类结果自动绘图
- **灵活配置** - 支持参数调优

---

## 初始化

```python
from yihuier import Yihuier
import pandas as pd

data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

cluster = yh.cluster_module
```

---

## API 参考

### 支持的聚类算法

| 算法 | 方法名 | 适用场景 |
|------|--------|---------|
| K-Means | `cluster_KMeans` | 球形簇，大小相近 |
| DBSCAN | `cluster_DBSCAN` | 发现任意形状的簇 |
| 层次聚类 | `cluster_AgglomerativeClustering` | 树状结构，层级关系 |
| BIRCH | `cluster_Birch` | 大数据集，内存效率高 |
| 高斯混合 | `cluster_GaussianMixture` | 软聚类，概率分布 |
| 均值漂移 | `cluster_MeanShift` | 密度峰值，无需指定簇数 |
| 亲和力传播 | `cluster_AffinityPropagation` | 基于示例传递 |
| Mini-Batch K-Means | `cluster_MiniBatchKMeans` | 大数据集快速聚类 |
| OPTICS | `cluster_OPTICS` | 变密度聚类 |
| 光谱聚类 | `cluster_SpectralClustering` | 图聚类，非线性 |

---

### 1. cluster_KMeans() - K-Means聚类

最常用的聚类算法。

#### 语法

```python
cluster.cluster_KMeans(
    col_list: List[str],
    n_clusters: int = 2
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 用于聚类的变量（2个变量） |
| `n_clusters` | int | 2 | 聚类数量 |

#### 使用示例

```python
# 选择2个变量进行聚类
cluster.cluster_KMeans(
    col_list=['age', 'income'],
    n_clusters=3
)
```

⚠️ **注意**：目前版本只支持2个变量的可视化。

---

### 2. cluster_DBSCAN() - DBSCAN聚类

基于密度的聚类，可发现任意形状的簇。

#### 语法

```python
cluster.cluster_DBSCAN(
    col_list: List[str],
    eps: float = 0.30,
    min_samples: int = 9
) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `col_list` | List[str] | 必填 | 用于聚类的变量 |
| `eps` | float | 0.30 | 邻域半径 |
| `min_samples` | int | 9 | 最小样本数 |

#### 使用示例

```python
cluster.cluster_DBSCAN(
    col_list=['age', 'income'],
    eps=0.5,
    min_samples=10
)
```

---

### 3. cluster_AgglomerativeClustering() - 层次聚类

凝聚的层次聚类方法。

#### 语法

```python
cluster.cluster_AgglomerativeClustering(
    col_list: List[str],
    n_clusters: int = 2
) -> None
```

#### 使用示例

```python
cluster.cluster_AgglomerativeClustering(
    col_list=['age', 'debt_ratio'],
    n_clusters=4
)
```

---

### 4. cluster_GaussianMixture() - 高斯混合模型

基于概率的聚类，可计算样本属于各簇的概率。

#### 语法

```python
cluster.cluster_GaussianMixture(
    col_list: List[str],
    n_components: int = 2
) -> None
```

#### 使用示例

```python
cluster.cluster_GaussianMixture(
    col_list=['age', 'income'],
    n_components=3
)
```

---

## 完整聚类流程

### 客户分群示例

```python
from yihuier import Yihuier
import pandas as pd
import numpy as np

# 1. 准备数据
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 数据预处理
yh.data = yh.dp_module.delete_missing_var(threshold=0.15)

# 选择用于聚类的变量
cluster_vars = ['age', 'income', 'debt_ratio', 'employment_years']

# 标准化（重要）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(yh.data[cluster_vars])
data_scaled = pd.DataFrame(data_scaled, columns=cluster_vars)

# 2. K-Means聚类
print("=== K-Means聚类 ===")
yh_temp = Yihuier(data_scaled, target=None)  # 无目标变量

# 尝试不同的K值
inertias = []
K_range = range(2, 11)

for K in K_range:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)

# 绘制肘部图
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K值')
plt.ylabel('簇内误差平方和')
plt.title('肘部法则选择最优K值')
plt.show()

# 选择最优K（假设为4）
optimal_K = 4

# 可视化聚类结果（选择2个变量）
yh.cluster_module.cluster_KMeans(
    col_list=['age', 'income'],
    n_clusters=optimal_K
)

# 3. 对所有变量进行聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=optimal_K, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# 添加聚类标签
yh.data['cluster'] = clusters

# 4. 分析聚类结果
print("\n=== 聚类分析 ===")
for i in range(optimal_K):
    cluster_data = yh.data[yh.data['cluster'] == i]
    print(f"\n簇 {i}:")
    print(f"  样本数: {len(cluster_data)}")
    print(f"  平均年龄: {cluster_data['age'].mean():.1f}")
    print(f"  平均收入: {cluster_data['income'].mean():.0f}")
    print(f"  坏样本率: {cluster_data[yh.target].mean():.2%}")

# 5. 保存聚类结果
yh.data[['user_id', 'cluster']].to_csv('output/cluster_result.csv', index=False)
```

---

## 聚类算法选择

### 选择指南

```python
# 根据数据特点选择算法

def recommend_clustering_algorithm(data):
    """根据数据特点推荐聚类算法"""

    n_samples = len(data)
    n_features = data.shape[1]

    print(f"样本数: {n_samples}, 特征数: {n_features}")

    # 1. 大数据集（>10万样本）
    if n_samples > 100000:
        print("推荐: Mini-Batch K-Means 或 BIRCH")
        print("理由: 计算效率高，内存占用少")
        return ['cluster_MiniBatchKMeans', 'cluster_Birch']

    # 2. 小样本（<1000样本）
    elif n_samples < 1000:
        print("推荐: 层次聚类 或 高斯混合")
        print("理由: 对小样本效果好，可解释性强")
        return ['cluster_AgglomerativeClustering', 'cluster_GaussianMixture']

    # 3. 球形簇
    else:
        print("推荐: K-Means 或 DBSCAN")
        print("理由: 通用性强，适合大多数场景")
        return ['cluster_KMeans', 'cluster_DBSCAN']

# 使用
recommend_clustering_algorithm(yh.data[cluster_vars])
```

---

## 聚类评估

### 聚类质量评估

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(data, labels):
    """评估聚类质量"""

    metrics = {}

    # 1. 轮廓系数（-1到1，越大越好）
    metrics['silhouette'] = silhouette_score(data, labels)

    # 2. Calinski-Harabasz指数（越大越好）
    metrics['ch'] = calinski_harabasz_score(data, labels)

    # 3. Davies-Bouldin指数（越小越好）
    metrics['db'] = davies_bouldin_score(data, labels)

    return pd.Series(metrics)

# 评估不同K值的聚类效果
for K in range(2, 8):
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    metrics = evaluate_clustering(data_scaled, labels)
    print(f"\nK={K}:")
    print(f"  轮廓系数: {metrics['silhouette']:.3f}")
    print(f"  CH指数: {metrics['ch']:.1f}")
    print(f"  DB指数: {metrics['db']:.3f}")
```

---

## 注意事项

### 1. 变量选择

```python
# 聚类前应该选择有区分度的变量

# 计算变量变异系数
cv = lambda x: x.std() / x.mean()
variable_cv = yh.data[cluster_vars].apply(cv)

print("变量变异系数:")
print(variable_cv)

# 选择变异系数较大的变量
selected_vars = variable_cv[variable_cv > 0.3].index.tolist()
print(f"选择的变量: {selected_vars}")
```

### 2. 数据标准化

```python
# 聚类前必须标准化！

# 方法1: Z-score标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(yh.data[cluster_vars])

# 方法2: Min-Max标准化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(yh.data[cluster_vars])

# 方法3: Robust标准化（对异常值鲁棒）
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data_scaled = scaler.fit_transform(yh.data[cluster_vars])
```

### 3. 离群值处理

```python
# 聚类对离群值敏感，建议先处理

# 方法1: 删除离群值
from scipy import stats
z_scores = stats.zscore(yh.data[cluster_vars])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_clean = yh.data[filtered_entries]

# 方法2: 使用鲁棒聚类算法
# DBSCAN对离群值相对鲁棒
cluster.cluster_DBSCAN(
    col_list=cluster_vars[:2],
    eps=0.5,
    min_samples=10
)
```

---

## 常见问题

### Q1: 如何确定最优聚类数？

```python
# 方法1: 肘部法则（已展示）
# 方法2: 轮廓系数

silhouette_scores = []
K_range = range(2, 11)

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    silhouette_scores.append(score)

optimal_K = K_range[np.argmax(silhouette_scores)]
print(f"最优K值: {optimal_K}")

# 方法3: Gap统计量
from sklearn.cluster import KMeans
from gap_statistic import OptimalK

optimalK = OptimalK(parallel_backend='joblib')
optimalK(data_scaled, cluster_range=range(2, 11))
print(f"最优K值: {optimalK.k}")
```

### Q2: 聚类结果不稳定？

```python
# 聚类结果可能受初始值影响

# 解决方案1: 多次聚类取最优
best_score = -1
best_labels = None

for i in range(10):
    kmeans = KMeans(n_clusters=4, random_state=i)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)

    if score > best_score:
        best_score = score
        best_labels = labels

print(f"最优轮廓系数: {best_score:.3f}")

# 解决方案2: 使用层次聚类（结果稳定）
cluster.cluster_AgglomerativeClustering(
    col_list=['age', 'income'],
    n_clusters=4
)
```

### Q3: 如何解释聚类结果？

```python
# 分析各簇的特征
def analyze_clusters(data, cluster_col, target_col):
    """分析各簇的特征"""

    results = []

    for cluster_id in sorted(data[cluster_col].unique()):
        cluster_data = data[data[cluster_col] == cluster_id]

        result = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'bad_rate': cluster_data[target_col].mean()
        }

        # 添加各变量的均值
        for var in cluster_vars:
            result[f'{var}_mean'] = cluster_data[var].mean()

        results.append(result)

    return pd.DataFrame(results)

# 使用
cluster_analysis = analyze_clusters(yh.data, 'cluster', yh.target)
print(cluster_analysis)

# 给每个簇命名
cluster_names = {
    0: "低风险年轻客户",
    1: "高风险中年客户",
    2: "低风险老年客户",
    3: "中等风险客户"
}

cluster_analysis['cluster_name'] = cluster_analysis['cluster'].map(cluster_names)
print(cluster_analysis)
```

---

## 相关文档

- [EDA模块](01-eda.md) - 聚类前的数据分析
- [数据处理模块](02-data-processing.md) - 聚类前的数据清洗
- [变量分箱模块](03-binning.md) - 聚类变量的分箱处理
