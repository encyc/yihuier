# Yihuier

一会儿轻松解决信用评分卡建模

基于 [Scorecard--Function](https://github.com/taenggu0309/Scorecard--Function) 重构的面向对象版本。

## 特性

- **面向对象设计** - 统一的 Yihuier 类管理数据和状态
- **完整建模流程** - EDA → 数据处理 → 分箱 → 变量选择 → 模型评估 → 评分卡实现 → 监控
- **模块化架构** - 9个独立模块，职责清晰
- **类型提示** - 完整的类型注解，更好的IDE支持

## 快速开始

### 安装

```bash
# 使用 pip
pip install yihuier

# 或使用 uv
uv pip install yihuier
```

### 基础使用

```python
import pandas as pd
from yihuier import Yihuier

# 加载数据
data = pd.read_csv('data.csv')
yh = Yihuier(data, target='dlq_flag')

# 数据预处理
data_clean = yh.dp_module.delete_missing_var(threshold=0.15)

# 变量分箱
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=['v1', 'v2', 'v3'],
    max_bin=5,
    method='ChiMerge'
)

# WOE转换
woe_df = yh.binning_module.woe_df_concat()
data_woe = yh.binning_module.woe_transform()

# 变量选择
xg_imp, xg_rank, xg_cols = yh.var_select_module.select_xgboost(
    col_list=data_woe.drop(['dlq_flag'], axis=1).columns.tolist(),
    imp_num=10
)

# 模型训练
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data_woe[xg_cols], data_woe['dlq_flag'])

# 模型评估
y_pred = model.predict_proba(x_test)[:, 1]
yh.me_module.plot_roc(y_test, y_pred)
yh.me_module.plot_model_ks(y_test, y_pred)
```

## 模块概览

| 模块 | 功能 | 文档 |
|------|------|------|
| `EDAModule` | 探索性数据分析 | [📖 EDA 模块](https://encyc.github.io/yihuier/guide/modules/eda.html) |
| `DataProcessingModule` | 数据预处理 | [📖 数据预处理](https://encyc.github.io/yihuier/guide/modules/data-processing.html) |
| `BinningModule` | 变量分箱 | [📖 分箱模块](https://encyc.github.io/yihuier/guide/modules/binning.html) |
| `VarSelectModule` | 变量选择 | [📖 变量选择](https://encyc.github.io/yihuier/guide/modules/var-select.html) |
| `ModelEvaluationModule` | 模型评估 | [📖 模型评估](https://encyc.github.io/yihuier/guide/modules/model-evaluation.html) |
| `ScorecardImplementModule` | 评分卡实现 | [📖 评分卡实现](https://encyc.github.io/yihuier/guide/modules/scorecard-implement.html) |
| `ScorecardMonitorModule` | 评分卡监控 | [📖 评分卡监控](https://encyc.github.io/yihuier/guide/modules/scorecard-monitor.html) |
| `ClusterModule` | 聚类分析 | [📖 聚类模块](https://encyc.github.io/yihuier/guide/modules/cluster.html) |
| `PipelineModule` | 流水线 | [📖 流水线模块](https://encyc.github.io/yihuier/guide/modules/pipeline.html) |

## 📚 完整文档

- **[在线文档](https://encyc.github.io/yihuier/)** - VitePress 部署的完整文档网站
- **[快速开始](https://encyc.github.io/yihuier/guide/quick-start)** - 5分钟快速上手
- **[API 参考](https://encyc.github.io/yihuier/guide/api)** - 完整的 API 文档
- **[最佳实践](https://encyc.github.io/yihuier/guide/best-practices)** - 行业最佳实践指南
- **[示例集合](https://encyc.github.io/yihuier/guide/examples)** - 常用场景代码示例

## 开发

```bash
# 克隆项目
git clone https://github.com/ency/yihuier.git
cd yihuier

# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码格式化
ruff format yihuier/
ruff check yihuier/
```

## 许可证

MIT License

## 致谢

- 原项目: [Scorecard--Function](https://github.com/taenggu0309/Scorecard--Function)
- 相关文章: [知乎专栏](https://zhuanlan.zhihu.com/p/675830391)
