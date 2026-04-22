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
| `EDAModule` | 探索性数据分析 | [docs/01-eda.md](docs/01-eda.md) |
| `DataProcessingModule` | 数据预处理 | [docs/02-data-processing.md](docs/02-data-processing.md) |
| `BinningModule` | 变量分箱 | [docs/03-binning.md](docs/03-binning.md) |
| `VarSelectModule` | 变量选择 | [docs/04-var-select.md](docs/04-var-select.md) |
| `ModelEvaluationModule` | 模型评估 | [docs/05-model-evaluation.md](docs/05-model-evaluation.md) |
| `ScorecardImplementModule` | 评分卡实现 | [docs/06-scorecard-implement.md](docs/06-scorecard-implement.md) |
| `ScorecardMonitorModule` | 评分卡监控 | [docs/07-scorecard-monitor.md](docs/07-scorecard-monitor.md) |
| `ClusterModule` | 聚类分析 | [docs/08-cluster.md](docs/08-cluster.md) |
| `PipelineModule` | 流水线 | [docs/09-pipeline.md](docs/09-pipeline.md) |

## 详细文档

每个模块的详细使用说明和API文档，请查看 [docs/](docs/) 目录。

## 开发

```bash
# 克隆项目
git clone https://github.com/your-username/yihuier.git
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
