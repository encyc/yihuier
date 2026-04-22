"""
Yihuier - 评分卡模型实现函数模块

轻松解决逻辑回归建模的 Python 工具包。

主要模块:
- Yihuier: 主类，提供统一的建模接口
- EDAModule: 探索性数据分析
- DataProcessingModule: 数据预处理
- ClusterModule: 聚类分析
- BinningModule: 变量分箱
- VarSelectModule: 变量选择
- ModelEvaluationModule: 模型评估
- ScorecardImplementModule: 评分卡实现
- ScorecardMonitorModule: 评分卡监控
- PipelineModule: 流水线

示例:
    >>> from yihuier import Yihuier
    >>> import pandas as pd
    >>> data = pd.read_csv('data.csv')
    >>> project = Yihuier(data, target='dlq_flag')
    >>> # 进行 EDA、数据处理、分箱等操作
"""

from yihuier.yihuier import Yihuier

__version__ = "0.1.0"
__all__ = ["Yihuier"]
