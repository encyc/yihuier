from typing import Optional, List, Union
from yihuier.binning import BinningModule
from yihuier.eda import EDAModule
from yihuier.data_processing import DataProcessingModule
from yihuier.cluster import ClusterModule
from yihuier.var_select import VarSelectModule
from yihuier.model_evaluation import ModelEvaluationModule
from yihuier.scorecard_implement import ScorecardImplementModule
from yihuier.pipeline import PipelineModule

import numpy as np
import pandas as pd


class Yihuier:
    """评分卡建模主类

    提供统一的接口来进行信用评分卡建模，包括 EDA、数据处理、
    分箱、变量选择、模型评估和监控等功能。

    Args:
        data: 输入数据集（pandas DataFrame）
        target: 目标变量列名，默认为 None

    Attributes:
        data: 数据集
        target: 目标变量名
        eda_module: 探索性数据分析模块
        dp_module: 数据处理模块
        cluster_module: 聚类分析模块
        binning_module: 分箱模块
        var_select_module: 变量选择模块
        me_module: 模型评估模块
        si_module: 评分卡实现模块
        pipeline_module: 流水线模块
    """

    def __init__(self, data: pd.DataFrame, target: Optional[str] = None) -> None:
        """初始化 Yihuier 实例

        Args:
            data: 输入数据集
            target: 目标变量列名
        """
        self.data: pd.DataFrame = data
        self.target: Optional[str] = target
        self.eda_module: EDAModule = EDAModule(self)
        self.dp_module: DataProcessingModule = DataProcessingModule(self)
        self.cluster_module: ClusterModule = ClusterModule(self)
        self.binning_module: BinningModule = BinningModule(self)
        self.var_select_module: VarSelectModule = VarSelectModule(self)
        self.me_module: ModelEvaluationModule = ModelEvaluationModule(self)
        self.si_module: ScorecardImplementModule = ScorecardImplementModule(self)
        self.pipeline_module: PipelineModule = PipelineModule(self)

    def get_categorical_variables(self) -> List[str]:
        """提取字符型/类别型变量的名字并返回列表

        Returns:
            类别型变量名称列表
        """
        cate_vars = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.target in cate_vars:
            cate_vars.remove(self.target)
        return cate_vars

    def get_numeric_variables(self) -> List[str]:
        """提取数值型变量的名字并返回列表

        Returns:
            数值型变量名称列表
        """
        num_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # 剔除日期型变量
        date_vars = self.get_date_variables()
        num_vars = [var for var in num_vars if var not in date_vars]

        if self.target in num_vars:
            num_vars.remove(self.target)
        return num_vars

    def get_date_variables(self) -> List[str]:
        """提取日期型变量的名字并返回列表

        Returns:
            日期型变量名称列表
        """
        date_vars = []

        for col in self.data.columns:
            if self.__is_numeric_date_format(col):
                date_vars.append(col)

        return date_vars

    def __is_numeric_date_format(self, col: str) -> bool:
        """检查列是否为日期格式

        Args:
            col: 列名

        Returns:
            是否为日期格式
        """
        if self.data[col].dropna().empty:
            return False

        date_formats = ['%Y%m%d', '%Y%m', '%m%d', '%Y%m%d%H%M%S', '%Y-%m-%d', '%Y-%m', '%m-%d']

        for date_format in date_formats:
            try:
                # 尝试转换为日期
                pd.to_datetime(self.data[col], format=date_format, errors='raise')
                return True
            except (ValueError, TypeError):
                continue

        return False
