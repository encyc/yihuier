from Yihuier.binning import BinningModule
from Yihuier.eda import EDAModule
from Yihuier.data_processing import DataProcessingModule
from Yihuier.cluster import ClusterMuodule
from Yihuier.var_selelct import VarSelectModule
from Yihuier.model_evaluation import ModelEvaluationModule
from Yihuier.scorecard_implement import ScorecardImplementModule
from Yihuier.pipeline import PipelineModule

import numpy as np
import pandas as pd

# 导入相关库


class Yihuier:
    def __init__(self, data, target=None):
        self.data = data
        self.target = target
        self.eda_module = EDAModule(self)
        self.dp_module = DataProcessingModule(self)
        self.cluster_module = ClusterMuodule(self)
        self.binning_module = BinningModule(self)
        self.var_select_module = VarSelectModule(self)
        self.me_module = ModelEvaluationModule(self)
        self.si_module = ScorecardImplementModule(self)
        self.pipeline_module = PipelineModule(self)
        # 其他模块的初始化...

    # 提取字符型变量的名字并返回一个list
    def get_categorical_variables(self):
        # cate_vars = list(self.data.select_dtypes(include='object').columns)
        cate_vars = self.data.select_dtypes(include=[np.object, 'category']).columns.tolist()

        if self.target in cate_vars:
            cate_vars.remove(self.target)
        return cate_vars

    # 提取数值型变量的名字并返回一个list
    def get_numeric_variables(self):
        # num_vars = list(self.data.select_dtypes(exclude='object').columns)
        num_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # 剔除日期型变量
        date_vars = self.get_date_variables()
        num_vars = [var for var in num_vars if var not in date_vars]

        if self.target in num_vars:
            num_vars.remove(self.target)
        return num_vars

    def get_date_variables(self):
        date_vars = []

        for col in self.data.columns:
            if self.__is_numeric_date_format(col):
                date_vars.append(col)

        return date_vars

    def __is_numeric_date_format(self, col):
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
