from Yihui.binning import BinningModule
from Yihui.eda import EDAModule
from Yihui.data_processing import DataProcessingModule
from Yihui.cluster import ClusterMuodule
from Yihui.var_selelct import VarSelectModule


# 导入相关库


class Yihui:
    def __init__(self, data, target=None):
        self.data = data
        self.target = target
        self.eda_module = EDAModule(self)
        self.dp_module = DataProcessingModule(self)
        self.cluster_module = ClusterMuodule(self)
        self.binning_module = BinningModule(self)
        self.var_select_module = VarSelectModule(self)
        # 其他模块的初始化...

    # 提取字符型变量的名字并返回一个list
    def get_categorical_variables(self):
        cate_vars = list(self.data.select_dtypes(include='object').columns)
        if self.data[self.target].dtype == 'object':
            cate_vars.remove(self.target)
        return cate_vars

    # 提取数值型变量的名字并返回一个list
    def get_numeric_variables(self):
        num_vars = list(self.data.select_dtypes(exclude='object').columns)
        if self.data[self.target].dtype != 'object':
            num_vars.remove(self.target)
        return num_vars

