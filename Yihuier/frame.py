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

class EDAModule:
    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance
        self.data = self.yihuier_instance.data.copy()
        self.variables = self.yihuier_instance.data.columns
        self.category_variables = self.yihuier_instance.get_categorical_variables()
        self.numeric_variables = self.yihuier_instance.get_numeric_variables()


class DataProcessingModule:
    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance


class BinningModule:

    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance
        self.bin_df = None
        self.woe_list = None
        self.iv_df = None
        self.ks_df = None
        self.woe_result_df = None
        self.data_woe = None


class VarSelectModule:

    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance
        self.xg_fea_imp = None
        self.rf_fea_imp = None
        self.selected_var = None


class ModelEvaluationModule:

    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance
        self.y_label = None
        self.y_pred = None
