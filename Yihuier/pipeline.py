# 在 main.py 或者 __init__.py 中创建主类 Yihui
import warnings
import pandas as pd
import numpy as np


class PipelineModule:

    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance

    def product_test(self):
        # 打印变量
        print("Categorical Variables:", self.yihuier_instance.get_categorical_variables())
        print("Numeric Variables:", self.yihuier_instance.get_numeric_variables())
        print("Date Variables:", self.yihuier_instance.get_date_variables())
        # 日期变量转成二分类变量
        self.yihuier_instance.data = self.yihuier_instance.dp_module.date_var_shift_binary(self.yihuier_instance.get_date_variables(), replace=True)
        # 填充类别型变量空值
        self.yihuier_instance.dp_module.fillna_cate_var(self.yihuier_instance.get_categorical_variables(), 'class', 'unkonwn')
        # 打印类别型变量的违约率
        # self.yihuier_instance.eda_module.plot_default_cate(self.yihuier_instance.get_categorical_variables(),plt_size=(100,100),plt_num=3,x=3,y=1)
        # 删除类别型变量
        # self.yihuier_instance.data = self.yihuier_instance.data.drop(self.yihuier_instance.get_categorical_variables(),axis = 1)

        # 填充数值型变量空值
        self.yihuier_instance.data = self.yihuier_instance.dp_module.fillna_num_var(self.yihuier_instance.get_numeric_variables(), fill_type='class', fill_class_num=-999)
        # 删除数值型变量
        self.yihuier_instance.data = self.yihuier_instance.dp_module.delete_missing_var(threshold=0.01)
        # 展示空值分布
        # self.yihuier_instance.dp_module.plot_bar_missing_var()

        # 对每个数值变量做分箱，并计算IV
        iv_list = []
        col_list = []
        for col in self.yihuier_instance.get_numeric_variables():
            try:
                _, iv_value = self.yihuier_instance.binning_module.binning_num([col], 10, 0, 'freq')
                # _, iv_value = self.yihuier_instance.binning_module.binning_num([col], 10, 0, 'ChiMerge')

                # 处理 iv_value 或其他逻辑
                col_list.append(col)
                iv_list.append(iv_value[0])
            except Exception as e:
                col_list.append(col)
                iv_list.append('error')

                print(f"Error processing variable {col}: {str(e)}")
                continue  # 如果出现异常，跳过当前变量，继续下一个变量
        iv = pd.DataFrame({'col': col_list, 'iv': iv_list})
        xg_fea_imp, _, _ = self.yihuier_instance.var_select_module.select_xgboost(self.yihuier_instance.get_numeric_variables())
        rf_fea_imp, _ = self.yihuier_instance.var_select_module.select_rf(self.yihuier_instance.get_numeric_variables())

        # 合并DataFrame
        fea_csv = iv.merge(xg_fea_imp, on='col', how='outer')
        fea_csv = fea_csv.merge(rf_fea_imp, on='col', how='outer', suffixes=('_xgb', '_rf'))
        print(fea_csv)
        return fea_csv


# 示例
# if __name__ == "__main__":
#     from Yihuier.yihuier import Yihuier
#
#     # ban FutureWarning
#     warnings.filterwarnings('ignore')
#
#     # generate data.csv
#     with open("Data/result_hebing.csv", "r") as f:
#         data = pd.read_csv(f)
#     print(data.head())
#
#
#     df = data.copy()
#     yi = Yihuier(df, 'dlq')
#     yi.pipeline_module.product_test()