# 在 main.py 或者 __init__.py 中创建主类 Yihui
import warnings
import pandas as pd
import numpy as np
from Yihuier.yihuier import Yihuier

# 在主程序中使用 Yihui 类
if __name__ == "__main__":
    # ban FutureWarning
    warnings.filterwarnings('ignore')

    # generate data.csv
    with open("Data/data_yinlian.csv", "r") as f:
        # with open("Data/yunyinshang.csv", "r") as f:
        data = pd.read_csv(f)

    df = data.copy()
    # df = df.drop(
    #     ['Unnamed: 0', 'APPLY_SERIAL_NO', 'apply_time', 'CUSTOMER_ID', 'CERT_ID', 'PHONE_NUMBER', 'CUSTOMER_NAME'],
    #     axis=1)

    yi = Yihuier(df, 'dlq')

    print("Categorical Variables:", yi.get_categorical_variables())
    print("Numeric Variables:", yi.get_numeric_variables())
    print("Date Variables:", yi.get_date_variables())

    yi.data = yi.dp_module.date_var_shift_binary(yi.get_date_variables(), replace=True)

    print(yi.get_numeric_variables())

    print(yi.get_categorical_variables())
    yi.dp_module.fillna_cate_var(yi.get_categorical_variables(), 'class', 'unkonwn')
    #     # yi.eda_module.plot_default_cate(yi.get_categorical_variables(),plt_size=(100,100),plt_num=3,x=3,y=1)
    #
    #     yi.data = yi.data.drop(yi.get_categorical_variables(),axis = 1)
    # )
    #     # l = []
    #     # for i in yi.get_numeric_variables():
    #     #     if len(l) == 100:
    #     #         yi.eda_module.plot_num_col(l,plt_type='box',plt_size=(100,100),plt_num=100,x=10,y=10)
    #     #         l = []
    #     #     else:
    #     #         l.append(i)
    #
    yi.data = yi.dp_module.fillna_num_var(yi.get_numeric_variables(), fill_type='class', fill_class_num=-999)
    yi.data = yi.dp_module.delete_missing_var(threshold=0.01)
    # yi.dp_module.plot_bar_missing_var()


    # iv_list = []
    # col_list = []
    # for col in yi.get_numeric_variables():
    #     try:
    #         _, iv_value = yi.binning_module.binning_num([col], 10, 0, 'freq')
    #         # _, iv_value = yi.binning_module.binning_num([col], 10, 0, 'ChiMerge')
    #
    #         # 处理 iv_value 或其他逻辑
    #         col_list.append(col)
    #         iv_list.append(iv_value[0])
    #     except Exception as e:
    #         col_list.append(col)
    #         iv_list.append('error')
    #
    #         print(f"Error processing variable {col}: {str(e)}")
    #         continue  # 如果出现异常，跳过当前变量，继续下一个变量
    # iv = pd.DataFrame({'col': col_list, 'iv': iv_list})
    # print("iv_value:{}".format(iv))
    #
    # iv.to_csv("Data/iv_yinlian.csv", index=False)
    #
    # xg_fea_imp, _, _ = yi.var_select_module.select_xgboost(yi.get_numeric_variables())
    # xg_fea_imp.to_csv("Data/xg_fea_imp.csv", index=False)
    # print("xg_fea_imp:{}".format(xg_fea_imp))
    #
    # rf_fea_imp, _ = yi.var_select_module.select_rf(yi.get_numeric_variables())
    # rf_fea_imp.to_csv("Data/rf_fea_imp.csv", index=False)
    # print("rf_fea_imp:{}".format(rf_fea_imp))

    twice = yi.var_select_module.forward_delete_corr_ivfirst(yi.get_numeric_variables(),threshold=0.3)
    print(twice)