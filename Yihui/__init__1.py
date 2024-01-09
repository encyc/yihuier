# 在 main.py 或者 __init__.py 中创建主类 Yihui
import warnings
import pandas as pd
import numpy as np
from Yihui.yihui import Yihui

# 在主程序中使用 Yihui 类
if __name__ == "__main__":
    # ban FutureWarning
    warnings.filterwarnings('ignore')

    # generate data.csv
    with open("../Data/data_yinlian.csv", "r") as f:
        data = pd.read_csv(f)

    df = data.copy()
    df = df.drop(
        ['Unnamed: 0', 'APPLY_SERIAL_NO', 'apply_time', 'CUSTOMER_ID', 'CERT_ID', 'PHONE_NUMBER', 'CUSTOMER_NAME'],
        axis=1)

    yi = Yihui(df, 'dlq')

    print("Categorical Variables:", yi.get_categorical_variables())
    print("Numeric Variables:", yi.get_numeric_variables())
    print("Date Variables:", yi.get_date_variables())

    df = yi.dp_module.date_var_shift_binary(yi.get_date_variables(),replace=True)
    print(df['UPPA007_binary'])
#
#     print(yi.get_categorical_variables())
#     # yi.dp_module.fillna_cate_var(yi.get_categorical_variables(),'class','unkonwn')
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
#     yi.data = yi.dp_module.fillna_num_var(yi.get_numeric_variables(),fill_type='class',fill_class_num=-999)
#     # yi.dp_module.plot_bar_missing_var()
#
#     yi.binning_module.binning_num(yi.get_numeric_variables(),20,0)