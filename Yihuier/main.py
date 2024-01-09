# 在 main.py 或者 __init__.py 中创建主类 Yihui
import warnings
import pandas as pd

from yihui import Yihui



# 在主程序中使用 Yihui 类
if __name__ == "__main__":
    # ban FutureWarning
    warnings.filterwarnings('ignore')

    # generate data.csv
    with open("../Data/data.csv", "r") as f:
        data = pd.read_csv(f)
    data['customer_no'] = str(data['customer_no'])

    # loading Titanic dataset
    # data = sns.load_dataset('titanic')
    # data.head()

    # Create Yihui Class
    yihui_project = Yihui(data,'dlq_flag')
    print(yihui_project.data.head())

    print("Categorical Variables:", yihui_project.categorical_vars)
    print("Numeric Variables:", yihui_project.numeric_vars)

    # 直接访问 Yihui 类的属性获取字符型和数值型变量的名字
    categorical_vars_list = yihui_project.categorical_vars
    numeric_vars_list = yihui_project.numeric_vars

    # yihui_project.eda_module.auto_eda_profiling()

    # yihui_project.eda_module.plot_num_col(numeric_vars_list,plt_type='hist',hspace=0.4,wspace=0.4,plt_size=(100,100),plt_num=100,x=10,y=10)
    # yihui_project.eda_module.plot_num_col(numeric_vars_list,plt_type='box',hspace=0.4,wspace=0.4,plt_size=(100,100),plt_num=100,x=10,y=10)



    # eda_module.plot_default_num
    # yihui_project.eda_module.plot_default_num(numeric_vars_list,hspace=0.4,wspace=0.4,q=10,plt_size=(100,100),plt_num=100,x=10,y=10)

    # eda_module.plot_default_cate
    # yihui_project.eda_module.plot_default_cate(categorical_vars_list,hspace=0.4,wspace=0.4,plt_size=(10,10),plt_num=1,x=1,y=1)
