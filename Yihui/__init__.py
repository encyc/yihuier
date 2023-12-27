# 在 main.py 或者 __init__.py 中创建主类 Yihui
import warnings
import pandas as pd

from Yihui.yihui import Yihui



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

    # ### eda 阶段
    #
    # # 使用ydata_profiling 自动生成eda报告
    # # 根据dataset数据量大小，生成报告的时间会不同。建议慎重操作。
    # yihui_project.eda_module.auto_eda_profiling()
    #
    # # 手动查看变量分布情况
    # yihui_project.eda_module.plot_num_col(numeric_vars_list,plt_type='hist',plt_size=(100,100),plt_num=100,x=10,y=10)
    # yihui_project.eda_module.plot_num_col(numeric_vars_list,plt_type='box',plt_size=(100,100),plt_num=100,x=10,y=10)
    # yihui_project.eda_module.plot_cate_var(categorical_vars_list,plt_size=(100,100),plt_num=100,x=10,y=10)
    #
    # # 数值型变量的违约率分析
    # yihui_project.eda_module.plot_default_num(numeric_vars_list,q=10,plt_size=(100,100),plt_num=100,x=10,y=10)
    #
    # # 类别型变量的违约率分析
    # yihui_project.eda_module.plot_default_cate(categorical_vars_list,plt_size=(10,10),plt_num=1,x=1,y=1)
    #

    ### data processing 阶段

    # 所有变量缺失值分布图
    print(yihui_project.dp_module.plot_bar_missing_var())

    # 使用 '0','median','class','rf'
    yihui_project.data = yihui_project.dp_module.fillna_num_var(numeric_vars_list, fill_type='0')

    yihui_project.data = yihui_project.dp_module.fillna_cate_var(categorical_vars_list, fill_type='class', fill_str='missing')
    yihui_project.data = yihui_project.dp_module.fillna_cate_var(categorical_vars_list, fill_type='mode')

    # 缺失值剔除
    yihui_project.data = yihui_project.dp_module.delete_missing_var(threshold=0.2)
    yihui_project.data = yihui_project.dp_module.delete_missing_obs(threshold=5)

    # 常变量/同值化处理
    yihui_project.dp_module.const_delete(threshold=0.9)

    # 再检查一下缺失值
    print(yihui_project.dp_module.plot_bar_missing_var())


    # cluster 阶段

    # yihui_project.cluster_module.cluster_AffinityPropagation(['v3','v5'])