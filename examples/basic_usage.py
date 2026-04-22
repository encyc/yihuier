"""
评分卡建模基本使用示例

演示如何使用 yihuier 包进行完整的信用评分卡建模流程。
"""

import warnings
import pandas as pd
import numpy as np
from yihuier.yihuier import Yihuier


def main():
    """主函数：演示完整的评分卡建模流程"""

    # 屏蔽警告
    warnings.filterwarnings('ignore')

    # 读取数据
    # 注意：实际使用时需要根据你的数据路径调整
    with open("data/data.csv", "r") as f:
        data = pd.read_csv(f)

    # 数据预处理示例
    data['customer_no'] = str(data['customer_no'])
    data['v101'] = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], size=len(data))

    # 创建 Yihuier 实例
    yihuier_project = Yihuier(data, 'dlq_flag')
    print(yihuier_project.data.head())

    # 获取变量类型
    print("Categorical Variables:", yihuier_project.get_categorical_variables())
    print("Numeric Variables:", yihuier_project.get_numeric_variables())
    print("Date Variables:", yihuier_project.get_date_variables())

    # EDA 阶段
    categorical_vars_list = yihuier_project.get_categorical_variables()
    numeric_vars_list = yihuier_project.get_numeric_variables()

    # 自动 EDA 报告
    eda_result = yihuier_project.eda_module.auto_eda_simple()
    print(eda_result)

    # 手动查看变量分布情况
    yihuier_project.eda_module.plot_num_col(
        numeric_vars_list,
        plt_type='box',
        plt_size=(100, 100),
        plt_num=100,
        x=10,
        y=10
    )

    # 数据处理阶段

    # 缺失值填充
    yihuier_project.data = yihuier_project.dp_module.fillna_num_var(
        numeric_vars_list,
        fill_type='0'
    )

    # 常变量/同值化处理
    yihuier_project.data = yihuier_project.dp_module.const_delete(threshold=0.5)
    print(yihuier_project.get_numeric_variables())

    # 删除目标变量缺失的观测
    yihuier_project.data = yihuier_project.dp_module.target_missing_delete()

    # 变量选择阶段

    # 考虑 IV 大小的相关变量删除
    twice = yihuier_project.var_select_module.forward_delete_corr_ivfirst(
        ['v1', 'v2', 'v3', 'v4'],
        threshold=0.5
    )
    print(twice)

    # 考虑特征重要性的相关变量删除
    twice = yihuier_project.var_select_module.forward_delete_corr_impfirst(
        ['v1', 'v2', 'v3', 'v4'],
        type='xgboost',
        threshold=0.5
    )
    print(twice)


if __name__ == "__main__":
    main()
