"""
Yihuier 基础使用示例

本示例展示 Yihuier 包的基本功能，包括：
- 数据探索性分析 (EDA)
- 数据预处理
- 变量分箱
- WOE 转换
- 变量选择
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from yihuier import Yihuier


def main():
    """基础使用示例"""

    # 屏蔽警告信息
    warnings.filterwarnings("ignore")

    # ==================== 1. 数据加载 ====================
    print("=" * 60)
    print("1. 数据加载")
    print("=" * 60)

    # 从 CSV 文件加载数据
    # 实际使用时请修改为你的数据路径
    data_path = Path("data/data.csv")

    if not data_path.exists():
        print(f"警告: 数据文件 {data_path} 不存在")
        print("使用示例数据进行演示...")
        # 创建示例数据
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "customer_id": range(1000),
                "age": np.random.randint(18, 70, 1000),
                "income": np.random.uniform(10000, 100000, 1000),
                "debt_ratio": np.random.uniform(0, 1, 1000),
                "credit_history": np.random.choice(["good", "average", "poor"], 1000),
                "default": np.random.randint(0, 2, 1000),
            }
        )
    else:
        data = pd.read_csv(data_path)

    print(f"数据形状: {data.shape}")
    print(f"\n前 5 行数据:\n{data.head()}")

    # ==================== 2. 初始化 Yihuier ====================
    print("\n" + "=" * 60)
    print("2. 初始化 Yihuier")
    print("=" * 60)

    # 创建 Yihuier 实例，指定目标变量
    # 注意：如果使用真实数据集，目标变量是 'dlq_flag'
    if 'dlq_flag' in data.columns:
        target_var = 'dlq_flag'
    else:
        target_var = 'default'

    yh = Yihuier(data, target=target_var)
    print(f"目标变量: {yh.target}")
    print(f"样本数: {len(yh.data)}")
    print(f"变量数: {len(yh.data.columns)}")

    # ==================== 3. 数据探索 (EDA) ====================
    print("\n" + "=" * 60)
    print("3. 数据探索")
    print("=" * 60)

    # 获取不同类型的变量
    categorical_vars = yh.get_categorical_variables()
    numeric_vars = yh.get_numeric_variables()

    print(f"类别型变量 ({len(categorical_vars)}): {categorical_vars[:5]}...")
    print(f"数值型变量 ({len(numeric_vars)}): {numeric_vars[:5]}...")

    # 快速生成 EDA 报告
    print("\n生成 EDA 统计报告...")
    eda_stats = yh.eda_module.auto_eda_simple()
    print(f"\nEDA 统计摘要:\n{eda_stats}")

    # 可视化数值型变量分布（选择前 4 个）
    if len(numeric_vars) >= 4:
        print("\n绘制数值型变量分布图...")
        yh.eda_module.plot_num_col(
            col_list=numeric_vars[:4],
            plt_type="hist",
            plt_size=(12, 8),
            plt_num=4,
            x=2,
            y=2,
        )

    # ==================== 4. 数据预处理 ====================
    print("\n" + "=" * 60)
    print("4. 数据预处理")
    print("=" * 60)

    print(f"原始数据形状: {yh.data.shape}")

    # 删除高缺失率变量
    yh.data = yh.dp_module.delete_missing_var(threshold=0.2)
    print(f"删除缺失变量后: {yh.data.shape}")

    # 填充数值型变量缺失值
    if numeric_vars:
        yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type="0")
        print("数值型变量缺失值已填充")

    # 填充分类型变量缺失值
    if categorical_vars:
        yh.data = yh.dp_module.fillna_cate_var(categorical_vars, fill_type="mode")
        print("类别型变量缺失值已填充")

    # 删除目标变量缺失的样本
    yh.data = yh.dp_module.target_missing_delete()
    print(f"删除目标缺失样本后: {yh.data.shape}")

    # 删除常变量
    # 注意：此操作会删除同值比例高的变量，请根据实际数据调整阈值
    # yh.data = yh.dp_module.const_delete(threshold=0.99)
    print("跳过常变量删除步骤（实际项目中请谨慎使用）")

    # ==================== 5. 变量分箱 ====================
    print("\n" + "=" * 60)
    print("5. 变量分箱")
    print("=" * 60)

    # 获取当前数值型变量
    numeric_vars = yh.get_numeric_variables()

    if len(numeric_vars) >= 3:
        # 选择前 3 个数值型变量进行分箱
        binning_vars = numeric_vars[:3]

        # 使用 ChiMerge 方法进行分箱
        bin_df, iv_value = yh.binning_module.binning_num(
            col_list=binning_vars, max_bin=5, method="ChiMerge"
        )

        print(f"\n分箱变量: {binning_vars}")
        print(f"IV 值: {iv_value}")

        # 显示第一个变量的分箱结果
        if bin_df:
            print(f"\n{binning_vars[0]} 分箱结果:")
            print(bin_df[0].head())

    # ==================== 6. WOE 转换 ====================
    print("\n" + "=" * 60)
    print("6. WOE 转换")
    print("=" * 60)

    # 拼接所有分箱结果
    woe_df = yh.binning_module.woe_df_concat()
    print(f"WOE 数据形状: {woe_df.shape}")

    # 转换数据集
    data_woe = yh.binning_module.woe_transform()
    print(f"WOE 转换后数据形状: {data_woe.shape}")
    print(f"\nWOE 转换后前 5 行:\n{data_woe.head()}")

    # ==================== 7. 变量选择 ====================
    print("\n" + "=" * 60)
    print("7. 变量选择")
    print("=" * 60)

    # 获取 WOE 转换后的特征变量
    feature_cols = [col for col in data_woe.columns if col != yh.target]

    if len(feature_cols) >= 3:
        select_cols = feature_cols[:3]

        # XGBoost 特征选择
        xg_imp, xg_rank, xg_cols = yh.var_select_module.select_xgboost(
            col_list=select_cols, imp_num=2
        )

        print(f"\n原始变量: {select_cols}")
        print(f"XGBoost 选择的变量: {xg_cols}")
        print(f"特征重要性:\n{xg_rank}")

    print("\n" + "=" * 60)
    print("基础使用示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
