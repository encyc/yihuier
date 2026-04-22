"""
Yihuier 高级建模流程示例

本示例展示完整的信用评分卡建模流程，包括：
1. 完整的数据探索性分析
2. 数据预处理和清洗
3. 变量分箱（数值型和类别型）
4. WOE 转换和 IV 值计算
5. 变量选择（多种方法组合）
6. 逻辑回归模型训练
7. 模型评估（ROC、KS、交叉验证）
8. 评分卡实现和转换
9. 模型验证和监控
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yihuier import Yihuier


def create_sample_data():
    """创建示例数据集"""
    np.random.seed(42)

    n_samples = 5000

    data = pd.DataFrame(
        {
            # 客户基本信息
            "age": np.random.randint(18, 70, n_samples),
            "income": np.random.uniform(10000, 150000, n_samples),
            "debt_ratio": np.random.uniform(0, 1, n_samples),
            "employment_length": np.random.randint(0, 40, n_samples),
            # 信用历史
            "credit_history": np.random.choice(["excellent", "good", "average", "poor"], n_samples),
            "late_payment_count": np.random.randint(0, 10, n_samples),
            # 财务指标
            "monthly_payment": np.random.uniform(500, 10000, n_samples),
            "loan_amount": np.random.uniform(5000, 200000, n_samples),
            "asset_value": np.random.uniform(0, 500000, n_samples),
            # 行为特征
            "inquiry_count_6m": np.random.randint(0, 20, n_samples),
            "account_count": np.random.randint(1, 30, n_samples),
            # 目标变量
            "default": np.random.randint(0, 2, n_samples),
        }
    )

    # 添加一些缺失值
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data.loc[missing_indices, "income"] = np.nan

    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    data.loc[missing_indices, "debt_ratio"] = np.nan

    return data


def main():
    """完整的建模流程"""

    # 屏蔽警告
    warnings.filterwarnings("ignore")

    # ==================== 1. 数据加载 ====================
    print("=" * 80)
    print("信用评分卡完整建模流程")
    print("=" * 80)
    print("\n[步骤 1/10] 数据加载")

    data_path = Path("data/data.csv")

    if data_path.exists():
        data = pd.read_csv(data_path)
        print(f"从 {data_path} 加载数据")
        # 确定目标变量名称
        target_var = 'dlq_flag' if 'dlq_flag' in data.columns else 'default'
    else:
        print("使用生成的示例数据")
        data = create_sample_data()
        target_var = 'default'

    print(f"数据形状: {data.shape}")
    print(f"目标变量: {target_var}")
    print(f"目标变量分布:\n{data[target_var].value_counts()}")

    # ==================== 2. 初始化和数据探索 ====================
    print("\n[步骤 2/10] 初始化和数据探索")

    yh = Yihuier(data, target=target_var)

    # 获取变量类型
    categorical_vars = yh.get_categorical_variables()
    numeric_vars = yh.get_numeric_variables()

    print(f"类别型变量: {categorical_vars}")
    print(f"数值型变量: {numeric_vars}")

    # 生成 EDA 报告
    eda_stats = yh.eda_module.auto_eda_simple()
    print(f"\n数据质量报告:\n{eda_stats}")

    # ==================== 3. 数据预处理 ====================
    print("\n[步骤 3/10] 数据预处理")

    original_shape = yh.data.shape
    print(f"原始数据: {original_shape}")

    # 删除高缺失率变量
    yh.data = yh.dp_module.delete_missing_var(threshold=0.15)
    print(f"删除高缺失变量后: {yh.data.shape}")

    # 填充缺失值
    yh.data = yh.dp_module.fillna_num_var(numeric_vars, fill_type="0")
    yh.data = yh.dp_module.fillna_cate_var(categorical_vars, fill_type="class")
    print("缺失值填充完成")

    # 删除目标变量缺失的样本
    yh.data = yh.dp_module.target_missing_delete()
    print(f"删除目标缺失样本后: {yh.data.shape}")

    # 删除常变量（可选：根据实际需求调整阈值）
    # yh.data = yh.dp_module.const_delete(threshold=0.95)
    # print(f"删除常变量后: {yh.data.shape}")
    print("跳过常变量删除步骤（实际项目中请谨慎使用）")

    # ==================== 4. 变量分箱 ====================
    print("\n[步骤 4/10] 变量分箱")

    # 更新变量列表
    numeric_vars = yh.get_numeric_variables()
    categorical_vars = yh.get_categorical_variables()

    # 数值型变量分箱（选择前 10 个进行演示）
    if len(numeric_vars) > 0:
        # 只选择前 10 个数值型变量进行演示
        binning_vars = numeric_vars[:10] if len(numeric_vars) > 10 else numeric_vars
        # 排除 customer_no 等非特征变量
        binning_vars = [v for v in binning_vars if v not in ['customer_no', yh.target]]

        print(f"对 {len(binning_vars)} 个数值型变量进行分箱...")
        # 使用等频分箱，更稳定
        bin_df_num, iv_value_num = yh.binning_module.binning_num(
            col_list=binning_vars, max_bin=5, method="freq"
        )

        # 显示高 IV 值变量
        iv_df_num = pd.DataFrame({"var": binning_vars, "iv": iv_value_num})
        iv_df_num = iv_df_num.sort_values("iv", ascending=False)
        print(f"\n数值型变量 IV 值 Top 5:\n{iv_df_num.head()}")

    # 类别型变量分箱
    if len(categorical_vars) > 0:
        print(f"\n对 {len(categorical_vars)} 个类别型变量进行分箱...")
        # 限制类别型变量数量
        cate_vars = categorical_vars[:5] if len(categorical_vars) > 5 else categorical_vars
        bin_df_cate, iv_value_cate, ks_value_cate = yh.binning_module.binning_cate(cate_vars)

        # 显示高 IV 值变量
        iv_df_cate = pd.DataFrame({"var": cate_vars, "iv": iv_value_cate})
        iv_df_cate = iv_df_cate.sort_values("iv", ascending=False)
        print(f"\n类别型变量 IV 值:\n{iv_df_cate}")

    # ==================== 5. WOE 转换 ====================
    print("\n[步骤 5/10] WOE 转换")

    # 获取完整的 IV 值表
    if yh.binning_module.iv_df is not None:
        iv_df = yh.binning_module.iv_df.copy()
        print(f"完整 IV 值统计:\n{iv_df['iv'].describe()}")

        # 选择 IV > 0.02 的变量
        selected_vars = iv_df[iv_df["iv"] > 0.02]["col"].tolist()
        print(f"\nIV > 0.02 的变量数量: {len(selected_vars)}")

    # WOE 转换
    # 先拼接 WOE 结果表
    woe_df = yh.binning_module.woe_df_concat()
    print(f"WOE 结果表: {woe_df.shape if woe_df is not None else 'None'}")

    data_woe = yh.binning_module.woe_transform()
    print(f"WOE 转换后数据: {data_woe.shape}")

    # ==================== 6. 高级变量选择 ====================
    print("\n[步骤 6/10] 高级变量选择")

    feature_cols = [col for col in data_woe.columns if col != yh.target]

    if len(feature_cols) >= 5:
        # 方法 1: XGBoost 特征选择
        print("\n方法 1: XGBoost 特征重要性")
        xg_imp, xg_rank, xg_cols = yh.var_select_module.select_xgboost(col_list=feature_cols, imp_num=10)
        print(f"选择的变量: {xg_cols}")

        # 方法 2: 相关性筛选（考虑 IV）
        print("\n方法 2: 相关性筛选（考虑 IV）")
        final_vars = yh.var_select_module.forward_delete_corr_ivfirst(col_list=xg_cols, threshold=0.6)
        print(f"最终变量: {final_vars}")

        # 方法 3: 随机森林特征选择
        print("\n方法 3: 随机森林特征重要性")
        rf_imp, rf_cols = yh.var_select_module.select_rf(col_list=final_vars, imp_num=8)
        print(f"RF 选择的变量: {rf_cols}")

        model_features = rf_cols
    else:
        model_features = feature_cols
        print(f"使用全部变量: {model_features}")

    # ==================== 7. 模型训练 ====================
    print("\n[步骤 7/10] 模型训练")

    # 准备数据
    X = data_woe[model_features]
    y = data_woe[yh.target]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")

    # 训练逻辑回归模型
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # 预测
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"模型训练完成")

    # ==================== 8. 模型评估 ====================
    print("\n[步骤 8/10] 模型评估")

    # ROC 曲线和 AUC
    yh.me_module.plot_roc(y_test, y_pred_proba)

    # KS 曲线
    yh.me_module.plot_model_ks(y_test, y_pred_proba)

    # 计算 KS 值
    ks_value = yh.me_module.model_ks(y_test, y_pred_proba)
    print(f"KS 值: {ks_value:.4f}")

    # 混淆矩阵和分类报告
    yh.me_module.plot_matrix_report(y_test, y_pred)

    # 交叉验证
    print("\n执行 5 折交叉验证...")
    yh.me_module.cross_verify(X_train, y_train, lr_model, fold=5, scoring="roc_auc")

    # ==================== 9. 评分卡实现 ====================
    print("\n[步骤 9/10] 评分卡实现")

    # 计算评分卡刻度
    A, B, base_score = yh.si_module.cal_scale(score=600, odds=50, PDO=20, model=lr_model)

    # 计算测试集分数
    # 直接使用线性回归系数计算分数
    test_scores = pd.DataFrame({"score": base_score + X_test.dot(lr_model.coef_[0]) * B})
    test_scores[yh.target] = y_test.values
    test_scores["score"] = test_scores["score"].astype(float)

    print(f"\n分数统计:")
    print(test_scores["score"].describe())

    # 绘制分数分布
    plt.figure(figsize=(10, 6))
    plt.hist(test_scores[test_scores[yh.target] == 0]["score"], bins=50, alpha=0.5, label="正常")
    plt.hist(test_scores[test_scores[yh.target] == 1]["score"], bins=50, alpha=0.5, label="违约")
    plt.xlabel("分数")
    plt.ylabel("人数")
    plt.title("分数分布图")
    plt.legend()
    plt.show()

    # ==================== 10. 模型监控 ====================
    print("\n[步骤 10/10] 模型监控")

    # 分数区间统计
    score_info = yh.si_module.score_info(
        test_scores, score_col="score", target=yh.target, x=200, y=800, step=20
    )
    print(f"分数区间明细:\n{score_info.head()}")

    # 设置 cutoff 并验证
    cutoff = test_scores["score"].quantile(0.7)  # 70% 分位数
    print(f"\nCut-off 分数: {cutoff:.2f}")

    matrix = yh.si_module.rule_verify(test_scores, col_score="score", target=yh.target, cutoff=cutoff)
    print(f"\n混淆矩阵:\n{matrix}")

    # ==================== 总结 ====================
    print("\n" + "=" * 80)
    print("建模流程完成!")
    print("=" * 80)
    print(f"\n最终模型特征数量: {len(model_features)}")
    print(f"模型特征: {model_features}")
    print(f"\n建议: 根据业务需求调整 cutoff 值和监控指标")


if __name__ == "__main__":
    main()
