"""
信用评分卡建模完整模板脚本
解决了 yihuier 工具包的实际 API 问题
"""

import pandas as pd
import numpy as np
from yihuier import Yihuier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================
DATA_PATH = "your_data.xlsx"  # 数据文件路径
TARGET = "target_column"       # 目标变量名
IV_THRESHOLD = 0.05            # IV 筛选阈值
CORR_THRESHOLD = 0.6           # 相关性阈值
MAX_BIN = 5                    # 分箱数量
TEST_SIZE = 0.3                # 测试集比例
BASE_SCORE = 600               # 基础分
PDO = 20                       # PDO
ODDS = 50                      # Odds

# ============================================================
# 第 1 步：数据加载和初始化
# ============================================================
print("=" * 60)
print("第 1 步：数据加载和初始化")
print("=" * 60)

# 加载数据（支持 csv, xlsx）
if DATA_PATH.endswith('.xlsx'):
    data = pd.read_excel(DATA_PATH)
else:
    data = pd.read_csv(DATA_PATH)

print(f"数据形状: {data.shape}")

# 初始化 Yihuier
yh = Yihuier(data, target=TARGET)

# 目标变量分布
target_dist = yh.data[yh.target].value_counts()
print(f"目标变量分布:\n{target_dist}")
print(f"正样本占比: {target_dist[1] / len(yh.data):.2%}")

# ============================================================
# 第 2 步：数据预处理
# ============================================================
print("\n" + "=" * 60)
print("第 2 步：数据预处理")
print("=" * 60)

numeric_vars = yh.get_numeric_variables()
categorical_vars = yh.get_categorical_variables()

print(f"数值型变量: {len(numeric_vars)} 个")
print(f"类别型变量: {len(categorical_vars)} 个")

# 缺失值处理
yh.data = yh.dp_module.fillna_num_var(
    col_list=numeric_vars,
    fill_type='0'
)

if categorical_vars:
    yh.data = yh.dp_module.fillna_cate_var(
        col_list=categorical_vars,
        fill_type='mode'
    )

yh.data = yh.dp_module.target_missing_delete()
print(f"处理后样本量: {len(yh.data)}")

# ============================================================
# 第 3 步：变量分箱
# ============================================================
print("\n" + "=" * 60)
print("第 3 步：变量分箱")
print("=" * 60)

binning_vars = [col for col in numeric_vars if col != yh.target]

# ChiMerge 可能失败，使用等频分箱作为备选
try:
    print("尝试 ChiMerge 分箱...")
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=binning_vars,
        max_bin=MAX_BIN,
        method='ChiMerge',
        min_binpct=0.05
    )
except Exception as e:
    print(f"ChiMerge 失败 ({e}), 使用等频分箱...")
    bin_df, iv_value = yh.binning_module.binning_num(
        col_list=binning_vars,
        max_bin=MAX_BIN,
        method='freq'
    )

# 构建 IV 字典
iv_dict = dict(zip(binning_vars, iv_value))
iv_sorted = sorted(iv_dict.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 IV 变量:")
for var, iv in iv_sorted[:10]:
    print(f"  {var}: IV = {iv:.4f}")

# ============================================================
# 第 4 步：WOE 转换
# ============================================================
print("\n" + "=" * 60)
print("第 4 步：WOE 转换")
print("=" * 60)

# 拼接 WOE 结果表（重要：生成 woe_result_df）
yh.binning_module.woe_df_concat()

# WOE 转换
data_woe = yh.binning_module.woe_transform()
print(f"WOE 转换后形状: {data_woe.shape}")

# 注意：woe_result_df 包含所有变量的分箱详情
# 列名: col, index, bin, min_bin, max_bin, total, totalrate,
#       bad, badrate, good, goodrate, woe, bin_iv, IV 等

# ============================================================
# 第 5 步：变量选择
# ============================================================
print("\n" + "=" * 60)
print("第 5 步：变量选择")
print("=" * 60)

# IV 筛选
high_iv_vars = [var for var, iv in iv_sorted if iv > IV_THRESHOLD]
print(f"IV > {IV_THRESHOLD} 的变量: {len(high_iv_vars)} 个")

# 相关性去重
try:
    final_vars = yh.var_select_module.forward_delete_corr_ivfirst(
        col_list=high_iv_vars,
        threshold=CORR_THRESHOLD
    )
    print(f"相关性去重后: {len(final_vars)} 个")
except Exception as e:
    print(f"相关性去重失败 ({e}), 使用 Top 15 IV 变量")
    final_vars = [var for var, iv in iv_sorted[:15]]

print("\n最终入模变量:")
for var in final_vars:
    print(f"  {var}: IV = {iv_dict.get(var, 0):.4f}")

# ============================================================
# 第 6 步：模型训练
# ============================================================
print("\n" + "=" * 60)
print("第 6 步：模型训练")
print("=" * 60)

X = data_woe[final_vars]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    penalty='l2',
    C=1.0
)
model.fit(X_train, y_train)

print("\n模型系数:")
for var, coef in zip(final_vars, model.coef_[0]):
    print(f"  {var}: {coef:.4f}")

# ============================================================
# 第 7 步：模型评估
# ============================================================
print("\n" + "=" * 60)
print("第 7 步：模型评估")
print("=" * 60)

y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, y_pred_train)
auc_test = roc_auc_score(y_test, y_pred_test)

# 注意：model_ks 返回单个 KS 值
ks_train = yh.me_module.model_ks(y_train, y_pred_train)
ks_test = yh.me_module.model_ks(y_test, y_pred_test)

print(f"训练集 AUC: {auc_train:.4f}")
print(f"测试集 AUC: {auc_test:.4f}")
print(f"训练集 KS: {ks_train:.4f}")
print(f"测试集 KS: {ks_test:.4f}")

# 交叉验证（使用 sklearn，不依赖 yihuier 的 cross_verify）
print("\n--- 5折交叉验证 ---")
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    X, y, cv=5, scoring='roc_auc'
)
print(f"交叉验证 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 第 8 步：评分卡实现
# ============================================================
print("\n" + "=" * 60)
print("第 8 步：评分卡实现")
print("=" * 60)

A, B, base_score = yh.si_module.cal_scale(
    score=BASE_SCORE,
    odds=ODDS,
    PDO=PDO,
    model=model
)

print(f"基础分: {base_score:.2f}")
print(f"评分卡参数: A={A:.4f}, B={B:.4f}")

train_scores = base_score + X_train.dot(model.coef_[0]) * B
test_scores = base_score + X_test.dot(model.coef_[0]) * B

print(f"训练集分数: {train_scores.min():.2f} - {train_scores.max():.2f}")
print(f"测试集分数: {test_scores.min():.2f} - {test_scores.max():.2f}")

# ============================================================
# 第 9 步：生成评分卡详情
# ============================================================
print("\n" + "=" * 60)
print("第 9 步：生成评分卡详情")
print("=" * 60)

# 使用 woe_result_df（不是 bin_df）
woe_result_df = yh.binning_module.woe_result_df

scorecard_data = []
for var in final_vars:
    var_woe = woe_result_df[woe_result_df['col'] == var]
    coef = model.coef_[0][final_vars.index(var)]

    for _, row in var_woe.iterrows():
        score_points = -coef * row['woe'] * B
        scorecard_data.append({
            '变量名': var,
            '分箱': row['bin'],
            '下限': row['min_bin'],
            '上限': row['max_bin'],
            '样本数': row['total'],
            '占比': row['totalrate'],
            '坏样本率': row['badrate'],
            'WOE': row['woe'],
            'IV': row['IV'],
            '系数': coef,
            '得分': round(score_points, 2)
        })

scorecard_df = pd.DataFrame(scorecard_data)
print(f"评分卡行数: {len(scorecard_df)}")

# ============================================================
# 第 10 步：KS Bucket 表格
# ============================================================
print("\n" + "=" * 60)
print("第 10 步：KS Bucket 表格")
print("=" * 60)

test_score_df = pd.DataFrame({
    'score': test_scores,
    'target': y_test.values
})

# 分成 10 个 bucket
test_score_df['bucket'] = pd.qcut(test_score_df['score'], q=10, labels=False, duplicates='drop')

ks_bucket_data = []
for bucket in sorted(test_score_df['bucket'].unique()):
    bucket_data = test_score_df[test_score_df['bucket'] == bucket]
    total = len(bucket_data)
    good = (bucket_data['target'] == 0).sum()
    bad = (bucket_data['target'] == 1).sum()
    ks_bucket_data.append({
        'Bucket': bucket + 1,
        '分数范围': f"{bucket_data['score'].min():.2f}-{bucket_data['score'].max():.2f}",
        '样本数': total,
        '好样本': good,
        '坏样本': bad,
        '坏样本率': bad / total if total > 0 else 0
    })

ks_bucket_df = pd.DataFrame(ks_bucket_data)
ks_bucket_df = ks_bucket_df.sort_values('分数范围', ascending=False)

# 计算累计 KS
ks_bucket_df['累计好样本'] = ks_bucket_df['好样本'].cumsum()
ks_bucket_df['累计坏样本'] = ks_bucket_df['坏样本'].cumsum()
total_good = ks_bucket_df['好样本'].sum()
total_bad = ks_bucket_df['坏样本'].sum()
ks_bucket_df['累计好占比'] = ks_bucket_df['累计好样本'] / total_good
ks_bucket_df['累计坏占比'] = ks_bucket_df['累计坏样本'] / total_bad
ks_bucket_df['KS'] = abs(ks_bucket_df['累计好占比'] - ks_bucket_df['累计坏占比'])

print(ks_bucket_df)

# ============================================================
# 第 11 步：PSI 稳定性分析（手动实现）
# ============================================================
print("\n" + "=" * 60)
print("第 11 步：PSI 稳定性分析")
print("=" * 60)

def calculate_psi(expected, actual, buckets=10):
    """计算 PSI (Population Stability Index)"""
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    expected_percents = np.percentile(expected, breakpoints)

    expected_counts = np.histogram(expected, bins=expected_percents)[0]
    actual_counts = np.histogram(actual, bins=expected_percents)[0]

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # 处理零值
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

psi_value = calculate_psi(train_scores.values, test_scores.values)
print(f"训练集 vs 测试集 PSI: {psi_value:.4f}")

if psi_value < 0.1:
    print("PSI < 0.1: 稳定")
elif psi_value < 0.25:
    print("PSI 0.1-0.25: 需要监控")
else:
    print("PSI > 0.25: 显著变化")

# ============================================================
# 第 12 步：输出结果
# ============================================================
print("\n" + "=" * 60)
print("第 12 步：输出结果")
print("=" * 60)

OUTPUT_DIR = "./output"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存 Excel
excel_path = f"{OUTPUT_DIR}/评分卡模型输出.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    scorecard_df.to_excel(writer, sheet_name='评分卡', index=False)
    ks_bucket_df.to_excel(writer, sheet_name='KS_Bucket', index=False)

    iv_summary = pd.DataFrame(iv_sorted, columns=['变量名', 'IV值'])
    iv_summary.to_excel(writer, sheet_name='IV值汇总', index=False)

    coef_df = pd.DataFrame({
        '变量名': final_vars,
        '系数': model.coef_[0],
        'IV值': [iv_dict.get(v, 0) for v in final_vars]
    })
    coef_df.to_excel(writer, sheet_name='模型系数', index=False)

    performance_df = pd.DataFrame({
        '指标': ['训练集AUC', '测试集AUC', '训练集KS', '测试集KS',
                 '交叉验证AUC均值', '交叉验证AUC标准差', 'PSI', '基础分'],
        '值': [auc_train, auc_test, ks_train, ks_test,
               cv_scores.mean(), cv_scores.std(), psi_value, base_score]
    })
    performance_df.to_excel(writer, sheet_name='模型性能', index=False)

print(f"Excel 已保存: {excel_path}")

print("\n" + "=" * 60)
print("建模完成！")
print("=" * 60)