import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def initialize_bins(df, col, n=10):
    # 检测缺失值
    if df[col].isnull().sum() > 0:
        print('missing obs exit, quit')
        pass
    # 初始化分箱，等距离划分为 n 个箱体
    bins = np.linspace(df[col].min(), df[col].max(), n + 1)
    return bins


def calculate_chi2(df, col, target, bins):
    # # 计算每个分箱的正负样本数
    df.loc[:, 'bin'] = pd.cut(df[col], bins, include_lowest=True)
    observed = pd.crosstab(df['bin'], df[target])
    print('df')
    print(df)
    print('observed')
    print(observed)

    # 计算期望频数
    # expected = np.outer(df['bin'].value_counts(), observed.sum(axis=1)) / df.shape[0]
    # print('expected values')
    # print(expected)
    # 计算卡方值
    chi2, p_value, dof, _ = chi2_contingency(observed, correction=False)
    print('chi2')
    print(chi2)
    print('dof')
    print(dof)
    return chi2, dof



def merge_bins(df, col, bins_to_merge):
    # 合并指定的相邻分箱
    df['bin'] = df['bin'].apply(lambda x: bins_to_merge[x] if x in bins_to_merge else x)

    # 将 Interval 转换为左边界的数值
    df['bin'] = df['bin'].apply(lambda x: x.left if isinstance(x, pd.Interval) else x)

    bins = sorted(df['bin'].unique())
    return bins


def check_significance(df, col, target, bins):
    # 检验合并后的分箱的显著性
    observed = pd.crosstab(df['bin'], df[target])
    chi2, _, _, _ = chi2_contingency(observed, correction=False)

    # 计算p值
    p_value = 1 - chi2_contingency(observed)[1]

    return chi2, p_value


def chi_merge_iteration(df, col, target, max_bin=None, min_binpct=None):
    # 初始化分箱
    bins = initialize_bins(df, col, n=10)
    chi2_value, dof = calculate_chi2(df, col, target, bins)

    # 开始迭代
    while len(bins) > 2:  # 直到只剩下两个分箱
        # 找到卡方值最小的相邻分箱
        bins_to_merge = find_merge_bins(df, col, target, bins)

        # 合并相邻分箱
        bins = merge_bins(df, col, bins_to_merge)

        # 检验合并后的分箱的显著性
        chi2, p_value = check_significance(df, col, target, bins)

        # 判断卡方值是否显著
        if p_value > 0.05:
            break  # 不再显著时结束迭代

    return bins


def find_merge_bins(df, col, target, bins):
    # 计算相邻分箱的卡方值
    chi2_values = {}

    for i in range(len(bins) - 1):
        bin1 = bins[i]
        bin2 = bins[i + 1]

        # 合并相邻分箱
        merged_bins = merge_bins(df, col, {bin1: bin1, bin2: bin2})

        # 计算合并后的卡方值
        chi2, _ = calculate_chi2(df, col, target, merged_bins)
        chi2_values[(bin1, bin2)] = chi2

    # 找到卡方值最小的相邻分箱
    min_chi2_bins = min(chi2_values, key=chi2_values.get)

    return {min_chi2_bins[0]: min_chi2_bins[1]}

# 示例使用
# 假设 df 是你的 DataFrame，target 是目标变量，col 是分箱变量
# bins = chi_merge_iteration(df, col='col', target='target')


if __name__ == "__main__":
    # ban FutureWarning
    # warnings.filterwarnings('ignore')

    # generate data.csv
    with open("../Data/data.csv", "r") as f:
        data = pd.read_csv(f)
    data['customer_no'] = str(data['customer_no'])
    data['v1'].fillna(0, inplace=True)

    bins = initialize_bins(data, 'v1', 200)
    calculate_chi2(data, 'v1', 'dlq_flag', bins=bins)
    print()



    # bins = chi_merge_iteration(data, 'v1', 'dlq_flag')