import math
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import spearmanr


# 01 计算IV
def iv_count(data, var, target, weight):
    """ 计算iv值
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟计算IV值的变量名称
        target: String，Y列名称
    Returns:
        IV值， float
    """
    # 获取变量的唯一值
    value_list = set(list(np.unique(data[var])))
    iv = 0
    # 计算每个唯一值的错误率
    data_bad = pd.Series(data[data[target] == 1][var].values, index=data[data[target] == 1].index)
    data_good = pd.Series(data[data[target] == 0][var].values, index=data[data[target] == 0].index)
    len_bad = len(data_bad)
    len_good = len(data_good)
    # 遍历变量的唯一值
    for value in value_list:
        # 判断是否某类是否为0，避免出现无穷小值和无穷大值
        if sum(data_bad == value) == 0:
            bad_rate = 1 / len_bad
        else:
            bad_rate = sum(data_bad == value) / len_bad
        if sum(data_good == value) == 0:
            good_rate = 1 / len_good
        else:
            good_rate = sum(data_good == value) / len_good
        # 计算iv值
        iv += (good_rate - bad_rate) * math.log(good_rate / bad_rate, 2)
        # print(value,iv)
    # 返回iv值
    return iv


# In[ ]:
# 02 基于CART算法的最优分箱代码实现


def get_var_median(data, var):
    """ 得到指定连续变量的所有元素的中位数列表
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
    Returns:
        关于连续变量的所有元素的中位列表，List
    """
    var_value_list = list(np.unique(data[var]))
    var_median_list = []
    for i in range(len(var_value_list) - 1):
        var_median = (var_value_list[i] + var_value_list[i + 1]) / 2
        var_median_list.append(var_median)
    return var_median_list


def calculate_gini(y):
    """ 计算基尼指数
    Args:
        y: Array，待计算数据的target，即0和1的数组
    Returns:
        基尼指数，float
    """
    # 将数组转化为列表
    y = y.tolist()
    probs = [y.count(i) / len(y) for i in np.unique(y)]
    gini = sum([p * (1 - p) for p in probs])
    return gini


def get_cart_split_point(data, var, target, min_sample):
    """ 获得最优的二值划分点（即基尼指数下降最大的点）
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        min_sample: int，分箱的最小数据样本，也就是数据量至少达到多少才需要去分箱，一般作用在开头或者结尾处的分箱点

    Returns:
        BestSplit_Point: 返回本次迭代的最优划分点，float
        BestSplit_Position: 返回最优划分点的位置，最左边为0，最右边为1，float
    """

    # 初始化
    Gini = calculate_gini(data[target].values)
    Best_Gini = 0.0
    BestSplit_Point = -99999
    BestSplit_Position = 0.0
    median_list = get_var_median(data, var)  # 获取当前数据集指定元素的所有中位数列表

    # 遍历中位数列表
    for i in range(len(median_list)):
        left = data[data[var] < median_list[i]]
        right = data[data[var] > median_list[i]]

        # 如果切分后的数据量少于指定阈值，跳出本次分箱计算
        if len(left) < min_sample or len(right) < min_sample:
            continue

        Left_Gini = calculate_gini(left[target].values)
        Right_Gini = calculate_gini(right[target].values)
        Left_Ratio = len(left) / len(data)
        Right_Ratio = len(right) / len(data)

        Temp_Gini = Gini - (Left_Gini * Left_Ratio + Right_Gini * Right_Ratio)
        if Temp_Gini > Best_Gini:
            Best_Gini = Temp_Gini
            BestSplit_Point = median_list[i]
            # 获取切分点的位置，最左边为0，最右边为1
            if len(median_list) > 1:
                BestSplit_Position = i / (len(median_list) - 1)
            else:
                BestSplit_Position = i / len(len(median_list))
        else:
            continue
    Gini = Gini - Best_Gini
    # print("最优切分点：", BestSplit_Point)
    return BestSplit_Point, BestSplit_Position


def get_cart_bincut(data, var, target, leaf_stop_percent=0.05):
    """ 计算最优分箱切分点
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        leaf_stop_percent: 叶子节点占比，作为停止条件，默认5%

    Returns:
        best_bincut: 最优的切分点列表，List
    """
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []

    def cutting_data(data, var, target, min_sample, best_bincut):
        split_point, position = get_cart_split_point(data, var, target, min_sample)

        if split_point != -99999:
            best_bincut.append(split_point)

        # 根据最优切分点切分数据集，并对切分后的数据集递归计算切分点，直到满足停止条件
        # print("本次分箱的值域范围为{0} ~ {1}".format(data[var].min(), data[var].max()))
        left = data[data[var] < split_point]
        right = data[data[var] > split_point]

        # 当切分后的数据集仍大于最小数据样本要求，则继续切分
        if len(left) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(left, var, target, min_sample, best_bincut)
        else:
            pass
        if len(right) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(right, var, target, min_sample, best_bincut)
        else:
            pass
        return best_bincut

    best_bincut = cutting_data(data, var, target, min_sample, best_bincut)

    # 把切分点补上头尾
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)

    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min() - 1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


# In[]:

# 03 基于卡方检验的最优分箱代码实现


def calculate_chi(freq_array):
    """ 计算卡方值
    Args:
        freq_array: Array，待计算卡方值的二维数组，频数统计结果
    Returns:
        卡方值，float
    """
    # 检查是否为二维数组
    assert (freq_array.ndim == 2)

    # 计算每列的频数之和
    col_nums = freq_array.sum(axis=0)
    # 计算每行的频数之和
    row_nums = freq_array.sum(axis=1)
    # 计算总频数
    nums = freq_array.sum()
    # 计算期望频数
    E_nums = np.ones(freq_array.shape) * col_nums / nums
    E_nums = (E_nums.T * row_nums).T
    # 计算卡方值
    tmp_v = (freq_array - E_nums) ** 2 / E_nums
    # 如果期望频数为0，则计算结果记为0
    tmp_v[E_nums == 0] = 0
    chi_v = tmp_v.sum()
    return chi_v


def get_chimerge_bincut(data, var, target, max_group=None, chi_threshold=None):
    """ 计算卡方分箱的最优分箱点
    Args:
        data: DataFrame，待计算卡方分箱最优切分点列表的数据集
        var: 待计算的连续型变量名称
        target: 待计算的目标列Y的名称
        max_group: 最大的分箱数量（因为卡方分箱实际上是合并箱体的过程，需要限制下最大可以保留的分箱数量）
        chi_threshold: 卡方阈值，如果没有指定max_group，我们默认选择类别数量-1，置信度95%来设置阈值

    Returns:
        最优切分点列表，List
    """

    '''
    如果不知道卡方阈值怎么取，可以生成卡方表来看看，代码如下：  
    import pandas as pd
    import numpy as np
    from scipy.stats import chi2
    p = [0.995, 0.99, 0.975, 0.95, 0.9, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
    pd.DataFrame(np.array([chi2.isf(p, df=i) for i in range(1,10)]), columns=p, index=list(range(1,10)))
    '''

    freq_df = pd.crosstab(index=data[var], columns=data[target])
    # 转化为二维数组
    freq_array = freq_df.values

    # 初始化箱体，每个元素单独一组
    best_bincut = freq_df.index.values

    # 初始化阈值 chi_threshold，如果没有指定 chi_threshold，则默认选择target数量-1，置信度95%来设置阈值
    if max_group is None:
        if chi_threshold is None:
            chi_threshold = chi2.isf(0.05, df=freq_array.shape[-1])

    # 开始迭代
    while True:
        min_chi = None
        min_idx = None
        for i in range(len(freq_array) - 1):
            # 两两计算相邻两组的卡方值，得到最小卡方值的两组
            v = calculate_chi(freq_array[i: i + 2])
            if min_chi is None or min_chi > v:
                min_chi = v
                min_idx = i

        # 是否继续迭代条件判断
        # 条件1：当前箱体数仍大于 最大分箱数量阈值
        # 条件2：当前最小卡方值仍小于制定卡方阈值
        if (max_group is not None and max_group < len(freq_array)) or (
                chi_threshold is not None and min_chi < chi_threshold):
            tmp = freq_array[min_idx] + freq_array[min_idx + 1]
            freq_array[min_idx] = tmp
            freq_array = np.delete(freq_array, min_idx + 1, 0)
            best_bincut = np.delete(best_bincut, min_idx + 1, 0)
        else:
            break

    # 把切分点补上头尾
    best_bincut = best_bincut.tolist()
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)

    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min() - 1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


# In[ ]:
# 04 基于最优KS的最优分箱代码实现

def get_maxks_split_point(data, var, target, min_sample=0.05):
    """ 计算KS值
    Args:
        data: DataFrame，待计算卡方分箱最优切分点列表的数据集
        var: 待计算的连续型变量名称
        target: 待计算的目标列Y的名称
        min_sample: int，分箱的最小数据样本，也就是数据量至少达到多少才需要去分箱，一般作用在开头或者结尾处的分箱点
    Returns:
        ks_v: KS值，float
        BestSplit_Point: 返回本次迭代的最优划分点，float
        BestSplit_Position: 返回最优划分点的位置，最左边为0，最右边为1，float
    """
    if len(data) < min_sample:
        ks_v, BestSplit_Point, BestSplit_Position = 0, -9999, 0.0
    else:
        # 计算每个组的卡方分箱点
        freq_df = pd.crosstab(index=data[var], columns=data[target])
        freq_array = freq_df.values
        if freq_array.shape[1] == 1:  # 如果某一组只有一个枚举值，如0或1，则数组形状会有问题，跳出本次计算
            # tt = np.zeros(freq_array.shape).T
            # freq_array = np.insert(freq_array, 0, values=tt, axis=1)
            ks_v, BestSplit_Point, BestSplit_Position = 0, -99999, 0.0
        else:
            bincut = freq_df.index.values
            tmp = freq_array.cumsum(axis=0) / (np.ones(freq_array.shape) * freq_array.sum(axis=0).T)
            tmp_abs = abs(tmp.T[0] - tmp.T[1])
            ks_v = tmp_abs.max()
            BestSplit_Point = bincut[tmp_abs.tolist().index(ks_v)]
            BestSplit_Position = tmp_abs.tolist().index(ks_v) / max(len(bincut) - 1, 1)

    return ks_v, BestSplit_Point, BestSplit_Position


def get_bestks_bincut(data, var, target, leaf_stop_percent=0.05):
    """ 计算最优分箱切分点
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        leaf_stop_percent: 叶子节点占比，作为停止条件，默认5%

    Returns:
        best_bincut: 最优的切分点列表，List
    """
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []

    def cutting_data(data, var, target, min_sample, best_bincut):
        ks, split_point, position = get_maxks_split_point(data, var, target, min_sample)

        if split_point != -99999:
            best_bincut.append(split_point)

        # 根据最优切分点切分数据集，并对切分后的数据集递归计算切分点，直到满足停止条件
        # print("本次分箱的值域范围为{0} ~ {1}".format(data[var].min(), data[var].max()))
        left = data[data[var] < split_point]
        right = data[data[var] > split_point]

        # 当切分后的数据集仍大于最小数据样本要求，则继续切分
        if len(left) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(left, var, target, min_sample, best_bincut)
        else:
            pass
        if len(right) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(right, var, target, min_sample, best_bincut)
        else:
            pass
        return best_bincut

    best_bincut = cutting_data(data, var, target, min_sample, best_bincut)

    # 把切分点补上头尾
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)

    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min() - 1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


# In[ ]:


# 等频分箱
def bin_frequency(x, y, n):
    total = y.count()
    bad = y.sum()
    good = total - bad
    d1 = pd.DataFrame({"x": x, "y": y, 'bin': pd.qcut(x, n, duplicates='drop')})
    d2 = d1.groupby('bin', as_index=True)
    d3 = pd.DataFrame()
    d3['total'] = d2.y.count()  ##每个箱中的总样本数
    d3['bad'] = d2.y.sum()  ##每个箱中的坏样本数
    d3['good'] = d3['total'] - d3['bad']  ##每个箱中的好样本数
    d3['bad_rate'] = d3['bad'] / d3['total'] * 100  ##每个箱中的坏样本率
    d3['%bad'] = d3['bad'] / bad * 100  ##每个箱中的坏样本占总坏样本的比重
    d3['%good'] = d3['good'] / good * 100  ##每个箱中的好样本占总好样本的比重
    d3['%cum_bad'] = d3['%bad'].cumsum()
    d3['%cum_good'] = d3['%good'].cumsum()
    d3['woe'] = np.log(d3['%bad'] / d3['%good'])
    d3['bin_iv'] = (d3['%bad'] - d3['%good']) * d3['woe']
    d3['iv'] = d3['bin_iv'].sum()
    d3['bin_ks'] = d3['%cum_bad'] - d3['%cum_good']
    d3['ks'] = d3['bin_ks'].max()
    d3.reset_index(inplace=True)
    return d3


'''
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    all_odds = good/bad
    inf = float('inf')
    ninf = float('-inf')
    bin_df=[]
    iv_value=[]
    d1 = df.groupby(bucket)
    d2 = pd.DataFrame()
    d2['min_bin'] = d1[col].min()
    d2['max_bin'] = d1[col].max()
    d2['total'] = d1[target].count()
    d2['totalrate'] = d2['total']/total
    d2['bad'] = d1[target].sum()
    d2['badrate'] = d2['bad']/d2['total']
    d2['good'] = d2['total'] - d2['bad']
    d2['goodrate'] = d2['good']/d2['total']
    d2['badattr'] = d2['bad']/bad
    d2['goodattr'] = (d2['total']-d2['bad'])/good
    d2['cumgoodrate'] =(d2['goodrate']*d2['totalrate']/(good/total)).cumsum()
    d2['cumbadrate'] =(d2['badrate']*d2['totalrate']/(bad/total)).cumsum()
    d2['odds'] = d2['good']/d2['bad']
    GB_list=[]
    for i in d2.odds:
        if i>=all_odds:
            GB_index = str(round((i/all_odds)*100,0))+str('G')
        else:
            GB_index = str(round((all_odds/i)*100,0))+str('B')
        GB_list.append(GB_index)
    d2['GB_index'] = GB_list
    d2['woe'] = np.log(d2['badattr']/d2['goodattr'])
    d2['bin_iv'] = (d2['badattr']-d2['goodattr'])*d2['woe']
    d2['IV'] = d2['bin_iv'].sum()
    iv = d2['bin_iv'].sum().round(3)
    print('变量名:{}'.format(col))
    print('IV:{}'.format(iv))
    print('\t')
    bin_df.append(d2)
    iv_value.append(iv)
    d2.reset_index(inplace=True)
'''


# 等距分箱
def bin_distance(x, y, n=10):  ##主要woe有可能为-inf
    total = y.count()
    bad = y.sum()
    good = total - bad
    d1 = pd.DataFrame({"x": x, "y": y, 'bin': pd.cut(x, n)})  ##等距分箱
    d2 = d1.groupby('bin', as_index=True)
    d3 = pd.DataFrame()
    d3['total'] = d2.y.count()  ##每个箱中的总样本数
    d3['bad'] = d2.y.sum()  ##每个箱中的坏样本数
    d3['good'] = d3['total'] - d3['bad']  ##每个箱中的好样本数
    d3['bad_rate'] = d3['bad'] / d3['total'] * 100  ##每个箱中的坏样本率
    d3['%bad'] = d3['bad'] / bad * 100  ##每个箱中的坏样本占总坏样本的比重
    d3['%good'] = d3['good'] / good * 100  ##每个箱中的好样本占总好样本的比重
    d3['%cum_bad'] = d3['%bad'].cumsum()
    d3['%cum_good'] = d3['%good'].cumsum()
    d3['woe'] = np.log(d3['%bad'] / d3['%good'])
    d3['bin_iv'] = (d3['%bad'] - d3['%good']) * d3['woe']
    d3['iv'] = d3['bin_iv'].sum()
    d3['bin_ks'] = d3['%cum_bad'] - d3['%cum_good']
    d3['ks'] = d3['bin_ks'].max()
    d3.reset_index(inplace=True)
    return d3


# 单调分箱
def mono_bin(x, y, n=20):
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'x': x,
                           'y': y,
                           'bin': pd.qcut(x, n)})
        d2 = d1.groupby('bin', as_index=True)
        print(d1)
        r, p = spearmanr(d2.mean().x, d2.mean().y)
        print(r, p)
        n = n + 1
        print(n)
    total = y.count()
    bad = y.sum()
    good = total - bad
    d3 = pd.DataFrame()
    d3['total'] = d2.y.count()  ##每个箱中的总样本数
    d3['bad'] = d2.y.sum()  ##每个箱中的坏样本数
    d3['good'] = d3['total'] - d3['bad']  ##每个箱中的好样本数
    d3['bad_rate'] = d3['bad'] / d3['total'] * 100  ##每个箱中的坏样本率
    d3['%bad'] = d3['bad'] / bad * 100  ##每个箱中的坏样本占总坏样本的比重
    d3['%good'] = d3['good'] / good * 100  ##每个箱中的好样本占总好样本的比重
    d3['%cum_bad'] = d3['%bad'].cumsum()
    d3['%cum_good'] = d3['%good'].cumsum()
    d3['woe'] = np.log(d3['%bad'] / d3['%good'])
    d3['bin_iv'] = (d3['%bad'] - d3['%good']) * d3['woe']
    d3['iv'] = d3['bin_iv'].sum()
    d3['bin_ks'] = d3['%cum_bad'] - d3['%cum_good']
    d3['ks'] = d3['bin_ks'].max()
    d3.reset_index(inplace=True)
    return d3


# 自定义分箱
def bin_self(x, y, cut):  ##cut:自定义分箱（list）
    total = y.count()
    bad = y.sum()
    good = total - bad
    d1 = pd.DataFrame({"x": x, "y": y, 'bin': pd.cut(x, cut)})  ##等距分箱
    d2 = d1.groupby('bin', as_index=True)
    d3 = pd.DataFrame()
    d3['total'] = d2.y.count()  ##每个箱中的总样本数
    d3['bad'] = d2.y.sum()  ##每个箱中的坏样本数
    d3['good'] = d3['total'] - d3['bad']  ##每个箱中的好样本数
    d3['bad_rate'] = d3['bad'] / d3['total'] * 100  ##每个箱中的坏样本率
    d3['%bad'] = d3['bad'] / bad * 100  ##每个箱中的坏样本占总坏样本的比重
    d3['%good'] = d3['good'] / good * 100  ##每个箱中的好样本占总好样本的比重
    d3['%cum_bad'] = d3['%bad'].cumsum()
    d3['%cum_good'] = d3['%good'].cumsum()
    d3['woe'] = np.log(d3['%bad'] / d3['%good'])
    d3['bin_iv'] = (d3['%bad'] - d3['%good']) * d3['woe']
    d3['iv'] = d3['bin_iv'].sum()
    d3['bin_ks'] = d3['%cum_bad'] - d3['%cum_good']
    d3['ks'] = d3['bin_ks'].max()
    d3.reset_index(inplace=True)
    return d3


# In[ ]:
# Examples:

# 先卡方分箱
# a = get_chimerge_bincut(df, '决策分数', 'target')
# 再自定义画分箱
# bins_chimerge = bin_self(df['决策分数'], df['target'], a)

# a = get_bestks_bincut(df, '决策分数', 'target')
# bins_bestks = bin_self(df['决策分数'], df['target'], a)

# In[ ]:


# In[ ]:
# 测试
'''
df['score_bins1'] = pd.cut(df['决策分数'], bins=get_cart_bincut(df, '决策分数', 'target'))
df['score_bins2'] = pd.cut(df['决策分数'], bins=get_chimerge_bincut(df, '决策分数', 'target'))
df['score_bins3'] = pd.cut(df['决策分数'], bins=get_bestks_bincut(df, '决策分数', 'target'))
print("变量 决策分数 的分箱结果如下：")
print("score_cart_bins:", get_cart_bincut(df, 'score', 'target'))
print("score_chimerge_bins:", get_chimerge_bincut(df, '决策分数', 'target'))
print("score_bestks_bins:", get_bestks_bincut(df, 'score', 'target'))
print("IV值如下：")
print("score:", iv_count(df, '决策分数', 'target'))
print("score_cart_bins:", iv_count(df, 'score_bins1', 'target'))
print("score_chimerge_bins:", iv_count(df, 'score_bins2', 'target'))
print("score_bestks_bins:", iv_count(df, 'score_bins3', 'target'))


df['income_bins1'] = pd.cut(df['income'], bins=get_cart_bincut(df, 'income', 'target'))
df['income_bins2'] = pd.cut(df['income'], bins=get_chimerge_bincut(df, 'income', 'target'))
df['income_bins3'] = pd.cut(df['income'], bins=get_bestks_bincut(df, 'income', 'target'))
print("变量 income 的分箱结果如下：")
print("income_cart_bins:", get_cart_bincut(df, 'income', 'target'))
print("income_chimerge_bins:", get_chimerge_bincut(df, 'income', 'target'))
print("income_bestks_bins:", get_bestks_bincut(df, 'income', 'target'))
print("IV值如下：")
print("income:", iv_count(df, 'income', 'target'))
print("income_cart_bins:", iv_count(df, 'income_bins1', 'target'))
print("income_chimerge_bins:", iv_count(df, 'income_bins2', 'target'))
print("income_bestks_bins:", iv_count(df, 'income_bins3', 'target'))
'''
