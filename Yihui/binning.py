import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr



class BinningModule:

    def __init__(self, yihui_instance):
        self.yihui_instance = yihui_instance
        self.bin_df = None
        self.woe_list = None
        self.iv_value = None
        self.ks_value = None

    def binning_cate(self, col_list):
        df = self.yihui_instance.data
        target = self.yihui_instance.target

        total = df[target].count()
        bad = df[target].sum()
        good = total - bad

        all_odds = good * 1.0 / bad

        bin_df = []
        iv_value = []
        ks_value = []
        for col in col_list:
            d1 = df.groupby([col], as_index=True)
            d2 = pd.DataFrame()
            d2['min_bin'] = d1[col].min()
            d2['max_bin'] = d1[col].max()
            d2['total'] = d1[target].count()
            d2['totalrate'] = d2['total'] / total
            d2['bad'] = d1[target].sum()
            d2['badrate'] = d2['bad'] / d2['total']
            d2['good'] = d2['total'] - d2['bad']
            d2['goodrate'] = d2['good'] / d2['total']
            d2['badattr'] = d2['bad'] / bad
            d2['goodattr'] = d2['good'] / good
            d2['cumgoodrate'] = (d2['goodrate'] * d2['totalrate'] / (good / total)).cumsum()
            d2['cumbadrate'] = (d2['badrate'] * d2['totalrate'] / (bad / total)).cumsum()
            d2['odds'] = d2['good'] / d2['bad']
            GB_list = []
            for i in d2.odds:
                if i >= all_odds:
                    GB_index = str(round((i / all_odds) * 100, 0)) + str('G')
                else:
                    GB_index = str(round((all_odds / i) * 100, 0)) + str('B')
                GB_list.append(GB_index)
            d2['GB_index'] = GB_list
            # WOEi = log( (BADi/BADtotal)/(GOODi/GODDtotal) )
            d2['woe'] = np.log(d2['badattr'] / d2['goodattr'])
            # IVi = ( (BADi/BADtotal) - (GOODi/GODDtotal) ) * WOEi
            d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['woe']
            # IV = sum(IVi)
            d2['IV'] = d2['bin_iv'].sum()
            d2['ks'] = np.abs(d2['badattr'] - d2['goodattr'])
            iv = d2['bin_iv'].sum().round(3)
            ks = d2['ks'].max()
            print('var:{}'.format(col))
            print('IV:{}'.format(iv))
            print('KS:{}'.format(ks))
            bin_df.append(d2)
            iv_value.append(iv)
            ks_value.append(ks)
        self.bin_df = bin_df
        self.iv_value = iv_value
        self.ks_value = ks_value
        return bin_df, iv_value, ks_value

    # 数值型变量的分箱

    # 先用卡方分箱输出变量的分割点
    def __split_data(df2, col, split_num):
        """
        df: 原始数据集
        col:需要分箱的变量
        split_num:分割点的数量
        """
        count = df2.shape[0]  # 总样本数
        n = math.floor(count / split_num)  # 按照分割点数目等分后每组的样本数
        split_index = [i * n for i in range(1, split_num)]  # 分割点的索引
        values = sorted(list(df2[col]))  # 对变量的值从小到大进行排序
        split_value = [values[i] for i in split_index]  # 分割点对应的value
        split_value = sorted(list(set(split_value)))  # 分割点的value去重排序
        return split_value


    def __bin_bad_rate(self, col, target, grant_rate_indicator=0):
        """
        df:原始数据集
        col:原始变量/变量映射后的字段
        target:目标变量的字段
        grantRateIndicator:是否输出总体的违约率
        """
        df = self.yihui_instance.data
        total = df.groupby([col])[target].count()
        bad = df.groupby([col])[target].sum()
        total_df = pd.DataFrame({'total': total})
        bad_df = pd.DataFrame({'bad': bad})
        regroup = pd.merge(total_df, bad_df, left_index=True, right_index=True, how='left')
        regroup = regroup.reset_index()
        regroup['bad_rate'] = regroup['bad'] / regroup['total']  # 计算根据col分组后每组的违约率
        dict_bad = dict(zip(regroup[col], regroup['bad_rate']))  # 转为字典形式
        if grantRateIndicator == 0:
            return dict_bad, regroup
        total_all = df.shape[0]
        bad_all = df[target].sum()
        all_bad_rate = bad_all / total_all  # 计算总体的违约率
        return dict_bad, regroup, all_bad_rate

    def __cal_chi2(self, all_bad_rate):
        """
        df:bin_bad_rate得出的regroup
        all_bad_rate:bin_bad_rate得出的总体违约率
        """
        df2 = self.yihui_instance.data.copy()

        df2['expected'] = df2['total'] * all_bad_rate  # 计算每组的坏用户期望数量
        combined = zip(df2['expected'], df2['bad'])  # 遍历每组的坏用户期望数量和实际数量
        chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]  # 计算每组的卡方值
        chi2 = sum(chi)  # 计算总的卡方值
        return chi2

    def __assign_group(x, split_bin):
        """
        x:变量的value
        split_bin:split_data得出的分割点list
        """
        n = len(split_bin)
        if x <= min(split_bin):
            return min(split_bin)  # 如果x小于分割点的最小值，则x映射为分割点的最小值
        elif x > max(split_bin):  # 如果x大于分割点的最大值，则x映射为分割点的最大值
            return 10e10
        else:
            for i in range(n - 1):
                if split_bin[i] < x <= split_bin[i + 1]:  # 如果x在两个分割点之间，则x映射为分割点较大的值
                    return split_bin[i + 1]

    def __assign_bin(self, x, cutoffpoints):
        """
        x:变量的value
        cutoffpoints:分箱的切割点
        """
        bin_num = len(cutoffpoints) + 1  # 箱体个数
        if x <= cutoffpoints[0]:  # 如果x小于最小的cutoff点，则映射为Bin 0
            return 'Bin 0'
        elif x > cutoffpoints[-1]:  # 如果x大于最大的cutoff点，则映射为Bin(bin_num-1)
            return 'Bin {}'.format(bin_num - 1)
        else:
            for i in range(0, bin_num - 1):
                if cutoffpoints[i] < x <= cutoffpoints[i + 1]:  # 如果x在两个cutoff点之间，则x映射为Bin(i+1)
                    return 'Bin {}'.format(i + 1)

    def __chi_merge(self, df, col, target, **kwargs):
        max_bin = kwargs.get('max_bin', None)
        min_binpct = kwargs.get('min_binpct', None)

        df = self.yihui_instance.data.copy()

        col_unique = sorted(list(set(df[col])))  # 变量的唯一值并排序
        n = len(col_unique)  # 变量唯一值得个数
        df2 = df.copy()
        if n > 100:  # 如果变量的唯一值数目超过100，则将通过__split_data和__assign_group将x映射为split对应的value
            split_col = self.__split_data(df2, col, 100)  # 通过这个目的将变量的唯一值数目人为设定为100
            df2['col_map'] = df2[col].map(lambda x: self.__assign_group(x, split_col))
        else:
            df2['col_map'] = df2[col]  # 变量的唯一值数目没有超过100，则不用做映射
        # 生成dict_bad,regroup,all_bad_rate的元组
        (dict_bad, regroup, all_bad_rate) = self.__bin_bad_rate(df2, 'col_map', target, grant_rate_indicator=1)
        col_map_unique = sorted(list(set(df2['col_map'])))  # 对变量映射后的value进行去重排序
        group_interval = [[i] for i in col_map_unique]  # 对col_map_unique中每个值创建list并存储在group_interval中

        while (len(group_interval) > max_bin):  # 当group_interval的长度大于max_bin时，执行while循环
            chi_list = []
            for i in range(len(group_interval) - 1):
                temp_group = group_interval[i] + group_interval[i + 1]  # temp_group 为生成的区间,list形式，例如[1,3]
                chi_df = regroup[regroup['col_map'].isin(temp_group)]
                chi_value = self.__cal_chi2(chi_df, all_bad_rate)  # 计算每一对相邻区间的卡方值
                chi_list.append(chi_value)
            best_combined = chi_list.index(min(chi_list))  # 最小的卡方值的索引
            # 将卡方值最小的一对区间进行合并
            group_interval[best_combined] = group_interval[best_combined] + group_interval[best_combined + 1]
            # 删除合并前的右区间
            group_interval.remove(group_interval[best_combined + 1])
            # 对合并后每个区间进行排序
        group_interval = [sorted(i) for i in group_interval]
        # cutoff点为每个区间的最大值
        cutoffpoints = [max(i) for i in group_interval[:-1]]

        # 检查是否有箱只有好样本或者只有坏样本
        df2['col_map_bin'] = df2['col_map'].apply(lambda x: self.__assign_bin(x, cutoffpoints))  # 将col_map映射为对应的区间Bin
        # 计算每个区间的违约率
        (dict_bad, regroup) = self.__bin_bad_rate(df2, 'col_map_bin', target)
        # 计算最小和最大的违约率
        [min_bad_rate, max_bad_rate] = [min(dict_bad.values()), max(dict_bad.values())]
        # 当最小的违约率等于0，说明区间内只有好样本，当最大的违约率等于1，说明区间内只有坏样本
        while min_bad_rate == 0 or max_bad_rate == 1:
            bad01_index = regroup[regroup['bad_rate'].isin([0, 1])].col_map_bin.tolist()  # 违约率为1或0的区间
            bad01_bin = bad01_index[0]
            if bad01_bin == max(regroup.col_map_bin):
                cutoffpoints = cutoffpoints[:-1]  # 当bad01_bin是最大的区间时，删除最大的cutoff点
            elif bad01_bin == min(regroup.col_map_bin):
                cutoffpoints = cutoffpoints[1:]  # 当bad01_bin是最小的区间时，删除最小的cutoff点
            else:
                bad01_bin_index = list(regroup.col_map_bin).index(bad01_bin)  # 找出bad01_bin的索引
                prev_bin = list(regroup.col_map_bin)[bad01_bin_index - 1]  # bad01_bin前一个区间
                df3 = df2[df2.col_map_bin.isin([prev_bin, bad01_bin])]
                (dict_bad, regroup1) = self.__bin_bad_rate(df3, 'col_map_bin', target)
                chi1 = self.__cal_chi2(regroup1, all_bad_rate)  # 计算前一个区间和bad01_bin的卡方值
                later_bin = list(regroup.col_map_bin)[bad01_bin_index + 1]  # bin01_bin的后一个区间
                df4 = df2[df2.col_map_bin.isin([later_bin, bad01_bin])]
                (dict_bad, regroup2) = self.__bin_bad_rate(df4, 'col_map_bin', target)
                chi2 = self.__cal_chi2(regroup2, all_bad_rate)  # 计算后一个区间和bad01_bin的卡方值
                if chi1 < chi2:  # 当chi1<chi2时,删除前一个区间对应的cutoff点
                    cutoffpoints.remove(cutoffpoints[bad01_bin_index - 1])
                else:  # 当chi1>=chi2时,删除bin01对应的cutoff点
                    cutoffpoints.remove(cutoffpoints[bad01_bin_index])
            df2['col_map_bin'] = df2['col_map'].apply(lambda x: self.__assign_bin(x, cutoffpoints))
            (dict_bad, regroup) = self.__bin_bad_rate(df2, 'col_map_bin', target)
            # 重新将col_map映射至区间，并计算最小和最大的违约率，直达不再出现违约率为0或1的情况，循环停止
            [min_bad_rate, max_bad_rate] = [min(dict_bad.values()), max(dict_bad.values())]

        # 检查分箱后的最小占比
        if min_binpct > 0:
            group_values = df2['col_map'].apply(lambda x: self.__assign_bin(x, cutoffpoints))
            df2['col_map_bin'] = group_values  # 将col_map映射为对应的区间Bin
            group_df = group_values.value_counts().to_frame()
            group_df['bin_pct'] = group_df['col_map'] / n  # 计算每个区间的占比
            min_pct = group_df.bin_pct.min()  # 得出最小的区间占比
            while min_pct < min_binpct and len(cutoffpoints) > 2:  # 当最小的区间占比小于min_pct且cutoff点的个数大于2，执行循环
                # 下面的逻辑基本与“检验是否有箱体只有好/坏样本”的一致
                min_pct_index = group_df[group_df.bin_pct == min_pct].index.tolist()
                min_pct_bin = min_pct_index[0]
                if min_pct_bin == max(group_df.index):
                    cutoffpoints = cutoffpoints[:-1]
                elif min_pct_bin == min(group_df.index):
                    cutoffpoints = cutoffpoints[1:]
                else:
                    minpct_bin_index = list(group_df.index).index(min_pct_bin)
                    prev_pct_bin = list(group_df.index)[minpct_bin_index - 1]
                    df5 = df2[df2['col_map_bin'].isin([min_pct_bin, prev_pct_bin])]
                    (dict_bad, regroup3) = self.__bin_bad_rate(df5, 'col_map_bin', target)
                    chi3 = self.__cal_chi2(regroup3, all_bad_rate)
                    later_pct_bin = list(group_df.index)[minpct_bin_index + 1]
                    df6 = df2[df2['col_map_bin'].isin([min_pct_bin, later_pct_bin])]
                    (dict_bad, regroup4) = self.__bin_bad_rate(df6, 'col_map_bin', target)
                    chi4 = self.__cal_chi2(regroup4, all_bad_rate)
                    if chi3 < chi4:
                        cutoffpoints.remove(cutoffpoints[minpct_bin_index - 1])
                    else:
                        cutoffpoints.remove(cutoffpoints[minpct_bin_index])
        return cutoffpoints

    # 数值型变量的分箱
    def binning_num(self, col_list, n=10, max_bin=None, min_binpct=None, method=None, leaf_stop_percent=0.05):
        """
        df:数据集
        target:目标变量的字段名
        col_list:变量list集合
        max_bin:最大的分箱个数
        min_binpct:区间内样本所占总体的最小比
        method:分箱方式（freq,count,ChiMerge(default),cart)
        leaf_stop_percent:叶子节点占比，作为停止条件，默认5%()
    
        return:
        bin_df :list形式，里面存储每个变量的分箱结果
        iv_value:list形式，里面存储每个变量的IV值
        """
        df = self.yihui_instance.data.copy()
        target = self.yihui_instance.target

        total = df[target].count()
        bad = df[target].sum()
        good = total - bad
        all_odds = good / bad
        inf = float('inf')
        ninf = float('-inf')
        bin_df = []
        iv_value = []
        ks_value = []

        for col in col_list:
            if method == 'freq':  # 等频分箱
                bucket = pd.qcut(df[col], n, duplicates='drop')
            elif method == 'count':  # 等距分箱
                bucket = pd.cut(df[col], n)
            # elif method == 'cart':  # 决策树cart
            #     cut = sc.binning_function.get_cart_bincut(df, col, target, leaf_stop_percent=leaf_stop_percent)
            #     bucket = pd.cut(df[col], cut)
            elif method == 'monotonic':
                r = 0
                while np.abs(r) < 1:
                    d_1 = pd.DataFrame({'col': df[col],
                                        'target': df[target],
                                        'bin': pd.qcut(df[col], n)})
                    d_2 = d_1.groupby('bin', as_index=True)
                    r, p = spearmanr(d_2['col'].mean(), d_2['target'].mean())
                    n = n + 1
                bucket = pd.qcut(df[col], n - 1)
                print(bucket)
            elif method == 'ChiMerge':  # 卡方
                cut = self.__chi_merge(df, col, target, max_bin=max_bin, min_binpct=min_binpct)
                cut.insert(0, ninf)
                cut.append(inf)
                bucket = pd.cut(df[col], cut)
            d1 = df.groupby(bucket)
            d2 = pd.DataFrame()
            d2['min_bin'] = d1[col].min()
            d2['max_bin'] = d1[col].max()
            d2['total'] = d1[target].count()
            d2['totalrate'] = d2['total'] / total
            d2['bad'] = d1[target].sum()
            d2['badrate'] = d2['bad'] / d2['total']
            d2['good'] = d2['total'] - d2['bad']
            d2['goodrate'] = d2['good'] / d2['total']
            d2['badattr'] = d2['bad'] / bad
            d2['goodattr'] = (d2['total'] - d2['bad']) / good
            d2['cumgoodrate'] = (d2['goodrate'] * d2['totalrate'] / (good / total)).cumsum()
            d2['cumbadrate'] = (d2['badrate'] * d2['totalrate'] / (bad / total)).cumsum()
            d2['odds'] = d2['good'] / d2['bad']
            GB_list = []
            for i in d2.odds:
                if i >= all_odds:
                    GB_index = str(round((i / all_odds) * 100, 0)) + str('G')
                else:
                    GB_index = str(round((all_odds / i) * 100, 0)) + str('B')
                GB_list.append(GB_index)
            d2['GB_index'] = GB_list
            d2['woe'] = np.log(d2['badattr'] / d2['goodattr'])
            d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['woe']
            d2['IV'] = d2['bin_iv'].sum()
            iv = d2['bin_iv'].sum().round(3)
            ks = d2['ks'].max()
            print('var:{}'.format(col))
            print('IV:{}'.format(iv))
            print('KS:{}'.format(ks))
            bin_df.append(d2)
            iv_value.append(iv)
            ks_value.append(ks)

        return bin_df, iv_value, ks_value
