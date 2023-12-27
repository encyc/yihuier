import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

class DataProcessingModule:
    def __init__(self, yihui_instance):
        self.yihui_instance = yihui_instance

    def __missing_var_cal(self):
        """
        calculate var missing pct
        """
        total = self.yihui_instance.data.shape[0]
        missing_count = self.yihui_instance.data.isnull().sum()
        missing_pct = missing_count/total
        missing_data = pd.DataFrame({
            'index': self.yihui_instance.data.columns.tolist(),
            'total_obs': total,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
        return missing_data

    def __missing_obs_cal(self):
        """
        calculate obs missing pct
        """
        total = len(self.yihui_instance.data.columns)
        missing_count = self.yihui_instance.data.isnull().sum(axis=1)
        missing_pct = missing_count / total
        missing_data = pd.DataFrame({
            'index': self.yihui_instance.data.index.tolist(),
            'total_obs': total,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
        print(missing_data)
        return missing_data



    # 所有变量缺失值分布图
    def plot_bar_missing_var(self, plt_size=None):
        """
        plt_size: plot chart size: (10, 10)

        return: bar chart of missing variables
        """
        missing_data = self.__missing_var_cal()
        print('plot_bar')
        print(missing_data)
        if plt_size is not None:
            plt.figure(figsize=plt_size)
        else:
            plt.figure()

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        x = missing_data['index']
        y = missing_data['missing_pct']

        plt.bar(x, y, color='hotpink', edgecolor='k', alpha=0.8)
        plt.title('各列缺失率')
        plt.ylabel('缺失率')
        plt.xlabel('列名')
        plt.xticks(rotation=45)

        plt.show()

    # TODO:
    #* 单个样本缺失值分布图
    # def plot_bar_missing_obs(self):
        


    # 缺失值填充（类别型变量）
    def fillna_cate_var(self, col_list, fill_type=None, fill_str=None):
        """
        data:数据集
        col_list:变量list集合
        fill_type: 填充方式：众数/当做一个类别

        return :填充后的数据集
        """
        data2 = self.yihui_instance.data.copy()
        for col in col_list:
            if fill_type == 'class':
                data2[col] = data2[col].fillna(fill_str)
            if fill_type == 'mode':
                data2[col] = data2[col].fillna(data2[col].mode()[0])
        return data2

    # 数值型变量的填充
    # 针对缺失率在5%以下的变量用中位数填充
    # 缺失率在5%--15%的变量用随机森林填充,可先对缺失率较低的变量先用中位数填充，在用没有缺失的样本来对变量作随机森林填充
    # 缺失率超过15%的变量建议当作一个类别
    def fillna_num_var(self, col_list, fill_type=None, fill_class_num=None, filled_data=None):
        """
        data:数据集
        col_list:变量list集合
        fill_type:填充方式：0/中位数/当作一个类别/随机森林
        fill_class_num:用特殊数值填充，当填充方式为当作一个类别时使用
        filled_data :已填充好的数据集，当填充方式为随机森林时使用

        return:已填充好的数据集
        """
        data2 = self.yihui_instance.data.copy()
        for col in col_list:
            if fill_type == '0':
                data2[col] = data2[col].fillna(0)
            if fill_type == 'median':
                data2[col] = data2[col].fillna(data2[col].median())
            if fill_type == 'class':
                data2[col] = data2[col].fillna(fill_class_num)
            if fill_type == 'rf':
                rf_data = pd.concat([data2[col], filled_data], axis=1)
                known = rf_data[rf_data[col].notnull()]
                unknown = rf_data[rf_data[col].isnull()]
                x_train = known.drop([col], axis=1)
                y_train = known[col]
                x_pre = unknown.drop([col], axis=1)
                rf = RandomForestRegressor(random_state=0)
                rf.fit(x_train, y_train)
                y_pre = rf.predict(x_pre)
                data2.loc[data2[col].isnull(), col] = y_pre
        return data2

    # 缺失值剔除（单个变量）
    def delete_missing_var(self, threshold=None):
        """
        data:数据集
        threshold:缺失率删除的阈值

        return :删除缺失后的数据集
        """
        data2 = self.yihui_instance.data.copy()
        missing_data = self.__missing_var_cal()
        missing_col_num = missing_data[missing_data.missing_pct >= threshold].shape[0]
        missing_col = list(missing_data[missing_data.missing_pct >= threshold].index)
        data2 = data2.drop(missing_col, axis=1)
        print('缺失率超过{}的变量个数为{}'.format(threshold, missing_col_num))
        return data2

    def delete_missing_obs(self, threshold=None):
        """
        删除包含超过阈值数量的缺失值的 observation

        Parameters:
        - threshold: 允许的最大缺失变量数量

        Returns:
        删除缺失后的数据集
        """
        data2 = self.yihui_instance.data.copy()
        # 计算每个 observation 中缺失值的数量
        missing_data = self.__missing_obs_cal()

        # 找到缺失值数量超过阈值的 observation 的索引
        if threshold >= 1:
            obs_to_remove = missing_data[missing_data['missing_count'] >= threshold].index
        else:
            obs_to_remove = missing_data[missing_data['missing_pct'] >= threshold].index
        data2 = data2.drop(obs_to_remove, axis=0)
        print('含有超过{}个缺失值的样本数量为{}'.format(threshold, len(obs_to_remove)))
        return data2

    # 常变量/同值化处理
    def const_delete(self, threshold=0.9):
        """
        删除常变量/同值化处理

        Parameters:
        - threshold: 同值化处理的阈值，默认为 0.9

        Returns:
        删除常变量/同值化处理后的数据集
        """
        # 计算每一列中唯一值的比例
        unique_ratio = self.yihui_instance.data.nunique() / len(self.yihui_instance.data)

        # 找到同值比例超过阈值的列
        const_columns = unique_ratio[unique_ratio >= threshold].index

        # 删除常变量/同值化处理后的数据集
        data_after_const_delete = self.yihui_instance.data.drop(columns=const_columns)

        print('删除常变量/同值化处理后的变量个数为{},名字为{}'.format(len(const_columns),const_columns))
        print(data_after_const_delete.shape[1])
        return data_after_const_delete

    # 缺失目标变量删除
    def target_missing_delete(self):
        """
        删除目标变量为空的观测

        Returns:
        删除缺失目标变量后的数据集
        """
        if self.yihui_instance.target is not None:
            data_without_missing_target = self.yihui_instance.data.dropna(subset=[self.yihui_instance.target])
            missing_target_count = len(self.yihui_instance.data) - len(data_without_missing_target)
            print('删除目标变量缺失的观测数: {}'.format(missing_target_count))
            return data_without_missing_target
        else:
            print('未指定目标变量，无法执行删除操作。')