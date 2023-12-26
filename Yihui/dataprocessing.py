import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

class DataProcessingModule:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def missing_cal(self):
        """
        calculate data missing pct
        """
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        missing_data = pd.DataFrame(missing_series).reset_index()
        missing_data = missing_data.rename(columns={'index': 'col',
                                                0: 'missing_pct'})
        missing_data = missing_data.sort_values('missing_pct', ascending=False)
        return missing_data.sort_values('missing_pct', ascending=False)


    # def plot_hist_missing_var(self, plt_size=None):
    #     """
    #     plt_size: plot chart size: (10, 10)
    #
    #     return: hist chart of missing variables
    #     """
    #     missing_data = self.missing_cal()
    #     if plt_size is not None:
    #         plt.figure(figsize=plt_size)
    #     else:
    #         plt.figure()
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #     plt.rcParams['axes.unicode_minus'] = False
    #
    #     x = missing_data['missing_pct']
    #     bins = np.arange(0, 1.1, 0.1)
    #
    #     plt.hist(x=x, bins=bins, color='hotpink', edgecolor='k', alpha=0.8)
    #     plt.title('缺失值分布')
    #     plt.ylabel('缺失值个数')
    #     plt.xlabel('缺失率')
    #
    #     plt.show()

    # 所有变量缺失值分布图
    def plot_bar_missing_var(self, plt_size=None):
        """
        plt_size: plot chart size: (10, 10)

        return: bar chart of missing variables
        """
        missing_data = self.missing_cal()  # 假设 missing_cal() 返回一个 DataFrame，包含 'col' 和 'missing_pct' 列

        if plt_size is not None:
            plt.figure(figsize=plt_size)
        else:
            plt.figure()

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        x = missing_data['col']
        y = missing_data['missing_pct']

        plt.bar(x, y, color='hotpink', edgecolor='k', alpha=0.8)
        plt.title('各列缺失率')
        plt.ylabel('缺失率')
        plt.xlabel('列名')
        plt.xticks(rotation=45)

        plt.show()

    # 缺失值填充（类别型变量）
    def fillna_cate_var(self, col_list, fill_type=None, fill_str=None):
        """
        data:数据集
        col_list:变量list集合
        fill_type: 填充方式：众数/当做一个类别

        return :填充后的数据集
        """
        for col in col_list:
            if fill_type == 'class':
                self.data[col] = self.data[col].fillna(fill_str)
            if fill_type == 'mode':
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

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
        data2 = self.data.copy()
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
        data2 = self.data.copy()
        missing_data = self.missing_cal()
        missing_col_num = missing_data[missing_data.missing_pct >= threshold].shape[0]
        missing_col = list(missing_data[missing_data.missing_pct >= threshold].col)
        self.data = data2.drop(missing_col, axis=1)
        print('缺失率超过{}的变量个数为{}'.format(threshold, missing_col_num))

