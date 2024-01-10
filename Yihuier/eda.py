import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

import pandas as pd
import numpy as np
from scipy.stats import entropy

class EDAModule:
    def __init__(self, yihuier_instance):
        self.yihuier_instance = yihuier_instance
        self.data = self.yihuier_instance.data.copy()
        self.variables = self.yihuier_instance.data.columns
        self.category_variables = self.yihuier_instance.get_categorical_variables()
        self.numeric_variables = self.yihuier_instance.get_numeric_variables()

    # 类别型变量的分布
    def plot_cate_var(self, col_list, hspace=0.4, wspace=0.4, plt_size=None, plt_num=None, x=None, y=None):
        """
        self.yihuier_instance.data:数据集
        col_list:变量list集合
        hspace :子图之间的间隔(y轴方向)
        wspace :子图之间的间隔(x轴方向)
        plt_size :图纸的尺寸
        plt_num :子图的数量
        x :子图矩阵中一行子图的数量
        y :子图矩阵中一列子图的数量

        return :变量的分布图（柱状图形式）
        """
        plt.figure(figsize=plt_size)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        for i, col in zip(range(1, plt_num + 1, 1), col_list):
            plt.subplot(x, y, i)
            plt.title(col)
            sns.countplot(data=self.yihuier_instance.data, y=col)
            plt.ylabel('')
        return plt.show()

    # 数值型变量的分布
    def plot_num_col(self, col_list, plt_type='hist', hspace=0.4, wspace=0.4, plt_size=None, plt_num=None, x=None,
                     y=None):
        """
        col_list:变量list集合
        hspace :子图之间的间隔(y轴方向)
        wspace :子图之间的间隔(x轴方向)
        plt_type: 选择直方图/箱线图
        plt_size :图纸的尺寸
        plt_num :子图的数量
        x :子图矩阵中一行子图的数量
        y :子图矩阵中一列子图的数量

        return :变量的分布图（箱线图/直方图）
        """
        plt.figure(figsize=plt_size)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        if plt_type == 'hist':
            for i, col in zip(range(1, plt_num + 1, 1), col_list):
                plt.subplot(x, y, i)
                plt.title(col)
                sns.distplot(self.yihuier_instance.data[col].dropna())
                plt.xlabel('')
        if plt_type == 'box':
            for i, col in zip(range(1, plt_num + 1, 1), col_list):
                plt.subplot(x, y, i)
                plt.title(col)
                sns.boxplot(data=self.yihuier_instance.data, x=col, fliersize=5,
                            flierprops={'markerfacecolor': 'cornflowerblue', 'markeredgecolor': 'cornflowerblue',
                                        'markersize': 4}, )
                plt.xlabel('')
        if plt_type == 'stripplot':
            for i, col in zip(range(1, plt_num + 1, 1), col_list):
                plt.subplot(x, y, i)
                plt.title(col)
                sns.stripplot(data=self.yihuier_instance.data, x=col)
                plt.xlabel('')
        return plt.show()

    # 类别型变量的违约率分析
    def plot_default_cate(self, col_list, hspace=0.4, wspace=0.4, plt_size=None, plt_num=None, x=None, y=None):
        """
        col_list:变量list集合
        hspace :子图之间的间隔(y轴方向)
        wspace :子图之间的间隔(x轴方向)
        plt_size :图纸的尺寸
        plt_num :子图的数量
        x :子图矩阵中一行子图的数量
        y :子图矩阵中一列子图的数量

        return :违约率分布图（柱状图形式）
        """

        all_bad = self.yihuier_instance.data[self.yihuier_instance.target].sum()
        total = self.yihuier_instance.data[self.yihuier_instance.target].count()
        all_default_rate = all_bad * 1.0 / total

        plt.figure(figsize=plt_size)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        for i, col in zip(range(1, plt_num + 1, 1), col_list):
            d1 = self.yihuier_instance.data.groupby(col)
            d2 = pd.DataFrame()
            d2['total'] = d1[self.yihuier_instance.target].count()
            d2['bad'] = d1[self.yihuier_instance.target].sum()
            d2['default_rate'] = d2['bad'] / d2['total']
            d2 = d2.reset_index()
            plt.subplot(x, y, i)
            plt.title(col)
            plt.axvline(x=all_default_rate)
            sns.barplot(data=d2, y=col, x='default_rate')
            plt.ylabel('')
        return plt.show()

    # 数值型变量的违约率分析
    def plot_default_num(self, col_list, hspace=0.4, wspace=0.4, q=None, plt_size=None, plt_num=None, x=None,
                         y=None):
        """
        self.yihuier_instance.data:数据集
        col_list:变量list集合
        self.yihuier_instance.target ：目标变量的字段名
        hspace :子图之间的间隔(y轴方向)
        wspace :子图之间的间隔(x轴方向)
        q :等深分箱的箱体个数
        plt_size :图纸的尺寸
        plt_num :子图的数量
        x :子图矩阵中一行子图的数量
        y :子图矩阵中一列子图的数量

        return :违约率分布图（折线图形式）
        """
        all_bad = self.yihuier_instance.data[self.yihuier_instance.target].sum()
        total = self.yihuier_instance.data[self.yihuier_instance.target].count()
        all_default_rate = all_bad * 1.0 / total

        plt.figure(figsize=plt_size)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        for i, col in zip(range(1, plt_num + 1, 1), col_list):
            bucket = pd.qcut(self.yihuier_instance.data[col], q=q, duplicates='drop')
            d1 = self.yihuier_instance.data.groupby(bucket)
            d2 = pd.DataFrame()
            d2['total'] = d1[self.yihuier_instance.target].count()
            d2['bad'] = d1[self.yihuier_instance.target].sum()
            d2['default_rate'] = d2['bad'] / d2['total']
            d2 = d2.reset_index()
            plt.subplot(x, y, i)
            plt.title(col, color='cornflowerblue')
            plt.axhline(y=all_default_rate)
            sns.pointplot(data=d2, x=col, y='default_rate', color='cornflowerblue')
            plt.xticks(rotation=60)
            plt.xlabel('')
        return plt.show()

    # 使用ydata_profiling进行自动EDA，维度较高时，速度较慢
    def auto_eda_profiling(self):
        # 使用pandas profiling进行自动EDA
        profile = ProfileReport(self.yihuier_instance.yihuier_instance.data,
                                title="Report",
                                correlations={"auto": {"calculate": False}},
                                missing_diagrams={"Heatmap": False}
                                )  # object created
        profile.to_file(output_file='Data/output.html')



    def __calculate_category_stats(self):
        category_stats = {}
        for var in self.category_variables:
            unique_count = self.data[var].nunique()
            entropy_val = entropy(self.data[var].value_counts(normalize=True))
            missing_pct = self.data[var].isnull().mean() * 100

            category_stats[var] = {
                'unique_count': unique_count,
                'entropy': entropy_val,
                'missing_pct': missing_pct
            }

        return pd.DataFrame(category_stats).transpose()

    def __calculate_numeric_stats(self):
        numeric_stats = {}
        for var in self.numeric_variables:
            mean_val = self.data[var].mean()
            min_val = self.data[var].min()
            q1 = self.data[var].quantile(0.25)
            median_val = self.data[var].median()
            q3 = self.data[var].quantile(0.75)
            max_val = self.data[var].max()
            missing_pct = self.data[var].isnull().mean() * 100

            numeric_stats[var] = {
                'mean': mean_val,
                'min': min_val,
                'q1': q1,
                'median': median_val,
                'q3': q3,
                'max': max_val,
                'missing_pct': missing_pct
            }

        return pd.DataFrame(numeric_stats).transpose()

    # 快速自动分析数据集（无图）
    def auto_eda_simple(self):
        category_stats = self.__calculate_category_stats()
        numeric_stats = self.__calculate_numeric_stats()

        # Combine the results into a single DataFrame
        eda_results = pd.concat([category_stats, numeric_stats])

        return eda_results
