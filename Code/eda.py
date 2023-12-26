# 在单独的文件 eda.py 中创建 EDA 模块
# from dataprep.eda import create_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport




import seaborn as sns

class EDAModule:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def auto_eda_profiling(self):
        # 使用pandas profiling进行自动EDA
        profile = ProfileReport(self.data,
                                title="Report",
                                correlations={"auto": {"calculate":False}},
                                missing_diagrams={"Heatmap": False}
                                ) # object created
        profile.to_file(output_file='../Data/output.html')


    # #* plot_cate_var -- 类别型变量分布
    # def plot_cate_var(self, cat_var):
    #     """
    #     绘制类别型变量分布图
    # 
    #     Parameters:
    #     - data: DataFrame，包含类别型变量的数据集
    #     - cat_var: str，需要绘制分布图的类别型变量名
    # 
    #     Returns:
    #     无返回值，直接显示绘制的分布图
    #     """
    #     plt.figure(figsize=(10, 6))
    #     sns.countplot(x=cat_var, data=self.data[cat_var], palette='viridis')
    #     plt.title(f'Distribution of {cat_var}')
    #     plt.xlabel(cat_var)
    #     plt.ylabel('Count')
    #     plt.show()
    # 
    # # 示例使用
    # # 假设你的数据框为 self.data，包含一个名为 'category_column' 的类别型变量
    # # 调用函数
    # # plot_cate_var(self.data, 'category_column')
    # 
    # #* plot_num_col  -- 数值型变量分布
    # def plot_num_var(self, num_var):
    #     """
    #     绘制数值型变量分布图
    # 
    #     Parameters:
    #     - data: DataFrame，包含数值型变量的数据集
    #     - num_var: str，需要绘制分布图的数值型变量名
    # 
    #     Returns:
    #     无返回值，直接显示绘制的分布图
    #     """
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(self.data[num_var], bins=30, kde=True, color='skyblue')
    #     plt.title(f'Distribution of {num_var}')
    #     plt.xlabel(num_var)
    #     plt.ylabel('Frequency')
    #     plt.show()

    def plot_cate_var(self, col_list, hspace=0.4, wspace=0.4, plt_size=None, plt_num=None, x=None, y=None):
        """
        self.data:数据集
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
            sns.countplot(data=self.data, y=col)
            plt.ylabel('')
        return plt.show()

    # 数值型变量的分布
    def plot_num_col(self, col_list, plt_type='hist', hspace=0.4, wspace=0.4, plt_size=None, plt_num=None, x=None, y=None):
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
                sns.distplot(self.data[col].dropna())
                plt.xlabel('')
        if plt_type == 'box':
            for i, col in zip(range(1, plt_num + 1, 1), col_list):
                plt.subplot(x, y, i)
                plt.title(col)
                sns.boxplot(data=self.data, x=col, fliersize=5,
                            flierprops={'markerfacecolor': 'cornflowerblue', 'markeredgecolor': 'cornflowerblue',
                                        'markersize': 4}, )
                plt.xlabel('')
        if plt_type == 'stripplot':
            for i, col in zip(range(1, plt_num + 1, 1), col_list):
                plt.subplot(x, y, i)
                plt.title(col)
                sns.stripplot(data=self.data, x=col)
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

        all_bad = self.data[self.target].sum()
        total = self.data[self.target].count()
        all_default_rate = all_bad * 1.0 / total

        plt.figure(figsize=plt_size)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        for i, col in zip(range(1, plt_num + 1, 1), col_list):
            d1 = self.data.groupby(col)
            d2 = pd.DataFrame()
            d2['total'] = d1[self.target].count()
            d2['bad'] = d1[self.target].sum()
            d2['default_rate'] = d2['bad'] / d2['total']
            d2 = d2.reset_index()
            plt.subplot(x, y, i)
            plt.title(col)
            plt.axvline(x=all_default_rate)
            sns.barplot(data=d2, y=col, x='default_rate')
            plt.ylabel('')
        return plt.show()

    # plot_default_cate(self.data,['order_status_chinese'],'self.target',hspace=0.4,wspace=0.4,plt_size=(10,10),plt_num=1,x=1,y=1)

    # 数值型变量的违约率分析
    def plot_default_num(self, col_list, hspace=0.4, wspace=0.4, q=None, plt_size=None, plt_num=None, x=None,
                         y=None):
        """
        self.data:数据集
        col_list:变量list集合
        self.target ：目标变量的字段名
        hspace :子图之间的间隔(y轴方向)
        wspace :子图之间的间隔(x轴方向)
        q :等深分箱的箱体个数
        plt_size :图纸的尺寸
        plt_num :子图的数量
        x :子图矩阵中一行子图的数量
        y :子图矩阵中一列子图的数量

        return :违约率分布图（折线图形式）
        """
        all_bad = self.data[self.target].sum()
        total = self.data[self.target].count()
        all_default_rate = all_bad * 1.0 / total

        plt.figure(figsize=plt_size)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        for i, col in zip(range(1, plt_num + 1, 1), col_list):
            bucket = pd.qcut(self.data[col], q=q, duplicates='drop')
            d1 = self.data.groupby(bucket)
            d2 = pd.DataFrame()
            d2['total'] = d1[self.target].count()
            d2['bad'] = d1[self.target].sum()
            d2['default_rate'] = d2['bad'] / d2['total']
            d2 = d2.reset_index()
            plt.subplot(x, y, i)
            plt.title(col, color='cornflowerblue')
            plt.axhline(y=all_default_rate)
            sns.pointplot(data=d2, x=col, y='default_rate', color='cornflowerblue')
            plt.xticks(rotation=60)
            plt.xlabel('')
        return plt.show()

    # plot_default_num(data,['score'],'self.target',hspace=0.4,wspace=0.4,q=10,plt_size=(10,10),plt_num=1,x=1,y=1)