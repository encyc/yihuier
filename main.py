import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas_profiling

# class DF():
#     def __init__(self):
#         self.dataframe = None
#         self.columns = self.dataframe.columns
#         self.columns_type = self.dataframe.dtypes
#         self.target = None
#
#     def get_dataframe(self, file_path):
#         # 根据文件扩展名选择合适的读取函数
#         if file_path.endswith('.csv'):
#             df = pd.read_csv(file_path)
#         elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
#             df = pd.read_excel(file_path)
#         elif file_path.endswith('.txt'):
#             df = pd.read_csv(file_path, delimiter='\t')
#         elif file_path.endswith('.json'):
#             df = pd.read_json(file_path)
#         else:
#             raise ValueError("No support file; 不支持的文件格式")
#         return df

class EDA(object):
    def __init__(self):
        self.dataframe = None
        self.columns = self.dataframe.columns
        self.target = None
        self.columns_type = self.dataframe.dtypes

    # def auto_eda(self):
    #     profile = pandas_profiling.ProfileReport(self.dataframe)
    #     profile.to_file('auto_eda.html')

    def manual_eda(self):
        pass
    def cate_var_distribution(self,column_name):
        self.dataframe[column_name].value_counts().plot(kind='bar')

    def num_var_distribution(self,column_name):
        self.dataframe[column_name].plot(kind='hist')


    def plot_cate_var(self, col_list, hspace=0.4, wspace=0.4, plt_size=None, plt_num=None, x=None, y=None):
        """
        df:数据集
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
            sns.countplot(data=df, y=col)
            plt.ylabel('')
        return plt.show()


# In[]:
df = pd.read_excel(r'D:\pythonProject\zhizu-work\modeling\同盾多头模型\data\tongdun_duotou_testdataset.xlsx')
print(df.head())

yihui = EDA()
yihui.dataframe = df
yihui.target = 'dlq_flag'
print(yihui.columns)
print(yihui.columns_type)