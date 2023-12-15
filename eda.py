# 在单独的文件 eda.py 中创建 EDA 模块
from dataprep.eda import create_report

import matplotlib.pyplot as plt
plt.style.use('science')

import pandas as pd
import seaborn as sns

class EDAModule:
    def __init__(self, data):
        self.data = data

    def auto_eda_dataprep(self):
        # 使用DataPrep进行自动EDA
        report = create_report(self.data)
        # 可以选择在控制台显示报告或者保存为文件
        report.show_browser()
        # report.save("dataprep_eda_report.html")



    # 其他 EDA 功能...
