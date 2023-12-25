# 在 main.py 或者 __init__.py 中创建主类 Yihui
from eda import EDAModule
import pandas as pd

class Yihui:
    def __init__(self, data):
        self.data = data
        self.categorical_vars = self.get_categorical_variables()
        self.numeric_vars = self.get_numeric_variables()
        self.eda_module = EDAModule(data)
        # 其他模块的初始化...

    def get_categorical_variables(self):
        # 提取字符型变量的名字并返回一个list
        return list(self.data.select_dtypes(include='object').columns)

    def get_numeric_variables(self):
        # 提取数值型变量的名字并返回一个list
        return list(self.data.select_dtypes(exclude='object').columns)


# 在主程序中使用 Yihui 类
if __name__ == "__main__":
    # 假设 data 是你的数据
    with open("data.csv", "r") as f:
        data = pd.read_csv(f)

    # 创建 Yihui 类的实例
    yihui_project = Yihui(data)
    print(yihui_project.data.head())

    # 直接访问 Yihui 类的属性获取字符型和数值型变量的名字
    categorical_vars_list = yihui_project.categorical_vars
    numeric_vars_list = yihui_project.numeric_vars

    print("Categorical Variables:", categorical_vars_list)
    print("Numeric Variables:", numeric_vars_list)
