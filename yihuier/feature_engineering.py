"""特征工程模块

提供评分卡场景常用的宽表特征衍生函数：
- 交叉特征（比率/求和/差分/通用四则运算）
- 数学变换（log/sqrt/square 等）
- 缺失指示符
- 批量交叉 + IV 预筛
"""

import numpy as np
import pandas as pd


class FeatureEngineeringModule:
    """特征工程模块

    所有方法操作 yh.data，返回新的 DataFrame（不原地修改），
    由用户赋值回 yh.data，与 dp_module 风格一致。

    Attributes:
        yihuier_instance: Yihuier 主实例
        feature_log: 已生成特征的记录列表，每项 {name, source, method}
    """

    def __init__(self, yihuier_instance) -> None:
        """初始化特征工程模块

        Args:
            yihuier_instance: Yihuier 主实例
        """
        self.yihuier_instance = yihuier_instance
        self.feature_log: list[dict] = []

    def gen_cross(
        self, a: str, b: str, op: str, name: str
    ) -> pd.DataFrame:
        """通用四则运算交叉特征

        Args:
            a: 第一个列名
            b: 第二个列名
            op: 运算符，支持 '+', '-', '*', '/'
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）

        Raises:
            ValueError: op 不在 {'+', '-', '*', '/'} 中时
        """
        data = self.yihuier_instance.data.copy()

        if op == "+":
            result = data[a] + data[b]
        elif op == "-":
            result = data[a] - data[b]
        elif op == "*":
            result = data[a] * data[b]
        elif op == "/":
            # 除零 → NaN（pandas 默认得 inf，这里显式转为 NaN）
            with np.errstate(divide="ignore", invalid="ignore"):
                result = data[a] / data[b].replace(0, np.nan)
        else:
            raise ValueError(f"不支持的运算符: {op}。必须是 '+', '-', '*', '/' 之一")

        data[name] = result
        self.feature_log.append({"name": name, "source": (a, b), "method": f"cross:{op}"})
        return data

    def gen_ratio(self, num: str, den: str, name: str) -> pd.DataFrame:
        """比率特征 num / den

        Args:
            num: 分子列名
            den: 分母列名（den==0 时结果为 NaN）
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        # 除零 → NaN（pandas 默认得 inf，这里显式转为 NaN）
        with np.errstate(divide="ignore", invalid="ignore"):
            data[name] = data[num] / data[den].replace(0, np.nan)
        self.feature_log.append({"name": name, "source": (num, den), "method": "ratio"})
        return data

    def gen_sum(self, cols: list[str], name: str) -> pd.DataFrame:
        """多列求和

        Args:
            cols: 要求和的列名列表
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        result = data[cols[0]].copy()
        for col in cols[1:]:
            result = result + data[col]
        data[name] = result
        self.feature_log.append({"name": name, "source": tuple(cols), "method": "sum"})
        return data

    def gen_diff(self, a: str, b: str, name: str) -> pd.DataFrame:
        """差分特征 a - b

        Args:
            a: 被减数列名
            b: 减数列名
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）
        """
        data = self.yihuier_instance.data.copy()
        data[name] = data[a] - data[b]
        self.feature_log.append({"name": name, "source": (a, b), "method": "diff"})
        return data

    def gen_transform(self, col: str, method: str, name: str) -> pd.DataFrame:
        """数学变换

        对指定列做数学变换。log/sqrt 对非正值的处理：返回 NaN。

        Args:
            col: 原始列名
            method: 变换方法，支持 'log', 'log1p', 'sqrt', 'square', 'abs', 'reciprocal'
            name: 新特征列名

        Returns:
            新增 name 列的 DataFrame（副本）

        Raises:
            ValueError: method 不支持时
        """
        data = self.yihuier_instance.data.copy()
        s = data[col]

        if method == "log":
            # 非正值 → NaN
            valid = s > 0
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[valid] = np.log(s[valid])
        elif method == "log1p":
            # x > -1 有效；其余 → NaN
            valid = s > -1
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[valid] = np.log1p(s[valid])
        elif method == "sqrt":
            valid = s >= 0
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[valid] = np.sqrt(s[valid])
        elif method == "square":
            result = s ** 2
        elif method == "abs":
            result = s.abs()
        elif method == "reciprocal":
            result = 1.0 / s  # s==0 → inf，但 pandas 默认行为；保持自然语义
        else:
            raise ValueError(
                f"不支持的变换方法: {method}。必须是 'log', 'log1p', 'sqrt', 'square', 'abs', 'reciprocal' 之一"
            )

        data[name] = result
        self.feature_log.append({"name": name, "source": col, "method": f"transform:{method}"})
        return data
