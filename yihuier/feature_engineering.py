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
