"""
测试特征工程模块
"""

import pandas as pd
import numpy as np
import pytest


def test_fe_module_mounted(yihuier_instance):
    """测试 fe_module 正确挂载到 Yihuier 实例"""
    from yihuier.feature_engineering import FeatureEngineeringModule
    assert hasattr(yihuier_instance, 'fe_module')
    assert isinstance(yihuier_instance.fe_module, FeatureEngineeringModule)


def test_feature_log_initialized(yihuier_instance):
    """测试 feature_log 初始化为空列表"""
    assert hasattr(yihuier_instance.fe_module, 'feature_log')
    assert yihuier_instance.fe_module.feature_log == []
