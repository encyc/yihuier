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


def test_gen_cross_add(yihuier_instance):
    """测试加法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '+', 'v1_plus_v2')
    assert 'v1_plus_v2' in result.columns
    # 逐行验证（忽略 v1 缺失的行）
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] + yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_plus_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_subtract(yihuier_instance):
    """测试减法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '-', 'v1_minus_v2')
    assert 'v1_minus_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] - yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_minus_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_multiply(yihuier_instance):
    """测试乘法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '*', 'v1_mul_v2')
    assert 'v1_mul_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] * yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_mul_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_divide(yihuier_instance):
    """测试除法交叉"""
    result = yihuier_instance.fe_module.gen_cross('v1', 'v2', '/', 'v1_div_v2')
    assert 'v1_div_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] / yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v1_div_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_cross_invalid_op(yihuier_instance):
    """测试非法运算符抛 ValueError"""
    with pytest.raises(ValueError, match="不支持的运算符"):
        yihuier_instance.fe_module.gen_cross('v1', 'v2', '%', 'v1_mod_v2')


def test_gen_cross_divide_by_zero():
    """测试除零产生 NaN 而非 inf/异常"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'a': [10.0, 20.0, 30.0], 'b': [0.0, 0.0, 5.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_cross('a', 'b', '/', 'a_div_b')
    # b==0 的行应为 NaN（不是 inf）
    assert pd.isna(result.loc[0, 'a_div_b'])
    assert pd.isna(result.loc[1, 'a_div_b'])
    assert result.loc[2, 'a_div_b'] == 6.0


def test_gen_cross_does_not_mutate_original(yihuier_instance):
    """测试返回新 DataFrame，不修改 yh.data"""
    original_cols = list(yihuier_instance.data.columns)
    yihuier_instance.fe_module.gen_cross('v1', 'v2', '+', 'new_col')
    assert list(yihuier_instance.data.columns) == original_cols
    assert 'new_col' not in yihuier_instance.data.columns


def test_gen_cross_logs_feature(yihuier_instance):
    """测试特征被记录到 feature_log"""
    yihuier_instance.fe_module.gen_cross('v1', 'v2', '+', 'v1_plus_v2')
    log = yihuier_instance.fe_module.feature_log
    assert any(entry['name'] == 'v1_plus_v2' for entry in log)
    entry = [e for e in log if e['name'] == 'v1_plus_v2'][0]
    assert entry['source'] == ('v1', 'v2')
    assert entry['method'] == 'cross:+'
