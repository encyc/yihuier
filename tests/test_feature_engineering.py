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


def test_gen_ratio(yihuier_instance):
    """测试比率特征"""
    result = yihuier_instance.fe_module.gen_ratio('v1', 'v2', 'ratio_v1_v2')
    assert 'ratio_v1_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] / yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'ratio_v1_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_ratio_divide_by_zero():
    """测试比率除零 → NaN"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1], 'a': [10.0, 20.0], 'b': [0.0, 5.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_ratio('a', 'b', 'a_over_b')
    assert pd.isna(result.loc[0, 'a_over_b'])
    assert result.loc[1, 'a_over_b'] == 4.0


def test_gen_sum(yihuier_instance):
    """测试多列求和"""
    result = yihuier_instance.fe_module.gen_sum(['v1', 'v2', 'v4'], 'sum_v124')
    assert 'sum_v124' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = (
        yihuier_instance.data.loc[mask, 'v1']
        + yihuier_instance.data.loc[mask, 'v2']
        + yihuier_instance.data.loc[mask, 'v4']
    )
    pd.testing.assert_series_equal(
        result.loc[mask, 'sum_v124'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_sum_single_column(yihuier_instance):
    """测试单列求和（等于原列复制）"""
    result = yihuier_instance.fe_module.gen_sum(['v1'], 'sum_v1')
    assert 'sum_v1' in result.columns


def test_gen_diff(yihuier_instance):
    """测试差分"""
    result = yihuier_instance.fe_module.gen_diff('v1', 'v2', 'diff_v1_v2')
    assert 'diff_v1_v2' in result.columns
    mask = yihuier_instance.data['v1'].notna()
    expected = yihuier_instance.data.loc[mask, 'v1'] - yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'diff_v1_v2'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_shortcuts_log_features(yihuier_instance):
    """测试快捷函数记录 feature_log"""
    yihuier_instance.fe_module.gen_ratio('v1', 'v2', 'r')
    yihuier_instance.fe_module.gen_sum(['v1', 'v2'], 's')
    yihuier_instance.fe_module.gen_diff('v1', 'v2', 'd')
    names = [e['name'] for e in yihuier_instance.fe_module.feature_log]
    assert 'r' in names and 's' in names and 'd' in names


def test_gen_transform_log1p(yihuier_instance):
    """测试 log1p 变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'log1p', 'v2_log1p')
    assert 'v2_log1p' in result.columns
    mask = yihuier_instance.data['v2'].notna() & (yihuier_instance.data['v2'] > -1)
    expected = np.log1p(yihuier_instance.data.loc[mask, 'v2'])
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_log1p'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_transform_square(yihuier_instance):
    """测试 square 变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'square', 'v2_sq')
    assert 'v2_sq' in result.columns
    mask = yihuier_instance.data['v2'].notna()
    expected = yihuier_instance.data.loc[mask, 'v2'] ** 2
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_sq'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_transform_abs(yihuier_instance):
    """测试 abs 变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'abs', 'v2_abs')
    assert 'v2_abs' in result.columns
    mask = yihuier_instance.data['v2'].notna()
    expected = yihuier_instance.data.loc[mask, 'v2'].abs()
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_abs'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_transform_sqrt_negative_returns_nan():
    """测试 sqrt 对负数返回 NaN"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'x': [4.0, -1.0, 9.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_transform('x', 'sqrt', 'x_sqrt')
    assert result.loc[0, 'x_sqrt'] == 2.0
    assert pd.isna(result.loc[1, 'x_sqrt'])  # 负数 → NaN
    assert result.loc[2, 'x_sqrt'] == 3.0


def test_gen_transform_log_nonpositive_returns_nan():
    """测试 log 对零/负数返回 NaN"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'x': [1.0, 0.0, -2.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_transform('x', 'log', 'x_log')
    assert result.loc[0, 'x_log'] == 0.0  # log(1) = 0
    assert pd.isna(result.loc[1, 'x_log'])  # log(0) → NaN
    assert pd.isna(result.loc[2, 'x_log'])  # log(-2) → NaN


def test_gen_transform_invalid_method(yihuier_instance):
    """测试非法 method 抛 ValueError"""
    with pytest.raises(ValueError, match="不支持的变换方法"):
        yihuier_instance.fe_module.gen_transform('v1', 'cube', 'v1_cube')


def test_gen_transform_reciprocal(yihuier_instance):
    """测试倒数变换"""
    result = yihuier_instance.fe_module.gen_transform('v2', 'reciprocal', 'v2_rec')
    assert 'v2_rec' in result.columns
    mask = yihuier_instance.data['v2'].notna() & (yihuier_instance.data['v2'] != 0)
    expected = 1.0 / yihuier_instance.data.loc[mask, 'v2']
    pd.testing.assert_series_equal(
        result.loc[mask, 'v2_rec'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_gen_missing_flag_basic(yihuier_instance):
    """测试缺失指示符：有缺失=1，无缺失=0"""
    # v1 在 sample_data 中有 5% 缺失
    result = yihuier_instance.fe_module.gen_missing_flag(['v1'])
    assert 'miss_v1' in result.columns
    # 缺失行应为 1
    missing_mask = yihuier_instance.data['v1'].isna()
    assert (result.loc[missing_mask, 'miss_v1'] == 1).all()
    # 非缺失行应为 0
    present_mask = yihuier_instance.data['v1'].notna()
    assert (result.loc[present_mask, 'miss_v1'] == 0).all()


def test_gen_missing_flag_multiple_cols(yihuier_instance):
    """测试多列各生成独立 flag"""
    result = yihuier_instance.fe_module.gen_missing_flag(['v1', 'v2'])
    assert 'miss_v1' in result.columns
    assert 'miss_v2' in result.columns


def test_gen_missing_flag_custom_prefix(yihuier_instance):
    """测试自定义前缀"""
    result = yihuier_instance.fe_module.gen_missing_flag(['v1'], prefix='is_na_')
    assert 'is_na_v1' in result.columns
    assert 'miss_v1' not in result.columns


def test_gen_missing_flag_preserves_originals(yihuier_instance):
    """测试原始列保留不变"""
    original_v1 = yihuier_instance.data['v1'].copy()
    result = yihuier_instance.fe_module.gen_missing_flag(['v1'])
    pd.testing.assert_series_equal(result['v1'], original_v1, check_names=False)


def test_gen_missing_flag_no_missing():
    """测试无缺失列全部为 0"""
    from yihuier.yihuier import Yihuier
    data = pd.DataFrame({'target': [0, 1, 0], 'x': [1.0, 2.0, 3.0]})
    yh = Yihuier(data, 'target')
    result = yh.fe_module.gen_missing_flag(['x'])
    assert (result['miss_x'] == 0).all()


def test_gen_missing_flag_logs_features(yihuier_instance):
    """测试每个 flag 列都记录到 feature_log"""
    yihuier_instance.fe_module.gen_missing_flag(['v1', 'v2'])
    names = [e['name'] for e in yihuier_instance.fe_module.feature_log]
    assert 'miss_v1' in names
    assert 'miss_v2' in names
