"""
测试数据处理模块
"""

import pandas as pd
import numpy as np
import pytest
from yihuier.constants import MISSING_VALUE_NEG_999


def test_fillna_num_var_with_zero(yihuier_instance):
    """测试用 0 填充数值型变量缺失值"""
    original_len = len(yihuier_instance.data)

    # 填充缺失值
    result = yihuier_instance.dp_module.fillna_num_var(['v1'], fill_type='0')

    assert len(result) == original_len
    assert result['v1'].isna().sum() == 0  # 不应该有缺失值


def test_fillna_num_var_with_class(yihuier_instance):
    """测试用特殊值填充数值型变量缺失值"""
    result = yihuier_instance.dp_module.fillna_num_var(
        ['v1'],
        fill_type='class',
        fill_class_num=MISSING_VALUE_NEG_999
    )

    # 检查原始缺失值位置是否被填充
    assert result['v1'].isna().sum() == 0


def test_const_delete(yihuier_instance):
    """测试常变量删除功能"""
    # 添加一个常变量列
    yihuier_instance.data['const_var'] = 1.0

    original_cols = len(yihuier_instance.data.columns)
    result = yihuier_instance.dp_module.const_delete(threshold=0.95)

    # 常变量应该被删除
    assert len(result.columns) < original_cols
    assert 'const_var' not in result.columns


def test_target_missing_delete(yihuier_instance):
    """测试删除目标变量缺失的观测"""
    # 添加一些目标变量缺失的行
    yihuier_instance.data.loc[0:2, 'dlq_flag'] = np.nan

    original_len = len(yihuier_instance.data)
    result = yihuier_instance.dp_module.target_missing_delete()

    # 缺失目标变量的行应该被删除
    assert len(result) < original_len
    assert result['dlq_flag'].isna().sum() == 0


def test_fillna_cate_var_with_mode(yihuier_instance):
    """测试用众数填充类别型变量缺失值"""
    # 创建一些缺失值
    yihuier_instance.data.loc[0:2, 'category_var'] = np.nan

    result = yihuier_instance.dp_module.fillna_cate_var(
        ['category_var'],
        fill_type='mode'
    )

    # 缺失值应该被填充
    assert result['category_var'].isna().sum() == 0


def test_fillna_cate_var_with_class(yihuier_instance):
    """测试用特殊类别填充缺失值"""
    # 创建一些缺失值
    yihuier_instance.data.loc[0:2, 'category_var'] = np.nan

    result = yihuier_instance.dp_module.fillna_cate_var(
        ['category_var'],
        fill_type='class',
        fill_str='MISSING'
    )

    # 缺失值应该被填充为 MISSING
    assert result['category_var'].isna().sum() == 0
    assert (result['category_var'] == 'MISSING').sum() >= 3


def test_delete_missing_var(yihuier_instance):
    """测试删除缺失率高的变量"""
    # 创建一个高缺失率的变量
    yihuier_instance.data['high_missing_var'] = np.nan
    yihuier_instance.data.loc[0:10, 'high_missing_var'] = 1.0  # 只有10个非缺失值

    original_cols = len(yihuier_instance.data.columns)
    result = yihuier_instance.dp_module.delete_missing_var(threshold=0.95)

    # 高缺失率变量应该被删除
    assert len(result.columns) < original_cols
    assert 'high_missing_var' not in result.columns


def test_get_numeric_variables(yihuier_instance):
    """测试获取数值型变量列表"""
    num_vars = yihuier_instance.get_numeric_variables()

    assert isinstance(num_vars, list)
    assert 'v1' in num_vars
    assert 'v2' in num_vars
    assert 'dlq_flag' not in num_vars  # 目标变量不应该在列表中


def test_get_categorical_variables(yihuier_instance):
    """测试获取类别型变量列表"""
    cat_vars = yihuier_instance.get_categorical_variables()

    assert isinstance(cat_vars, list)
    assert 'category_var' in cat_vars
    assert 'dlq_flag' not in cat_vars  # 目标变量不应该在列表中
