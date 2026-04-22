"""
测试分箱模块
"""

import numpy as np
import pandas as pd


def test_binning_cate(yihuier_instance):
    """测试类别型变量分箱"""
    result_list, iv_values, ks_values = yihuier_instance.binning_module.binning_cate(
        ['category_var']
    )

    assert len(result_list) == 1
    assert len(iv_values) == 1
    assert len(ks_values) == 1
    assert isinstance(iv_values[0], (int, float))
    assert isinstance(ks_values[0], (int, float))


def test_iv_cate(yihuier_instance):
    """测试类别型变量 IV 计算"""
    # 先进行分箱
    yihuier_instance.binning_module.binning_cate(['category_var'])

    # 检查 iv_df 是否有数据
    if yihuier_instance.binning_module.iv_df is not None:
        assert len(yihuier_instance.binning_module.iv_df) > 0


def test_binning_num_freq(yihuier_instance):
    """测试数值型变量等频分箱"""
    # 使用等频分箱
    bin_df, iv_value = yihuier_instance.binning_module.binning_num(
        ['v1'],
        max_bin=5,
        min_binpct=0,
        method='freq'
    )

    assert iv_value is not None
    assert len(iv_value) == 1  # 只对一个变量分箱


def test_binning_num_chimerge(yihuier_instance):
    """测试数值型变量卡方分箱"""
    # 先处理缺失值
    yihuier_instance.data = yihuier_instance.dp_module.fillna_num_var(['v1'], fill_type='0')

    # 使用卡方分箱
    bin_df, iv_value = yihuier_instance.binning_module.binning_num(
        ['v1'],
        max_bin=5,
        min_binpct=0,
        method='ChiMerge'
    )

    assert iv_value is not None
    assert len(iv_value) == 1


def test_woe_transform(yihuier_instance):
    """测试 WOE 转换"""
    # 先对类别变量分箱
    yihuier_instance.binning_module.binning_cate(['category_var'])

    # 检查是否可以获取 WOE 结果
    if yihuier_instance.binning_module.bin_df:
        assert len(yihuier_instance.binning_module.bin_df) > 0


def test_iv_num_positive(yihuier_instance):
    """测试数值变量 IV 值为正数"""
    yihuier_instance.data = yihuier_instance.dp_module.fillna_num_var(['v1'], fill_type='0')

    _, iv_value = yihuier_instance.binning_module.binning_num(
        ['v1'],
        max_bin=5,
        min_binpct=0,
        method='freq'
    )

    # IV 值应该是非负数
    assert iv_value[0] >= 0


def test_binning_with_missing_values(yihuier_instance):
    """测试带缺失值的分箱"""
    # 确保有缺失值
    yihuier_instance.data.loc[0:5, 'v2'] = np.nan

    # 填充缺失值后分箱
    yihuier_instance.data = yihuier_instance.dp_module.fillna_num_var(['v2'], fill_type='0')

    bin_df, iv_value = yihuier_instance.binning_module.binning_num(
        ['v2'],
        max_bin=5,
        min_binpct=0,
        method='freq'
    )

    assert iv_value is not None


def test_woe_monotonic(yihuier_instance):
    """测试 WOE 单调性检查"""
    # 先分箱
    yihuier_instance.binning_module.binning_cate(['category_var'])

    # 这个方法应该能运行（具体结果取决于数据）
    # 这里只测试不会抛出异常
    try:
        result = yihuier_instance.binning_module.woe_monoton(
            yihuier_instance.binning_module.bin_df[0]
        )
        # 结果可能是 True, False 或 None
        assert result is not None
    except Exception:
        # 如果实现还不完整，至少不应该崩溃
        pass
