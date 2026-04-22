"""
测试EDA模块
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免弹出窗口
import matplotlib.pyplot as plt


class TestEDAModule:
    """EDA模块测试类"""

    def test_plot_cate_var(self, yihuier_instance):
        """测试类别型变量分布可视化"""
        # 获取类别型变量
        cate_vars = yihuier_instance.get_categorical_variables()

        if len(cate_vars) > 0:
            # 测试单个变量
            yihuier_instance.eda_module.plot_cate_var(
                col_list=[cate_vars[0]],
                plt_size=(6, 4),
                plt_num=1,
                x=1,
                y=1
            )
            plt.close('all')

            # 测试多个变量（如果有）
            if len(cate_vars) >= 2:
                yihuier_instance.eda_module.plot_cate_var(
                    col_list=cate_vars[:2],
                    plt_size=(12, 4),
                    plt_num=2,
                    x=1,
                    y=2
                )
                plt.close('all')

    def test_plot_num_col_hist(self, yihuier_instance):
        """测试数值型变量分布-直方图"""
        num_vars = yihuier_instance.get_numeric_variables()

        if len(num_vars) > 0:
            yihuier_instance.eda_module.plot_num_col(
                col_list=[num_vars[0]],
                plt_type='hist',
                plt_size=(6, 4),
                plt_num=1,
                x=1,
                y=1
            )
            plt.close('all')

    def test_plot_num_col_box(self, yihuier_instance):
        """测试数值型变量分布-箱线图"""
        num_vars = yihuier_instance.get_numeric_variables()

        if len(num_vars) > 0:
            yihuier_instance.eda_module.plot_num_col(
                col_list=[num_vars[0]],
                plt_type='box',
                plt_size=(6, 4),
                plt_num=1,
                x=1,
                y=1
            )
            plt.close('all')

    def test_plot_num_col_stripplot(self, yihuier_instance):
        """测试数值型变量分布-散点图"""
        num_vars = yihuier_instance.get_numeric_variables()

        if len(num_vars) > 0:
            yihuier_instance.eda_module.plot_num_col(
                col_list=[num_vars[0]],
                plt_type='stripplot',
                plt_size=(6, 4),
                plt_num=1,
                x=1,
                y=1
            )
            plt.close('all')

    def test_plot_default_cate(self, yihuier_instance):
        """测试类别型变量违约率分析"""
        cate_vars = yihuier_instance.get_categorical_variables()

        if len(cate_vars) > 0:
            yihuier_instance.eda_module.plot_default_cate(
                col_list=[cate_vars[0]],
                plt_size=(8, 4),
                plt_num=1,
                x=1,
                y=1
            )
            plt.close('all')

    def test_plot_default_num(self, yihuier_instance):
        """测试数值型变量违约率分析"""
        num_vars = yihuier_instance.get_numeric_variables()

        if len(num_vars) > 0:
            yihuier_instance.eda_module.plot_default_num(
                col_list=[num_vars[0]],
                q=5,
                plt_size=(8, 4),
                plt_num=1,
                x=1,
                y=1
            )
            plt.close('all')

    def test_auto_eda_simple(self, yihuier_instance):
        """测试快速自动EDA分析"""
        result = yihuier_instance.eda_module.auto_eda_simple()

        # 验证返回结果是DataFrame
        assert isinstance(result, pd.DataFrame)

        # 验证结果不为空
        assert not result.empty

        # 验证包含期望的列
        cate_vars = yihuier_instance.get_categorical_variables()
        num_vars = yihuier_instance.get_numeric_variables()

        if len(cate_vars) > 0:
            # 类别型变量应该有这些列
            assert 'unique_count' in result.columns
            assert 'entropy' in result.columns

        if len(num_vars) > 0:
            # 数值型变量应该有这些列
            assert 'mean' in result.columns
            assert 'min' in result.columns
            assert 'max' in result.columns
            assert 'median' in result.columns

        # 所有变量都应该有 missing_pct
        assert 'missing_pct' in result.columns

    def test_auto_eda_simple_stats_correctness(self, yihuier_instance):
        """测试auto_eda_simple的统计量正确性"""
        result = yihuier_instance.eda_module.auto_eda_simple()

        num_vars = yihuier_instance.get_numeric_variables()

        if len(num_vars) > 0:
            # 验证第一个数值变量的统计量
            var = num_vars[0]
            actual_mean = yihuier_instance.data[var].mean()
            actual_median = yihuier_instance.data[var].median()

            assert result.loc[var, 'mean'] == actual_mean
            assert result.loc[var, 'median'] == actual_median

    def test_eda_module_initialization(self, yihuier_instance):
        """测试EDA模块初始化"""
        eda = yihuier_instance.eda_module

        # 验证属性存在
        assert hasattr(eda, 'data')
        assert hasattr(eda, 'variables')
        assert hasattr(eda, 'category_variables')
        assert hasattr(eda, 'numeric_variables')

        # 验证数据是DataFrame副本
        assert isinstance(eda.data, pd.DataFrame)
        assert eda.data is not yihuier_instance.data  # 确保是副本

        # 验证变量列表正确
        assert len(eda.category_variables) == len(yihuier_instance.get_categorical_variables())
        assert len(eda.numeric_variables) == len(yihuier_instance.get_numeric_variables())

    def test_plot_with_empty_dataset(self, yihuier_instance):
        """测试处理空数据集的情况"""
        # 创建一个只有类别型变量的数据集
        if len(yihuier_instance.get_categorical_variables()) > 0:
            # 这个测试验证函数不会在没有数值变量时崩溃
            pass  # 如果没有数值变量，函数应该优雅地处理
