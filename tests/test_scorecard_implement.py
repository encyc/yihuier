"""
测试评分卡实现模块
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class TestScorecardImplementModule:
    """评分卡实现模块测试类"""

    @pytest.fixture
    def trained_model_and_woe(self, yihuier_instance):
        """创建训练好的模型和WOE数据"""
        # 使用简单变量进行分箱
        num_vars = yihuier_instance.get_numeric_variables()
        if len(num_vars) < 2:
            pytest.skip("Not enough numeric variables")

        # 使用简单变量进行分箱
        simple_vars = num_vars[:2]
        bin_df, iv_value = yihuier_instance.binning_module.binning_num(
            col_list=simple_vars,
            max_bin=3,
            method='freq'
        )

        # 获取WOE结果
        woe_df = yihuier_instance.binning_module.woe_df_concat()

        # 转换为WOE
        data_woe = yihuier_instance.binning_module.woe_transform()

        # 训练逻辑回归模型
        X = data_woe[simple_vars].dropna()
        y = data_woe.loc[X.index, yihuier_instance.target]

        if len(X) < 10 or y.nunique() < 2:
            pytest.skip("Not enough data or only one class")

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        return model, woe_df, data_woe, simple_vars

    def test_cal_scale(self, yihuier_instance, trained_model_and_woe):
        """测试评分卡刻度计算"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        A, B, base_score = yihuier_instance.si_module.cal_scale(
            score=600,
            odds=19,
            PDO=40,
            model=model
        )

        # 验证返回值类型
        assert isinstance(A, float)
        assert isinstance(B, float)
        assert isinstance(base_score, float)

        # 验证值不为零
        assert B != 0
        assert base_score > 0

    def test_score_df_concat(self, yihuier_instance, trained_model_and_woe):
        """测试变量得分表生成"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数
        B = 20 / (np.log(19) - np.log(2 * 19))

        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 验证返回结果是DataFrame
        assert isinstance(score_df, pd.DataFrame)

        # 验证包含score列
        assert 'score' in score_df.columns

        # 验证分数不为空
        assert not score_df['score'].isnull().all()

    def test_score_transform(self, yihuier_instance, trained_model_and_woe):
        """测试分数转换"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 验证返回结果是DataFrame
        assert isinstance(data_score, pd.DataFrame)

        # 验证数据形状
        assert data_score.shape[0] == data_woe.shape[0]

    def test_plot_score_ks(self, yihuier_instance, trained_model_and_woe):
        """测试评分卡KS曲线"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 绘制KS曲线
        yihuier_instance.si_module.plot_score_ks(
            df=data_score,
            score_col='total_score',
            target=yihuier_instance.target
        )
        plt.close('all')

    def test_plot_pr(self, yihuier_instance, trained_model_and_woe):
        """测试PR曲线"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 绘制PR曲线
        yihuier_instance.si_module.plot_PR(
            df=data_score,
            score_col='total_score',
            target=yihuier_instance.target,
            plt_size=(6, 4)
        )
        plt.close('all')

    def test_plot_score_hist(self, yihuier_instance, trained_model_and_woe):
        """测试好坏用户得分分布图"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 绘制得分分布图
        yihuier_instance.si_module.plot_score_hist(
            df=data_score,
            target=yihuier_instance.target,
            score_col='total_score',
            plt_size=(8, 6),
            cutoff=400
        )
        plt.close('all')

    def test_score_info(self, yihuier_instance, trained_model_and_woe):
        """测试得分明细表"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 生成得分明细表
        score_info_df = yihuier_instance.si_module.score_info(
            df=data_score,
            score_col='total_score',
            target=yihuier_instance.target,
            x=200,
            y=800,
            step=50
        )

        # 验证返回结果是DataFrame
        assert isinstance(score_info_df, pd.DataFrame)

        # 验证包含期望的列
        expected_cols = ['用户数', '坏用户', '好用户', '违约占比', '累计用户',
                        '坏用户累计', '好用户累计', '坏用户累计占比', '好用户累计占比',
                        '累计用户占比', '累计违约占比']
        for col in expected_cols:
            assert col in score_info_df.columns

    def test_plot_lifting(self, yihuier_instance, trained_model_and_woe):
        """测试提升图和洛伦兹曲线"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 绘制提升图和洛伦兹曲线
        yihuier_instance.si_module.plot_lifting(
            df=data_score,
            score_col='total_score',
            target=yihuier_instance.target,
            bins=5,
            plt_size=(12, 4)
        )
        plt.close('all')

    def test_rule_verify(self, yihuier_instance, trained_model_and_woe):
        """测试cutoff点验证"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 进行规则验证
        matrix_df = yihuier_instance.si_module.rule_verify(
            df=data_score,
            col_score='total_score',
            target=yihuier_instance.target,
            cutoff=400
        )

        # 验证返回结果是DataFrame
        assert isinstance(matrix_df, pd.DataFrame)

    def test_si_module_initialization(self, yihuier_instance):
        """测试评分卡实现模块初始化"""
        si = yihuier_instance.si_module

        # 验证属性存在
        assert hasattr(si, 'yihuier_instance')

    def test_cal_scale_different_parameters(self, yihuier_instance, trained_model_and_woe):
        """测试不同参数的刻度计算"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 测试不同的参数组合
        A1, B1, base_score1 = yihuier_instance.si_module.cal_scale(
            score=600,
            odds=19,
            PDO=40,
            model=model
        )

        A2, B2, base_score2 = yihuier_instance.si_module.cal_scale(
            score=500,
            odds=10,
            PDO=20,
            model=model
        )

        # 验证不同参数产生不同的结果
        assert A1 != A2 or B1 != B2

    def test_score_transform_preserves_data_shape(self, yihuier_instance, trained_model_and_woe):
        """测试分数转换保持数据形状"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 验证数据形状保持不变
        assert data_score.shape[0] == data_woe.shape[0]
        assert data_score.shape[1] == data_woe.shape[1]

    def test_plot_score_hist_with_no_cutoff(self, yihuier_instance, trained_model_and_woe):
        """测试没有cutoff的得分分布图"""
        model, woe_df, data_woe, simple_vars = trained_model_and_woe

        # 计算B参数和得分表
        B = 20 / (np.log(19) - np.log(2 * 19))
        score_df = yihuier_instance.si_module.score_df_concat(
            woe_df=woe_df,
            model=model,
            B=B
        )

        # 转换分数
        data_score = yihuier_instance.si_module.score_transform(
            df=data_woe,
            target=yihuier_instance.target,
            df_score=score_df
        )

        # 创建总分列
        data_score['total_score'] = data_score[simple_vars].sum(axis=1)

        # 不设置cutoff
        yihuier_instance.si_module.plot_score_hist(
            df=data_score,
            target=yihuier_instance.target,
            score_col='total_score',
            plt_size=(8, 6),
            cutoff=None
        )
        plt.close('all')
