"""
测试模型评估模块
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class TestModelEvaluationModule:
    """模型评估模块测试类"""

    @pytest.fixture
    def sample_predictions(self, yihuier_instance):
        """创建样本预测数据"""
        # 使用简单的逻辑回归生成预测
        X = yihuier_instance.data[['v1', 'v2']].dropna()
        y = yihuier_instance.data.loc[X.index, yihuier_instance.target]

        if len(X) < 10:
            pytest.skip("Not enough data for prediction tests")

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:, 1]

        return y, y_pred, model

    def test_plot_roc(self, yihuier_instance, sample_predictions):
        """测试ROC曲线绘制"""
        y_label, y_pred, _ = sample_predictions

        yihuier_instance.me_module.plot_roc(y_label, y_pred)
        plt.close('all')

    def test_plot_model_ks(self, yihuier_instance, sample_predictions):
        """测试KS曲线绘制"""
        y_label, y_pred, _ = sample_predictions

        yihuier_instance.me_module.plot_model_ks(y_label, y_pred)
        plt.close('all')

    def test_model_ks(self, yihuier_instance, sample_predictions):
        """测试KS值计算"""
        y_label, y_pred, _ = sample_predictions

        ks_value = yihuier_instance.me_module.model_ks(y_label, y_pred)

        # 验证KS值在合理范围内
        assert isinstance(ks_value, float)
        assert 0 <= ks_value <= 1

    def test_model_ks_edge_cases(self, yihuier_instance):
        """测试KS计算的边界情况"""
        # 完美预测
        y_label = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        ks = yihuier_instance.me_module.model_ks(y_label, y_pred)
        assert ks > 0.8  # 应该有很高的KS值

        # 随机预测
        y_label = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        ks = yihuier_instance.me_module.model_ks(y_label, y_pred)
        assert ks == 0.0  # 完全随机的预测KS应该为0

    def test_plot_learning_curve(self, yihuier_instance, sample_predictions):
        """测试学习曲线绘制"""
        y_label, y_pred, model = sample_predictions
        X = yihuier_instance.data[['v1', 'v2']].dropna()
        y = yihuier_instance.data.loc[X.index, yihuier_instance.target]

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        yihuier_instance.me_module.plot_learning_curve(
            estimator=model,
            x=X,
            y=y,
            cv=cv,
            train_size=np.linspace(0.5, 1.0, 3),
            plt_size=(8, 6)
        )
        plt.close('all')

    def test_cross_verify(self, yihuier_instance, sample_predictions):
        """测试交叉验证"""
        y_label, y_pred, model = sample_predictions
        X = yihuier_instance.data[['v1', 'v2']].dropna()
        y = yihuier_instance.data.loc[X.index, yihuier_instance.target]

        fold = KFold(n_splits=3, shuffle=True, random_state=42)

        # 测试函数能正常执行
        yihuier_instance.me_module.cross_verify(
            x=X,
            y=y,
            estimators=model,
            fold=fold,
            scoring='roc_auc'
        )
        plt.close('all')

    def test_plot_matrix_report(self, yihuier_instance, sample_predictions):
        """测试混淆矩阵和分类报告"""
        y_label, y_pred, _ = sample_predictions

        # 将概率转换为类别预测
        y_pred_class = (y_pred > 0.5).astype(int)

        yihuier_instance.me_module.plot_matrix_report(y_label, y_pred_class)
        plt.close('all')

    def test_me_module_initialization(self, yihuier_instance):
        """测试模型评估模块初始化"""
        me = yihuier_instance.me_module

        # 验证属性存在
        assert hasattr(me, 'y_label')
        assert hasattr(me, 'y_pred')

        # 初始值应该为None
        assert me.y_label is None
        assert me.y_pred is None

    def test_roc_with_same_predictions(self, yihuier_instance):
        """测试所有预测值相同的情况"""
        y_label = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])

        # 应该能处理但AUC会是0.5
        yihuier_instance.me_module.plot_roc(y_label, y_pred)
        plt.close('all')

    def test_ks_with_perfect_separation(self, yihuier_instance):
        """测试完美分离的情况"""
        y_label = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        ks = yihuier_instance.me_module.model_ks(y_label, y_pred)

        # 完美分离应该有很高的KS值
        assert ks > 0.8

    def test_cross_verify_with_different_scoring(self, yihuier_instance, sample_predictions):
        """测试不同的评分指标"""
        y_label, y_pred, model = sample_predictions
        X = yihuier_instance.data[['v1', 'v2']].dropna()
        y = yihuier_instance.data.loc[X.index, yihuier_instance.target]

        fold = KFold(n_splits=3, shuffle=True, random_state=42)

        # 测试accuracy评分
        yihuier_instance.me_module.cross_verify(
            x=X,
            y=y,
            estimators=model,
            fold=fold,
            scoring='accuracy'
        )
        plt.close('all')

    def test_plot_matrix_report_classification_metrics(self, yihuier_instance, sample_predictions):
        """测试混淆矩阵和分类报告的指标"""
        y_label = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_class = np.array([0, 0, 1, 0, 1, 1, 1, 1])

        # 这个函数会打印classification_report并绘制混淆矩阵
        yihuier_instance.me_module.plot_matrix_report(y_label, y_pred_class)
        plt.close('all')

    def test_learning_curve_with_small_dataset(self, yihuier_instance):
        """测试小数据集的学习曲线"""
        # 创建一个很小的数据集
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = KFold(n_splits=2, shuffle=True, random_state=42)

        yihuier_instance.me_module.plot_learning_curve(
            estimator=model,
            x=X,
            y=y,
            cv=cv,
            train_size=np.linspace(0.3, 1.0, 2),
            plt_size=(6, 4)
        )
        plt.close('all')
