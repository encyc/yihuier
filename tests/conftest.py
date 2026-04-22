"""
Pytest 配置和共享 fixtures

提供测试所需的数据和 fixtures。
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'customer_no': [f'CUST{i:06d}' for i in range(n_samples)],
        'dlq_flag': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'v1': np.random.normal(100, 20, n_samples),
        'v2': np.random.normal(50, 10, n_samples),
        'v3': np.random.choice([1, 2, 3, 4, 5], size=n_samples),
        'v4': np.random.uniform(0, 100, n_samples),
        'v5': np.random.exponential(10, n_samples),
        'category_var': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
    })

    # 添加一些缺失值
    data.loc[data.sample(frac=0.05).index, 'v1'] = np.nan
    data.loc[data.sample(frac=0.03).index, 'v2'] = -999
    data.loc[data.sample(frac=0.02).index, 'v4'] = -1111

    return data


@pytest.fixture
def real_data_path():
    """获取真实数据文件的路径"""
    data_path = Path(__file__).parent.parent / "data" / "data.csv"
    if data_path.exists():
        return str(data_path)
    return None


@pytest.fixture
def real_data(real_data_path):
    """加载真实数据（如果存在）"""
    if real_data_path and Path(real_data_path).exists():
        return pd.read_csv(real_data_path)
    return None


@pytest.fixture
def yihuier_instance(sample_data):
    """创建 Yihuier 实例用于测试"""
    from yihuier.yihuier import Yihuier
    return Yihuier(sample_data.copy(), 'dlq_flag')
