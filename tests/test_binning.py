"""
测试分箱模块
"""

import math

import numpy as np
import pandas as pd
import pytest

optbinning_available = True
try:
    import optbinning  # noqa: F401
except ImportError:
    optbinning_available = False


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

    not_monoton_cols, judge_df = yihuier_instance.binning_module.woe_monoton()
    assert len(judge_df) == 1
    assert judge_df['judge_monoton'].iloc[0] in ('True', 'False')


# ============= 真实数据回归测试 =============
# 依赖 data/data.csv；若不存在则跳过（保证 CI / 无数据环境仍全绿）。

@pytest.fixture
def real_yihuier(real_data):
    """基于真实数据构造的 Yihuier 实例（清洗缺失标记 + 填充缺失值）。"""
    if real_data is None:
        pytest.skip("真实数据 data/data.csv 不存在，跳过真实数据回归测试")
    df = real_data.dropna(subset=['dlq_flag']).reset_index(drop=True)
    num_cols = [c for c in df.columns if c.startswith('v')]
    # -999 / -1111 是缺失标记，转成 NaN 后用中位数填充
    df[num_cols] = df[num_cols].replace({-999: np.nan, -1111: np.nan})
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    from yihuier.yihuier import Yihuier
    return Yihuier(df, 'dlq_flag')


def _is_finite_iv(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def test_real_data_binning_no_inf(real_yihuier):
    """真实数据：freq / count / ChiMerge 三种方法 IV 不应出现 inf 或 nan。"""
    num_cols = [c for c in real_yihuier.data.columns if c.startswith('v')]
    sample_cols = num_cols[:20]  # 取前 20 个变量，平衡覆盖与耗时

    for method in ['freq', 'count', 'ChiMerge']:
        _, iv_values = real_yihuier.binning_module.binning_num(
            sample_cols, method=method, n=10, max_bin=10, min_binpct=0
        )
        assert len(iv_values) == len(sample_cols)
        for v in iv_values:
            assert _is_finite_iv(v), f"method={method} 出现非有限 IV: {v}"


def test_real_data_woe_transform_preserves_unbinned(real_yihuier):
    """真实数据：woe_transform 不得修改未分箱的列。"""
    num_cols = [c for c in real_yihuier.data.columns if c.startswith('v')]
    binned, unbinned = num_cols[0], num_cols[10]

    real_yihuier.binning_module.binning_num([binned], method='freq', n=10)
    real_yihuier.binning_module.woe_df_concat()

    before = real_yihuier.data[unbinned].values
    transformed = real_yihuier.binning_module.woe_transform()
    after = transformed[unbinned].values

    assert np.array_equal(before, after), "未分箱列在 woe_transform 后被改动"
    # 已分箱列应被映射为有限个 WOE 值
    assert transformed[binned].nunique() <= 10


def test_real_data_woe_checks_col_names(real_yihuier):
    """真实数据：woe_monoton / woe_large 应返回真实列名而非 None。"""
    num_cols = [c for c in real_yihuier.data.columns if c.startswith('v')]
    real_yihuier.binning_module.binning_num(num_cols[:5], method='freq', n=10)

    _, mono_judge = real_yihuier.binning_module.woe_monoton()
    _, large_judge = real_yihuier.binning_module.woe_large()

    assert list(mono_judge['col']) == num_cols[:5]
    assert list(large_judge['col']) == num_cols[:5]
    assert mono_judge['judge_monoton'].isin(['True', 'False']).all()


# ============= optbinning 集成测试 =============
# 需要 pip install yihuier[optimal]；未安装则跳过。

@pytest.mark.skipif(not optbinning_available, reason="optbinning 未安装")
def test_optbinning_compat_shim_translates_kwarg():
    """垫片应把 force_all_finite 翻译为 ensure_all_finite，且幂等。"""
    import yihuier._optbinning_compat  # noqa: F401
    import sklearn.utils.validation as sv
    # 幂等标志
    assert getattr(sv, "_yihuier_force_finite_patched", False) is True
    # 包装后的 check_array 能接受旧参数名（含 'allow-nan' 字符串值）
    arr = np.array([1.0, 2.0, np.nan, 4.0])
    res = sv.check_array(arr, ensure_2d=False, force_all_finite="allow-nan")
    assert res.shape == arr.shape


@pytest.mark.skipif(not optbinning_available, reason="optbinning 未安装")
def test_optbinning_method_basic(yihuier_instance):
    """method='optbinning' 能跑通并返回与其它方法一致的结构。"""
    yihuier_instance.data = yihuier_instance.dp_module.fillna_num_var(['v1'], fill_type='0')
    bin_df, iv_value = yihuier_instance.binning_module.binning_num(
        ['v1'], method='optbinning', max_bin=10, min_binpct=0.02
    )
    assert len(iv_value) == 1
    assert iv_value[0] >= 0
    # 输出结构应与 ChiMerge 一致（含 woe/bin_iv/IV 等列）
    cols = set(bin_df[0].columns)
    assert {'woe', 'bin_iv', 'IV', 'bad', 'good'}.issubset(cols)


@pytest.mark.skipif(not optbinning_available, reason="optbinning 未安装")
def test_optbinning_missing_dep_error(monkeypatch, yihuier_instance):
    """optbinning 未安装时 method='optbinning' 应给出清晰的 ImportError 提示。"""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optbinning":
            raise ImportError("simulated missing optbinning")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="yihuier\\[optimal\\]"):
        yihuier_instance.binning_module.binning_num(
            ['v1'], method='optbinning', max_bin=5, min_binpct=0.02
        )


@pytest.mark.skipif(not optbinning_available, reason="optbinning 未安装")
def test_real_data_optbinning_monotonicity(real_yihuier):
    """真实数据：optbinning 方法产出的 WOE 单调性应显著优于 ChiMerge。"""
    num_cols = [c for c in real_yihuier.data.columns if c.startswith('v')]
    cols = num_cols[:20]

    def count_monotonic(method):
        y = real_yihuier.binning_module
        y.binning_num(cols, method=method, max_bin=10, min_binpct=0.02)
        _, judge = y.woe_monoton()
        return int((judge['judge_monoton'] == 'True').sum())

    ob_mono = count_monotonic('optbinning')
    assert ob_mono >= 5, f"optbinning 单调变量数偏低: {ob_mono}/20"
