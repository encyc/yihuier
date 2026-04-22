"""
评分卡项目常量定义

定义项目中使用的特殊值、默认参数等常量。
"""

# ============= 缺失值标记 =============

# 负数形式的缺失值标记
MISSING_VALUE_NEG_999 = -999
MISSING_VALUE_NEG_1111 = -1111

# ============= 初始化占位符 =============

# 分箱算法初始化占位符（表示未找到有效分割点）
SPLIT_POINT_NOT_FOUND = -99999
SPLIT_POINT_NOT_FOUND_ALT = -9999

# ============= 分箱相关 =============

# 默认分箱数
DEFAULT_MAX_BIN = 10
DEFAULT_MIN_BIN_PCT = 0.05

# 分箱方法
BINNING_METHOD_CHI_MERGE = "ChiMerge"
BINNING_METHOD_FREQ = "freq"
BINNING_METHOD_DISTANCE = "distance"
BINNING_METHOD_CART = "cart"
BINNING_METHOD_KS = "ks"

# ============= IV 值解释标准 =============

# IV 值预测能力分类标准
IV_STRENGTH_WEAK = 0.02
IV_STRENGTH_MEDIUM = 0.1
IV_STRENGTH_STRONG = 0.3

# ============= WOE 相关 =============

# WOE 合理范围
WOE_MIN_REASONABLE = -1.0
WOE_MAX_REASONABLE = 1.0

# ============= 相关性阈值 =============

# 默认高相关性阈值
DEFAULT_CORRELATION_THRESHOLD = 0.5

# ============= 数据处理 =============

# 默认缺失率阈值
DEFAULT_MISSING_THRESHOLD_VAR = 0.2  # 变量缺失率
DEFAULT_MISSING_THRESHOLD_OBS = 5  # 样本缺失变量数量

# 默认常变量阈值
DEFAULT_CONST_THRESHOLD = 0.9

# ============= 模型评估 =============

# 默认交叉验证折数
DEFAULT_CV_FOLDS = 5

# ============= 评分卡 =============

# 默认 PDO (Points to Double the Odds)
DEFAULT_PDO = 20
DEFAULT_ODDS = 1.0
DEFAULT_SCORE = 600

# ============= PSI =============

# PSI 稳定性阈值
PSI_STABLE = 0.1
PSI_WARNING = 0.2
