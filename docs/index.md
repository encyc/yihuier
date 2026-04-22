---
layout: home

hero:
  name: Yihuier
  text: 专业的信用评分卡建模 Python 工具包
  tagline: 从数据探索到模型部署的全流程支持
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/quick-start
    - theme: alt
      text: 项目简介
      link: /guide/intro

features:
  - title: 🤖 AI 智能建模
    details: 内置风控建模 Skill，AI 助手自动引导完成 10 步建模全流程，确保模型质量
  - title: 🎯 开箱即用
    details: 简洁的 API 设计，几行代码完成复杂建模流程
  - title: 🔧 专业完整
    details: 覆盖评分卡建模全流程的各个环节，从数据预处理到模型监控
  - title: 🚀 高度可定制
    details: 支持自定义参数和方法，灵活应对各种业务场景
  - title: 📦 模块化架构
    details: 面向对象设计，各模块职责清晰，易于维护和扩展
  - title: 📚 完整文档
    details: 详细的 API 文档和使用示例，助你快速上手
  - title: ✅ 生产就绪
    details: 完善的测试覆盖，代码质量保证，可直接用于生产环境
---

## 快速体验

```python
from yihuier import Yihuier

# 初始化
yh = Yihuier(data, target='dlq_flag')

# 数据预处理
yh.data = yh.dp_module.delete_missing_var(threshold=0.2)

# 变量分箱
bin_df, iv_value = yh.binning_module.binning_num(
    col_list=['v1', 'v2', 'v3'],
    max_bin=5,
    method='ChiMerge'
)

# WOE 转换
data_woe = yh.binning_module.woe_transform()

# 变量选择
xg_imp, _, xg_cols = yh.var_select_module.select_xgboost(
    col_list=data_woe.drop(['dlq_flag'], axis=1).columns.tolist(),
    imp_num=10
)
```

## 核心功能

### 数据预处理
- 缺失值处理（数值型/类别型）
- 常变量删除
- 异常值处理

### 变量分箱
- ChiMerge 卡方分箱
- 等频/等距分箱
- 单调性分箱
- WOE 编码和 IV 计算

### 变量选择
- XGBoost/随机森林特征重要性
- 相关性分析
- IV 筛选

### 模型评估
- ROC/KS 曲线
- 交叉验证
- 学习曲线
- 混淆矩阵

### 评分卡实现
- 刻度参数计算
- 分数转换
- Cutoff 验证

### 模型监控
- PSI 分析
- 稳定性监控
- 偏移检测

## 为什么选择 Yihuier？

::: tip 与其他工具对比

| 特性 | Yihuier | Scorecardpy | Toad |
|------|---------|-------------|------|
| 分箱方法 | 4种 | 2种 | 3种 |
| WOE 转换 | ✅ | ✅ | ✅ |
| 变量选择 | 3种策略 | 1种 | 2种 |
| 模型评估 | 完整 | 基础 | 完整 |
| 评分卡实现 | 完整 | 部分 | 部分 |
| 模型监控 | ✅ | ❌ | ❌ |
| Python 版本 | 3.13+ | 3.6+ | 3.6+ |
| 类型提示 | 完整 | 无 | 无 |
| 中文文档 | ✅ | ❌ | 部分 |

:::

## 学习路径

### 初学者
1. 阅读 [项目简介](/guide/intro)
2. 跟随 [快速开始](/guide/quick-start)
3. 学习 [核心概念](/guide/concepts/)
4. 查阅 [模块文档](/guide/modules/)

### 进阶用户
1. 深入学习 [架构设计](/develop/architecture)
2. 研究 [最佳实践](/guide/best-practices)
3. 探索 [高级示例](/guide/examples)

### 贡献者
1. 阅读 [贡献指南](/develop/contributing)
2. 查看 [开发文档](/develop/)
3. 参与 [GitHub](https://github.com/encyc/yihuier) 讨论

## 许可证

本项目采用 [MIT](https://opensource.org/licenses/MIT) 许可证发布。

## 致谢

感谢开源社区的所有贡献者！

## 相关
**GitHub**: [encyc/yihuier](https://github.com/encyc/yihuier)

**PyPI**: [yihuier](https://pypi.org/project/yihuier/)
