# Yihuier 文档中心

欢迎使用 Yihuier（一会儿）文档！这里包含了从入门到精通的完整指南。

## 📚 文档导航

### 新手入门

- [项目简介](guide/intro.md) - 了解 Yihuier 的设计理念、核心优势和适用场景
- [快速开始](guide/quick-start.md) - 5分钟快速上手，完成第一个评分卡模型
- [安装指南](guide/installation.md) - 详细的安装步骤和环境配置

### 用户指南

#### 核心概念

- [WOE 和 IV 介绍](guide/concepts/woe-iv.md) - 理解 WOE 编码和 IV 值的原理
- [评分卡基础](guide/concepts/scorecard-basics.md) - 评分卡的数学原理和业务含义
- [模型评估指标](guide/concepts/evaluation-metrics.md) - AUC、KS、PSI 等指标详解

#### 功能模块

- [EDA 模块](guide/modules/eda.md) - 数据探索性分析
- [数据预处理模块](guide/modules/data-processing.md) - 数据清洗和预处理
- [分箱模块](guide/modules/binning.md) - 变量分箱和 WOE 转换
- [变量选择模块](guide/modules/var-select.md) - 特征选择和降维
- [模型评估模块](guide/modules/model-evaluation.md) - 模型性能评估
- [评分卡实现模块](guide/modules/scorecard-implement.md) - 评分卡刻度和分数转换
- [评分卡监控模块](guide/modules/scorecard-monitor.md) - 模型监控和 PSI 分析
- [聚类模块](guide/modules/cluster.md) - 客户聚类分析
- [流水线模块](guide/modules/pipeline.md) - 端到端建模流程

### API 参考

- [API 文档](guide/api.md) - 完整的 API 参考手册

### 示例集合

- [基础示例](guide/examples.md) - 常用场景的代码示例
- [高级示例](guide/advanced-examples.md) - 复杂场景的完整案例
- [最佳实践](guide/best-practices.md) - 行业最佳实践和经验总结

### 开发者指南

- [架构设计](develop/architecture.md) - 系统架构和设计模式
- [贡献指南](develop/contributing.md) - 如何贡献代码和文档
- [更新日志](develop/changelog.md) - 版本更新记录

## 🚀 快速开始

### 安装

```bash
pip install yihuier
```

### 基础使用

```python
import pandas as pd
from yihuier import Yihuier

# 加载数据
data = pd.read_csv('data.csv')

# 初始化
yh = Yihuier(data, target='dlq_flag')

# 快速建模
yh.dp_module.fillna_num_var(yh.get_numeric_variables(), fill_type='0')
yh.binning_module.binning_num(['v1', 'v2', 'v3'], max_bin=5, method='ChiMerge')
data_woe = yh.binning_module.woe_transform()

# 变量选择
xg_imp, _, xg_cols = yh.var_select_module.select_xgboost(
    yh.get_numeric_variables()[:10], imp_num=5
)

# 模型训练
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = data_woe[xg_cols]
y = data_woe[yh.target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict_proba(X_test)[:, 1]
yh.me_module.plot_roc(y_test, y_pred)
yh.me_module.plot_model_ks(y_test, y_pred)
```

## 📖 学习路径

### 初学者路径

1. 阅读 [项目简介](guide/intro.md)，了解 Yihuier 是什么
2. 跟随 [快速开始](guide/quick-start.md)，运行第一个示例
3. 学习 [核心概念](guide/concepts/)，理解评分卡基础理论
4. 查阅 [模块文档](guide/modules/)，掌握各模块用法
5. 运行 [示例代码](guide/examples.md)，动手实践

### 进阶者路径

1. 深入学习 [架构设计](develop/architecture.md)，理解系统设计
2. 研究 [最佳实践](guide/best-practices.md)，学习行业经验
3. 探索 [高级示例](guide/advanced-examples.md)，处理复杂场景
4. 参考 [API 文档](guide/api.md)，掌握高级功能

### 贡献者路径

1. 阅读 [贡献指南](develop/contributing.md)，了解贡献流程
2. 查看 [开发文档](develop/)，掌握开发规范
3. Fork 项目，提交 Pull Request
4. 参与 [Issues](https://github.com/encyc/yihuier/issues) 讨论

## 💡 常见问题

### Q1: Yihuier 和其他评分卡工具有什么区别？

Yihuier 相比其他工具的主要优势：
- ✅ 面向对象设计，使用更简洁
- ✅ 模块化架构，易于扩展
- ✅ 完整的类型提示，开发体验好
- ✅ 丰富的文档和示例
- ✅ 活跃维护，持续更新

### Q2: 如何处理大数据集？

Yihuier 支持大数据集处理：
- 使用向量化计算，提高性能
- 支持分批处理，降低内存占用
- 可以选择性处理变量，避免全量计算

### Q3: 可以自定义分箱方法吗？

当然！Yihuier 支持多种扩展方式：
- 继承 `BinningModule` 类，重写分箱方法
- 使用 `binning_num_manual()` 方法，传入自定义分箱规则
- 结合 `binning_function.py` 中的辅助函数实现

### Q4: 如何部署到生产环境？

Yihuier 提供了完整的部署支持：
- 模型导出为 PMML/Pickle 格式
- 评分卡转换为 SQL/规则引擎
- PSI 监控用于模型稳定性追踪

## 🔗 相关资源

- **GitHub 仓库**: https://github.com/encyc/yihuier
- **PyPI 页面**: https://pypi.org/project/yihuier/
- **示例数据**: [examples/data/data.csv](../examples/data/data.csv)
- **测试报告**: [测试覆盖率报告](../test-coverage/)

## 📞 获取帮助

- 📖 查阅文档：本文档中心
- 💻 提交 Issue：[GitHub Issues](https://github.com/encyc/yihuier/issues)
- 💬 参与讨论：[GitHub Discussions](https://github.com/encyc/yihuier/discussions)

---

**最后更新**: 2026-04-22  
**文档版本**: v0.1.0
