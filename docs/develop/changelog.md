# 更新日志

本文档记录 Yihuier 的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中
- 添加更多分箱方法（决策树分箱、K-Means 分箱）
- 支持分布式计算（Dask、Ray）
- 添加自动化特征工程
- 支持 GPU 加速

### 已知问题
- ChiMerge 分箱在极端分布下可能失败
- 大数据集（>100万样本）内存占用较高

## [0.1.0] - 2024-01-22

### Added
- ✨ 初始版本发布
- ✨ 完整的评分卡建模流程支持
- ✨ 9 个功能模块
  - EDA 模块
  - 数据预处理模块
  - 分箱模块
  - 变量选择模块
  - 模型评估模块
  - 评分卡实现模块
  - 评分卡监控模块
  - 聚类模块
  - 流水线模块

### Features

#### 数据预处理
- 缺失值处理（数值型和类别型）
- 常变量删除
- 目标变量缺失删除
- 高缺失率变量删除

#### 变量分箱
- ChiMerge 卡方分箱
- 等频分箱
- 等距分箱
- 单调性分箱
- 类别型变量分箱
- WOE 转换
- IV 值计算

#### 变量选择
- XGBoost 特征重要性
- 随机森林特征重要性
- IV 筛选
- 相关性分析（IV 优先/重要性优先）

#### 模型评估
- ROC 曲线
- KS 曲线
- 学习曲线
- 交叉验证
- 混淆矩阵
- 分类报告

#### 评分卡实现
- 刻度参数计算（A, B, base_score）
- 变量得分表生成
- 分数转换
- Cutoff 验证
- PR 曲线
- 提升图
- 洛伦兹曲线

#### 评分卡监控
- PSI 计算
- 分数分布对比
- 变量稳定性分析
- 变量偏移检测

#### 聚类分析
- K-Means 聚类
- DBSCAN 聚类
- 层次聚类
- 聚类可视化

### Documentation
- 📖 完整的 API 文档
- 📖 项目简介
- 📖 快速开始指南
- 📖 核心概念文档
  - WOE 和 IV 介绍
  - 评分卡基础
  - 模型评估指标
- 📖 示例集合
  - 基础用法示例
  - 高级流程示例
- 📖 最佳实践指南
- 📖 模块详细文档（9 个模块）
- 📖 系统架构文档
- 📖 贡献指南

### Testing
- ✅ 61 个单元测试
- ✅ 测试覆盖率约 30%（核心模块 100%）
- ✅ 集成测试
- ✅ 测试数据 fixture

### Code Quality
- 🎨 完整的类型提示（Python 3.13+）
- 🎨 代码格式化（ruff）
- 🎨 代码质量检查（41 个问题全部修复）
- 🎨 常量定义文件

### Examples
- 📚 basic_usage.py - 基础用法示例
- 📚 advanced_pipeline.py - 完整建模流程示例

### Development
- 🔧 pyproject.toml 配置
- 🔧 ruff 配置
- 🔧 pytest 配置
- 🔧 .gitignore 配置

## 版本说明

### 语义化版本

Yihuier 遵循语义化版本规范：MAJOR.MINOR.PATCH

- **MAJOR**：不兼容的 API 变更
- **MINOR**：向后兼容的功能新增
- **PATCH**：向后兼容的 Bug 修复

### 更新类型

- **Added**：新增功能
- **Changed**：功能变更
- **Deprecated**：即将废弃的功能
- **Removed**：已删除的功能
- **Fixed**：Bug 修复
- **Security**：安全相关修复

## 贡献指南

如果您想为 Yihuier 做贡献，请查看：

- [贡献指南](contributing.md)
- [系统架构](architecture.md)

## 引用

如果 Yihuier 对您的研究或工作有帮助，请引用：

```bibtex
@software{yihuier2024,
  title = {Yihuier: A Python Package for Credit Scoring Card Modeling},
  author = {Justin Gao},
  year = {2024},
  url = {https://github.com/encyc/yihuier}
}
```

## 许可证

本项目采用 MIT 许可证发布。

---

**最后更新**：2024-01-22
