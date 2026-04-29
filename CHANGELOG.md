# 更新日志

本文档记录 Yihuier 的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.2.2] - 2026-04-29

### 修复
- 修复 ChiMerge 分箱的重复分箱边界 bug（`bins must increase monotonically` 错误）
  - 问题：ChiMerge 可能产生重复的分箱边界，导致 `pd.cut` 失败
  - 修复：使用 `sorted(set(cut))` 去重并排序分箱边界

## [0.2.1] - 2026-04-28

### 修复
- 修复 ChiMerge 分箱方法的 bug（`value_counts().to_frame()` 列名错误）
  - 问题：当变量唯一值超过 100 时，ChiMerge 分箱失败并抛出 `KeyError: 'col_map'`
  - 修复：正确设置 DataFrame 列名为 `'count'`

### 新增
- 新增 `ScorecardMonitorModule.calculate_psi()` 方法
  - 支持直接传入分数数组计算 PSI，无需指定分箱边界
  - 参数：`expected`, `actual`, `buckets=10`, `return_detail=False`
  - 返回 PSI 值或详细 PSI 表格
- Yihuier 类新增 `sm_module` 属性（评分卡监控模块）

### 改进
- 更新 risk-modeling Skill 文档，修复 API 使用示例

## [0.2.0] - 2026-04-23

### 新增
- 🤖 **AI 智能建模 Skill**：内置专业的风控建模 Skill，让 AI 助手自动引导完成 10 步建模全流程
  - 自动触发：当询问信用评分卡建模时自动激活
  - 质量保证：内置 AUC ≥ 0.65、KS ≥ 0.15、PSI < 0.25 质量标准
  - 智能诊断：自动检测问题并提供优化建议
- 📚 **完整的 VitePress 文档网站**
  - 模块化文档结构（指南、概念、API、示例）
  - GitHub Pages 自动部署
  - Mermaid 图表支持
  - 搜索功能
- ✅ **完善的测试框架**
  - 61 个单元测试，核心功能全覆盖
  - pytest 配置和 CI 集成
  - 测试覆盖率报告
- 🔧 **类型提示和代码规范化**
  - 完整的类型注解（Python 3.13+）
  - ruff 代码格式化和检查
  - 统一的代码风格

### 修复
- 修复文档中的死链问题（18 个）
- 修复绘图函数返回值类型不一致
- 修复分箱边界处理逻辑
- 修复数据预处理模块中的常变量删除逻辑
- 修复 XGBoost API 变更导致的兼容性问题

### 文档
- 新增快速开始指南
- 新增 API 参考文档
- 新增最佳实践指南
- 新增高级示例集合
- 新增 AI Skill 专题文档
- 新增架构设计文档

### 改进
- 优化项目结构和模块组织
- 改进错误处理和异常提示
- 增强 API 一致性和易用性
- 提升代码可维护性

### 技术栈
- Python 3.13+
- pandas >= 2.1.4
- scikit-learn >= 1.3.2
- xgboost >= 2.0.3
- matplotlib >= 3.8.2
- seaborn >= 0.12.2

## [0.1.13] - 2024-XX-XX

### 新增
- 🎉 首次发布 Yihuier 评分卡建模工具包
- 完整的面向对象建模架构
- 9 个核心模块：EDA、数据预处理、分箱、变量选择、模型评估、评分卡实现、评分卡监控、聚类、流水线
- 支持多种分箱方法：ChiMerge、等频、等距、单调性分箱
- 完整的 WOE 转换和 IV 计算
- 多种变量选择策略：XGBoost、随机森林、相关性筛选
- 评分卡刻度计算和分数转换
- PSI 稳定性分析和模型监控
- 完整的类型提示（Python 3.13+）
- 61 个单元测试，核心功能全覆盖
- AI 智能建模 Skill（Claude Code 支持）

### 文档
- 完整的 VitePress 文档站点
- 快速开始指南
- API 参考文档
- 最佳实践指南
- 高级示例集合
- 模块化文档结构

### 技术栈
- Python 3.13+
- pandas >= 2.1.4
- scikit-learn >= 1.3.2
- xgboost >= 2.0.3
- matplotlib >= 3.8.2
- seaborn >= 0.12.2

## [未发布]

### 计划中
- 更多分箱方法（决策树分箱、最优分箱）
- 自动化特征工程模块
- 模型对比和自动选择
- 更多评估指标和可视化
- GPU 加速支持
- 分布式计算支持

---

## 版本说明

- **[0.1.0]** - 初始发布版本，包含完整的评分卡建模功能
- **[未发布]** - 计划中的功能和改进
