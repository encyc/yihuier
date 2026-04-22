# 开发者文档

欢迎查阅 Yihuier 开发者文档！

## 文档目录

- [架构设计](architecture.md) - 系统架构和设计模式
- [贡献指南](contributing.md) - 如何贡献代码和文档
- [更新日志](changelog.md) - 版本更新记录

## 开发资源

### 代码结构

```
yihuier/
├── yihuier/              # 源代码
│   ├── __init__.py
│   ├── yihuier.py        # 主类
│   ├── eda.py            # EDA 模块
│   ├── data_processing.py
│   ├── binning.py
│   ├── var_select.py
│   ├── model_evaluation.py
│   ├── scorecard_implement.py
│   ├── scorecard_monitor.py
│   ├── cluster.py
│   └── pipeline.py
├── tests/                # 测试
├── examples/             # 示例
└── docs/                 # 文档
```

### 开发指南

1. **设置开发环境**
   ```bash
   git clone https://github.com/encyc/yihuier.git
   cd yihuier
   uv pip install -e ".[dev]"
   ```

2. **运行测试**
   ```bash
   pytest tests/ -v
   ```

3. **代码风格**
   ```bash
   ruff format yihuier/
   ruff check yihuier/
   ```

## 贡献方式

我们欢迎各种形式的贡献：

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码

查看 [贡献指南](contributing.md) 了解详情。

## 许可证

本项目采用 MIT 许可证发布。

---

**下一步**: [架构设计](architecture.md) 或 [贡献指南](contributing.md)
