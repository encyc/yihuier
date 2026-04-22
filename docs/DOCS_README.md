# Yihuier 文档

本目录包含 Yihuier 项目的完整文档，使用 [VitePress](https://vitepress.dev/) 构建。

## 本地开发

### 前置要求

- Node.js >= 18
- npm 或 yarn

### 安装依赖

```bash
cd docs
npm install
```

### 启动开发服务器

```bash
npm run docs:dev
```

访问 https://encyc.github.io/yihuier/ 查看文档。

### 构建静态文件

```bash
npm run docs:build
```

构建产物在 `docs/.vitepress/dist` 目录。

### 预览构建结果

```bash
npm run docs:preview
```

## 文档结构

```
docs/
├── index.md              # 首页
├── .vitepress/
│   └── config.ts         # VitePress 配置
├── guide/                # 用户指南
│   ├── intro.md          # 项目简介
│   ├── quick-start.md    # 快速开始
│   ├── installation.md   # 安装指南
│   ├── concepts/         # 核心概念
│   ├── modules/          # 功能模块文档
│   ├── examples.md       # 示例集合
│   ├── best-practices.md # 最佳实践
│   └── api.md            # API 文档
└── develop/              # 开发者文档
    ├── architecture.md   # 架构设计
    ├── contributing.md   # 贡献指南
    └── changelog.md      # 更新日志
```

## 部署到 GitHub Pages

文档会自动在以下情况部署：

1. 推送到 `main` 分支
2. `docs/` 目录下的文件有变更

部署后的文档地址：https://encyc.github.io/yihuier/

## 编写文档

### Markdown 语法

VitePress 支持 GitHub Flavored Markdown，并扩展了以下功能：

- 自定义容器
- 语法高亮代码块
- 数学公式
- Mermaid 图表

### 自定义容器

::: tip 提示
这是一条提示信息
:::

::: warning 警告
这是一条警告信息
:::

::: danger 危险
这是一条危险信息
:::

### Mermaid 图表

\`\`\`mermaid
flowchart LR
    A[开始] --> B[处理]
    B --> C[结束]
\`\`\`

### 代码块

\`\`\`python
from yihuier import Yihuier

yh = Yihuier(data, target='dlq_flag')
\`\`\`

## 相关资源

- [VitePress 官方文档](https://vitepress.dev/)
- [项目贡献指南](../docs/develop/contributing.md)
