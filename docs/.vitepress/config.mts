import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'Yihuier',
  description: '专业的信用评分卡建模 Python 工具包',
  base: '/yihuier/',

  markdown: {
    config: (md) => {
      // 可以在这里添加其他 markdown-it 插件
    }
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '指南', link: '/guide/intro' },
      { text: 'AI Skill', link: '/guide/skill' },
      { text: 'API 参考', link: '/guide/api' },
      {
        text: '开发',
        items: [
          { text: '架构设计', link: '/develop/architecture' },
          { text: '贡献指南', link: '/develop/contributing' },
        ]
      },
      {
        text: 'GitHub',
        link: 'https://github.com/encyc/yihuier'
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: '开始',
          items: [
            { text: '项目简介', link: '/guide/intro' },
            { text: 'AI 智能建模', link: '/guide/skill' },
            { text: '快速开始', link: '/guide/quick-start' },
            { text: '安装指南', link: '/guide/installation' },
          ]
        },
        {
          text: '核心概念',
          collapsed: false,
          items: [
            { text: '概念索引', link: '/guide/concepts/index' },
            { text: 'WOE 和 IV', link: '/guide/concepts/woe-iv' },
            { text: '评分卡基础', link: '/guide/concepts/scorecard-basics' },
            { text: '评估指标', link: '/guide/concepts/evaluation-metrics' },
          ]
        },
        {
          text: '功能模块',
          collapsed: false,
          items: [
            { text: '模块索引', link: '/guide/modules/index' },
            { text: 'EDA 模块', link: '/guide/modules/eda' },
            { text: '数据预处理', link: '/guide/modules/data-processing' },
            { text: '分箱模块', link: '/guide/modules/binning' },
            { text: '变量选择', link: '/guide/modules/var-select' },
            { text: '模型评估', link: '/guide/modules/model-evaluation' },
            { text: '评分卡实现', link: '/guide/modules/scorecard-implement' },
            { text: '评分卡监控', link: '/guide/modules/scorecard-monitor' },
            { text: '聚类模块', link: '/guide/modules/cluster' },
            { text: '流水线模块', link: '/guide/modules/pipeline' },
          ]
        },
        {
          text: '更多',
          items: [
            { text: '示例集合', link: '/guide/examples' },
            { text: '高级示例', link: '/guide/advanced-examples' },
            { text: '最佳实践', link: '/guide/best-practices' },
            { text: 'API 文档', link: '/guide/api' },
          ]
        },
      ],
      '/develop/': [
        {
          text: '开发者指南',
          items: [
            { text: '开发者索引', link: '/develop/index' },
            { text: '架构设计', link: '/develop/architecture' },
            { text: '贡献指南', link: '/develop/contributing' },
            { text: '更新日志', link: '/develop/changelog' },
          ]
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/encyc/yihuier' }
    ],

    footer: {
      message: '基于 MIT 许可证发布',
      copyright: 'Copyright © 2024-Present Justin Gao'
    },

    editLink: {
      pattern: 'https://github.com/encyc/yihuier/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },

    search: {
      provider: 'local'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    }
  },
}))
