#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# DAMA-DMBOK2 章节标题映射
chapters = {
    "01": "数据管理",
    "02": "数据处理伦理", 
    "03": "数据治理",
    "04": "数据架构",
    "05": "数据建模和设计",
    "06": "数据存储和操作",
    "07": "数据安全",
    "08": "数据集成和互操作性",
    "09": "文档和内容管理",
    "10": "参考数据和主数据",
    "11": "数据仓库和商务智能",
    "12": "元数据管理",
    "13": "数据质量",
    "14": "大数据和数据科学",
    "15": "数据管理成熟度评估",
    "16": "数据管理组织和角色期望"
}

# 通用模板
template = """使用 baoyu-image-gen 技能生成专业 DAMA-DMBOK2 教程架构图：

**精确的提示词要求：**
- Light background (white or very light gray)
- High contrast text (dark text on light background for maximum readability)
- 16:9 aspect ratio, 2K resolution (2752x1536 pixels)
- PNG format output
- Professional blueprint style with clear typography
- 使用 gemini-3-pro-image-preview 模型

**章节主题：DAMA-DMBOK2 Chapter {chapter_num}: {chapter_title}**

**视觉元素要求：**
- 中央放置章节标题 "DAMA-DMBOK2 Chapter {chapter_num}: {chapter_title}"
- 围绕中央标题展示该章节的核心概念、流程和组件
- 使用清晰的箭头表示数据流向和关系
- 保持专业的技术架构图风格
- 所有文字使用中文
- 背景使用统一的深蓝色渐变 (#093572 to #103D78)

**内容重点：**
- 突出显示该章节在 DAMA 框架中的位置和作用
- 展示关键的流程、组件和最佳实践
- 体现企业级数据管理的专业性和严谨性
"""

# 创建 images 目录（如果不存在）
os.makedirs("images", exist_ok=True)

# 为每个章节生成提示词文件
for chapter_num, chapter_title in chapters.items():
    filename = f"images/chapter_{chapter_num}_prompt.md"
    content = template.format(chapter_num=chapter_num, chapter_title=chapter_title)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Created {filename}")

print("All chapter prompt files generated successfully!")