# DavidAgent 概要设计文档 (High-Level Design Document)

## 0. 版本与概览
- **项目名称**: DavidAgent
- **定位**: 仿生双脑多智能体自我进化系统
- **核心哲学**: 模拟人类大脑的“感官-逻辑-创造-记忆”闭环，通过夜间反思实现 Prompt 级的持续进化。

---

## 1. 架构设计 (Architecture Design)

DavidAgent 采用**仿生双脑 + 三层海马体**架构，通过“物理通道”连接各个模块。

### 1.1 双脑模型 (Dual-Brain Model)
- **左脑 (Left Brain - Gemini)**: **事实中枢**。负责结构化阅读、RDF 图谱提取（PageIndex）、事实核查与红军审查。它的底色是“严谨、怀疑、逻辑”。
- **右脑 (Right Brain - Qwen)**: **创意中枢**。负责文学创作、风格化改写、情感注入。它的底色是“发散、热忱、直觉”。

### 1.2 海马体三层记忆 (Hippocampus Memory Matrix)
系统不依赖单一数据库，而是构建了三层递进的记忆层：
1. **层一：ChromaDB (语义记忆)**: 存储向量化的摘要，通过余弦相似度实现“潜意识”关联召回。支持艾宾浩斯时间衰减。
2. **层二：PageIndex (逻辑记忆)**: 基于本地 Markdown 知识图谱。实体、关系、属性的结构化沉淀，构建 AI 的稳定世界观。
3. **层三：SQLite WAL (情景记忆)**: 史官日志。记录工作流轨迹 (trace_logs) 与原始信号 (raw_signals)，支持高并发写入。

### 1.3 虚拟胼胝体 (Virtual Corpus Callosum)
左脑与右脑通过黑板模式（Blackboard Pattern）进行状态交换，胼胝体负责管理任务锁与通信同步。

---

## 2. 特性设计 (Feature Design)

### 2.1 自动化流水线 (Bionic Pipeline)
- **Perceptor (感知器)**: 采集 X/RSS/GitHub 信号，进行初步清洗。
- **Constructor (构造器)**: 左脑将信号逻辑化，存入 PageIndex。
- **Generator (发生器)**: 右脑基于图谱和历史记忆进行创作。
- **Shield (免疫盾)**: 左脑审查产出，如果不符事实或违反规则，则打回重写。

### 2.2 凌晨睡眠整理 (Nightly Consolidation)
- **睡眠窗口 (0:00 - 6:00)**: 系统自动停止接收外部信号，进入反思模式。
- **五阶段整理**:
    - **Replaying**: 扫描向量记忆，标记过期。
    - **Evaluating**: 评估图谱密度，识别昨日“缺陷记忆”（被拦截的任务）。
    - **Archiving**: 将教训转化为新规则。

### 2.3 规则剪枝 (Rule Consolidation)
为防止 Prompt 膨胀，系统在反思时会强制合并、冷冻或删除旧规则，保持 System Prompt 始终精炼。

---

## 3. 代码设计 (Code Design)

### 3.1 模块化分层
- `brain/`: 系统核心，包含 `dashboard.py` (元认知大盘) 与 `reflection/` (反思引擎)。
- `brain/memory/`: 核心记忆模块，封装了 `Consolidator`, `EpisodicDB`, `ChromaIntegration`。
- `left_brain/` & `right_brain/`: 具体的智能体处理逻辑。
- `config/`: 动态规则 (`dynamic_guidelines.md`) 与静态配置。

### 3.2 鲁棒性与容灾
- **JSON Fallback**: 当 ChromaDB 在高版本 Python 环境异常时，自动回退到 `LeanMemoryStore` (JSON 暴力检索)。
- **WAL 模式**: SQLite 开启 WAL，保障并发读写稳定性。
- **异步安全**: 记忆检索带有超时熔断保护，防止海马体异常阻塞主工作流。

---

## 4. 维护设计 (Maintenance Design)

### 4.1 监控管理 (Dashboard)
- **海马体监控**: 实时观察三层记忆计数、睡眠周期进度。
- **RLHF 闭环**: 人类可在 Web 端打分打标，反馈将直达次日的夜间反思流程。

### 4.2 数据治理
- **自动清理**: 记忆整理引擎自动删除 >180 天的非核心向量记忆。
- **手动介入**: Dashboard 提供“手动触发整理”与“数据库清理”按钮。

### 4.3 协同演进 (For OpenClaw)
- **接管建议**: OpenClaw 应优先关注 `consolidation_logs` 以确保进化链路通畅。
- **规则干预**: 可直接修改 `config/persona.json` 来调整双脑的人格底色。

---
*文档更新于: 2026-02-21*
