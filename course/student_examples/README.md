# 学生示例代码

本目录包含985高校工科大二学生完成《Python数据分析》课程四个项目作业的示例代码。

## 学生信息
- **姓名**: 张明
- **学校**: 985高校工科专业
- **年级**: 大二
- **技术背景**: 具备基础Python编程知识，对数据分析和人工智能感兴趣

## 项目列表

### 1. 项目一：数据清洗与预处理实战
- **文件**: `project_01_data_cleaning_example.py`
- **内容**: 演示真实世界数据的清洗技巧，包括缺失值处理、异常值检测、数据质量验证等
- **技术栈**: Pandas, Polars, Great Expectations

### 2. 项目一：探索性数据分析 (EDA)
- **文件**: `project_01_eda_example.py`  
- **内容**: 全球AI发展趋势分析，包含完整的EDA流程和交互式可视化
- **技术栈**: Pandas, Seaborn, Plotly

### 3. 项目二：机器学习预测管道
- **文件**: `project_02_ml_pipeline_example.py`
- **内容**: 房价预测机器学习管道，从数据预处理到模型部署
- **技术栈**: Scikit-learn, Pandas, Joblib

### 4. 项目三：大数据分析实战
- **文件**: `project_03_big_data_example.py`
- **内容**: NYC出租车大数据分析，性能对比Pandas vs Dask vs Polars
- **技术栈**: Pandas, Dask, Polars, Vaex

### 5. 项目四：AI增强的数据分析系统
- **文件**: `project_04_ai_enhanced_example.py`
- **内容**: 智能数据分析助手，集成LLM进行自然语言查询和代码生成
- **技术栈**: Streamlit, Plotly, Mock LLM

### 6. 作业总结
- **文件**: `student_assignment_summary.md`
- **内容**: 学生对四个项目的完整总结和学习心得

## 使用说明

所有Python脚本都可以直接运行（需要安装相应的依赖包）：

```bash
# 安装依赖
pip install pandas numpy matplotlib seaborn plotly scikit-learn dask polars streamlit

# 运行项目示例
python project_01_data_cleaning_example.py
python project_01_eda_example.py
python project_02_ml_pipeline_example.py  
python project_03_big_data_example.py
python project_04_ai_enhanced_example.py  # 可以用streamlit run运行获得完整Web界面
```

## 项目特点

- **符合2026年技术标准**: 使用最新的Python 3.12+、Pandas 3.0、Polars等技术栈
- **完整的项目结构**: 包含数据加载、清洗、分析、可视化、模型训练等完整流程
- **教育价值**: 代码注释详细，适合其他学生学习参考
- **实用性**: 所有代码都经过本地测试验证，可以直接运行