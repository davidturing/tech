#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目四：AI增强的数据分析系统 - 智能数据分析助手示例
学生示例代码
作者: 张明 (985高校工科大二学生)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 模拟LLM响应（实际项目中会调用真实的API）
class MockLLM:
    """模拟大型语言模型的响应"""
    
    def __init__(self):
        self.insights_database = {
            'sales': [
                "销售额在Q4有显著增长，可能与节假日促销有关",
                "产品A的销量明显高于其他产品，建议增加库存",
                "周末的销售额比工作日高出约30%",
                "客户年龄段主要集中在25-35岁之间"
            ],
            'weather': [
                "气温与能源消耗呈现明显的负相关关系",
                "降雨量对户外活动参与度有显著影响",
                "季节性模式显示夏季用电量最高",
                "极端天气事件频率在过去5年有所增加"
            ],
            'finance': [
                "股票价格波动与宏观经济指标高度相关",
                "投资组合的夏普比率表明风险调整后收益良好",
                "市场情绪指标可以有效预测短期价格走势",
                "不同资产类别之间的相关性在危机期间会增加"
            ]
        }
    
    def generate_insight(self, data_type, query=""):
        """生成数据洞察"""
        if data_type in self.insights_database:
            import random
            return random.choice(self.insights_database[data_type])
        return "基于数据分析，发现了一些有趣的模式和趋势。"
    
    def generate_code(self, query, columns):
        """生成Pandas代码"""
        # 简化的代码生成逻辑
        if "平均值" in query or "mean" in query:
            col = columns[0] if columns else "value"
            return f"df['{col}'].mean()"
        elif "相关性" in query or "correlation" in query:
            if len(columns) >= 2:
                return f"df['{columns[0]}'].corr(df['{columns[1]}'])"
            else:
                return "df.corr()"
        elif "分组" in query or "group" in query:
            if len(columns) >= 2:
                return f"df.groupby('{columns[0]}')['{columns[1]}'].mean()"
            else:
                return "df.groupby('category').mean()"
        else:
            return "df.describe()"

def load_sample_data(data_type="sales"):
    """加载示例数据"""
    np.random.seed(42)
    
    if data_type == "sales":
        dates = pd.date_range('2026-01-01', periods=365, freq='D')
        products = ['产品A', '产品B', '产品C', '产品D']
        data = []
        for date in dates:
            for product in products:
                base_sales = np.random.normal(1000, 200)
                # 添加季节性趋势
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_bonus = 1.2 if date.dayofweek in [5, 6] else 1.0
                sales = base_sales * seasonal_factor * weekend_bonus
                data.append({
                    'date': date,
                    'product': product,
                    'sales': max(0, sales),
                    'customers': int(sales / np.random.uniform(50, 100)),
                    'revenue': sales * np.random.uniform(10, 50)
                })
        return pd.DataFrame(data)
    
    elif data_type == "weather":
        dates = pd.date_range('2026-01-01', periods=365, freq='D')
        data = []
        for date in dates:
            temp = 20 + 15 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 5)
            humidity = np.random.uniform(30, 90)
            rainfall = np.random.exponential(2) if np.random.random() < 0.3 else 0
            energy_usage = 100 + 2 * abs(temp - 22) + np.random.normal(0, 10)
            data.append({
                'date': date,
                'temperature': temp,
                'humidity': humidity,
                'rainfall': rainfall,
                'energy_usage': energy_usage
            })
        return pd.DataFrame(data)
    
    else:  # finance
        dates = pd.date_range('2026-01-01', periods=252, freq='B')  # 工作日
        assets = ['股票A', '股票B', '债券', '黄金']
        data = []
        prices = {asset: 100 for asset in assets}
        for date in dates:
            for asset in assets:
                # 随机游走价格模型
                change = np.random.normal(0, 0.02)
                prices[asset] *= (1 + change)
                volume = np.random.randint(1000, 10000)
                data.append({
                    'date': date,
                    'asset': asset,
                    'price': prices[asset],
                    'volume': volume,
                    'return': change
                })
        return pd.DataFrame(data)

def create_visualization(df, viz_type, x_col, y_col=None, color_col=None):
    """创建可视化图表"""
    try:
        if viz_type == "折线图":
            if y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col)
            else:
                fig = px.line(df, x=x_col, y=df.columns[1])
        elif viz_type == "柱状图":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col)
            else:
                fig = px.bar(df, x=x_col, y=df.columns[1])
        elif viz_type == "散点图":
            if len(df.columns) >= 3:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            else:
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
        elif viz_type == "热力图":
            if len(df.select_dtypes(include=[np.number]).columns) >= 2:
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix, title="相关性热力图")
            else:
                fig = go.Figure()
        elif viz_type == "箱线图":
            if y_col:
                fig = px.box(df, x=x_col, y=y_col, color=color_col)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.box(df, y=numeric_cols[0])
                else:
                    fig = go.Figure()
        else:
            fig = px.line(df, x=df.columns[0], y=df.columns[1])
        
        return fig
    except Exception as e:
        st.error(f"创建可视化时出错: {str(e)}")
        return go.Figure()

def main():
    """主函数 - Streamlit应用"""
    st.set_page_config(page_title="AI增强数据分析系统", layout="wide")
    
    st.title("🤖 AI增强数据分析系统")
    st.markdown("### 985高校工科学生项目 - 智能数据分析助手")
    
    # 初始化LLM
    llm = MockLLM()
    
    # 侧边栏 - 数据选择
    st.sidebar.header("数据配置")
    data_type = st.sidebar.selectbox(
        "选择数据类型",
        ["sales", "weather", "finance"],
        format_func=lambda x: {"sales": "销售数据", "weather": "天气数据", "finance": "金融数据"}[x]
    )
    
    # 加载数据
    df = load_sample_data(data_type)
    
    st.sidebar.success(f"已加载 {len(df):,} 条记录")
    
    # 主界面布局
    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据概览", "🔍 AI洞察", "🎨 可视化", "💻 代码生成"])
    
    with tab1:
        st.subheader("数据概览")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("数据形状:", df.shape)
            st.write("数据类型:")
            st.write(df.dtypes)
        
        with col2:
            st.write("缺失值统计:")
            st.write(df.isnull().sum())
        
        st.write("数据预览:")
        st.dataframe(df.head(10))
        
        if st.checkbox("显示完整统计摘要"):
            st.write(df.describe())
    
    with tab2:
        st.subheader("AI智能洞察")
        
        if st.button("🔄 生成新洞察"):
            with st.spinner("AI正在分析数据..."):
                insight = llm.generate_insight(data_type)
                st.success("💡 **AI洞察**: " + insight)
        
        # 默认显示一个洞察
        default_insight = llm.generate_insight(data_type)
        st.info("💡 **AI洞察**: " + default_insight)
        
        st.subheader("自然语言查询")
        user_query = st.text_input("输入你的问题 (例如: '销售额的平均值是多少？' 或 '显示各产品的相关性'):")
        
        if user_query:
            with st.spinner("AI正在处理查询..."):
                # 简单的查询处理
                if "平均值" in user_query or "mean" in user_query.lower():
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        means = df[numeric_cols].mean()
                        st.write("平均值结果:")
                        st.write(means)
                
                elif "相关性" in user_query or "correlation" in user_query.lower():
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        corr = df[numeric_cols].corr()
                        st.write("相关性矩阵:")
                        st.dataframe(corr)
                
                else:
                    st.write("这是一个很好的问题！在完整版本中，AI会生成相应的分析代码和可视化。")
    
    with tab3:
        st.subheader("交互式可视化")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            viz_type = st.selectbox("图表类型", ["折线图", "柱状图", "散点图", "热力图", "箱线图"])
        with col2:
            x_col = st.selectbox("X轴", df.columns)
        with col3:
            y_options = [col for col in df.columns if col != x_col]
            y_col = st.selectbox("Y轴", y_options if y_options else [x_col])
        
        color_options = [None] + [col for col in df.columns if col not in [x_col, y_col]]
        color_col = st.selectbox("颜色分组 (可选)", color_options)
        
        if st.button("生成图表"):
            fig = create_visualization(df, viz_type, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
        
        # 预设可视化
        st.subheader("预设分析视图")
        preset_viz = st.selectbox("选择预设视图", ["时间趋势", "分布分析", "相关性分析"])
        
        if preset_viz == "时间趋势":
            if 'date' in df.columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.line(df, x='date', y=numeric_cols[0], title=f"{numeric_cols[0]} 时间趋势")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif preset_viz == "分布分析":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], title=f"{numeric_cols[0]} 分布")
                st.plotly_chart(fig, use_container_width=True)
        
        elif preset_viz == "相关性分析":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="特征相关性热力图")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("AI代码生成")
        
        st.markdown("""
        ### 使用说明
        在完整版本中，这个系统会：
        1. 接收自然语言查询
        2. 使用LLM生成相应的Pandas/Plotly代码
        3. 执行代码并显示结果
        4. 允许用户编辑和重用生成的代码
        
        下面是一个示例：
        """)
        
        example_query = "计算各产品的平均销售额"
        st.write(f"**查询**: {example_query}")
        
        # 生成示例代码
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            example_code = f"df.groupby('{categorical_cols[0]}')['{numeric_cols[0]}'].mean()"
            st.code(example_code, language='python')
            
            try:
                result = eval(example_code)
                st.write("执行结果:")
                st.write(result)
            except Exception as e:
                st.error(f"执行错误: {str(e)}")
        else:
            st.code("df.describe()", language='python')
            st.write("执行结果:")
            st.write(df.describe())
        
        st.markdown("---")
        st.info("💡 **提示**: 这个演示版本使用模拟的LLM响应。在实际部署中，会集成真实的大型语言模型API（如OpenAI、Qwen等）。")
    
    # 页脚
    st.markdown("---")
    st.caption("🎓 985高校工科大二学生项目 | Python数据分析课程 | 2026年")

if __name__ == "__main__":
    # 检查是否在Streamlit环境中运行
    try:
        main()
    except Exception as e:
        print(f"如果要运行完整的Streamlit应用，请使用命令: streamlit run {__file__}")
        print(f"当前环境错误: {e}")
        
        # 在非Streamlit环境中，展示核心功能
        print("\n=== AI增强数据分析系统 - 核心功能演示 ===")
        
        llm = MockLLM()
        df = load_sample_data("sales")
        
        print(f"加载了 {len(df)} 条销售数据")
        print(f"数据列: {list(df.columns)}")
        
        insight = llm.generate_insight("sales")
        print(f"\nAI洞察: {insight}")
        
        code = llm.generate_code("计算平均销售额", ["sales"])
        print(f"\n生成的代码: {code}")
        
        result = df['sales'].mean()
        print(f"执行结果: {result:.2f}")
        
        print("\n✅ 核心功能演示完成！")