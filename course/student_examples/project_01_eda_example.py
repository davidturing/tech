#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1: 探索性数据分析 (EDA) - 全球AI发展趋势分析
学生示例代码
作者: 张明 (985高校工科大二学生)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_explore_data():
    """数据加载与初步探索"""
    print("=== 数据加载与探索 ===")
    
    # 创建模拟数据（实际项目中会从CSV文件加载）
    np.random.seed(42)
    countries = ['中国', '美国', '英国', '德国', '日本', '韩国', '法国', '加拿大', '澳大利亚', '印度']
    years = list(range(2020, 2027))
    
    data = []
    for country in countries:
        base_investment = np.random.uniform(100, 1000)
        base_patents = np.random.randint(50, 500)
        base_talent = np.random.randint(1000, 10000)
        base_companies = np.random.randint(10, 100)
        gov_support = np.random.uniform(5, 9)
        
        for year in years:
            # 模拟逐年增长趋势
            growth_factor = 1.1 ** (year - 2020)
            investment = base_investment * growth_factor + np.random.normal(0, 50)
            patents = int(base_patents * growth_factor + np.random.normal(0, 20))
            talent = int(base_talent * growth_factor + np.random.normal(0, 100))
            companies = int(base_companies * growth_factor + np.random.normal(0, 5))
            
            data.append({
                'country': country,
                'year': year,
                'ai_investment_millions': max(0, investment),
                'ai_patents': max(0, patents),
                'ai_talent_count': max(0, talent),
                'ai_companies': max(0, companies),
                'government_support_score': gov_support
            })
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值和异常值用于演示清洗过程
    df.loc[df.sample(frac=0.05).index, 'ai_investment_millions'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'ai_patents'] = -1  # 异常值
    
    print(f"数据形状: {df.shape}")
    print(f"数据基本信息:")
    print(df.info())
    print(f"\n数值列统计摘要:")
    print(df.describe())
    
    return df

def clean_data(df):
    """数据清洗"""
    print("\n=== 数据清洗 ===")
    
    # 1. 处理缺失值
    print(f"缺失值统计:")
    print(df.isnull().sum())
    
    # 使用前向填充处理投资金额的缺失值
    df['ai_investment_millions'] = df.groupby('country')['ai_investment_millions'].fillna(method='ffill')
    
    # 对于仍然存在的缺失值，使用国家平均值填充
    country_means = df.groupby('country')['ai_investment_millions'].mean()
    for country in df['country'].unique():
        mask = (df['country'] == country) & (df['ai_investment_millions'].isnull())
        df.loc[mask, 'ai_investment_millions'] = country_means[country]
    
    # 2. 处理异常值
    print(f"\n异常值处理前 - AI专利最小值: {df['ai_patents'].min()}")
    df.loc[df['ai_patents'] < 0, 'ai_patents'] = 0
    print(f"异常值处理后 - AI专利最小值: {df['ai_patents'].min()}")
    
    # 3. 数据类型优化
    df['year'] = df['year'].astype('int32')
    df['ai_investment_millions'] = df['ai_investment_millions'].astype('float32')
    df['ai_patents'] = df['ai_patents'].astype('int32')
    df['ai_talent_count'] = df['ai_talent_count'].astype('int32')
    df['ai_companies'] = df['ai_companies'].astype('int32')
    
    print(f"\n清洗后数据形状: {df.shape}")
    print(f"清洗后缺失值: {df.isnull().sum().sum()}")
    
    return df

def basic_analysis(df):
    """基础分析"""
    print("\n=== 基础分析 ===")
    
    # 1. 计算各国AI投资的年增长率
    df_sorted = df.sort_values(['country', 'year'])
    df_sorted['investment_growth'] = df_sorted.groupby('country')['ai_investment_millions'].pct_change()
    df_sorted['investment_growth'] = df_sorted['investment_growth'].fillna(0)
    
    print(f"各国AI投资年均增长率 (2020-2026):")
    growth_by_country = df_sorted.groupby('country')['investment_growth'].mean()
    print(growth_by_country.sort_values(ascending=False))
    
    # 2. 找出AI人才密度最高的前10个国家
    # 这里简化为直接按人才数量排序（实际项目中会考虑人口等因素）
    talent_by_country = df[df['year'] == 2026].groupby('country')['ai_talent_count'].mean()
    print(f"\nAI人才数量最多的前5个国家 (2026年):")
    print(talent_by_country.sort_values(ascending=False).head())
    
    # 3. 分析政府支持度与AI投资的相关性
    correlation = df[['government_support_score', 'ai_investment_millions']].corr().iloc[0, 1]
    print(f"\n政府支持度与AI投资的相关系数: {correlation:.3f}")
    
    return df_sorted

def create_visualizations(df):
    """创建可视化"""
    print("\n=== 可视化生成 ===")
    
    # 1. 时间序列图展示全球AI投资趋势
    global_investment = df.groupby('year')['ai_investment_millions'].sum().reset_index()
    
    fig1 = px.line(global_investment, x='year', y='ai_investment_millions',
                   title='全球AI投资趋势 (2020-2026)',
                   labels={'ai_investment_millions': 'AI投资金额 (百万美元)', 'year': '年份'})
    fig1.write_html("global_ai_investment_trend.html")
    print("已保存: global_ai_investment_trend.html")
    
    # 2. 热力图显示各国AI发展指标相关性
    numeric_cols = ['ai_investment_millions', 'ai_patents', 'ai_talent_count', 'ai_companies', 'government_support_score']
    correlation_matrix = df[numeric_cols].corr()
    
    fig2 = px.imshow(correlation_matrix, 
                     labels=dict(color="相关系数"),
                     title="AI发展指标相关性热力图",
                     color_continuous_scale='RdBu_r')
    fig2.write_html("ai_correlation_heatmap.html")
    print("已保存: ai_correlation_heatmap.html")
    
    # 3. 散点图矩阵展示多变量关系
    sample_df = df[df['year'] == 2026].sample(n=min(100, len(df)), random_state=42)
    
    fig3 = px.scatter_matrix(sample_df, 
                            dimensions=['ai_investment_millions', 'ai_patents', 'ai_talent_count', 'government_support_score'],
                            color='country',
                            title="AI发展指标散点图矩阵 (2026年)")
    fig3.update_traces(diagonal_visible=False)
    fig3.write_html("ai_scatter_matrix.html")
    print("已保存: ai_scatter_matrix.html")
    
    # 4. 地理可视化 - 交互式世界地图
    latest_data = df[df['year'] == 2026].groupby('country').first().reset_index()
    
    # 创建国家代码映射（简化版）
    country_codes = {
        '中国': 'CHN', '美国': 'USA', '英国': 'GBR', '德国': 'DEU', '日本': 'JPN',
        '韩国': 'KOR', '法国': 'FRA', '加拿大': 'CAN', '澳大利亚': 'AUS', '印度': 'IND'
    }
    latest_data['country_code'] = latest_data['country'].map(country_codes)
    
    fig4 = px.choropleth(latest_data, 
                        locations='country_code',
                        color='ai_investment_millions',
                        hover_name='country',
                        hover_data=['ai_talent_count', 'ai_companies', 'government_support_score'],
                        color_continuous_scale='Viridis',
                        title='2026年各国AI投资分布')
    fig4.write_html("ai_investment_world_map.html")
    print("已保存: ai_investment_world_map.html")
    
    return True

def main():
    """主函数"""
    print("🚀 开始执行项目1: 探索性数据分析 (EDA) - 全球AI发展趋势分析")
    
    # 1. 数据加载与探索
    df = load_and_explore_data()
    
    # 2. 数据清洗
    df_clean = clean_data(df)
    
    # 3. 基础分析
    df_analyzed = basic_analysis(df_clean)
    
    # 4. 可视化
    create_visualizations(df_clean)
    
    # 5. 保存清洗后的数据
    df_clean.to_csv("cleaned_ai_trends_data.csv", index=False)
    print("\n✅ 已保存清洗后的数据: cleaned_ai_trends_data.csv")
    
    print("\n🎉 项目1执行完成！所有输出文件已保存到当前目录。")
    print("📋 生成的文件包括:")
    print("   - cleaned_ai_trends_data.csv")
    print("   - global_ai_investment_trend.html")  
    print("   - ai_correlation_heatmap.html")
    print("   - ai_scatter_matrix.html")
    print("   - ai_investment_world_map.html")

if __name__ == "__main__":
    main()