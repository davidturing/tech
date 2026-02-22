#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目一：数据清洗与预处理实战 - 示例代码
学生：李明（985高校计算机科学与技术专业大二学生）
学号：2024123456
完成时间：2026年2月22日

本项目使用模拟的电商平台用户行为数据，演示完整的数据清洗流程。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_data():
    """生成模拟的电商平台用户行为数据"""
    np.random.seed(42)
    
    # 生成基础数据
    n_records = 10000
    user_ids = np.random.randint(1000, 9999, n_records)
    session_ids = np.random.randint(100000, 999999, n_records)
    
    # 时间戳（最近30天）
    base_time = pd.Timestamp('2026-01-01')
    time_offsets = np.random.randint(0, 30*24*60, n_records)
    timestamps = base_time + pd.to_timedelta(time_offsets, unit='m')
    
    # 页面类型
    page_types = np.random.choice(['home', 'product', 'cart', 'checkout', 'profile'], 
                                 n_records, p=[0.3, 0.4, 0.15, 0.1, 0.05])
    
    # 设备类型
    device_types = np.random.choice(['mobile', 'desktop', 'tablet'], 
                                   n_records, p=[0.6, 0.35, 0.05])
    
    # 浏览时长（秒）
    view_durations = np.random.exponential(120, n_records)  # 平均2分钟
    
    # 购买金额（部分为0，表示未购买）
    purchase_amounts = np.where(
        np.random.random(n_records) < 0.15,  # 15%的转化率
        np.random.lognormal(3, 1, n_records) * 10,  # 对数正态分布
        0
    )
    
    # 创建DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'session_id': session_ids,
        'timestamp': timestamps,
        'page_type': page_types,
        'device_type': device_types,
        'view_duration_seconds': view_durations,
        'purchase_amount': purchase_amounts
    })
    
    # 故意引入一些数据质量问题
    # 1. 缺失值
    df.loc[np.random.choice(df.index, 200), 'user_id'] = np.nan
    df.loc[np.random.choice(df.index, 150), 'page_type'] = None
    df.loc[np.random.choice(df.index, 100), 'view_duration_seconds'] = np.nan
    
    # 2. 异常值
    df.loc[np.random.choice(df.index, 50), 'view_duration_seconds'] = 100000  # 10万秒（异常）
    df.loc[np.random.choice(df.index, 30), 'purchase_amount'] = -100  # 负金额（异常）
    
    # 3. 数据类型不一致
    df.loc[np.random.choice(df.index, 20), 'device_type'] = 'MOBILE'  # 大写
    df.loc[np.random.choice(df.index, 15), 'device_type'] = 'Phone'   # 不同命名
    
    return df

def data_quality_assessment(df):
    """数据质量评估"""
    print("=== 数据质量评估报告 ===")
    print(f"数据集形状: {df.shape}")
    print(f"缺失值统计:")
    missing_stats = df.isnull().sum()
    for col, missing in missing_stats.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    print(f"\n数据类型:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print(f"\n数值列基本统计:")
    print(df.describe())
    
    return missing_stats

def clean_missing_values(df):
    """处理缺失值"""
    df_clean = df.copy()
    
    # user_id缺失：删除记录（关键字段）
    df_clean = df_clean.dropna(subset=['user_id'])
    
    # page_type缺失：用众数填充
    mode_page = df_clean['page_type'].mode()[0]
    df_clean['page_type'] = df_clean['page_type'].fillna(mode_page)
    
    # view_duration_seconds缺失：用中位数填充
    median_duration = df_clean['view_duration_seconds'].median()
    df_clean['view_duration_seconds'] = df_clean['view_duration_seconds'].fillna(median_duration)
    
    print(f"缺失值处理后数据集形状: {df_clean.shape}")
    return df_clean

def detect_and_handle_outliers(df):
    """检测和处理异常值"""
    df_clean = df.copy()
    
    # 处理view_duration_seconds异常值（使用IQR方法）
    Q1 = df_clean['view_duration_seconds'].quantile(0.25)
    Q3 = df_clean['view_duration_seconds'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    # 将超过上限的值设为上限值（而不是删除，保留记录）
    df_clean['view_duration_seconds'] = np.where(
        df_clean['view_duration_seconds'] > upper_bound,
        upper_bound,
        df_clean['view_duration_seconds']
    )
    
    # 处理purchase_amount负值
    df_clean['purchase_amount'] = np.where(
        df_clean['purchase_amount'] < 0,
        0,
        df_clean['purchase_amount']
    )
    
    print(f"异常值处理完成")
    return df_clean

def standardize_categorical_data(df):
    """标准化分类数据"""
    df_clean = df.copy()
    
    # 标准化device_type
    device_mapping = {
        'MOBILE': 'mobile',
        'Phone': 'mobile',
        'DESKTOP': 'desktop',
        'TABLET': 'tablet'
    }
    
    df_clean['device_type'] = df_clean['device_type'].replace(device_mapping)
    df_clean['device_type'] = df_clean['device_type'].str.lower()
    
    # 确保只有预期的值
    valid_devices = ['mobile', 'desktop', 'tablet']
    df_clean = df_clean[df_clean['device_type'].isin(valid_devices)]
    
    print(f"分类数据标准化完成")
    return df_clean

def create_visualizations(df_original, df_clean):
    """创建可视化对比"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始数据 vs 清洗后数据的浏览时长分布
    axes[0,0].hist(df_original['view_duration_seconds'].dropna(), bins=50, alpha=0.7, label='原始数据')
    axes[0,0].hist(df_clean['view_duration_seconds'], bins=50, alpha=0.7, label='清洗后数据')
    axes[0,0].set_xlabel('浏览时长（秒）')
    axes[0,0].set_ylabel('频次')
    axes[0,0].set_title('浏览时长分布对比')
    axes[0,0].legend()
    
    # 购买金额分布
    axes[0,1].hist(df_original['purchase_amount'], bins=50, alpha=0.7, label='原始数据')
    axes[0,1].hist(df_clean['purchase_amount'], bins=50, alpha=0.7, label='清洗后数据')
    axes[0,1].set_xlabel('购买金额')
    axes[0,1].set_ylabel('频次')
    axes[0,1].set_title('购买金额分布对比')
    axes[0,1].legend()
    
    # 设备类型分布
    device_counts_original = df_original['device_type'].value_counts()
    device_counts_clean = df_clean['device_type'].value_counts()
    
    axes[1,0].bar(range(len(device_counts_original)), device_counts_original.values, alpha=0.7, label='原始数据')
    axes[1,0].bar(range(len(device_counts_clean)), device_counts_clean.values, alpha=0.7, label='清洗后数据')
    axes[1,0].set_xlabel('设备类型')
    axes[1,0].set_ylabel('数量')
    axes[1,0].set_title('设备类型分布对比')
    axes[1,0].set_xticks(range(len(device_counts_clean)))
    axes[1,0].set_xticklabels(device_counts_clean.index, rotation=45)
    axes[1,0].legend()
    
    # 页面类型分布
    page_counts_original = df_original['page_type'].value_counts()
    page_counts_clean = df_clean['page_type'].value_counts()
    
    axes[1,1].bar(range(len(page_counts_original)), page_counts_original.values, alpha=0.7, label='原始数据')
    axes[1,1].bar(range(len(page_counts_clean)), page_counts_clean.values, alpha=0.7, label='清洗后数据')
    axes[1,1].set_xlabel('页面类型')
    axes[1,1].set_ylabel('数量')
    axes[1,1].set_title('页面类型分布对比')
    axes[1,1].set_xticks(range(len(page_counts_clean)))
    axes[1,1].set_xticklabels(page_counts_clean.index, rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('data_cleaning_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("项目一：数据清洗与预处理实战")
    print("=" * 50)
    
    # 1. 生成模拟数据
    print("1. 生成模拟数据...")
    df_raw = generate_sample_data()
    print(f"原始数据集大小: {df_raw.shape}")
    
    # 2. 数据质量评估
    print("\n2. 数据质量评估...")
    data_quality_assessment(df_raw)
    
    # 3. 数据清洗流程
    print("\n3. 开始数据清洗...")
    
    # 处理缺失值
    df_step1 = clean_missing_values(df_raw)
    
    # 处理异常值
    df_step2 = detect_and_handle_outliers(df_step1)
    
    # 标准化分类数据
    df_cleaned = standardize_categorical_data(df_step2)
    
    print(f"\n清洗完成！最终数据集大小: {df_cleaned.shape}")
    print(f"数据清洗成功率: {df_cleaned.shape[0]/df_raw.shape[0]*100:.2f}%")
    
    # 4. 保存清洗后的数据
    df_cleaned.to_csv('cleaned_ecommerce_data.csv', index=False)
    print("\n清洗后的数据已保存到: cleaned_ecommerce_data.csv")
    
    # 5. 创建可视化
    print("\n6. 创建可视化对比...")
    create_visualizations(df_raw, df_cleaned)
    
    print("\n项目一完成！")

if __name__ == "__main__":
    main()