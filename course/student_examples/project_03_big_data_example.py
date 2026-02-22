#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目三：大数据分析实战 - NYC出租车数据分析示例
学生示例代码
作者: 张明 (985高校工科大二学生)
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import polars as pl
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_sample_taxi_data(n_samples=100000):
    """生成模拟NYC出租车数据（实际项目中会使用真实的大数据集）"""
    print(f"生成 {n_samples:,} 条模拟出租车数据...")
    
    np.random.seed(42)
    
    # 生成时间数据
    start_date = pd.Timestamp('2026-01-01')
    end_date = pd.Timestamp('2026-12-31')
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # 生成位置数据（简化版，实际会有更复杂的地理坐标）
    pickup_lat = np.random.uniform(40.5, 40.9, n_samples)
    pickup_lon = np.random.uniform(-74.2, -73.7, n_samples)
    dropoff_lat = np.random.uniform(40.5, 40.9, n_samples)
    dropoff_lon = np.random.uniform(-74.2, -73.7, n_samples)
    
    # 生成其他特征
    passenger_count = np.random.randint(1, 6, n_samples)
    trip_distance = np.random.exponential(3, n_samples)  # 指数分布模拟行程距离
    fare_amount = trip_distance * 2.5 + np.random.normal(3, 1, n_samples)  # 基础费用 + 距离费用
    tip_amount = np.where(fare_amount > 10, np.random.exponential(2, n_samples), 0)
    total_amount = fare_amount + tip_amount
    
    # 确保金额为正
    fare_amount = np.maximum(fare_amount, 0)
    tip_amount = np.maximum(tip_amount, 0)
    total_amount = np.maximum(total_amount, 0)
    
    data = {
        'tpep_pickup_datetime': date_range,
        'tpep_dropoff_datetime': date_range + pd.to_timedelta(np.random.exponential(15, n_samples), unit='m'),
        'passenger_count': passenger_count,
        'trip_distance': trip_distance,
        'pickup_latitude': pickup_lat,
        'pickup_longitude': pickup_lon,
        'dropoff_latitude': dropoff_lat,
        'dropoff_longitude': dropoff_lon,
        'fare_amount': fare_amount,
        'tip_amount': tip_amount,
        'total_amount': total_amount,
        'payment_type': np.random.choice([1, 2], n_samples, p=[0.8, 0.2])  # 1:信用卡, 2:现金
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些异常值用于演示清洗过程
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[outlier_indices, 'trip_distance'] = np.random.uniform(100, 500, len(outlier_indices))
    df.loc[outlier_indices, 'total_amount'] = np.random.uniform(500, 2000, len(outlier_indices))
    
    return df

def performance_comparison():
    """性能对比：Pandas vs Dask vs Polars"""
    print("\n=== 性能对比测试 ===")
    
    # 生成不同规模的数据集进行测试
    sizes = [10000, 50000, 100000]
    results = []
    
    for size in sizes:
        print(f"\n测试数据规模: {size:,} 条记录")
        
        # 生成测试数据
        df_test = generate_sample_taxi_data(size)
        
        # 保存为CSV用于Dask和Polars测试
        csv_file = f'test_data_{size}.csv'
        df_test.to_csv(csv_file, index=False)
        
        # Pandas 测试
        start_time = time.time()
        df_pandas = pd.read_csv(csv_file)
        df_pandas_filtered = df_pandas[df_pandas['trip_distance'] < 50]
        df_pandas_agg = df_pandas_filtered.groupby('payment_type')['total_amount'].mean()
        pandas_time = time.time() - start_time
        
        # Dask 测试
        start_time = time.time()
        df_dask = dd.read_csv(csv_file)
        df_dask_filtered = df_dask[df_dask['trip_distance'] < 50]
        df_dask_agg = df_dask_filtered.groupby('payment_type')['total_amount'].mean().compute()
        dask_time = time.time() - start_time
        
        # Polars 测试
        start_time = time.time()
        df_polars = pl.read_csv(csv_file)
        df_polars_filtered = df_polars.filter(pl.col('trip_distance') < 50)
        df_polars_agg = df_polars_filtered.group_by('payment_type').agg(pl.col('total_amount').mean())
        polars_time = time.time() - start_time
        
        results.append({
            'size': size,
            'pandas_time': pandas_time,
            'dask_time': dask_time,
            'polars_time': polars_time,
            'pandas_result': df_pandas_agg.to_dict(),
            'dask_result': df_dask_agg.to_dict(),
            'polars_result': df_polars_agg.to_dict()
        })
        
        print(f"Pandas 时间: {pandas_time:.3f}s")
        print(f"Dask 时间: {dask_time:.3f}s") 
        print(f"Polars 时间: {polars_time:.3f}s")
        
        # 清理临时文件
        Path(csv_file).unlink()
    
    return results

def big_data_analysis_with_polars():
    """使用Polars进行大数据分析"""
    print("\n=== 使用Polars进行大数据分析 ===")
    
    # 生成大规模数据
    df_large = generate_sample_taxi_data(500000)  # 50万条记录
    
    # 保存为Parquet格式（高效存储）
    parquet_file = 'nyc_taxi_large.parquet'
    df_large.to_parquet(parquet_file)
    
    # 使用Polars加载和分析
    start_time = time.time()
    df_pl = pl.read_parquet(parquet_file)
    
    # 数据清洗
    df_clean = (
        df_pl
        .filter(pl.col('trip_distance') > 0)
        .filter(pl.col('trip_distance') < 100)  # 移除异常长距离
        .filter(pl.col('total_amount') > 0)
        .filter(pl.col('total_amount') < 1000)  # 移除异常高金额
    )
    
    # 高性能聚合分析
    daily_stats = (
        df_clean
        .with_columns([
            pl.col('tpep_pickup_datetime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col('tpep_pickup_datetime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.date().alias('pickup_date')
        ])
        .group_by('pickup_date')
        .agg([
            pl.count().alias('trip_count'),
            pl.col('total_amount').mean().alias('avg_fare'),
            pl.col('trip_distance').mean().alias('avg_distance'),
            pl.col('passenger_count').mean().alias('avg_passengers')
        ])
        .sort('pickup_date')
    )
    
    polars_analysis_time = time.time() - start_time
    
    print(f"Polars分析完成，耗时: {polars_analysis_time:.3f}s")
    print(f"清洗后数据量: {len(df_clean):,} 条记录")
    print(f"日期范围: {daily_stats['pickup_date'].min()} 到 {daily_stats['pickup_date'].max()}")
    
    # 可视化结果
    daily_stats_pd = daily_stats.to_pandas()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].plot(daily_stats_pd['pickup_date'], daily_stats_pd['trip_count'])
    axes[0,0].set_title('每日行程数量')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    axes[0,1].plot(daily_stats_pd['pickup_date'], daily_stats_pd['avg_fare'])
    axes[0,1].set_title('平均车费')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    axes[1,0].plot(daily_stats_pd['pickup_date'], daily_stats_pd['avg_distance'])
    axes[1,0].set_title('平均行程距离')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    axes[1,1].plot(daily_stats_pd['pickup_date'], daily_stats_pd['avg_passengers'])
    axes[1,1].set_title('平均乘客数量')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('big_data_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 清理文件
    Path(parquet_file).unlink()
    
    return daily_stats_pd

def distributed_computing_demo():
    """分布式计算演示（Dask）"""
    print("\n=== 分布式计算演示（Dask） ===")
    
    # 生成多个数据文件模拟分布式数据
    n_files = 5
    file_list = []
    
    for i in range(n_files):
        df_chunk = generate_sample_taxi_data(100000)  # 每个文件10万条
        csv_file = f'taxi_chunk_{i:02d}.csv'
        df_chunk.to_csv(csv_file, index=False)
        file_list.append(csv_file)
    
    # 使用Dask读取所有文件
    start_time = time.time()
    df_dask = dd.read_csv('taxi_chunk_*.csv')
    
    # 分布式数据清洗
    df_clean = df_dask[
        (df_dask['trip_distance'] > 0) & 
        (df_dask['trip_distance'] < 100) &
        (df_dask['total_amount'] > 0) &
        (df_dask['total_amount'] < 1000)
    ]
    
    # 分布式聚合
    payment_stats = df_clean.groupby('payment_type').agg({
        'total_amount': ['count', 'mean', 'sum'],
        'trip_distance': 'mean',
        'passenger_count': 'mean'
    }).compute()
    
    dask_time = time.time() - start_time
    
    print(f"Dask分布式分析完成，耗时: {dask_time:.3f}s")
    print(f"处理总数据量: {len(df_clean):,} 条记录")
    print(f"支付方式统计:")
    print(payment_stats)
    
    # 清理临时文件
    for file in file_list:
        Path(file).unlink()
    
    return payment_stats

def main():
    """主函数"""
    print("🚀 开始执行项目三：大数据分析实战 - NYC出租车数据分析")
    
    # 1. 性能对比测试
    performance_results = performance_comparison()
    
    # 2. 使用Polars进行大数据分析
    analysis_results = big_data_analysis_with_polars()
    
    # 3. 分布式计算演示
    distributed_results = distributed_computing_demo()
    
    # 4. 生成性能报告
    print("\n=== 性能对比总结 ===")
    for result in performance_results:
        size = result['size']
        print(f"\n数据规模: {size:,} 条记录")
        print(f"  Pandas: {result['pandas_time']:.3f}s")
        print(f"  Dask:   {result['dask_time']:.3f}s")  
        print(f"  Polars: {result['polars_time']:.3f}s")
        
        speedup_polars = result['pandas_time'] / result['polars_time']
        print(f"  Polars相对Pandas加速: {speedup_polars:.2f}x")
    
    print("\n🎉 项目三执行完成！所有输出文件已保存到当前目录。")
    print("📋 生成的文件包括:")
    print("   - big_data_analysis_results.png")
    print("   - 性能对比数据（控制台输出）")

if __name__ == "__main__":
    main()