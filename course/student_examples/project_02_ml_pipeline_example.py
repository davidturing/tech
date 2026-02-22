#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目二：机器学习预测管道 - 房价预测示例
学生示例代码
作者: 张明 (985高校工科大二学生)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_housing_data():
    """生成模拟房价数据（实际项目中会从CSV文件加载）"""
    np.random.seed(42)
    n_samples = 1000
    
    # 特征生成
    data = {
        'area': np.random.normal(100, 30, n_samples),  # 面积 (平方米)
        'bedrooms': np.random.randint(1, 6, n_samples),  # 卧室数量
        'bathrooms': np.random.randint(1, 4, n_samples),  # 浴室数量
        'age': np.random.randint(0, 50, n_samples),  # 房龄
        'location_score': np.random.uniform(1, 10, n_samples),  # 位置评分
        'school_district': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3]),  # 学区
        'has_garden': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # 是否有花园
        'has_parking': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # 是否有停车位
    }
    
    # 目标变量生成（基于特征的线性组合 + 噪声）
    price = (
        data['area'] * 1000 +
        data['bedrooms'] * 20000 +
        data['bathrooms'] * 15000 +
        data['location_score'] * 30000 +
        (50 - data['age']) * 1000 +
        np.where(data['school_district'] == 'A', 50000, 
                np.where(data['school_district'] == 'B', 30000, 10000)) +
        data['has_garden'] * 20000 +
        data['has_parking'] * 15000 +
        np.random.normal(0, 20000, n_samples)  # 噪声
    )
    
    data['price'] = np.maximum(price, 100000)  # 确保价格为正
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值用于演示处理过程
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'area'] = np.nan
    
    return df

def explore_data(df):
    """数据探索与可视化"""
    print("=== 数据探索 ===")
    
    print(f"数据形状: {df.shape}")
    print(f"\n数据基本信息:")
    print(df.info())
    print(f"\n数值列统计摘要:")
    print(df.describe())
    
    # 可视化目标变量分布
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['price'], bins=50, alpha=0.7, color='skyblue')
    plt.title('房价分布')
    plt.xlabel('价格 (元)')
    plt.ylabel('频次')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['price'])
    plt.title('房价箱线图')
    plt.ylabel('价格 (元)')
    
    plt.tight_layout()
    plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特征相关性热力图
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def prepare_features(df):
    """特征工程"""
    print("\n=== 特征工程 ===")
    
    # 创建新特征
    df['price_per_sqm'] = df['price'] / df['area']  # 每平方米价格
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']  # 总房间数
    df['is_new'] = (df['age'] <= 5).astype(int)  # 是否为新房
    
    # 处理分类变量
    le = LabelEncoder()
    df['school_district_encoded'] = le.fit_transform(df['school_district'])
    
    print(f"新增特征: price_per_sqm, total_rooms, is_new, school_district_encoded")
    
    return df, le

def build_models(X_train, X_test, y_train, y_test):
    """模型训练与评估"""
    print("\n=== 模型训练与评估 ===")
    
    # 定义模型
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name} 性能:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
    
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 最佳模型: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
    
    return results, best_model, best_model_name

def hyperparameter_tuning(X_train, y_train):
    """超参数调优"""
    print("\n=== 超参数调优 ===")
    
    # Random Forest 超参数调优
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def create_prediction_pipeline(best_model, feature_names):
    """创建预测管道"""
    print("\n=== 创建预测管道 ===")
    
    # 创建完整的管道（包括预处理和模型）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    
    # 保存管道
    joblib.dump(pipeline, 'house_price_prediction_pipeline.pkl')
    print("✅ 预测管道已保存为: house_price_prediction_pipeline.pkl")
    
    # 创建使用示例
    sample_input = pd.DataFrame({
        'area': [120],
        'bedrooms': [3],
        'bathrooms': [2],
        'age': [10],
        'location_score': [8.5],
        'school_district_encoded': [0],  # A类学区
        'has_garden': [1],
        'has_parking': [1],
        'price_per_sqm': [8000],
        'total_rooms': [5],
        'is_new': [0]
    })
    
    # 加载管道并预测
    loaded_pipeline = joblib.load('house_price_prediction_pipeline.pkl')
    prediction = loaded_pipeline.predict(sample_input)
    
    print(f"\n示例预测:")
    print(f"输入特征: {sample_input.iloc[0].to_dict()}")
    print(f"预测房价: ¥{prediction[0]:,.2f}")
    
    return pipeline

def main():
    """主函数"""
    print("🚀 开始执行项目二：机器学习预测管道 - 房价预测")
    
    # 1. 数据生成与探索
    df = generate_housing_data()
    df = explore_data(df)
    
    # 2. 特征工程
    df, label_encoder = prepare_features(df)
    
    # 3. 准备训练数据
    feature_cols = ['area', 'bedrooms', 'bathrooms', 'age', 'location_score', 
                   'school_district_encoded', 'has_garden', 'has_parking',
                   'price_per_sqm', 'total_rooms', 'is_new']
    X = df[feature_cols]
    y = df['price']
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # 4. 模型训练与评估
    results, best_model, best_model_name = build_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 5. 超参数调优
    best_model_tuned = hyperparameter_tuning(X_train_scaled, y_train)
    
    # 6. 创建预测管道
    pipeline = create_prediction_pipeline(best_model_tuned, feature_cols)
    
    print("\n🎉 项目二执行完成！所有输出文件已保存到当前目录。")
    print("📋 生成的文件包括:")
    print("   - price_distribution.png")
    print("   - feature_correlation.png") 
    print("   - house_price_prediction_pipeline.pkl")

if __name__ == "__main__":
    main()