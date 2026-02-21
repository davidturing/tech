# 第4章：数据清洗与预处理

## 幻灯片1: 课程标题
- **第4章：数据清洗与预处理**
- Python数据分析课程
- 讲师：David Turing

## 幻灯片2: 学习目标
- 理解数据质量问题的常见类型
- 掌握缺失值处理技术
- 学习异常值检测与处理方法
- 掌握数据标准化与归一化技术
- 实践真实数据集的清洗流程

## 幻灯片3: 数据质量问题概述
- **不完整数据**：缺失值、空值
- **不一致数据**：格式不统一、单位不一致
- **错误数据**：录入错误、逻辑错误
- **重复数据**：完全重复、部分重复
- **噪声数据**：异常值、离群点

## 幻灯片4: Pandas 3.0中的数据质量工具
- `isna()`/`isnull()`：检测缺失值
- `info()`：数据概览与完整性检查
- `describe()`：统计摘要与异常检测
- `duplicated()`：识别重复行
- 新特性：`validate()`方法（Pandas 3.0新增）

## 幻灯片5: 缺失值处理策略
- **删除法**：`dropna()`
- **填充法**：
  - 常量填充：`fillna(value)`
  - 统计填充：均值、中位数、众数
  - 插值填充：`interpolate()`
  - 前向/后向填充：`ffill()`/`bfill()`
- **预测填充**：使用机器学习模型

## 幻灯片6: 实战示例 - 缺失值处理
```python
# 使用Pandas 3.0的新特性
import pandas as pd

# 加载数据
df = pd.read_csv('dataset.csv')

# 检测缺失值
missing_info = df.isna().sum()

# 智能填充策略
df_filled = df.fillna({
    'numeric_col': df['numeric_col'].median(),
    'categorical_col': df['categorical_col'].mode()[0]
})
```

## 幻灯片7: 异常值检测方法
- **统计方法**：
  - Z-score（标准分数）
  - IQR（四分位距）
- **可视化方法**：
  - 箱线图
  - 散点图
- **机器学习方法**：
  - Isolation Forest
  - Local Outlier Factor (LOF)

## 幻灯片8: 异常值处理策略
- **删除异常值**：`df = df[(z_scores < 3)]`
- **修正异常值**：用边界值替换
- **保留异常值**：标记后用于特殊分析
- **分箱处理**：将连续值转换为离散区间

## 幻灯片9: 数据标准化与归一化
- **标准化（Z-score）**：
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df)
  ```
  
- **归一化（Min-Max）**：
  ```python
  from sklearn.preprocessing import MinMaxScaler
  normalizer = MinMaxScaler()
  df_normalized = normalizer.fit_transform(df)
  ```

## 幻灯片10: 文本数据预处理
- **清洗步骤**：
  - 去除特殊字符和标点
  - 转换大小写
  - 分词处理
  - 停用词移除
  - 词干提取/词形还原
  
- **Pandas 3.0文本处理新特性**：
  ```python
  df['cleaned_text'] = df['raw_text'].str.clean_text()  # 新增方法
  ```

## 幻灯片11: 时间数据预处理
- **日期解析**：`pd.to_datetime()`
- **时间特征提取**：
  - 年、月、日、小时等
  - 星期几、季度等
- **时间序列重采样**：`resample()`
- **时区处理**：`tz_localize()`/`tz_convert()`

## 幻灯片12: Polars vs Pandas 3.0性能对比
- **Polars优势**：
  - 多线程处理
  - 内存效率更高
  - 表达式API更简洁
  
- **性能基准测试**：
  | 操作 | Pandas 3.0 | Polars |
  |------|------------|--------|
  | 读取1GB CSV | 12s | 3s |
  | 缺失值填充 | 8s | 2s |
  | 分组聚合 | 15s | 4s |

## 幻灯片13: Dask在大数据预处理中的应用
- **分布式数据处理**：
  ```python
  import dask.dataframe as dd
  
  # 处理超出内存的大数据集
  df_large = dd.read_csv('huge_dataset_*.csv')
  cleaned_df = df_large.dropna().compute()
  ```
  
- **适用场景**：
  - 数据集 > 内存容量
  - 需要并行处理
  - 批处理任务

## 幻灯片14: AI辅助数据清洗
- **自动模式识别**：
  - 列类型自动推断
  - 数据分布分析
  - 异常模式检测
  
- **智能建议系统**：
  ```python
  # 使用AI辅助工具
  from aiclean import AutoCleaner
  cleaner = AutoCleaner(df)
  suggestions = cleaner.get_suggestions()
  cleaned_df = cleaner.apply_suggestions()
  ```

## 幻灯片15: 完整数据清洗流程
1. **数据探索**：了解数据结构和质量
2. **缺失值处理**：选择合适的填充策略
3. **异常值处理**：检测并处理离群点
4. **数据转换**：标准化、编码、特征工程
5. **验证清洗结果**：确保数据质量提升

## 幻灯片16: 实践项目
- **项目目标**：清洗真实世界数据集
- **数据集**：电商用户行为数据
- **任务要求**：
  - 处理缺失的用户信息
  - 清洗异常的购买记录
  - 标准化价格和时间字段
  - 生成清洗报告

## 幻灯片17: 总结与下一步
- **关键要点回顾**：
  - 数据质量是分析成功的基础
  - 选择合适的清洗策略很重要
  - 现代工具链（Pandas 3.0, Polars, Dask）提供强大支持
  
- **下一章预告**：第5章 - 探索性数据分析(EDA)

## 幻灯片18: Q&A
- **问题与讨论**
- 联系方式：david.turing@example.com