# 🚀餐饮服务评价情感分析
## 项目简介

本项目基于 **SVM** 和 **BERT** 机器学习模型，对餐饮行业的用户评价数据进行情感倾向分析。项目流程包括数据清洗、特征提取、模型训练、情感分类以及测试数据预测。最终目标是通过文本分析方法帮助餐饮商家优化服务，提高顾客满意度。

## 项目功能

**数据预处理**：读取用户评价数据，进行分词、去停用词、去重等操作，并保存清洗后的数据。

**词云生成**：生成并保存正向与负向评价的词云图。

**情感分类**：
    
**SVM**（支持向量机）模型：使用 TF-IDF 提取特征，训练 SVM 进行情感分类。
    
**BERT**（预训练语言模型）：使用 bert-base-chinese 进行深度学习建模，实现更精确的情感分析。

**模型评估**：输出分类模型的准确率、分类报告、均方误差，并将结果保存为图片。

**测试数据预测**：对测试数据集进行情感预测，并保存预测结果到 Excel 文件。

## 项目结构

```angular2html
|—— bert_logs                  # bert日志
|—— bert_results               # 训练权重返回
├── data/                      # 数据文件夹（存放原始数据）
│   ├── train.xlsx             # 训练数据集
│   ├── test.xlsx              # 测试数据集（待预测）
│   ├── stopwords.txt          # 停用词表
│   ├── clean_data.xlsx        # 预处理后的数据
├── output/                    # 结果输出目录
│   ├── positive_wordcloud.png # 正向评论词云
│   ├── negative_wordcloud.png # 负向评论词云
│   ├── sentiment_trend.png    # 情感趋势折线图
│   ├── evaluation_metrics.png # 评估结果（SVM/BERT）
│   ├── test_with_predictions.xlsx        # SVM 预测结果
│   ├── test_with_predictions_bert.xlsx   # BERT 预测结果
├── SVM_analysis.py             # SVM 模型训练与测试
├── BERT.py                     # BERT 模型训练与测试
├── data_preprocessing.py        # 数据清洗与词云生成
├── README.md                   # 项目说明文档
```


## 环境依赖
请确保安装以下依赖包：
```bash
pip install transformers torch pandas scikit-learn matplotlib wordcloud jieba openpyxl seaborn
```

## 使用方法
## 1. 数据预处理
运行 `Wordcloud_Shop_sentiment.py` 进行数据清洗、分词、停用词去除，并生成词云和折线图：
```bash
python data_preprocessing.py
```
**生成的结果：**

clean_data.xlsx（清洗后的数据）

[positive_wordcloud.png（正向词云）](./output/positive_wordcloud.png)

[negative_wordcloud.png（负向词云）](./output/negative_wordcloud.png)

sentiment_trend.png（情感趋势折线图）

## 2.训练 SVM 模型
运行 `SVM_analysis.py` 训练 SVM 模型，并在验证集上评估模型效果
```bash
python SVM_analysis.py
```

**输出**：

训练集模型评估（准确率、分类报告、均方误差）

[`SVM_evaluation_metrics.png`（模型评估结果图片）](./output/SVM_evaluation_metrics.png)

test_with_predictions.xlsx（SVM 对 test.xlsx 进行情感预测的结果）

## 3.训练 BERT 模型

运行 `BERT.py` 训练 BERT 模型，并在验证集上评估模型：
```bash
python BERT.py
```

**输出**：

训练集模型评估（准确率、分类报告、均方误差）

`test_with_predictions_bert.xlsx`（BERT 对 test.xlsx 进行情感预测的结果）

## 结果分析
#1. 词云分析
`positive_wordcloud.png` 显示了正向评论中出现频率最高的词。

`negative_wordcloud.png` 显示了负向评论中出现频率最高的词，有助于分析用户的不满点。

#2.情感趋势分析
`sentiment_trend.png` 展示了不同时间段的正负面评价数量变化趋势。

#3.模型评估
`evaluation_metrics.png` 展示了模型的准确率、分类报告和误差信息，便于比较 SVM 和 BERT 的效果。

#4.测试数据预测
`SVM_test_with_predictions.xlsx（SVM）`和 `BERT_test_with_predictions.xlsx（BERT）`存储了测试数据的情感预测结果，方便后续业务分析。

## 项目改进方向
**增强停用词库**：优化停用词表，去除无意义的常用词，提高分词质量。

**数据增强**：对评论数据进行扩充，增加模型的泛化能力。

**优化BERT模型**：尝试使用 `ERNIE`、`RoBERTa` 等更强大的中文预训练模型，提高分类效果。

**增加特征工程**：结合更多特征（如评论时间、评分）进行情感分类。

## 贡献者
**开发者**：`@innerfear1`

**数据来源**：`全国数据分析大赛数据集`

## 许可证
本项目基于 **MIT License** 开源，欢迎学习和使用，但请注明来源。