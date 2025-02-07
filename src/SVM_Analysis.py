# ----------------------
# 建立情感倾向模型及评估
# ----------------------
# 利用TF-IDF和SVM建立文本分类模型
# SVM_analysis.py

import os
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

def load_stopwords(file_path):
    """
    加载停用词列表
    """
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

# 加载停用词
stopwords = load_stopwords("../dataset/ChineseStopWords.txt")

def tokenize(text):
    """
    使用 jieba 分词，对文本进行分词，并过滤掉长度为 1 的词和停用词
    """
    words = jieba.lcut(text)
    filtered_words = [word for word in words if len(word) > 1 and word not in stopwords]
    return filtered_words

def main():
    # ---------------------------
    # 1. 数据读取与预处理（使用已清洗数据 clean_data.xlsx）
    # ---------------------------
    data_file = '../output/cleaned_data.xlsx'
    if not os.path.exists(data_file):
        print(f"文件 {data_file} 不存在，请检查路径。")
        return

    df = pd.read_excel(data_file)

    # 如果数据中已经有清洗后的文本（例如存储在 "cleaned_comment" 列中），则直接使用；
    # 否则使用 "comment" 列，并进行分词处理。
    if 'cleaned_comment' in df.columns:
        df['text_for_model'] = df['cleaned_comment']
    elif 'comment' in df.columns:
        df['text_for_model'] = df['comment'].apply(lambda x: " ".join(tokenize(x)) if isinstance(x, str) else "")
    else:
        print("未在数据中找到有效的文本列（cleaned_comment 或 comment）。")
        return

    # 去除缺失值和重复数据（要求 target 与 text_for_model 均有数据）
    df = df.dropna(subset=['target', 'text_for_model'])
    df = df.drop_duplicates()

    # ---------------------------
    # 2. 特征提取与 SVM 模型训练
    # ---------------------------
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        df['text_for_model'], df['target'], test_size=0.2, random_state=42
    )

    # 使用 TF-IDF 提取文本特征
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # 训练 SVM 模型（采用 LinearSVC）
    clf = LinearSVC(random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # ---------------------------
    # 3. 模型评估
    # ---------------------------
    y_pred = clf.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    print("验证集准确率：", acc)
    print("分类报告：\n", report)
    print("均方误差：", mse)

    # 将评估结果保存为图片
    eval_text = f"test_acc: {acc:.4f}\n\nevaluation:\n{report}\nmse: {mse:.4f}"

    # 创建画布，并将评估文本写入图中
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    ax.text(0.01, 0.99, eval_text, verticalalignment='top',
            fontfamily='monospace', fontsize=10)

    # 确保输出目录存在，这里保存到 output 目录
    output_dir = "../output/"
    os.makedirs(output_dir, exist_ok=True)
    eval_img_path = os.path.join(output_dir, "SVM_evaluation_metrics.png")
    plt.savefig(eval_img_path, bbox_inches='tight')
    plt.close()
    print("评估结果图片已保存到：", eval_img_path)

    # ---------------------------
    # 4. 对测试数据进行预测（如果存在 test.xlsx 文件）
    # ---------------------------
    test_file = '../dataset/test.xlsx'
    if os.path.exists(test_file):
        df_test = pd.read_excel(test_file)
        # 同样，判断是否已有清洗后的文本列
        if 'cleaned_comment' in df_test.columns:
            df_test['text_for_model'] = df_test['cleaned_comment']
        elif 'comment' in df_test.columns:
            df_test['text_for_model'] = df_test['comment'].apply(lambda x: " ".join(tokenize(x)) if isinstance(x, str) else "")
        else:
            print("测试数据中未找到有效的文本列（cleaned_comment 或 comment）。")
            return

        X_test_tfidf = tfidf.transform(df_test['text_for_model'])
        test_pred = clf.predict(X_test_tfidf)

        # 将预测结果写入 Excel 文件第一列，列名为 'target_prediction'
        df_test.insert(0, 'target_prediction', test_pred)
        output_file = '../output/SVM_test_with_predictions.xlsx'
        df_test.to_excel(output_file, index=False)
        print("测试数据预测结果已保存到：", output_file)
    else:
        print(f"测试数据文件 {test_file} 不存在，跳过测试数据预测。")

if __name__ == '__main__':
    main()
