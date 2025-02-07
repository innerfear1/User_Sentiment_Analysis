# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import datetime
import os


# 定义函数加载停用词
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

# 加载停用词
stopwords = load_stopwords("../dataset/ChineseStopWords.txt")

# 文本处理与分词函数
def tokenize(text):
    # 使用jieba分词，去除长度为1的词
    words = jieba.lcut(text)
    filtered_words = [word for word in words if len(word) > 1 and word not in stopwords]
    return filtered_words


def save_cleaned_data(df, output_path):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    cleaned_file = os.path.join(output_path, 'cleaned_data.xlsx')
    df.to_excel(cleaned_file, index=False)
    print(f"清洗后的数据已保存至：{cleaned_file}")

# 生成词云
def generate_wordcloud(word_freq, title, output_file=None):
    wc = WordCloud(
        font_path='../dataset/font.ttf',  # 指定支持中文的字体路径
        background_color='white',
        width=800,
        height=600
    )
    wc.generate_from_frequencies(word_freq)
    # 保存词云图到指定路径
    if output_file:
        wc.to_file(output_file)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        plt.show()


# 统计高频词
def top_n_words(text_series, n=10):
    # 对文本序列进行分词
    words = []
    for text in text_series.dropna():
        words.extend(tokenize(text))
    counter = Counter(words)
    return counter.most_common(n)


# 主流程
def main():
    # ----------------------
    # 1. 数据读取与清洗
    # ----------------------
    # 请根据实际情况修改文件路径和字段名
    train_file = '../dataset/train.xlsx'
    test_file = '../dataset/test.xlsx'

    # 读取训练数据
    df = pd.read_excel(train_file)
    # 示例字段：comment（评论内容），target（情感标签：1正向 0负向），sellerId（商家名称），timestamp（评价时间）
    # 数据清洗：去除空评论、重复数据
    df = df.dropna(subset=['comment', 'target'])
    df = df.drop_duplicates()

    # 将评价时间转换为 datetime 格式（假设字段名为'timestamp'）
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['cleaned_comment'] = df['comment'].apply(lambda x: " ".join(tokenize(x)))

    # ----------------------
    # 2. 词云与高频词统计
    # ----------------------
    # 分别针对正向和负向评论生成词云图并统计词频
    df_negative = df[df['target'] == 1]
    df_positive = df[df['target'] == 0]

    # 生成正向词云
    pos_counter = Counter()
    for text in df_positive['cleaned_comment']:
        pos_counter.update(text.split())
    generate_wordcloud(pos_counter, title="正向情绪词云", output_file='../output/positive_wordcloud.png')


    # 生成负向词云
    neg_counter = Counter()
    for text in df_negative['cleaned_comment']:
        neg_counter.update(text.split())
    generate_wordcloud(neg_counter, title="负向情绪词云", output_file='../output/negative_wordcloud.png')

    # 保存清洗后数据
    save_cleaned_data(df, '../output/')


    # ----------------------
    # 3. 评价时间与情感关系分析
    # ----------------------
    # 按时间（例如按月）统计正向和负向评论数量
    # 此处先将时间归一到年月
    df['year_month'] = df['timestamp'].dt.to_period('M')
    time_target = df.groupby(['year_month', 'target']).size().reset_index(name='count')
    # 将 Period 类型转换为字符串便于绘图
    time_target['year_month'] = time_target['year_month'].astype(str)

    # 绘制折线图
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_target, x='year_month', y='count', hue='target', marker="o")
    plt.xticks(rotation=45)
    plt.title("按月统计正向与负向评论数量")
    plt.xlabel("时间")
    plt.ylabel("评论数")
    plt.legend(title="情感（1:正向, 0:负向）")
    plt.tight_layout()
    plt.show()

    # 保存折线图
    os.makedirs('../output', exist_ok=True)
    lineplot_file = os.path.join('../output/', 'sentiment_trend.png')
    plt.savefig(lineplot_file)
    print(f"折线图已保存至：{lineplot_file}")
    plt.close()  # 关闭图形，避免显示多次

    # 分析：根据图形观察不同时间段正负评论的变化趋势，判断二者是否存在关系，
    # 例如：某些时间节点负向评论激增可能与特殊事件或服务波动相关。

    # ----------------------
    # 4. 商家评价分析
    # ----------------------
    # 分析各商家正向和负向评论数
    sellerId_target = df.groupby(['sellerId', 'target']).size().unstack(fill_value=0)
    sellerId_target.columns = ['negative', 'positive']
    sellerId_target = sellerId_target.reset_index()

    # 找出正向评论最多的商家
    best_sellerId = sellerId_target.sort_values(by='positive', ascending=False).iloc[0]
    print("正向评论最多的商家：", best_sellerId['sellerId'])
    # 根据正向评论内容总结优点（这里可以进一步分析该商家评论中的高频词）
    best_comments = df[(df['sellerId'] == best_sellerId['sellerId']) & (df['target'] == 1)]
    best_top_words = top_n_words(best_comments['comment'], n=10)
    print("该商家评论中正向高频词：", best_top_words)

    # 找出负向评论最多的商家
    worst_sellerId = sellerId_target.sort_values(by='negative', ascending=False).iloc[0]
    print("负向评论最多的商家：", worst_sellerId['sellerId'])
    worst_comments = df[(df['sellerId'] == worst_sellerId['sellerId']) & (df['target'] == 0)]
    worst_top_words = top_n_words(worst_comments['comment'], n=10)
    print("该商家评论中负向高频词：", worst_top_words)
    # 根据负向高频词，提出改进策略（例如：如果高频词中出现“服务慢”、“态度差”等，可建议加强培训和提升服务质量）



if __name__ == '__main__':
    main()
