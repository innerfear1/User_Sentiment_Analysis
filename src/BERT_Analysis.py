# BERT.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, BertModel

from sklearn.model_selection import train_test_split
import logging

# 设置日志级别，避免过多警告信息
logging.basicConfig(level=logging.INFO)


# 自定义用于训练/验证的 Dataset
class targetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # 去除 batch 维度
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding


# 自定义用于测试预测的 Dataset（无标签）
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding


# 评价指标函数，计算准确率
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def main():
    # ---------------------------
    # 1. 数据读取与预处理（使用已清洗数据 clean_data.xlsx）
    # ---------------------------
    data_file = '../output/cleaned_data.xlsx'
    if not os.path.exists(data_file):
        print(f"训练数据文件 {data_file} 不存在，请检查路径。")
        return

    df = pd.read_excel(data_file)

    # 使用 'cleaned_comment' 列，如果不存在则检查 'comment' 列
    if 'cleaned_comment' in df.columns:
        df['text_for_model'] = df['cleaned_comment']
    elif 'comment' in df.columns:
        df['text_for_model'] = df['comment']
    else:
        print("未找到有效的文本列（需包含 'cleaned_comment' 或 'comment'）。")
        return

    # 删除 target 或 text_for_model 中缺失的记录，并确保 target 为整数
    df = df.dropna(subset=['target', 'text_for_model'])
    df['target'] = df['target'].astype(int)

    texts = df['text_for_model'].tolist()
    labels = df['target'].tolist()

    # 划分训练集和验证集（80%训练，20%验证）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # ---------------------------
    # 2. 加载 BERT 分词器与模型
    # ---------------------------
    model_name = "../model/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 构造 Dataset
    train_dataset = targetDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = targetDataset(val_texts, val_labels, tokenizer, max_length=128)

    # 自动对 batch 内序列进行 padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # ---------------------------
    # 3. 训练参数设置及模型训练
    # ---------------------------
    training_args = TrainingArguments(
        output_dir='../bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='../bert_logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 开始训练
    trainer.train()

    # ---------------------------
    # 4. 模型评估
    # ---------------------------
    eval_results = trainer.evaluate()
    print("验证集评估结果：", eval_results)

    # ---------------------------
    # 5. 对测试数据进行预测（如果存在 test.xlsx）
    # ---------------------------
    test_file = '../dataset/test.xlsx'
    if os.path.exists(test_file):
        df_test = pd.read_excel(test_file)
        if 'cleaned_comment' in df_test.columns:
            df_test['text_for_model'] = df_test['cleaned_comment']
        elif 'comment' in df_test.columns:
            df_test['text_for_model'] = df_test['comment']
        else:
            print("测试数据中未找到有效的文本列（需包含 'cleaned_comment' 或 'comment'）。")
            return

        test_texts = df_test['text_for_model'].tolist()
        test_dataset = TestDataset(test_texts, tokenizer, max_length=128)
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        # 将预测结果插入到测试数据的第一列，列名为 'target_prediction'
        df_test.insert(0, 'target_prediction', pred_labels)
        output_file = '../output/BERT_test_with_predictions.xlsx'
        df_test.to_excel(output_file, index=False)
        print("测试数据预测结果已保存到：", output_file)
    else:
        print("未找到测试数据文件 test.xlsx，跳过预测。")


if __name__ == '__main__':
    main()
