"""
通过GPT-2模型训练一个文本分类器
步骤：
    1. 导入相关库
    2. 加载数据集
    3. 加载模型
"""

# ======================================1、导入相关库======================================
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import Trainer, Seq2SeqTrainer
import transformers
from transformers import DataCollatorWithPadding
from transformers import TextGenerationPipeline
import torch
import numpy as np
import os, re
from tqdm import tqdm
import torch.nn as nn

MAX_LEN=32             # 最大长度
SEED=42                 # 随机种子
LR=2e-5                 # 学习率
BATCH_SIZE=8          # BATCH_SIZE
WARMUP_RATIO=0.1        # warmup比例
INTERVAL=100             # 每多少步打一次 log / 做一次 eval

MODEL_NAME = "gpt2"                     # 模型名称
# MODEL_NAME = "gpt2-large"
MODEL_PATH = "/Volumes/WD_BLACK/models/gpt2/" # 模型路径

DATASET_NAME = "rotten_tomatoes"        # 数据集名称
DATA_BODY_KEY = "text"
DATA_LABEL_KEY = "label"

# ======================================2、加载数据集======================================
# 加载数据集
raw_datasets = load_dataset(DATASET_NAME, cache_dir="/Volumes/WD_BLACK/data")

# 训练集
raw_train_dataset = raw_datasets["train"]

# 验证集
raw_valid_dataset = raw_datasets["validation"]


columns = raw_train_dataset.column_names

# 设置随机种子
transformers.set_seed(SEED)

# =====================================4、加载tokenizer====================================
# 定义tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH,trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0

named_labels = ['neg','pos']

label_ids = [
    tokenizer(named_labels[i],add_special_tokens=False)["input_ids"][0] 
    for i in range(len(named_labels))
]

# ======================================5、数据处理======================================
# 定义数据处理函数，把原始数据转成input_ids, attention_mask, labels
"""
转成模型接受的输入格式
   - 拼接输入输出：<INPUT TOKEN IDS><EOS_TOKEN_ID><OUTPUT TOKEN IDS>
   - PAD成相等长度：
      - <INPUT 1.1><INPUT 1.2>...<EOS_TOKEN_ID><OUTPUT TOKEN IDS><PAD>...<PAD>
      - <INPUT 2.1><INPUT 2.2>...<EOS_TOKEN_ID><OUTPUT TOKEN IDS><PAD>...<PAD>
   - 标识出参与 Loss 计算的 Tokens (只有输出 Token 参与 Loss 计算)
      - <-100><-100>...<OUTPUT TOKEN IDS><-100>...<-100>
"""
def process_fn(examples):
    model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
    for i in range(len(examples[DATA_BODY_KEY])):
        inputs = tokenizer(examples[DATA_BODY_KEY][i],add_special_tokens=False)  # 输入文本转ID
        label = label_ids[examples[DATA_LABEL_KEY][i]]  # 标签
        input_ids = inputs["input_ids"] + [tokenizer.eos_token_id, label]  # 将输入和输出拼接，此处为输入+结束符+标签
        
        raw_len = len(input_ids)
        input_len = len(inputs["input_ids"]) + 1

        # 若长度超过最大长度，则截断，否则补齐
        if raw_len >= MAX_LEN:
            input_ids = input_ids[-MAX_LEN:]
            attention_mask = [1] * MAX_LEN
            labels = [-100]*(MAX_LEN - 1) + [label]
        else:
            input_ids = input_ids + [0] * (MAX_LEN - raw_len)
            attention_mask = [1] * raw_len + [tokenizer.pad_token_id] * (MAX_LEN - raw_len)
            labels = [-100]*input_len + [label] + [-100] * (MAX_LEN - raw_len)  # 只有输出参与loss计算，因此输入部分标记为-100
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    return model_inputs


# 处理训练数据集
tokenized_train_dataset = raw_train_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns,
    desc="Running tokenizer on train dataset",
)

# 处理验证数据集
tokenized_valid_dataset = raw_valid_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns,
    desc="Running tokenizer on validation dataset",
)

# ======================================6、数据校准器======================================
# 定义数据校准器（自动生成batch）
collater = DataCollatorWithPadding(
    tokenizer=tokenizer, return_tensors="pt",
)

# ======================================3、加载模型======================================
# 定义模型 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,trust_remote_code=True)


# 节省显存
model.gradient_checkpointing_enable()

# ======================================7、训练参数======================================
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./output",              # checkpoint保存路径
    evaluation_strategy="steps",        # 每N步做一次eval
    overwrite_output_dir=True,
    num_train_epochs=1,                 # 训练epoch数
    per_device_train_batch_size=BATCH_SIZE,     # 每张卡的batch大小
    gradient_accumulation_steps=1,              # 累加几个step做一次参数更新
    per_device_eval_batch_size=BATCH_SIZE,      # evaluation batch size
    logging_steps=INTERVAL,             # 每20步eval一次
    save_steps=INTERVAL,                # 每20步保存一个checkpoint
    learning_rate=LR,                   # 学习率
    warmup_ratio=WARMUP_RATIO,          # warmup比例
)


def compute_metric(eval_predictions):
    predictions, labels = eval_predictions

    label_indices = (labels != -100).nonzero()
    actual_labels = labels[label_indices]

    label_indices = (label_indices[0], label_indices[1]-1)
    selected_logits = predictions[label_indices]

    predicted_labels = selected_logits[:,label_ids].argmax(axis=-1)

    predicted_labels = np.array(label_ids)[predicted_labels]

    correct_predictions = (predicted_labels == actual_labels).sum()

    accuracy = correct_predictions / len(labels)

    return { "acc" : accuracy }

# ======================================8、训练器======================================
# 定义训练器
trainer = Trainer(
    model=model, # 待训练模型
    args=training_args, # 训练参数
    data_collator=collater, # 数据校准器
    train_dataset=tokenized_train_dataset,  # 训练集
    eval_dataset=tokenized_valid_dataset,   # 验证集
    compute_metrics=compute_metric,         # 计算自定义指标
)


# 开始训练
trainer.train()


# 可视化
# tensorboard --logdir=output/