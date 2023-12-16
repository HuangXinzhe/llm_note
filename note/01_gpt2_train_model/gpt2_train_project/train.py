from transformers import TrainingArguments
from config import LR, WARMUP_RATIO, BATCH_SIZE, INTERVAL
from model import model, label_ids
import numpy as np
from transformers import Trainer
from data_process import collater, tokenized_train_dataset, tokenized_valid_dataset

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

    predicted_labels = selected_logits[:, label_ids].argmax(axis=-1)

    predicted_labels = np.array(label_ids)[predicted_labels]

    correct_predictions = (predicted_labels == actual_labels).sum()

    accuracy = correct_predictions / len(labels)

    return {"acc": accuracy}


# 定义训练器
trainer = Trainer(
    model=model,  # 待训练模型
    args=training_args,  # 训练参数
    data_collator=collater,  # 数据校准器
    train_dataset=tokenized_train_dataset,  # 训练集
    eval_dataset=tokenized_valid_dataset,   # 验证集
    compute_metrics=compute_metric,         # 计算自定义指标
)


# 开始训练
trainer.train()


# 可视化
# 在output目录下执行
# tensorboard --logdir=output/
