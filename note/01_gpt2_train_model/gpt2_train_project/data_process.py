from config import MAX_LEN, DATA_BODY_KEY, DATA_LABEL_KEY
from model import tokenizer, label_ids
from datasets import raw_train_dataset, raw_valid_dataset, columns
from transformers import DataCollatorWithPadding


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
        inputs = tokenizer(examples[DATA_BODY_KEY]
                           [i], add_special_tokens=False)  # 输入文本转ID
        label = label_ids[examples[DATA_LABEL_KEY][i]]  # 标签
        # 将输入和输出拼接，此处为输入+结束符+标签
        input_ids = inputs["input_ids"] + [tokenizer.eos_token_id, label]

        raw_len = len(input_ids)
        input_len = len(inputs["input_ids"]) + 1

        # 若长度超过最大长度，则截断，否则补齐
        if raw_len >= MAX_LEN:
            input_ids = input_ids[-MAX_LEN:]
            attention_mask = [1] * MAX_LEN
            labels = [-100]*(MAX_LEN - 1) + [label]
        else:
            input_ids = input_ids + [0] * (MAX_LEN - raw_len)
            attention_mask = [1] * raw_len + \
                [tokenizer.pad_token_id] * (MAX_LEN - raw_len)
            labels = [-100]*input_len + [label] + [-100] * \
                (MAX_LEN - raw_len)  # 只有输出参与loss计算，因此输入部分标记为-100
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

# 定义数据校准器（自动生成batch）
collater = DataCollatorWithPadding(
    tokenizer=tokenizer, return_tensors="pt",
)
