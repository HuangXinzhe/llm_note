from datasets import load_dataset
from config import DATASET_NAME

# 加载数据集
raw_datasets = load_dataset(DATASET_NAME, cache_dir="/Volumes/WD_BLACK/data")

# 训练集
raw_train_dataset = raw_datasets["train"]

# 验证集
raw_valid_dataset = raw_datasets["validation"]


columns = raw_train_dataset.column_names
